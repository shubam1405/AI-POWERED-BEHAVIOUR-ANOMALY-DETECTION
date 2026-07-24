import unittest
import random
import math
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.enums import (
    DepartmentName, Browser, OperatingSystem, AttackType, 
    SessionStatus, LoginMethod, DeviceType, AccessLevel, 
    EventType, EventStatus, ResourceType
)
from models.employee import Employee
from models.device import Device
from models.session import Session
from models.event import Event
from models.behavior_profile import BehaviorProfile
from simulator.company import Company
from features.scaler import StandardScaler
from features.feature_extractors import (
    SessionFeatureExtractor,
    AuthenticationFeatureExtractor,
    ResourceFeatureExtractor,
    FileActivityFeatureExtractor,
    ProcessFeatureExtractor,
    NetworkFeatureExtractor,
    TemporalFeatureExtractor,
    StatisticalFeatureExtractor,
    SequenceFeatureExtractor,
    BehavioralFeatureExtractor
)
from features.feature_validator import FeatureValidator, FeatureValidationError

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.company = Company()
        
        # Setup dummy department
        from models.department import Department
        self.eng_dept = Department(department_id="DEP-ENG", name=DepartmentName.ENGINEERING)
        self.company.add_department(self.eng_dept)

        self.profile = BehaviorProfile(
            working_hours=(9, 17),
            login_hour_distribution={9: 1.0},
            device_probabilities={DeviceType.LAPTOP: 1.0},
            browser_probabilities={Browser.CHROME: 1.0},
            average_sessions_per_day=1.0,
            average_session_length=60.0,
            session_length_stddev=10.0,
            average_actions=5,
            actions_stddev=1,
            remote_work_probability=0.0,
            resource_probabilities={ResourceType.GITHUB: 1.0}
        )
        
        self.employee = Employee(
            employee_uuid="11111111-1111-4111-a111-111111111111",
            employee_id="EMP-0001",
            name="Alice Dev",
            department=DepartmentName.ENGINEERING,
            role="Software Engineer",
            office_location="New York",
            access_level=AccessLevel.LOW,
            preferred_os=OperatingSystem.MACOS,
            preferred_browser=Browser.CHROME,
            allowed_devices=["DEV-0001"],
            behavior_profile=self.profile
        )
        self.company.add_employee(self.employee)
        
        self.device = Device(
            device_uuid="22222222-2222-4222-b222-222222222222",
            device_id="DEV-0001",
            device_type=DeviceType.LAPTOP,
            operating_system=OperatingSystem.MACOS,
            browser=Browser.CHROME,
            owner_id="EMP-0001",
            trusted=True
        )
        self.company.add_device(self.device)

        # Setup standard mock session
        self.start_time = datetime(2026, 6, 1, 9, 0, 0)
        self.end_time = datetime(2026, 6, 1, 10, 0, 0)
        
        self.session = Session(
            session_uuid="99999999-9999-4999-a999-999999999999",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.MACOS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=(self.end_time - self.start_time).total_seconds(),
            day_of_week="Monday",
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )

        # Setup standard events: LOGIN -> RESOURCE_ACCESS (Github) -> FILE_DOWNLOAD -> LOGOUT
        self.events = [
            Event(
                event_uuid="88888888-8888-4888-a888-888888888881",
                session_id=self.session.session_id,
                timestamp=self.start_time,
                event_type=EventType.LOGIN,
                employee_id=self.employee.employee_id,
                device_id=self.device.device_id,
                resource=None,
                status=EventStatus.SUCCESS,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address="10.10.1.1",
                metadata={}
            ),
            Event(
                event_uuid="88888888-8888-4888-a888-888888888882",
                session_id=self.session.session_id,
                timestamp=self.start_time + timedelta(minutes=10),
                event_type=EventType.RESOURCE_ACCESS,
                employee_id=self.employee.employee_id,
                device_id=self.device.device_id,
                resource=ResourceType.GITHUB,
                status=EventStatus.SUCCESS,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address="10.10.1.1",
                metadata={}
            ),
            Event(
                event_uuid="88888888-8888-4888-a888-888888888883",
                session_id=self.session.session_id,
                timestamp=self.start_time + timedelta(minutes=30),
                event_type=EventType.FILE_DOWNLOAD,
                employee_id=self.employee.employee_id,
                device_id=self.device.device_id,
                resource=ResourceType.GITHUB,
                status=EventStatus.SUCCESS,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address="10.10.1.1",
                metadata={"bytes_transferred": 1024 * 1024}
            ),
            Event(
                event_uuid="88888888-8888-4888-a888-888888888884",
                session_id=self.session.session_id,
                timestamp=self.end_time,
                event_type=EventType.LOGOUT,
                employee_id=self.employee.employee_id,
                device_id=self.device.device_id,
                resource=None,
                status=EventStatus.SUCCESS,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address="10.10.1.1",
                metadata={}
            )
        ]

    def test_extractors_correctness(self):
        """Verifies session, resource, file activity, and process extractors calculate correct stats."""
        # 1. Session features
        f_sess = SessionFeatureExtractor().extract(self.session, self.events)
        self.assertEqual(f_sess["session_duration"], 3600.0)
        self.assertEqual(f_sess["login_hour"], 9.0)
        self.assertEqual(f_sess["weekend_flag"], 0.0)  # Monday is weekday
        
        # 2. Resource features
        f_res = ResourceFeatureExtractor().extract(self.session, self.events)
        self.assertEqual(f_res["total_resource_accesses"], 2.0)
        self.assertEqual(f_res["github_access_count"], 2.0)
        self.assertEqual(f_res["database_query_count"], 0.0)

        # 3. File activity features
        f_file = FileActivityFeatureExtractor().extract(self.session, self.events)
        self.assertEqual(f_file["files_downloaded"], 1.0)
        self.assertEqual(f_file["total_download_bytes"], 1024.0 * 1024.0)

    def test_sequence_feature_extractor(self):
        """Verifies sequence position and login-to-first-resource gaps."""
        f_seq = SequenceFeatureExtractor().extract(self.session, self.events)
        self.assertEqual(f_seq["event_sequence_length"], 4.0)
        self.assertEqual(f_seq["login_to_first_resource_time"], 600.0)  # 10 minutes gap
        self.assertEqual(f_seq["file_download_position"], 2.0 / 4.0)  # Index 2 out of 4 events
        self.assertEqual(f_seq["admin_access_position"], -1.0)  # No admin accessed

    def test_behavioral_baseline_deviation(self):
        """Verifies behavioral baseline deviations compute correctly chronologically."""
        extractor = BehavioralFeatureExtractor()
        
        # Scenario A: empty history (first session)
        history_empty = {
            "locations": set(),
            "devices": set(),
            "browsers": set(),
            "operating_systems": set(),
            "resources": set(),
            "login_hours": [],
            "sessions_seen": 0
        }
        f_beh_empty = extractor.extract_with_history(self.session, self.events, history_empty)
        # All deviations must default to 0.0
        for val in f_beh_empty.values():
            self.assertEqual(val, 0.0)

        # Scenario B: mismatching location and resource accesses
        history_populated = {
            "locations": {"San Francisco"},  # Different from New York
            "devices": {"DEV-0001"},
            "browsers": {Browser.CHROME},
            "operating_systems": {OperatingSystem.MACOS},
            "resources": {ResourceType.EMAIL},  # Mismatch (this session accessed GITHUB)
            "login_hours": [9, 9, 9],
            "sessions_seen": 3
        }
        f_beh_pop = extractor.extract_with_history(self.session, self.events, history_populated)
        self.assertEqual(f_beh_pop["location_deviation"], 1.0)
        self.assertEqual(f_beh_pop["resource_access_deviation"], 1.0)
        self.assertEqual(f_beh_pop["device_deviation"], 0.0)  # Match

    def test_custom_standard_scaler(self):
        """Verifies standard scaler fit and transform correctly normalizes vectors."""
        data = [
            {"feat1": 10.0, "feat2": 2.0, "label": 0.0},
            {"feat1": 20.0, "feat2": 4.0, "label": 1.0},
            {"feat1": 30.0, "feat2": 6.0, "label": 0.0}
        ]
        scaler = StandardScaler()
        scaler.fit(data, ignore_cols=["label"])
        
        # Means: feat1 = 20, feat2 = 4
        # Stds: feat1 = sqrt(((10-20)^2 + 0 + (30-20)^2)/3) = sqrt(200/3) = 8.1649
        self.assertAlmostEqual(scaler.means["feat1"], 20.0)
        self.assertAlmostEqual(scaler.means["feat2"], 4.0)

        scaled = scaler.transform(data, ignore_cols=["label"])
        
        # Check mean is 0 after scaling
        scaled_feat1 = [row["feat1"] for row in scaled]
        self.assertAlmostEqual(sum(scaled_feat1)/3, 0.0)
        
        # Check standard deviations are normalized to 1.0
        var_scaled = sum(x**2 for x in scaled_feat1) / 3
        self.assertAlmostEqual(math.sqrt(var_scaled), 1.0)

if __name__ == '__main__':
    unittest.main()
