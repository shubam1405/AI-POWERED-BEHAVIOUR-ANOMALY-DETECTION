import unittest
import random
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
from simulator.anomaly_injector import AnomalyInjector
from validators.anomaly_validator import AnomalyValidator, AnomalyValidationError

class TestAnomalyInjection(unittest.TestCase):
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
            remote_work_probability=0.5,
            resource_probabilities={
                ResourceType.GITHUB: 0.7,
                ResourceType.JIRA: 0.3
            }
        )
        
        self.employee = Employee(
            employee_uuid="11111111-1111-4111-a111-111111111111",
            employee_id="EMP-0001",
            name="Dev Engineer",
            department=DepartmentName.ENGINEERING,
            role="Software Engineer",
            office_location="San Francisco",
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

    def test_brute_force_injection(self):
        """Verifies brute force failed logins prepend, timestamp shifts, and risk score calculation."""
        start_time = datetime(2026, 6, 1, 9, 0, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        
        session = Session(
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
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            day_of_week="Monday",
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        # Pre-populate with standard events
        ev_login = Event(
            event_uuid="88888888-8888-4888-a888-888888888881",
            session_id=session.session_id,
            timestamp=start_time,
            event_type=EventType.LOGIN,
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=None,
            status=EventStatus.SUCCESS,
            attack_type=AttackType.NONE,
            risk_score=0.0,
            ip_address="10.10.1.1",
            metadata={}
        )
        ev_logout = Event(
            event_uuid="88888888-8888-4888-a888-888888888882",
            session_id=session.session_id,
            timestamp=end_time,
            event_type=EventType.LOGOUT,
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=None,
            status=EventStatus.SUCCESS,
            attack_type=AttackType.NONE,
            risk_score=0.0,
            ip_address="10.10.1.1",
            metadata={}
        )
        session.events = [ev_login, ev_logout]

        injector = AnomalyInjector(self.company, self.rng)
        injector._apply_attack_stage(session, AttackType.BRUTE_FORCE)
        
        # We expect failed logins prepended
        self.assertTrue(len(session.events) > 2)
        failed_logins = [e for e in session.events if e.status == EventStatus.FAILED]
        self.assertTrue(5 <= len(failed_logins) <= 10)
        
        # Ensure session boundaries shifted earlier
        self.assertTrue(session.start_time < start_time)
        self.assertEqual(session.events[0].timestamp, session.start_time)
        
        # Verify chronological sequence
        timestamps = [e.timestamp for e in session.events]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_rule_based_risk_score(self):
        """Verifies session risk score combines base severity and modifiers correctly."""
        start_time = datetime(2026, 6, 1, 9, 0, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        
        session = Session(
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
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            day_of_week="Monday",
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        # Add basic LOGIN and LOGOUT
        ev_login = Event(
            event_uuid="88888888-8888-4888-a888-888888888881",
            session_id=session.session_id,
            timestamp=start_time,
            event_type=EventType.LOGIN,
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=None,
            status=EventStatus.SUCCESS,
            attack_type=AttackType.NONE,
            risk_score=0.0,
            ip_address="10.10.1.1",
            metadata={}
        )
        ev_logout = Event(
            event_uuid="88888888-8888-4888-a888-888888888882",
            session_id=session.session_id,
            timestamp=end_time,
            event_type=EventType.LOGOUT,
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=None,
            status=EventStatus.SUCCESS,
            attack_type=AttackType.NONE,
            risk_score=0.0,
            ip_address="10.10.1.1",
            metadata={}
        )
        session.events = [ev_login, ev_logout]

        injector = AnomalyInjector(self.company, self.rng)
        
        # 1. Apply BRUTE_FORCE stage
        injector._apply_attack_stage(session, AttackType.BRUTE_FORCE)
        
        # 2. Add privilege escalation commands (sudo su)
        injector._apply_attack_stage(session, AttackType.PRIVILEGE_ESCALATION)
        
        session.attack_type = AttackType.PRIVILEGE_ESCALATION
        session.is_anomalous = True
        
        calculated_score = injector._compute_session_risk(session)
        
        # Base severity of PRIVILEGE_ESCALATION: 95.0
        # + failed logins (5.0 each, e.g. 5 failures = +25.0)
        # + unauthorized admin access (non-IT access SERVERS/ADMIN_CONSOLE = +15.0)
        # Total score should clamp to 100.0
        self.assertEqual(calculated_score, 100.0)

    def test_validator_fails_on_mismatches(self):
        """Verifies AnomalyValidator detects mock violations like label or score mismatches."""
        start_time = datetime(2026, 6, 1, 9, 0, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        
        session = Session(
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
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            day_of_week="Monday",
            risk_score=95.0,
            is_anomalous=True,
            attack_type=AttackType.NONE,  # Mismatch: is_anomalous=True but attack_type=NONE
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = AnomalyValidator()
        with self.assertRaises(AnomalyValidationError) as ctx:
            validator.validate([session], self.company)
            
        self.assertIn("Marked anomalous but attack_type is NONE", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
