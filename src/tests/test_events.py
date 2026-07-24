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
from generators.event_generator import EventGenerator
from validators.event_validator import EventValidator, EventValidationError

class TestEventGeneration(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.company = Company()
        
        # Setup dummy department
        from models.department import Department
        self.eng_dept = Department(department_id="DEP-ENG", name=DepartmentName.ENGINEERING)
        self.company.add_department(self.eng_dept)
        
        # Setup dummy employee
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
            name="Eng Dev",
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

    def test_event_generator_office_session(self):
        """Verifies event generation for an office (local) session."""
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
        
        generator = EventGenerator(self.company, self.rng)
        events = generator.generate_events([session])
        
        self.assertTrue(len(events) >= 6)  # LOGIN, NETWORK_CONNECT, at least 2 actions, NETWORK_DISCONNECT, LOGOUT
        self.assertEqual(session.events[0].event_type, EventType.LOGIN)
        self.assertEqual(session.events[1].event_type, EventType.NETWORK_CONNECT)
        self.assertEqual(session.events[-2].event_type, EventType.NETWORK_DISCONNECT)
        self.assertEqual(session.events[-1].event_type, EventType.LOGOUT)
        
        # Check IP address is in office subnet 10.10.x.y
        for ev in events:
            self.assertTrue(ev.ip_address.startswith("10.10."))
            
        # Verify chronological order and unique timestamps
        timestamps = [ev.timestamp for ev in events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(timestamps), len(set(timestamps)))

    def test_event_generator_remote_session(self):
        """Verifies event generation for a remote VPN session."""
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
            login_method=LoginMethod.VPN,
            remote_session=True,
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
        
        generator = EventGenerator(self.company, self.rng)
        events = generator.generate_events([session])
        
        self.assertTrue(len(events) >= 6)
        self.assertEqual(session.events[0].event_type, EventType.LOGIN)
        self.assertEqual(session.events[1].event_type, EventType.VPN_CONNECT)
        self.assertEqual(session.events[-2].event_type, EventType.VPN_DISCONNECT)
        self.assertEqual(session.events[-1].event_type, EventType.LOGOUT)
        
        # Check IP address is in VPN subnet 172.16.x.y
        for ev in events:
            self.assertTrue(ev.ip_address.startswith("172.16."))

    def test_event_validator_catches_violations(self):
        """Verifies that EventValidator raises errors on timestamp and boundary violations."""
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
        
        # Create incorrect events list
        ev1 = Event(
            event_uuid="88888888-8888-4888-a888-888888888881",
            session_id=session.session_id,
            timestamp=start_time,
            event_type=EventType.RESOURCE_ACCESS,  # Violates LOGIN first rule
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=ResourceType.GITHUB,
            status=EventStatus.SUCCESS,
            attack_type=AttackType.NONE,
            risk_score=0.0,
            ip_address="10.10.1.1",
            metadata={}
        )
        
        ev2 = Event(
            event_uuid="88888888-8888-4888-a888-888888888882",
            session_id=session.session_id,
            timestamp=start_time - timedelta(minutes=10),  # Violates start_time bounds
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
        
        session.events = [ev1, ev2]
        
        validator = EventValidator()
        with self.assertRaises(EventValidationError) as ctx:
            validator.validate([session])
            
        self.assertIn("expected LOGIN", str(ctx.exception))
        self.assertIn("falls outside session boundaries", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
