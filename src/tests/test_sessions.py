import unittest
import random
from datetime import datetime, timedelta
import sys
import os

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.enums import DepartmentName, Browser, OperatingSystem, AttackType, SessionStatus, LoginMethod, DeviceType, AccessLevel
from models.employee import Employee
from models.device import Device
from models.session import Session
from models.behavior_profile import BehaviorProfile
from simulator.company import Company
from utils.id_generator import IdGenerator
from simulator.session_scheduler import SessionScheduler
from simulator.session_generator import SessionGenerator
from validators.session_validator import SessionValidator, SessionValidationError

class TestSessionSimulation(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        
        # Setup dummy company
        self.company = Company()
        
        # Add a department
        from models.department import Department
        self.hr_dept = Department(department_id="DEP-HR", name=DepartmentName.HR)
        self.company.add_department(self.hr_dept)
        
        # Add employee
        self.profile = BehaviorProfile(
            working_hours=(9, 17),
            login_hour_distribution={8: 0.1, 9: 0.8, 10: 0.1},
            device_probabilities={DeviceType.LAPTOP: 1.0},
            browser_probabilities={Browser.CHROME: 1.0},
            average_sessions_per_day=2.0,
            average_session_length=30.0,
            session_length_stddev=5.0,
            average_actions=10,
            actions_stddev=2,
            remote_work_probability=0.0
        )
        
        self.employee = Employee(
            employee_uuid="11111111-1111-4111-a111-111111111111",
            employee_id="EMP-0001",
            name="HR Employee",
            department=DepartmentName.HR,
            role="Recruiter",
            office_location="New York",
            access_level=AccessLevel.from_str("Low") if hasattr(AccessLevel, "from_str") else AccessLevel.LOW,
            preferred_os=OperatingSystem.WINDOWS,
            preferred_browser=Browser.CHROME,
            allowed_devices=["DEV-0001"],
            behavior_profile=self.profile
        )
        self.company.add_employee(self.employee)
        
        # Add device
        self.device = Device(
            device_uuid="22222222-2222-4222-b222-222222222222",
            device_id="DEV-0001",
            device_type=DeviceType.LAPTOP,
            operating_system=OperatingSystem.WINDOWS,
            browser=Browser.CHROME,
            owner_id="EMP-0001",
            trusted=True
        )
        self.company.add_device(self.device)

    def test_id_generator(self):
        """Verifies IdGenerator produces correctly formatted sequenced IDs."""
        id_gen = IdGenerator(prefix="SES-", start=1, digits=6)
        self.assertEqual(id_gen.next_id(), "SES-000001")
        self.assertEqual(id_gen.next_id(), "SES-000002")

        id_gen_custom = IdGenerator(prefix="TST-", start=99, digits=3)
        self.assertEqual(id_gen_custom.next_id(), "TST-099")
        self.assertEqual(id_gen_custom.next_id(), "TST-100")

    def test_session_scheduler_validity(self):
        """Verifies SessionScheduler schedules sessions without overlaps and within work hours buffer."""
        start_date = datetime(2026, 6, 1, 9, 0, 0)
        scheduler = SessionScheduler(self.company, start_date, days=5, rng=self.rng)
        slots = scheduler.schedule_sessions()
        
        self.assertTrue(len(slots) > 0)
        
        # Check no overlaps
        slots_by_emp = {}
        for emp, start, end in slots:
            slots_by_emp.setdefault(emp.employee_id, []).append((start, end))
            
        for emp_id, emp_slots in slots_by_emp.items():
            for i in range(len(emp_slots) - 1):
                s1, e1 = emp_slots[i]
                s2, e2 = emp_slots[i+1]
                self.assertTrue(s1 <= s2)
                self.assertTrue(e1 <= s2, f"Overlapping slots: {s1}-{e1} and {s2}-{e2}")

    def test_session_validator_success(self):
        """Verifies SessionValidator validates a correct list of sessions successfully."""
        start_time = datetime(2026, 6, 1, 9, 30, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        duration = (end_time - start_time).total_seconds()
        
        session = Session(
            session_uuid="99999999-9999-4999-a999-999999999999",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            day_of_week=start_time.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = SessionValidator()
        self.assertTrue(validator.validate([session], self.company))

    def test_session_validator_detects_overlap(self):
        """Verifies SessionValidator raises SessionValidationError on overlapping sessions."""
        t1_start = datetime(2026, 6, 1, 9, 0, 0)
        t1_end = datetime(2026, 6, 1, 10, 0, 0)
        
        t2_start = datetime(2026, 6, 1, 9, 30, 0) # Overlap
        t2_end = datetime(2026, 6, 1, 10, 30, 0)
        
        s1 = Session(
            session_uuid="99999999-9999-4999-a999-999999999991",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=t1_start,
            end_time=t1_end,
            duration_seconds=(t1_end - t1_start).total_seconds(),
            day_of_week=t1_start.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        s2 = Session(
            session_uuid="99999999-9999-4999-a999-999999999992",
            session_id="SES-000002",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=t2_start,
            end_time=t2_end,
            duration_seconds=(t2_end - t2_start).total_seconds(),
            day_of_week=t2_start.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = SessionValidator()
        with self.assertRaises(SessionValidationError) as ctx:
            validator.validate([s1, s2], self.company)
            
        self.assertIn("overlapping sessions detected", str(ctx.exception))

    def test_session_validator_detects_invalid_duration(self):
        """Verifies SessionValidator raises SessionValidationError on mismatched duration_seconds."""
        start_time = datetime(2026, 6, 1, 9, 30, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        duration = 5.0 # Mismatched: should be 1800.0
        
        session = Session(
            session_uuid="99999999-9999-4999-a999-999999999999",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            day_of_week=start_time.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = SessionValidator()
        with self.assertRaises(SessionValidationError) as ctx:
            validator.validate([session], self.company)
            
        self.assertIn("duration_seconds mismatch", str(ctx.exception))

    def test_session_validator_detects_unowned_device(self):
        """Verifies SessionValidator raises SessionValidationError when session device is not owned by the employee."""
        start_time = datetime(2026, 6, 1, 9, 30, 0)
        end_time = datetime(2026, 6, 1, 10, 0, 0)
        duration = (end_time - start_time).total_seconds()
        
        session = Session(
            session_uuid="99999999-9999-4999-a999-999999999999",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id="DEV-FOREIGN", # Not owned
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            day_of_week=start_time.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = SessionValidator()
        with self.assertRaises(SessionValidationError) as ctx:
            validator.validate([session], self.company)
            
        self.assertIn("is not owned by Employee", str(ctx.exception))

    def test_session_validator_detects_working_hours_violation(self):
        """Verifies SessionValidator raises SessionValidationError on shift hours violations."""
        start_time = datetime(2026, 6, 1, 1, 0, 0) # 1 AM (Day shift employee works 9-17)
        end_time = datetime(2026, 6, 1, 2, 0, 0)
        duration = (end_time - start_time).total_seconds()
        
        session = Session(
            session_uuid="99999999-9999-4999-a999-999999999999",
            session_id="SES-000001",
            employee_uuid=self.employee.employee_uuid,
            employee_id=self.employee.employee_id,
            department=self.employee.department,
            role=self.employee.role,
            device_id=self.device.device_id,
            browser=Browser.CHROME,
            operating_system=OperatingSystem.WINDOWS,
            office_location=self.employee.office_location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            day_of_week=start_time.strftime("%A"),
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )
        
        validator = SessionValidator()
        with self.assertRaises(SessionValidationError) as ctx:
            validator.validate([session], self.company)
            
        self.assertIn("violates working hours", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
