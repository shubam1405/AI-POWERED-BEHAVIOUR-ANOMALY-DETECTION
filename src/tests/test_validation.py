import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.enums import DepartmentName, DeviceType, OperatingSystem, Browser, AccessLevel
from models.employee import Employee
from models.device import Device
from models.behavior_profile import BehaviorProfile
from utils.validator import CompanyValidator, ValidationError
from generators.enterprise_generator import EnterpriseGenerator

class TestEnterpriseValidation(unittest.TestCase):
    def setUp(self):
        # Generate the standard virtual enterprise using seed
        self.generator = EnterpriseGenerator(seed=42)
        self.company = self.generator.generate()
        self.validator = CompanyValidator()

    def test_company_generation_integrity(self):
        """Verifies that the default generated virtual enterprise passes all validator rules."""
        self.assertTrue(self.validator.validate(self.company))
        self.assertEqual(len(self.company.employees), 200)
        self.assertEqual(len(self.company.departments), 6)
        self.assertEqual(len(self.company.resources), 24)

    def test_device_counts_per_employee(self):
        """Checks that each employee is assigned between 1 and 3 devices."""
        for emp in self.company.employees.values():
            allowed_count = len(emp.allowed_devices)
            self.assertTrue(1 <= allowed_count <= 3, f"Employee {emp.employee_id} has {allowed_count} devices.")

    def test_no_orphan_devices(self):
        """Asserts that all devices are owned by an active employee."""
        for dev in self.company.devices.values():
            owner = self.company.get_employee(dev.owner_id)
            self.assertIsNotNone(owner, f"Device {dev.device_id} has no valid owner.")
            self.assertIn(dev.device_id, owner.allowed_devices)

    def test_department_distributions(self):
        """Ensures that OS distributions match department rules (e.g. Finance has Windows/macOS, plus optional mobile OS)."""
        for dev in self.company.devices.values():
            owner = self.company.get_employee(dev.owner_id)
            if owner.department == DepartmentName.FINANCE:
                self.assertIn(dev.operating_system, [OperatingSystem.WINDOWS, OperatingSystem.MACOS, OperatingSystem.IOS, OperatingSystem.ANDROID])
            elif owner.department == DepartmentName.HR:
                self.assertIn(dev.operating_system, [OperatingSystem.WINDOWS, OperatingSystem.MACOS, OperatingSystem.IOS, OperatingSystem.ANDROID])

    def test_fails_on_orphan_device(self):
        """Verifies the validator raises ValidationError when a device owner does not exist."""
        bad_device = Device(
            device_uuid="3dfde20c-c9e8-466a-b248-d3f3b9cde123",
            device_id="DEV-BAD99",
            device_type=DeviceType.LAPTOP,
            operating_system=OperatingSystem.WINDOWS,
            browser=Browser.CHROME,
            owner_id="EMP-NONEXISTENT",
            trusted=True
        )
        self.company.add_device(bad_device)
        with self.assertRaises(ValidationError) as context:
            self.validator.validate(self.company)
        self.assertIn("references missing owner", str(context.exception))

    def test_fails_on_duplicate_ids(self):
        """Verifies the validator raises ValidationError when duplicate employee IDs are present."""
        dup_employee = Employee(
            employee_uuid="4dfde20c-c9e8-466a-b248-d3f3b9cde124",
            employee_id="EMP-0001",  # Duplicate of an existing employee id
            name="John Clone",
            department=DepartmentName.HR,
            role="Recruiter",
            office_location="London",
            allowed_devices=["DEV-0001"],
            preferred_browser=Browser.CHROME,
            preferred_os=OperatingSystem.WINDOWS,
            vpn_probability=0.05,
            access_level=AccessLevel.LOW,
            behavior_profile=BehaviorProfile(working_hours=(9, 17))
        )
        # Force insert duplicate employee key in the lookup map
        self.company.employees["EMP-DUP-KEY"] = dup_employee
        with self.assertRaises(ValidationError) as context:
            self.validator.validate(self.company)
        self.assertIn("Duplicate employee ID", str(context.exception))

if __name__ == '__main__':
    unittest.main()
