import random
from typing import Dict

from config.config import (
    RANDOM_SEED,
    DEPARTMENTS,
    DEFAULT_DEVICE_PROBABILITIES,
)

from generators.device_generator import DeviceGenerator
from generators.employee_generator import EmployeeGenerator
from generators.resource_generator import ResourceGenerator

from models.department import Department
from models.enums import DepartmentName, DeviceType

from simulator.company import Company


class EnterpriseGenerator:
    """
    Generates the complete enterprise.

    Generation Pipeline

    Departments
        ↓
    Resources
        ↓
    Employees
        ↓
    Devices
        ↓
    Behavior Profile Finalization
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)

    def generate(self) -> Company:

        company = Company()

        self._generate_departments(company)
        self._generate_resources(company)

        employees = self._generate_employees(company)

        self._generate_devices(company, employees)

        self._finalize_behavior_profiles(company, employees)

        return company

    # --------------------------------------------------
    # Department Generation
    # --------------------------------------------------

    def _generate_departments(self, company: Company) -> None:

        for dept in DEPARTMENTS:

            department = Department(
                department_id=f"DEP-{dept.name}",
                name=dept,
                policies=self._default_department_policy(dept),
            )

            company.add_department(department)

    def _default_department_policy(self, dept: DepartmentName) -> Dict:

        policies = {
            DepartmentName.FINANCE: {
                "working_hours": (9, 18),
                "preferred_resources": [
                    "Email",
                    "FinanceDB",
                    "Payroll",
                ],
            },
            DepartmentName.ENGINEERING: {
                "working_hours": (10, 19),
                "preferred_resources": [
                    "GitHub",
                    "Slack",
                    "FileServer",
                ],
            },
            DepartmentName.HR: {
                "working_hours": (9, 18),
                "preferred_resources": [
                    "HRPortal",
                    "Email",
                ],
            },
            DepartmentName.SALES: {
                "working_hours": (9, 18),
                "preferred_resources": [
                    "CRM",
                    "Email",
                ],
            },
            DepartmentName.IT: {
                "working_hours": (0, 23),
                "preferred_resources": [
                    "AdminConsole",
                    "FileServer",
                ],
            },
        }

        return policies.get(dept, {})

    # --------------------------------------------------
    # Resource Generation
    # --------------------------------------------------

    def _generate_resources(self, company: Company):

        ResourceGenerator(company).generate()

    # --------------------------------------------------
    # Employee Generation
    # --------------------------------------------------

    def _generate_employees(self, company: Company):

        generator = EmployeeGenerator(company, self.rng)

        return generator.generate_all(count=200)

    # --------------------------------------------------
    # Device Generation
    # --------------------------------------------------

    def _generate_devices(self, company: Company, employees):
        import uuid
        from models.device import Device
        from models.enums import DeviceType, OperatingSystem, Browser

        generator = DeviceGenerator(company, self.rng)

        for employee in employees:
            entity_type = employee.entity_type if hasattr(employee, "entity_type") else "user"
            if entity_type != "user":
                # Register pre-configured devices for service accounts and device agents
                for dev_id in employee.allowed_devices:
                    device = Device(
                        device_uuid=str(uuid.UUID(int=self.rng.getrandbits(128), version=4)),
                        device_id=dev_id,
                        device_type=DeviceType.DESKTOP if "SERVER" in dev_id else DeviceType.LAPTOP,
                        operating_system=OperatingSystem.LINUX,
                        browser=Browser.CHROME,
                        owner_id=employee.employee_id
                    )
                    company.add_device(device)
                continue
            generator.generate_for_employee(employee)

    # --------------------------------------------------
    # Behavior Profile Finalization
    # --------------------------------------------------

    def _finalize_behavior_profiles(self, company: Company, employees):

        for employee in employees:

            devices = company.get_devices(employee.employee_id)

            if not devices:

                employee.behavior_profile.device_probabilities = {}

                continue

            probabilities = {}

            if len(devices) == 1:

                probabilities[devices[0].device_type] = 1.0

            else:

                remaining = 1.0

                for i, device in enumerate(devices):

                    if i == len(devices) - 1:

                        probabilities[device.device_type] = remaining

                    else:

                        weight = DEFAULT_DEVICE_PROBABILITIES.get(
                            device.device_type,
                            0.20,
                        )

                        probabilities[device.device_type] = weight

                        remaining -= weight

            total = sum(probabilities.values())

            employee.behavior_profile.device_probabilities = {
                device: probability / total
                for device, probability in probabilities.items()
            }