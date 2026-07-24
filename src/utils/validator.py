import re
from typing import List

from config.config import OFFICE_LOCATIONS
from models.enums import (
    Browser,
    DeviceType,
    OperatingSystem,
    SensitivityLevel,
)
from simulator.company import Company


class ValidationError(Exception):
    """Raised when company validation fails."""

    def __init__(self, errors: List[str]):
        message = (
            "Company validation failed with the following errors:\n"
            + "\n".join(errors)
        )
        super().__init__(message)
        self.errors = errors


class CompanyValidator:
    """
    Validates the structural integrity of a generated company.
    """

    @staticmethod
    def is_valid_uuid(uuid_str: str) -> bool:
        uuid_regex = re.compile(
            r"^[0-9a-f]{8}-"
            r"[0-9a-f]{4}-"
            r"4[0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-"
            r"[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_regex.match(uuid_str))

    def validate(self, company: Company) -> bool:
        errors: List[str] = []

        # --------------------------------------------------
        # Employee Validation
        # --------------------------------------------------

        employee_ids = set()
        employee_uuids = set()

        for employee in company.employees.values():

            if employee.employee_id in employee_ids:
                errors.append(
                    f"Duplicate employee ID: {employee.employee_id}"
                )
            employee_ids.add(employee.employee_id)

            if employee.employee_uuid in employee_uuids:
                errors.append(
                    f"Duplicate employee UUID: {employee.employee_uuid}"
                )
            employee_uuids.add(employee.employee_uuid)

            if not self.is_valid_uuid(employee.employee_uuid):
                errors.append(
                    f"Invalid UUID: {employee.employee_uuid}"
                )

            if employee.office_location not in OFFICE_LOCATIONS:
                errors.append(
                    f"{employee.employee_id} has invalid office "
                    f"location '{employee.office_location}'"
                )

            if not isinstance(employee.preferred_browser, Browser):
                errors.append(
                    f"{employee.employee_id} has invalid browser"
                )

            if not isinstance(employee.preferred_os, OperatingSystem):
                errors.append(
                    f"{employee.employee_id} has invalid OS"
                )

            if employee.behavior_profile is None:
                errors.append(
                    f"{employee.employee_id} has no behavior profile"
                )

            if len(employee.allowed_devices) < 1:
                errors.append(
                    f"{employee.employee_id} has no assigned devices"
                )

            if len(employee.allowed_devices) > 3:
                errors.append(
                    f"{employee.employee_id} has more than 3 devices"
                )

        # --------------------------------------------------
        # Device Validation
        # --------------------------------------------------

        device_ids = set()
        device_uuids = set()

        for device in company.devices.values():

            if device.device_id in device_ids:
                errors.append(
                    f"Duplicate device ID: {device.device_id}"
                )
            device_ids.add(device.device_id)

            if device.device_uuid in device_uuids:
                errors.append(
                    f"Duplicate device UUID: {device.device_uuid}"
                )
            device_uuids.add(device.device_uuid)

            if not self.is_valid_uuid(device.device_uuid):
                errors.append(
                    f"Invalid UUID: {device.device_uuid}"
                )

            if not isinstance(device.device_type, DeviceType):
                errors.append(
                    f"{device.device_id} has invalid device type"
                )

            if not isinstance(device.operating_system, OperatingSystem):
                errors.append(
                    f"{device.device_id} has invalid operating system"
                )

            if not isinstance(device.browser, Browser):
                errors.append(
                    f"{device.device_id} has invalid browser"
                )

            owner = company.get_employee(device.owner_id)

            if owner is None:
                errors.append(
                    f"{device.device_id} references missing owner "
                    f"{device.owner_id}"
                )
            elif device.device_id not in owner.allowed_devices:
                errors.append(
                    f"{device.device_id} missing from "
                    f"{owner.employee_id}'s allowed_devices"
                )

        # --------------------------------------------------
        # Department Validation
        # --------------------------------------------------

        department_names = set(company.departments.keys())

        for employee in company.employees.values():

            if employee.department not in department_names:
                errors.append(
                    f"{employee.employee_id} belongs to "
                    f"unknown department"
                )

        # --------------------------------------------------
        # Resource Validation
        # --------------------------------------------------

        for resource in company.resources.values():

            if not isinstance(
                resource.sensitivity_level,
                SensitivityLevel,
            ):
                errors.append(
                    f"{resource.resource_name} has invalid "
                    f"sensitivity level"
                )

            if len(resource.allowed_departments) == 0:
                errors.append(
                    f"{resource.resource_name} has no "
                    f"allowed departments"
                )

            for department in resource.allowed_departments:

                if department not in department_names:
                    errors.append(
                        f"{resource.resource_name} references "
                        f"unknown department {department}"
                    )

        # --------------------------------------------------
        # Department Membership Validation
        # --------------------------------------------------

        for department in company.departments.values():

            for employee in department.employees:

                if employee.employee_id not in company.employees:
                    errors.append(
                        f"{department.department_id} contains "
                        f"unknown employee "
                        f"{employee.employee_id}"
                    )

            for resource in department.resources:

                if resource.resource_name not in company.resources:
                    errors.append(
                        f"{department.department_id} contains "
                        f"unknown resource "
                        f"{resource.resource_name}"
                    )

        # --------------------------------------------------
        # Finish
        # --------------------------------------------------

        if errors:
            raise ValidationError(errors)

        return True