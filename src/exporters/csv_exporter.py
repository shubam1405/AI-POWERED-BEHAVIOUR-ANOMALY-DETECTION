import csv
import json
import os

from simulator.company import Company


class CSVExporter:
    """
    Exports the in-memory Company representation into CSV files.
    """

    def __init__(self, company: Company, output_dir: str = "data/raw"):
        self.company = company
        self.output_dir = output_dir

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def export_all(self) -> None:
        self._ensure_output_dir()
        self.export_employees()
        self.export_devices()
        self.export_resources()

    # ----------------------------------------------------------
    # Employees
    # ----------------------------------------------------------

    def export_employees(self) -> None:

        filepath = os.path.join(self.output_dir, "employees.csv")

        headers = [
            "employee_uuid",
            "employee_id",
            "name",
            "department",
            "role",
            "office_location",
            "work_start_time",
            "work_end_time",
            "allowed_devices",
            "preferred_browser",
            "preferred_os",
            "vpn_probability",
            "access_level",
            "behavior_profile",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for emp in self.company.employees.values():

                bp = emp.behavior_profile

                start_hour, end_hour = bp.working_hours

                behavior_profile = {
                    "working_hours": list(bp.working_hours),
                    "login_hour_distribution": bp.login_hour_distribution,
                    "resource_probabilities": {
                        (resource.value if hasattr(resource, "value") else str(resource)): probability
                        for resource, probability in bp.resource_probabilities.items()
                    },
                    "browser_probabilities": {
                        (browser.value if hasattr(browser, "value") else str(browser)): probability
                        for browser, probability in bp.browser_probabilities.items()
                    },
                    "device_probabilities": {
                        (device.value if hasattr(device, "value") else str(device)): probability
                        for device, probability in bp.device_probabilities.items()
                    },
                    "average_session_length": bp.average_session_length,
                    "session_length_stddev": bp.session_length_stddev,
                    "average_actions": bp.average_actions,
                    "actions_stddev": bp.actions_stddev,
                    "average_sessions_per_day": bp.average_sessions_per_day,
                    "remote_work_probability": bp.remote_work_probability,
                }

                writer.writerow(
                    {
                        "employee_uuid": emp.employee_uuid,
                        "employee_id": emp.employee_id,
                        "name": emp.name,
                        "department": emp.department.value,
                        "role": emp.role,
                        "office_location": emp.office_location,
                        "work_start_time": f"{start_hour:02d}:00",
                        "work_end_time": f"{end_hour:02d}:00",
                        "allowed_devices": ";".join(emp.allowed_devices),
                        "preferred_browser": emp.preferred_browser.value,
                        "preferred_os": emp.preferred_os.value,
                        "vpn_probability": emp.vpn_probability,
                        "access_level": emp.access_level.value,
                        "behavior_profile": json.dumps(behavior_profile),
                    }
                )

    # ----------------------------------------------------------
    # Devices
    # ----------------------------------------------------------

    def export_devices(self) -> None:

        filepath = os.path.join(self.output_dir, "devices.csv")

        headers = [
            "device_uuid",
            "device_id",
            "device_type",
            "operating_system",
            "browser",
            "owner_id",
            "trusted",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for device in self.company.devices.values():

                writer.writerow(
                    {
                        "device_uuid": device.device_uuid,
                        "device_id": device.device_id,
                        "device_type": device.device_type.value,
                        "operating_system": device.operating_system.value,
                        "browser": device.browser.value,
                        "owner_id": device.owner_id,
                        "trusted": device.trusted,
                    }
                )

    # ----------------------------------------------------------
    # Resources
    # ----------------------------------------------------------

    def export_resources(self) -> None:

        filepath = os.path.join(self.output_dir, "resources.csv")

        headers = [
            "resource_name",
            "resource_type",
            "sensitivity_level",
            "allowed_departments",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for resource in self.company.resources.values():

                writer.writerow(
                    {
                        "resource_name": resource.resource_name.value,
                        "resource_type": resource.resource_type,
                        "sensitivity_level": resource.sensitivity_level.value,
                        "allowed_departments": ";".join(
                            department.value
                            for department in resource.allowed_departments
                        ),
                    }
                )