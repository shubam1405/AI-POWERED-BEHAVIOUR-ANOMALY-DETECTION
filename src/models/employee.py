from dataclasses import dataclass, field
from typing import List

from models.behavior_profile import BehaviorProfile
from models.enums import (
    AccessLevel,
    Browser,
    DepartmentName,
    OperatingSystem,
)


@dataclass
class Employee:
    """
    Represents an employee within the simulated enterprise.
    """

    employee_uuid: str
    employee_id: str          # Human-readable ID (e.g. EMP-0001)

    name: str

    department: DepartmentName
    role: str

    office_location: str

    access_level: AccessLevel

    preferred_os: OperatingSystem
    preferred_browser: Browser

    vpn_probability: float = 0.0

    entity_type: str = "user"  # "user", "device", or "service account"

    # Store device IDs only.
    allowed_devices: List[str] = field(default_factory=list)

    # Every employee should always have a behavior profile.
    behavior_profile: BehaviorProfile = field(
        default_factory=BehaviorProfile
    )

    # ----------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------

    def is_during_working_hours(self, hour: int) -> bool:
        """
        Returns True if the supplied hour falls inside
        the employee's normal working hours.
        """

        start_hour, end_hour = self.behavior_profile.working_hours

        if start_hour <= end_hour:
            return start_hour <= hour < end_hour

        # Overnight shift
        return hour >= start_hour or hour < end_hour

    def has_device_access(self, device_id: str) -> bool:
        """Returns True if the employee owns the specified device."""

        return device_id in self.allowed_devices

    def add_device(self, device_id: str) -> None:
        """Registers a device with the employee."""

        if device_id not in self.allowed_devices:
            self.allowed_devices.append(device_id)

    @property
    def primary_device(self):
        """
        Returns the employee's primary device ID if available.
        """

        if not self.allowed_devices:
            return None

        return self.allowed_devices[0]