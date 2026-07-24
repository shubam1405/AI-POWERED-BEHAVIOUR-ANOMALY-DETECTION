from dataclasses import dataclass

from models.enums import Browser, DeviceType, OperatingSystem


@dataclass
class Device:
    """
    Represents a physical or virtual device assigned to an employee.
    """

    device_uuid: str

    device_id: str          # Human-readable ID (e.g. DEV-0001)

    device_type: DeviceType

    operating_system: OperatingSystem

    browser: Browser

    owner_id: str           # Employee.employee_id

    trusted: bool = True

    def mark_compromised(self) -> None:
        """Marks the device as compromised."""
        self.trusted = False

    def mark_trusted(self) -> None:
        """Marks the device as trusted."""
        self.trusted = True

    def __str__(self) -> str:
        return (
            f"{self.device_id} "
            f"({self.device_type.value}, "
            f"{self.operating_system.value})"
        )