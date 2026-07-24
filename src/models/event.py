from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List

from models.enums import (
    AttackType,
    EventStatus,
    EventType,
    ResourceType,
)


@dataclass
class Event:
    """
    Represents a single user activity standard log in the enterprise.
    Supports both synthetic logs and Honeywell schema logs transparently.
    """

    event_uuid: str

    session_id: str

    timestamp: datetime

    event_type: EventType

    employee_id: str

    device_id: str

    resource: ResourceType | None = None

    status: EventStatus = EventStatus.SUCCESS

    attack_type: AttackType = AttackType.NONE

    risk_score: float = 0.0

    ip_address: str = ""

    is_anomalous: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------
    # Honeywell Schema Adaptation (Attributes and Properties)
    # -------------------------------------------------------------
    entity_type: str = "user"  # "user", "device", or "service account"

    def __post_init__(self):
        # Coerce string arguments to proper Enums
        if isinstance(self.resource, str) and self.resource:
            try:
                self.resource = ResourceType.from_str(self.resource)
            except ValueError:
                try:
                    self.resource = ResourceType(self.resource)
                except ValueError:
                    self.resource = None
                    
        if isinstance(self.event_type, str) and self.event_type:
            try:
                self.event_type = EventType.from_str(self.event_type)
            except ValueError:
                self.event_type = EventType(self.event_type)

        if isinstance(self.status, str) and self.status:
            try:
                self.status = EventStatus.from_str(self.status)
            except ValueError:
                self.status = EventStatus(self.status)

        if isinstance(self.attack_type, str) and self.attack_type:
            try:
                self.attack_type = AttackType.from_str(self.attack_type)
            except ValueError:
                self.attack_type = AttackType(self.attack_type)

    @property
    def entity_id(self) -> str:
        return self.employee_id

    @entity_id.setter
    def entity_id(self, val: str):
        self.employee_id = val

    @property
    def source_ip(self) -> str:
        return self.ip_address

    @source_ip.setter
    def source_ip(self, val: str):
        self.ip_address = val

    @property
    def geo_location(self) -> str:
        return self.metadata.get("geo_location", "Office")

    @geo_location.setter
    def geo_location(self, val: str):
        self.metadata["geo_location"] = val

    @property
    def resource_accessed(self) -> str:
        return self.resource.value if self.resource else ""

    @resource_accessed.setter
    def resource_accessed(self, val: str):
        if val:
            try:
                self.resource = ResourceType.from_str(val)
            except ValueError:
                self.resource = ResourceType(val)
        else:
            self.resource = None

    @property
    def auth_method(self) -> str:
        return self.metadata.get("auth_method", "Password")

    @auth_method.setter
    def auth_method(self, val: str):
        self.metadata["auth_method"] = val

    @property
    def session_duration(self) -> float:
        return self.metadata.get("session_duration", 0.0)

    @session_duration.setter
    def session_duration(self, val: float):
        self.metadata["session_duration"] = val

    @property
    def command_sequence(self) -> List[str]:
        return self.metadata.get("command_sequence", [])

    @command_sequence.setter
    def command_sequence(self, val: List[str]):
        self.metadata["command_sequence"] = val

    @property
    def device_fingerprint(self) -> str:
        return self.device_id

    @device_fingerprint.setter
    def device_fingerprint(self, val: str):
        self.device_id = val

    @property
    def label(self) -> int:
        return 1 if self.is_anomalous else 0

    @label.setter
    def label(self, val: int):
        self.is_anomalous = (val == 1)