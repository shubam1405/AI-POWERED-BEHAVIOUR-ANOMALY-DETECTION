from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from models.event import Event
from models.enums import (
    DepartmentName,
    Browser,
    OperatingSystem,
    AttackType,
    SessionStatus,
    LoginMethod,
)

@dataclass
class Session:
    """
    Represents a user session consisting of an ordered sequence of events.
    """

    session_uuid: str
    session_id: str  # Human-readable ID (e.g. SES-000001)
    
    employee_uuid: str
    employee_id: str  # Human-readable ID (e.g. EMP-0001)
    
    department: DepartmentName
    role: str
    
    device_id: str
    browser: Browser
    operating_system: OperatingSystem
    office_location: str
    
    login_method: LoginMethod
    remote_session: bool
    
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    day_of_week: str

    risk_score: float = 0.0
    is_anomalous: bool = False
    attack_type: AttackType = AttackType.NONE
    session_status: SessionStatus = SessionStatus.COMPLETED
    
    # Store events in the session. Left empty for Phase 2A.
    events: List[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        """Adds an event to the session."""
        self.events.append(event)

    @property
    def event_count(self) -> int:
        """Returns the number of events in the session."""
        return len(self.events)