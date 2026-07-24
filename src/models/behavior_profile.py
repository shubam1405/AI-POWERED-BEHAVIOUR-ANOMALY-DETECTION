from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from uuid import UUID

from models.enums import Browser, ResourceType, OperatingSystem

@dataclass
class TemporalBehavior:
    """Stores patterns regarding active hours and weekday vs weekend properties."""
    working_hours: Tuple[int, int] = (9, 17)
    login_hour_distribution: Dict[int, float] = field(default_factory=dict)
    weekend_probability: float = 0.05

@dataclass
class AuthenticationBehavior:
    """Stores authentication methods, attempts, and success rates."""
    success_rate: float = 0.99
    preferred_methods: Dict[str, float] = field(default_factory=dict)

@dataclass
class ResourceBehavior:
    """Stores resource type counts and probability parameters."""
    resource_probabilities: Dict[str, float] = field(default_factory=dict)
    admin_access_probability: float = 0.01
    sensitive_access_probability: float = 0.05

@dataclass
class DeviceBehavior:
    """Stores device identifiers and browser/OS preferences."""
    device_probabilities: Dict[str, float] = field(default_factory=dict)
    primary_device_id: str = ""
    preferred_browsers: Dict[Browser, float] = field(default_factory=dict)
    preferred_os: Dict[OperatingSystem, float] = field(default_factory=dict)

@dataclass
class LocationBehavior:
    """Stores office location distributions and remote work ratios."""
    office_locations: Dict[str, float] = field(default_factory=dict)
    remote_work_probability: float = 0.25

@dataclass
class NetworkBehavior:
    """Stores networking subnets and connection rates."""
    office_subnet: str = "10.10.0.0/16"
    vpn_subnet: str = "172.16.0.0/16"
    average_external_connections: float = 2.0
    beacon_probability: float = 0.0

@dataclass
class CommandBehavior:
    """Stores habitual command sequences and command execution probability."""
    habitual_commands: List[str] = field(default_factory=list)
    powershell_probability: float = 0.05
    suspicious_command_probability: float = 0.001

@dataclass
class SessionBehavior:
    """Stores session lengths and average counts."""
    average_session_length: float = 60.0
    session_length_stddev: float = 10.0
    average_actions: int = 15
    actions_stddev: float = 3.0
    average_sessions_per_day: float = 1.5


@dataclass(init=False)
class BehaviorKnowledgeBase:
    """
    Stores the learned and default behavioral baseline properties of an entity.
    Reorganizes behavior profile statistics into 8 modular dimensions.
    """
    temporal: TemporalBehavior
    authentication: AuthenticationBehavior
    resource: ResourceBehavior
    device: DeviceBehavior
    location: LocationBehavior
    network: NetworkBehavior
    command: CommandBehavior
    session: SessionBehavior

    def __init__(
        self,
        temporal=None,
        authentication=None,
        resource=None,
        device=None,
        location=None,
        network=None,
        command=None,
        session=None,
        # Flat arguments for backward compatibility
        working_hours=None,
        login_hour_distribution=None,
        resource_probabilities=None,
        browser_probabilities=None,
        device_probabilities=None,
        average_session_length=None,
        session_length_stddev=None,
        average_actions=None,
        actions_stddev=None,
        average_sessions_per_day=None,
        remote_work_probability=None,
        **kwargs
    ):
        self.temporal = temporal or TemporalBehavior()
        self.authentication = authentication or AuthenticationBehavior()
        self.resource = resource or ResourceBehavior()
        self.device = device or DeviceBehavior()
        self.location = location or LocationBehavior()
        self.network = network or NetworkBehavior()
        self.command = command or CommandBehavior()
        self.session = session or SessionBehavior()

        if working_hours is not None:
            self.temporal.working_hours = working_hours
        if login_hour_distribution is not None:
            self.temporal.login_hour_distribution = login_hour_distribution
        if resource_probabilities is not None:
            self.resource.resource_probabilities = resource_probabilities
        if browser_probabilities is not None:
            self.device.preferred_browsers = browser_probabilities
        if device_probabilities is not None:
            self.device.device_probabilities = device_probabilities
        if average_session_length is not None:
            self.session.average_session_length = average_session_length
        if session_length_stddev is not None:
            self.session.session_length_stddev = session_length_stddev
        if average_actions is not None:
            self.session.average_actions = average_actions
        if actions_stddev is not None:
            self.session.actions_stddev = actions_stddev
        if average_sessions_per_day is not None:
            self.session.average_sessions_per_day = average_sessions_per_day
        if remote_work_probability is not None:
            self.location.remote_work_probability = remote_work_probability

    # -------------------------------------------------------------
    # Property Mappings for Backward Compatibility
    # -------------------------------------------------------------
    @property
    def working_hours(self) -> Tuple[int, int]:
        return self.temporal.working_hours
        
    @working_hours.setter
    def working_hours(self, val: Tuple[int, int]):
        self.temporal.working_hours = val

    @property
    def login_hour_distribution(self) -> Dict[int, float]:
        return self.temporal.login_hour_distribution

    @login_hour_distribution.setter
    def login_hour_distribution(self, val: Dict[int, float]):
        self.temporal.login_hour_distribution = val

    @property
    def resource_probabilities(self) -> Dict[str, float]:
        return self.resource.resource_probabilities

    @resource_probabilities.setter
    def resource_probabilities(self, val: Dict[str, float]):
        self.resource.resource_probabilities = val

    @property
    def browser_probabilities(self) -> Dict[Browser, float]:
        return self.device.preferred_browsers

    @browser_probabilities.setter
    def browser_probabilities(self, val: Dict[Browser, float]):
        self.device.preferred_browsers = val

    @property
    def device_probabilities(self) -> Dict[str, float]:
        return self.device.device_probabilities

    @device_probabilities.setter
    def device_probabilities(self, val: Dict[str, float]):
        self.device.device_probabilities = val

    @property
    def average_session_length(self) -> float:
        return self.session.average_session_length

    @average_session_length.setter
    def average_session_length(self, val: float):
        self.session.average_session_length = val

    @property
    def session_length_stddev(self) -> float:
        return self.session.session_length_stddev

    @session_length_stddev.setter
    def session_length_stddev(self, val: float):
        self.session.session_length_stddev = val

    @property
    def average_actions(self) -> int:
        return self.session.average_actions

    @average_actions.setter
    def average_actions(self, val: int):
        self.session.average_actions = val

    @property
    def actions_stddev(self) -> float:
        return self.session.actions_stddev

    @actions_stddev.setter
    def actions_stddev(self, val: float):
        self.session.actions_stddev = val

    @property
    def average_sessions_per_day(self) -> float:
        return self.session.average_sessions_per_day

    @average_sessions_per_day.setter
    def average_sessions_per_day(self, val: float):
        self.session.average_sessions_per_day = val

    @property
    def remote_work_probability(self) -> float:
        return self.location.remote_work_probability

    @remote_work_probability.setter
    def remote_work_probability(self, val: float):
        self.location.remote_work_probability = val


# Keep alias for backward compatibility with imports
BehaviorProfile = BehaviorKnowledgeBase