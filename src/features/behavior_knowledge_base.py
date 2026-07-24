import logging
import math
from typing import List, Dict, Any, Tuple, Set

from models.enums import DepartmentName, Browser, OperatingSystem, ResourceType
from models.session import Session
from models.event import Event
from models.behavior_profile import (
    BehaviorKnowledgeBase, TemporalBehavior, AuthenticationBehavior,
    ResourceBehavior, DeviceBehavior, LocationBehavior, NetworkBehavior,
    CommandBehavior, SessionBehavior
)
from simulator.company import Company
from config.config import ENABLE_DRIFT_MONITORING, DRIFT_WINDOW_SIZE

logger = logging.getLogger("BehaviorKnowledgeBaseService")

class ColdStartEngine:
    """
    Handles fallback behavior profiling for new entities with sparse history.
    Implements: Organization Baseline -> Department Baseline -> Role Baseline -> Personal Behavior.
    """
    def __init__(self, company: Company):
        self.company = company
        self._org_baseline: BehaviorKnowledgeBase = None
        self._dept_baselines: Dict[DepartmentName, BehaviorKnowledgeBase] = {}
        self._role_baselines: Dict[str, BehaviorKnowledgeBase] = {}

    def get_effective_baseline(self, entity_id: str, history_sessions: List[Session]) -> BehaviorKnowledgeBase:
        """
        Retrieves the appropriate baseline based on historical session counts.
        Falls back through Role -> Department -> Organization if personal history is insufficient.
        """
        emp = self.company.get_employee(entity_id)
        
        # 1. Personal Behavior (sufficient history >= 3 sessions)
        if len(history_sessions) >= 3:
            return self._build_personal_baseline(entity_id, history_sessions)

        # 2. Role Baseline fallback
        if emp and emp.role:
            role_base = self._get_role_baseline(emp.role)
            if role_base:
                return role_base

        # 3. Department Baseline fallback
        if emp and emp.department:
            dept_base = self._get_department_baseline(emp.department)
            if dept_base:
                return dept_base

        # 4. Organization Baseline fallback
        return self._get_org_baseline()

    def _build_personal_baseline(self, entity_id: str, sessions: List[Session]) -> BehaviorKnowledgeBase:
        """Computes a personal BehaviorKnowledgeBase dynamically from historical sessions."""
        login_hours = [s.start_time.hour for s in sessions]
        locations = {}
        devices = {}
        browsers = {}
        operating_systems = {}
        resources = {}
        methods = {}
        durations = []
        action_counts = []

        for s in sessions:
            locations[s.office_location] = locations.get(s.office_location, 0) + 1
            devices[s.device_id] = devices.get(s.device_id, 0) + 1
            browsers[s.browser] = browsers.get(s.browser, 0) + 1
            operating_systems[s.operating_system] = operating_systems.get(s.operating_system, 0) + 1
            methods[s.login_method.value] = methods.get(s.login_method.value, 0) + 1
            durations.append(s.duration_seconds)
            
            action_counts.append(len(s.events))
            for ev in s.events:
                if ev.resource:
                    resources[ev.resource.value] = resources.get(ev.resource.value, 0) + 1

        n_sess = len(sessions)
        
        # Norm distributions
        def norm(d):
            return {k: v/n_sess for k, v in d.items()}

        login_dist = {h: login_hours.count(h)/n_sess for h in set(login_hours)}
        
        avg_dur = sum(durations)/n_sess
        std_dur = math.sqrt(sum((x - avg_dur)**2 for x in durations)/n_sess) if n_sess > 1 else 10.0
        avg_acts = sum(action_counts)/n_sess
        std_acts = math.sqrt(sum((x - avg_acts)**2 for x in action_counts)/n_sess) if n_sess > 1 else 3.0

        return BehaviorKnowledgeBase(
            temporal=TemporalBehavior(
                working_hours=(min(login_hours), max(login_hours)),
                login_hour_distribution=login_dist,
                weekend_probability=sum(1 for s in sessions if s.day_of_week in ["Saturday", "Sunday"])/n_sess
            ),
            authentication=AuthenticationBehavior(
                success_rate=1.0,
                preferred_methods=norm(methods)
            ),
            resource=ResourceBehavior(
                resource_probabilities=norm(resources),
            ),
            device=DeviceBehavior(
                device_probabilities=norm(devices),
                preferred_browsers=norm(browsers),
                preferred_os=norm(operating_systems)
            ),
            location=LocationBehavior(
                office_locations=norm(locations),
                remote_work_probability=sum(1 for s in sessions if s.remote_session)/n_sess
            ),
            session=SessionBehavior(
                average_session_length=avg_dur / 60.0,
                session_length_stddev=std_dur / 60.0,
                average_actions=int(avg_acts),
                actions_stddev=std_acts
            )
        )

    def _get_org_baseline(self) -> BehaviorKnowledgeBase:
        """Returns the cached organization-wide average baseline."""
        if self._org_baseline is None:
            # Aggregate all default employee profiles
            self._org_baseline = BehaviorKnowledgeBase()
            # Try to build averages from company employees if available
            emps = list(self.company.employees.values())
            if emps:
                working_start = int(sum(e.behavior_profile.working_hours[0] for e in emps)/len(emps))
                working_end = int(sum(e.behavior_profile.working_hours[1] for e in emps)/len(emps))
                self._org_baseline.temporal.working_hours = (working_start, working_end)
        return self._org_baseline

    def _get_department_baseline(self, department: DepartmentName) -> BehaviorKnowledgeBase:
        """Returns the department-specific average baseline."""
        if department not in self._dept_baselines:
            # Gather employees in this department
            emps = [e for e in self.company.employees.values() if e.department == department]
            if emps:
                working_start = int(sum(e.behavior_profile.working_hours[0] for e in emps)/len(emps))
                working_end = int(sum(e.behavior_profile.working_hours[1] for e in emps)/len(emps))
                
                # Merge resource preferences
                merged_resources = {}
                for e in emps:
                    for r, p in e.behavior_profile.resource_probabilities.items():
                        r_val = r.value if isinstance(r, ResourceType) else str(r)
                        merged_resources[r_val] = merged_resources.get(r_val, 0.0) + p / len(emps)

                self._dept_baselines[department] = BehaviorKnowledgeBase(
                    temporal=TemporalBehavior(working_hours=(working_start, working_end)),
                    resource=ResourceBehavior(resource_probabilities=merged_resources)
                )
            else:
                self._dept_baselines[department] = self._get_org_baseline()
        return self._dept_baselines[department]

    def _get_role_baseline(self, role: str) -> BehaviorKnowledgeBase:
        """Returns the role-specific average baseline."""
        if role not in self._role_baselines:
            emps = [e for e in self.company.employees.values() if e.role == role]
            if emps:
                working_start = int(sum(e.behavior_profile.working_hours[0] for e in emps)/len(emps))
                working_end = int(sum(e.behavior_profile.working_hours[1] for e in emps)/len(emps))
                self._role_baselines[role] = BehaviorKnowledgeBase(
                    temporal=TemporalBehavior(working_hours=(working_start, working_end))
                )
            else:
                self._role_baselines[role] = None
        return self._role_baselines[role]


class DriftMonitor:
    """
    Monitors changes in employee behavior over time.
    Calculates behavioral drift using a sliding window of historical sessions.
    """
    def __init__(self, window_size: int = DRIFT_WINDOW_SIZE):
        self.window_size = window_size
        self.enable_drift_monitoring = ENABLE_DRIFT_MONITORING

    def process_and_update(self, current_session: Session, history_sessions: List[Session]) -> Tuple[Dict[str, float], float]:
        """
        Updates the running history list with current session properties
        and returns deviation metrics. Respects ENABLE_DRIFT_MONITORING configuration.
        """
        # Limit history list to window size
        if len(history_sessions) > self.window_size:
            history_sessions[:] = history_sessions[-self.window_size:]

        deviations = {
            "location_deviation": 0.0,
            "device_deviation": 0.0,
            "browser_deviation": 0.0,
            "operating_system_deviation": 0.0,
            "working_hours_deviation": 0.0,
            "resource_access_deviation": 0.0
        }

        # If full drift monitoring is disabled or history is too sparse, return standard static baseline comparison
        if not self.enable_drift_monitoring or len(history_sessions) < 3:
            return deviations, 0.0

        # Calculate dynamic deviations based on history window
        locs = {s.office_location for s in history_sessions}
        devs = {s.device_id for s in history_sessions}
        brows = {s.browser for s in history_sessions}
        oss = {s.operating_system for s in history_sessions}
        
        if current_session.office_location not in locs:
            deviations["location_deviation"] = 1.0
        if current_session.device_id not in devs:
            deviations["device_deviation"] = 1.0
        if current_session.browser not in brows:
            deviations["browser_deviation"] = 1.0
        if current_session.operating_system not in oss:
            deviations["operating_system_deviation"] = 1.0

        login_hours = [s.start_time.hour for s in history_sessions]
        mean_hour = sum(login_hours) / len(login_hours)
        var_hour = sum((h - mean_hour)**2 for h in login_hours) / len(login_hours)
        std_hour = math.sqrt(var_hour)
        
        if std_hour > 0.5:
            if abs(current_session.start_time.hour - mean_hour) > 2.0 * std_hour:
                deviations["working_hours_deviation"] = 1.0

        # Check resource accesses
        hist_resources = set()
        for s in history_sessions:
            for ev in s.events:
                if ev.resource:
                    hist_resources.add(ev.resource.value)

        for ev in current_session.events:
            if ev.resource and ev.resource.value not in hist_resources:
                deviations["resource_access_deviation"] = 1.0
                break

        # Calculate a weighted behavior drift score (0.0 to 100.0)
        drift_score = (
            deviations["location_deviation"] * 25.0 +
            deviations["device_deviation"] * 15.0 +
            deviations["browser_deviation"] * 5.0 +
            deviations["operating_system_deviation"] * 5.0 +
            deviations["working_hours_deviation"] * 20.0 +
            deviations["resource_access_deviation"] * 30.0
        )

        return deviations, drift_score
