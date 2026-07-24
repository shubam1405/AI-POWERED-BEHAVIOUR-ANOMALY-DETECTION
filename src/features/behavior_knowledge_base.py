import logging
import math
from collections import deque
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
from config.config import (
    ENABLE_DRIFT_MONITORING, DRIFT_WINDOW_SIZE,
    DRIFT_DEVIATION_WEIGHTS, DRIFT_UPDATE_THRESHOLD, DRIFT_MIN_HOUR_STDDEV,
    RESOURCE_DEVIATION_LAPLACE_ALPHA,
)

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
    Calculates behavioral drift using a sliding window of recent *normal* sessions.

    Key design principles:
    - Deviation scores are continuous probabilities in [0.0, 1.0] rather than
      binary 0/1 flags, giving downstream models richer signal.
    - Each entity maintains its own sliding window (collections.deque) so that
      one entity's activity cannot pollute another's baseline.
    - Sessions whose behavior score meets or exceeds DRIFT_UPDATE_THRESHOLD are
      excluded from the sliding window to prevent baseline poisoning.
    - All weights and thresholds are sourced from config.py.
    """

    def __init__(self, window_size: int = DRIFT_WINDOW_SIZE):
        self.window_size = window_size
        self.enable_drift_monitoring = ENABLE_DRIFT_MONITORING
        # Per-entity sliding windows of normal sessions only
        self._entity_windows: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_and_update(
        self,
        current_session: Session,
        history_sessions: List[Session],
        entity_id: str = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Computes continuous deviation metrics for *current_session* against
        the entity's sliding window of recent normal sessions.

        Parameters
        ----------
        current_session : Session
            The session being evaluated.
        history_sessions : List[Session]
            Externally-tracked history (used to seed the per-entity window
            the first time an entity is seen).  The caller's list is
            **no longer mutated**.
        entity_id : str, optional
            Defaults to ``current_session.employee_id``.

        Returns
        -------
        Tuple[Dict[str, float], float]
            (deviation_dict, behavior_score) where each deviation is in
            [0.0, 1.0] and the behaviour score is in [0.0, 100.0].
        """
        entity_id = entity_id or current_session.employee_id

        # Lazily initialise the entity window from the external history
        if entity_id not in self._entity_windows:
            seed = list(history_sessions[-self.window_size:])
            self._entity_windows[entity_id] = deque(seed, maxlen=self.window_size)

        window = self._entity_windows[entity_id]

        # Default zero-deviations
        deviations = {
            "location_deviation": 0.0,
            "device_deviation": 0.0,
            "browser_deviation": 0.0,
            "operating_system_deviation": 0.0,
            "working_hours_deviation": 0.0,
            "resource_access_deviation": 0.0,
        }

        # If drift monitoring is disabled or the window is too sparse,
        # return static zero baseline.
        if not self.enable_drift_monitoring :
            return deviations, 0.0

        if len(window) < 3:
            window.append(current_session)
            return deviations, 0.0

        n = len(window)

        # --- Frequency-based continuous deviations (categorical) -------
        deviations["location_deviation"] = self._categorical_deviation(
            current_session.office_location,
            [s.office_location for s in window],
        )
        deviations["device_deviation"] = self._categorical_deviation(
            current_session.device_id,
            [s.device_id for s in window],
        )
        deviations["browser_deviation"] = self._categorical_deviation(
            current_session.browser,
            [s.browser for s in window],
        )
        deviations["operating_system_deviation"] = self._categorical_deviation(
            current_session.operating_system,
            [s.operating_system for s in window],
        )

        # --- Gradual working-hours deviation (linear) ------------------
        login_hours = [s.start_time.hour for s in window]
        mean_hour = sum(login_hours) / n
        deviations["working_hours_deviation"] = min(
            abs(current_session.start_time.hour - mean_hour) / 12.0, 1.0
        )

        # --- Resource access deviation (frequency-based) ---------------
        deviations["resource_access_deviation"] = self._resource_deviation(
            current_session, window
        )

        # --- Weighted overall behavior score (0.0 – 100.0) -------------
        drift_score = sum(
            deviations[key] * DRIFT_DEVIATION_WEIGHTS[key]
            for key in deviations
        )

        # --- Baseline poisoning prevention -----------------------------
        # Only add the current session to the sliding window if its
        # behaviour score is below the configurable threshold.
        if drift_score < DRIFT_UPDATE_THRESHOLD:
            window.append(current_session)

        return deviations, drift_score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorical_deviation(current_value, history_values: list) -> float:
        """Returns ``1.0 - (frequency / n)``.

        A value seen in every historical session yields 0.0; a value
        never seen yields 1.0; intermediate frequencies produce a
        proportional score.
        """
        n = len(history_values)
        if n == 0:
            return 0.0
        frequency = sum(1 for v in history_values if v == current_value)
        return 1.0 - (frequency / n)

    @staticmethod
    def _resource_deviation(current_session: Session, window: deque) -> float:
        """Computes the mean per-resource frequency-based deviation with Laplace smoothing.

        Algorithm
        ---------
        1. Build a *session-level* frequency map from the sliding window:
           ``resource_session_freq[r]`` = number of window sessions that
           accessed resource ``r`` at least once.  (Session-level, not
           event-level, so sessions with many repeated events don\'t
           inflate the denominator.)

        2. For every unique resource ``r`` accessed in the current session
           compute a smoothed deviation::

               smoothed_freq = (freq + alpha) / (n + alpha)
               dev(r) = 1.0 - smoothed_freq

           Laplace (additive) smoothing prevents unseen resources from
           receiving a hard 1.0 score.  ``alpha`` is read from
           ``config.RESOURCE_DEVIATION_LAPLACE_ALPHA`` (default 2.0).

        3. **Return the mean** of the per-resource deviations (not the max).
           Mean aggregation correctly represents the *average novelty* of
           the session\'s resource footprint.  Normal employees access mostly
           familiar resources, so their mean stays low (~0.25–0.45).
           Attack sessions inject genuinely foreign resources whose high
           deviations pull the mean up significantly.

        Returns
        -------
        float
            Mean deviation in ``[0.0, 1.0]``.
        """
        n = len(window)
        if n == 0:
            return 0.0

        alpha = RESOURCE_DEVIATION_LAPLACE_ALPHA

        # Build a session-level frequency map: how many sessions in the window
        # accessed each resource at least once.
        resource_session_freq: Dict[str, int] = {}
        for s in window:
            seen_in_session: Set[str] = set()
            for ev in s.events:
                if ev.resource:
                    r_val = ev.resource.value if hasattr(ev.resource, "value") else str(ev.resource)
                    seen_in_session.add(r_val)
            for r_val in seen_in_session:
                resource_session_freq[r_val] = resource_session_freq.get(r_val, 0) + 1

        # Collect unique resources accessed in the current session
        current_resources: Set[str] = set()
        for ev in current_session.events:
            if ev.resource:
                r_val = ev.resource.value if hasattr(ev.resource, "value") else str(ev.resource)
                current_resources.add(r_val)

        if not current_resources:
            return 0.0

        # Compute Laplace-smoothed deviation for each accessed resource,
        # then return the mean (not the max) across the session.
        deviations_list: List[float] = []
        for r_val in current_resources:
            freq = resource_session_freq.get(r_val, 0)
            smoothed_freq = (freq + alpha) / (n + alpha)
            dev = 1.0 - smoothed_freq
            deviations_list.append(dev)

        mean_dev = sum(deviations_list) / len(deviations_list)
        return min(max(mean_dev, 0.0), 1.0)  # clamp to [0, 1]
