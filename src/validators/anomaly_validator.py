import logging
from typing import List

from models.session import Session
from models.event import Event
from models.enums import AttackType, EventType, EventStatus, ResourceType, DepartmentName
from config.config import ANOMALY_CONFIG, ATTACK_PAYLOADS

logger = logging.getLogger("AnomalyValidator")

class AnomalyValidationError(Exception):
    """Exception raised when anomaly validation checks fail."""
    def __init__(self, errors: List[str]):
        super().__init__("Anomaly validation failed with the following errors:\n" + "\n".join(errors))
        self.errors = errors

class AnomalyValidator:
    """
    Validates anomalous sessions and events to guarantee labeling correctness,
    chronological timeline validity, lack of timestamp duplication, and proper risk scoring.
    """
    def validate(self, sessions: List[Session], company) -> bool:
        """Validates anomaly attributes and calculations in the sessions. Raises AnomalyValidationError on failure."""
        errors: List[str] = []

        severities = ANOMALY_CONFIG.get("attack_severities", {})
        modifiers = ANOMALY_CONFIG.get("risk_modifiers", {})

        for session in sessions:
            sid = session.session_id
            events = session.events

            # 1. Check logical consistency of session labels
            if session.is_anomalous:
                if session.attack_type == AttackType.NONE:
                    errors.append(f"Session {sid}: Marked anomalous but attack_type is NONE.")
                if not (0.0 < session.risk_score <= 100.0):
                    errors.append(f"Session {sid}: Anomalous risk_score {session.risk_score} is out of bounds (0, 100].")
            else:
                if session.attack_type != AttackType.NONE:
                    errors.append(f"Session {sid}: Normal but has attack_type {session.attack_type.value}.")
                if session.risk_score != 0.0:
                    errors.append(f"Session {sid}: Normal but has risk_score {session.risk_score}.")

            # 2. Check logical consistency of event labels
            last_ts = None
            seen_timestamps = set()

            if not events:
                errors.append(f"Session {sid}: Contains no events.")
                continue

            # First event is LOGIN, last event is LOGOUT
            if events[0].event_type != EventType.LOGIN:
                errors.append(f"Session {sid}: First event is {events[0].event_type.value}, expected LOGIN.")
            if events[-1].event_type != EventType.LOGOUT:
                errors.append(f"Session {sid}: Last event is {events[-1].event_type.value}, expected LOGOUT.")

            # Recalculate risk score for verification
            recalculated_score = 0.0
            if session.is_anomalous:
                recalculated_score = severities.get(session.attack_type, 0.0)

            for idx, event in enumerate(events):
                ev_id = f"{sid}[event #{idx}]"

                # Check event/session anomaly label alignment
                if event.is_anomalous != session.is_anomalous:
                    errors.append(f"Event {ev_id}: Anomaly status mismatch (event={event.is_anomalous}, session={session.is_anomalous})")

                if event.is_anomalous:
                    if event.attack_type == AttackType.NONE:
                        errors.append(f"Event {ev_id}: Anomalous event has attack_type NONE.")
                    if event.risk_score != session.risk_score:
                        errors.append(f"Event {ev_id}: risk_score mismatch (event={event.risk_score}, session={session.risk_score})")
                else:
                    if event.attack_type != AttackType.NONE:
                        errors.append(f"Event {ev_id}: Normal event has attack_type {event.attack_type.value}.")
                    if event.risk_score != 0.0:
                        errors.append(f"Event {ev_id}: Normal event has risk_score {event.risk_score}.")

                # Check session bounds
                if not (session.start_time <= event.timestamp <= session.end_time):
                    errors.append(f"Event {ev_id}: Timestamp {event.timestamp} falls outside session boundaries ({session.start_time} to {session.end_time})")

                # Check chronological order (strictly increasing)
                if last_ts is not None and event.timestamp <= last_ts:
                    errors.append(f"Event {ev_id}: Timestamp {event.timestamp} is not strictly after previous timestamp {last_ts}")
                last_ts = event.timestamp

                # Check duplicate timestamps
                if event.timestamp in seen_timestamps:
                    errors.append(f"Event {ev_id}: Duplicate timestamp {event.timestamp} in session.")
                seen_timestamps.add(event.timestamp)

                # Recompute modifiers for score validation
                if session.is_anomalous:
                    if event.event_type == EventType.LOGIN and event.status == EventStatus.FAILED:
                        recalculated_score += modifiers.get("failed_login", 5.0)

                    employee = company.get_employee(event.employee_id)
                    if employee and employee.department != DepartmentName.IT:
                        if event.resource in [ResourceType.ADMIN_CONSOLE, ResourceType.SERVERS, ResourceType.FIREWALL]:
                            recalculated_score += modifiers.get("unauthorized_admin", 15.0)

                    bytes_transferred = event.metadata.get("bytes_transferred", 0)
                    if bytes_transferred > 100 * 1024 * 1024:
                        recalculated_score += modifiers.get("large_transfer", 20.0)

                    if event.event_type == EventType.PROCESS_START:
                        cmd = event.metadata.get("command_line", "").lower()
                        proc = event.metadata.get("process_name", "").lower()
                        is_suspicious = False
                        for p in ATTACK_PAYLOADS["malware_filenames"]:
                            if p.lower() in cmd or p.lower() in proc:
                                is_suspicious = True
                                break
                        for flag in ["bypass", "hidden", "encodedcommand"]:
                            if flag in cmd:
                                is_suspicious = True
                                break
                        if is_suspicious:
                            recalculated_score += modifiers.get("suspicious_cli", 20.0)

            # Validate risk score math
            if session.is_anomalous:
                recalculated_score = min(100.0, max(0.0, recalculated_score))
                if abs(session.risk_score - recalculated_score) > 1e-3:
                    errors.append(f"Session {sid}: risk_score calculation mismatch (got {session.risk_score}, expected {recalculated_score})")

        if errors:
            raise AnomalyValidationError(errors)

        logger.info("Anomaly validation successful. All integrity constraints verified.")
        return True
