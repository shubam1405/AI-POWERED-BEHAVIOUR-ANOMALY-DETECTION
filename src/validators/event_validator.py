import re
from typing import List
import logging

from models.session import Session
from models.event import Event
from models.enums import EventType, EventStatus, AttackType, ResourceType

logger = logging.getLogger("EventValidator")

class EventValidationError(Exception):
    """Exception raised when event validation checks fail."""
    def __init__(self, errors: List[str]):
        super().__init__("Event validation failed with the following errors:\n" + "\n".join(errors))
        self.errors = errors

class EventValidator:
    """
    Validates generated Event objects to ensure chronological consistency, session boundary respect,
    lack of timestamp duplication, and exact cross-linking with parent Sessions.
    """
    @staticmethod
    def is_valid_uuid(uuid_str: str) -> bool:
        """Verifies if a string is a valid UUID4 format."""
        uuid_regex = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_regex.match(uuid_str))

    def validate(self, sessions: List[Session]) -> bool:
        """Validates all events in the provided list of sessions. Raises EventValidationError on failure."""
        errors: List[str] = []
        seen_uuids = set()

        for session in sessions:
            sid = session.session_id
            events = session.events

            if not events:
                errors.append(f"Session {sid}: Contains no events.")
                continue

            # 1. First event is LOGIN, last event is LOGOUT
            first_event = events[0]
            last_event = events[-1]

            if first_event.event_type != EventType.LOGIN:
                errors.append(f"Session {sid}: First event is {first_event.event_type.value}, expected LOGIN.")
            if last_event.event_type != EventType.LOGOUT:
                errors.append(f"Session {sid}: Last event is {last_event.event_type.value}, expected LOGOUT.")

            # 2. Check VPN alignment
            if session.remote_session:
                # Second event should be VPN_CONNECT, penultimate should be VPN_DISCONNECT
                if len(events) >= 4:
                    sec_ev = events[1]
                    pen_ev = events[-2]
                    if sec_ev.event_type != EventType.VPN_CONNECT:
                        errors.append(f"Session {sid}: Remote session missing VPN_CONNECT as second event (got {sec_ev.event_type.value}).")
                    if pen_ev.event_type != EventType.VPN_DISCONNECT:
                        errors.append(f"Session {sid}: Remote session missing VPN_DISCONNECT as penultimate event (got {pen_ev.event_type.value}).")
            else:
                # Office session should connect to network
                if len(events) >= 4:
                    sec_ev = events[1]
                    pen_ev = events[-2]
                    if sec_ev.event_type != EventType.NETWORK_CONNECT:
                        errors.append(f"Session {sid}: Office session missing NETWORK_CONNECT as second event (got {sec_ev.event_type.value}).")
                    if pen_ev.event_type != EventType.NETWORK_DISCONNECT:
                        errors.append(f"Session {sid}: Office session missing NETWORK_DISCONNECT as penultimate event (got {pen_ev.event_type.value}).")

            # 3. Check timestamps, uniqueness, parent links, and enums
            last_ts = None
            seen_timestamps = set()

            for idx, event in enumerate(events):
                ev_id = f"{sid}[event #{idx}]"

                # Check unique UUID
                if event.event_uuid in seen_uuids:
                    errors.append(f"Event {ev_id}: Duplicate event_uuid {event.event_uuid}")
                seen_uuids.add(event.event_uuid)

                if not self.is_valid_uuid(event.event_uuid):
                    errors.append(f"Event {ev_id}: Invalid UUIDv4: {event.event_uuid}")

                # Check matching entity linking
                if event.session_id != session.session_id:
                    errors.append(f"Event {ev_id}: session_id mismatch (got {event.session_id}, expected {session.session_id})")
                if event.employee_id != session.employee_id:
                    errors.append(f"Event {ev_id}: employee_id mismatch (got {event.employee_id}, expected {session.employee_id})")
                if event.device_id != session.device_id:
                    errors.append(f"Event {ev_id}: device_id mismatch (got {event.device_id}, expected {session.device_id})")

                # Check session boundary bounds
                if not (session.start_time <= event.timestamp <= session.end_time):
                    errors.append(f"Event {ev_id}: Timestamp {event.timestamp} falls outside session boundaries ({session.start_time} to {session.end_time})")

                # Check chronological order (strictly increasing)
                if last_ts is not None and event.timestamp <= last_ts:
                    errors.append(f"Event {ev_id}: Timestamp {event.timestamp} is not strictly after previous timestamp {last_ts}")
                last_ts = event.timestamp

                # Check duplicate timestamps within session
                if event.timestamp in seen_timestamps:
                    errors.append(f"Event {ev_id}: Duplicate timestamp {event.timestamp} in session.")
                seen_timestamps.add(event.timestamp)

                # Check valid enums
                if not isinstance(event.event_type, EventType):
                    errors.append(f"Event {ev_id}: Invalid event_type: {event.event_type}")
                if not isinstance(event.status, EventStatus):
                    errors.append(f"Event {ev_id}: Invalid status: {event.status}")
                if not isinstance(event.attack_type, AttackType):
                    errors.append(f"Event {ev_id}: Invalid attack_type: {event.attack_type}")
                if event.resource is not None and not isinstance(event.resource, ResourceType):
                    errors.append(f"Event {ev_id}: Invalid resource: {event.resource}")

        if errors:
            raise EventValidationError(errors)

        logger.info("Event validation successful. All integrity constraints verified.")
        return True
