import csv
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

from models.enums import (
    Browser, OperatingSystem, DepartmentName, EventType, ResourceType, EventStatus, LoginMethod, AttackType, SessionStatus
)
from models.session import Session
from models.event import Event

logger = logging.getLogger("DataAdapter")

class DataAdapter:
    """
    Standardizes external log entries to the Honeywell schema
    and dynamically sessionizes flat event sequences based on a time gap threshold.
    """
    def __init__(self, inactivity_threshold_seconds: int = 1800):
        self.inactivity_threshold_seconds = inactivity_threshold_seconds

    def standardize_and_sessionize(self, events_raw: List[Dict[str, Any]], employee_lookup: Dict[str, Any] = None) -> List[Session]:
        """
        Takes raw event dictionaries, standardizes them, groups by entity_id,
        splits into sessions, and returns populated Session objects.
        """
        logger.info(f"Standardizing and sessionizing {len(events_raw)} events...")
        
        # 1. Standardize all events first
        standardized_events: List[Dict[str, Any]] = []
        for idx, row in enumerate(events_raw):
            std_ev = self._standardize_row(row, idx)
            standardized_events.append(std_ev)

        # 2. Group by entity_id
        grouped_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        for ev in standardized_events:
            grouped_by_entity.setdefault(ev["entity_id"], []).append(ev)

        sessions: List[Session] = []
        session_idx = 1

        # 3. Sessionize per entity
        for entity_id, entity_events in grouped_by_entity.items():
            # Sort chronologically
            entity_events.sort(key=lambda x: x["timestamp"])

            current_session_events: List[Dict[str, Any]] = []
            for ev in entity_events:
                if not current_session_events:
                    current_session_events.append(ev)
                else:
                    last_ev = current_session_events[-1]
                    gap = (ev["timestamp"] - last_ev["timestamp"]).total_seconds()
                    
                    if gap > self.inactivity_threshold_seconds:
                        # Close current session and start a new one
                        sessions.append(self._build_session(current_session_events, f"SES-AD-{session_idx:06d}", employee_lookup))
                        session_idx += 1
                        current_session_events = [ev]
                    else:
                        current_session_events.append(ev)

            if current_session_events:
                sessions.append(self._build_session(current_session_events, f"SES-AD-{session_idx:06d}", employee_lookup))
                session_idx += 1

        logger.info(f"Reconstructed {len(sessions)} sessions from log events.")
        return sessions

    def _standardize_row(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Standardizes a single raw row dictionary to the Honeywell event schema."""
        # 1. Standardize entity_id (map employee_id / user_id -> entity_id)
        entity_id = row.get("entity_id") or row.get("employee_id") or row.get("user_id") or "UNKNOWN_ENTITY"
        
        # 2. Standardize entity_type (default to "user")
        entity_type = row.get("entity_type") or "user"
        
        # 3. Standardize timestamp
        ts_val = row.get("timestamp")
        if isinstance(ts_val, str):
            try:
                timestamp = datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(ts_val)
                except ValueError:
                    timestamp = datetime.now()
        elif isinstance(ts_val, datetime):
            timestamp = ts_val
        else:
            timestamp = datetime.now()

        # 4. Standardize source_ip
        source_ip = row.get("source_ip") or row.get("ip_address") or "127.0.0.1"

        # 5. Standardize geo_location
        geo_location = row.get("geo_location") or row.get("office_location") or "Office"

        # 6. Standardize resource_accessed
        resource_accessed = row.get("resource_accessed") or row.get("resource") or ""

        # 7. Standardize auth_method
        auth_method = row.get("auth_method") or row.get("login_method") or "Password"

        # 8. Standardize session_duration
        duration = 0.0
        dur_val = row.get("session_duration") or row.get("duration_seconds")
        if dur_val:
            try:
                duration = float(dur_val)
            except (ValueError, TypeError):
                duration = 0.0

        # 9. Standardize command_sequence
        cmd_seq = row.get("command_sequence") or []
        if isinstance(cmd_seq, str):
            try:
                cmd_seq = json.loads(cmd_seq)
            except json.JSONDecodeError:
                if cmd_seq.strip():
                    cmd_seq = [cmd_seq]
                else:
                    cmd_seq = []

        # 10. Standardize device_fingerprint
        device_fingerprint = row.get("device_fingerprint") or row.get("device_id") or "UNKNOWN_DEVICE"

        # 11. Standardize label
        label = 0
        lbl_val = row.get("label") or row.get("is_anomalous")
        if lbl_val is not None:
            if str(lbl_val).lower() in ["1", "true", "yes", "anomalous"]:
                label = 1

        # Extra metadata
        session_id = row.get("session_id") or ""
        event_uuid = row.get("event_uuid") or str(uuid.uuid4())
        event_type = row.get("event_type") or ""
        status = row.get("status") or "SUCCESS"
        attack_type = row.get("attack_type") or "None"
        risk_score = 0.0
        risk_val = row.get("risk_score")
        if risk_val:
            try:
                risk_score = float(risk_val)
            except (ValueError, TypeError):
                risk_score = 0.0

        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "timestamp": timestamp,
            "source_ip": source_ip,
            "geo_location": geo_location,
            "resource_accessed": resource_accessed,
            "auth_method": auth_method,
            "session_duration": duration,
            "command_sequence": cmd_seq,
            "device_fingerprint": device_fingerprint,
            "label": label,
            "session_id": session_id,
            "event_uuid": event_uuid,
            "event_type": event_type,
            "status": status,
            "attack_type": attack_type,
            "risk_score": risk_score
        }

    def _build_session(self, events: List[Dict[str, Any]], session_id: str, employee_lookup: Dict[str, Any] = None) -> Session:
        """Helper to construct a filled Session model object from grouped event dicts."""
        first_ev = events[0]
        last_ev = events[-1]
        
        # Override session_id on events
        sid = first_ev["session_id"] if first_ev["session_id"] else session_id
        for ev in events:
            ev["session_id"] = sid

        entity_id = first_ev["entity_id"]
        
        # Resolve employee context if lookup table exists
        department = DepartmentName.IT
        role = "Staff"
        browser = Browser.CHROME
        operating_system = OperatingSystem.WINDOWS
        
        if employee_lookup and entity_id in employee_lookup:
            emp = employee_lookup[entity_id]
            # Emp can be an Employee object or dict
            if isinstance(emp, dict):
                department = DepartmentName(emp.get("department", "IT"))
                role = emp.get("role", "Staff")
                browser = Browser(emp.get("preferred_browser", "Chrome"))
                operating_system = OperatingSystem(emp.get("preferred_os", "Windows"))
            else:
                department = emp.department
                role = emp.role
                browser = emp.preferred_browser
                operating_system = emp.preferred_os

        # LoginMethod
        login_method = LoginMethod.VPN if "vpn" in first_ev["auth_method"].lower() else LoginMethod.OFFICE
        remote_session = (login_method == LoginMethod.VPN)

        # Durations
        duration_sec = (last_ev["timestamp"] - first_ev["timestamp"]).total_seconds()
        if first_ev["session_duration"] > 0.0:
            duration_sec = first_ev["session_duration"]

        is_anomalous = any(ev["label"] == 1 for ev in events)
        
        # Max attack type
        attack_type = AttackType.NONE
        for ev in events:
            if ev["attack_type"] != "None":
                try:
                    attack_type = AttackType(ev["attack_type"])
                except ValueError:
                    pass

        # Build Session
        session = Session(
            session_uuid=str(uuid.uuid4()),
            session_id=sid,
            employee_uuid=str(uuid.uuid4()),
            employee_id=entity_id,
            department=department,
            role=role,
            device_id=first_ev["device_fingerprint"],
            browser=browser,
            operating_system=operating_system,
            office_location=first_ev["geo_location"],
            login_method=login_method,
            remote_session=remote_session,
            day_of_week=first_ev["timestamp"].strftime("%A"),
            start_time=first_ev["timestamp"],
            end_time=last_ev["timestamp"],
            duration_seconds=duration_sec,
            risk_score=max(ev["risk_score"] for ev in events),
            is_anomalous=is_anomalous,
            attack_type=attack_type,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )

        # Build Event objects
        for ev in events:
            # Map event type
            ev_type_str = ev["event_type"]
            if ev_type_str:
                try:
                    ev_type = EventType(ev_type_str)
                except ValueError:
                    ev_type = EventType.RESOURCE_ACCESS
            else:
                # Infer from resource
                if ev["command_sequence"]:
                    ev_type = EventType.PROCESS_START
                elif not ev["resource_accessed"]:
                    ev_type = EventType.LOGIN
                else:
                    ev_type = EventType.RESOURCE_ACCESS

            res_str = ev["resource_accessed"]
            res = ResourceType(res_str) if res_str else None

            # Map status
            status = EventStatus.SUCCESS
            if ev["status"] == "FAILED":
                status = EventStatus.FAILED

            ev_obj = Event(
                event_uuid=ev["event_uuid"],
                session_id=sid,
                timestamp=ev["timestamp"],
                event_type=ev_type,
                employee_id=ev["entity_id"],
                device_id=ev["device_fingerprint"],
                resource=res,
                status=status,
                attack_type=AttackType(ev["attack_type"]) if ev["attack_type"] != "None" else AttackType.NONE,
                risk_score=ev["risk_score"],
                ip_address=ev["source_ip"],
                is_anomalous=(ev["label"] == 1),
                metadata={
                    "geo_location": ev["geo_location"],
                    "auth_method": ev["auth_method"],
                    "session_duration": ev["session_duration"],
                    "command_sequence": ev["command_sequence"]
                },
                entity_type=ev["entity_type"]
            )
            session.add_event(ev_obj)

        return session
