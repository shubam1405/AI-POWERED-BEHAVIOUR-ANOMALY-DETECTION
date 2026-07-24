import copy
import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

from models.session import Session
from models.event import Event
from models.enums import (
    AttackType, EventType, EventStatus, ResourceType, LoginMethod, SessionStatus, DepartmentName
)
from simulator.company import Company
from config.config import ANOMALY_CONFIG, ATTACK_PAYLOADS

logger = logging.getLogger("AnomalyInjector")

class AnomalyInjector:
    """
    Orchestrates the injection of realistic, multi-stage cyber attacks
    into simulated user sessions and calculates rule-based risk scores.
    Ensures second-level uniqueness for all event timestamps.
    """
    def __init__(self, company: Company, rng: random.Random):
        self.company = company
        self.rng = rng

    def inject_anomalies(self, sessions: List[Session]) -> Tuple[List[Session], List[Event]]:
        """
        Deep-copies normal sessions, injects cyber attacks based on configuration,
        calculates rule-based risk scores, and returns anomalous sessions and all events.
        """
        logger.info("Injecting anomalies into sessions...")
        
        anom_sessions = copy.deepcopy(sessions)
        
        global_rate = ANOMALY_CONFIG.get("global_anomaly_rate", 0.08)
        num_anomalous = int(len(anom_sessions) * global_rate)
        
        # Select sessions to inject anomalies
        target_sessions = self.rng.sample(anom_sessions, num_anomalous)
        logger.info(f"Targeting {len(target_sessions)} sessions for anomaly injection...")

        for session in target_sessions:
            # 1. Determine attack stages (chains)
            chain_roll = self.rng.random()
            if chain_roll < 0.15:
                # 2-stage chain
                stages = self.rng.choice([
                    [AttackType.CREDENTIAL_STUFFING, AttackType.PRIVILEGE_ESCALATION],
                    [AttackType.BRUTE_FORCE, AttackType.LATERAL_MOVEMENT],
                    [AttackType.MALWARE_EXECUTION, AttackType.BEACONING_C2],
                    [AttackType.INSIDER_THREAT, AttackType.DATA_EXFILTRATION],
                    [AttackType.DEVICE_SPOOFING, AttackType.LOW_SLOW_EXFILTRATION]
                ])
            elif chain_roll < 0.20:
                # 3-stage chain
                stages = self.rng.choice([
                    [AttackType.CREDENTIAL_STUFFING, AttackType.PRIVILEGE_ESCALATION, AttackType.DATA_EXFILTRATION],
                    [AttackType.BRUTE_FORCE, AttackType.LATERAL_MOVEMENT, AttackType.MALWARE_EXECUTION],
                    [AttackType.DEVICE_SPOOFING, AttackType.LATERAL_MOVEMENT, AttackType.LOW_SLOW_EXFILTRATION]
                ])
            else:
                # 1-stage attack
                att_types = list(ANOMALY_CONFIG["attack_probabilities"].keys())
                weights = list(ANOMALY_CONFIG["attack_probabilities"].values())
                stages = [self.rng.choices(att_types, weights=weights, k=1)[0]]

            # 2. Inject each attack stage
            for stage in stages:
                self._apply_attack_stage(session, stage)

            # 3. Deduplicate timestamps at the second level & adjust session bounds
            self._deduplicate_timestamps(session)

            # 4. Label session anomalous and compute risk scores
            session.is_anomalous = True
            
            # Overall attack type is the highest-severity stage in the chain
            severities = ANOMALY_CONFIG.get("attack_severities", {})
            session.attack_type = max(stages, key=lambda s: severities.get(s, 0))
            
            # Compute rule-based risk score (pre-ML placeholder computed dynamically)
            session.risk_score = self._compute_session_risk(session)
            
            # Propagate attributes to all events in this session
            for event in session.events:
                event.is_anomalous = True
                event.risk_score = session.risk_score
                if event.attack_type == AttackType.NONE:
                    event.attack_type = session.attack_type

        # Extract all flat events
        all_events: List[Event] = []
        for session in anom_sessions:
            all_events.extend(session.events)

        logger.info(f"Anomaly injection completed. Total events: {len(all_events)}")
        return anom_sessions, all_events

    def _deduplicate_timestamps(self, session: Session) -> None:
        """
        Ensures all events in the session have unique, strictly increasing timestamps
        at the second level, and updates session end_time and duration accordingly.
        """
        if not session.events:
            return

        # Ensure first event starts at session start_time (at second precision)
        session.events[0].timestamp = session.start_time.replace(microsecond=0)
        
        last_ts = session.events[0].timestamp
        for i in range(1, len(session.events)):
            ev = session.events[i]
            curr_ts = ev.timestamp.replace(microsecond=0)
            if curr_ts <= last_ts:
                curr_ts = last_ts + timedelta(seconds=1)
            ev.timestamp = curr_ts
            last_ts = curr_ts

        # Update session end time and duration to envelope all events
        session.end_time = max(session.end_time.replace(microsecond=0), last_ts)
        session.duration_seconds = (session.end_time - session.start_time).total_seconds()
        
        # Propagate session duration to all event metadata
        for ev in session.events:
            ev.metadata["session_duration"] = session.duration_seconds

    def _apply_attack_stage(self, session: Session, stage: AttackType) -> None:
        """Modifies session events in-place to insert specific attack indicators."""
        emp = self.company.get_employee(session.employee_id)
        entity_type = emp.entity_type if emp else "user"

        if stage == AttackType.BRUTE_FORCE:
            self._inject_failed_logins(session, stage, same_user=True)
            
        elif stage == AttackType.CREDENTIAL_STUFFING:
            self._inject_failed_logins(session, stage, same_user=False)
            
        elif stage == AttackType.IMPOSSIBLE_TRAVEL:
            # Change office location to distant city
            session.office_location = "Tokyo"
            session.remote_session = True
            session.login_method = LoginMethod.VPN
            # Update IP and location fields
            for ev in session.events:
                ev.ip_address = f"210.140.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"
                ev.metadata["geo_location"] = "Tokyo"
                ev.metadata["auth_method"] = "VPN - MFA"
                if ev.event_type in [EventType.LOGIN, EventType.VPN_CONNECT]:
                    ev.attack_type = stage

        elif stage == AttackType.DEVICE_SPOOFING:
            # Spoof device fingerprint
            spoofed_device = f"DEV-SPOOF-{self.rng.randint(1000, 9999)}"
            session.device_id = spoofed_device
            for ev in session.events:
                ev.device_id = spoofed_device
                ev.attack_type = stage

        elif stage == AttackType.LOW_SLOW_EXFILTRATION:
            # Inject multiple small, stealthy file downloads
            for i in range(4):
                self._inject_suspicious_actions(
                    session, stage, ResourceType.FILE_SERVER, EventType.FILE_DOWNLOAD,
                    {"file_name": f"stealth_chunk_{i}.bin", "bytes_transferred": 80 * 1024}
                )

        elif stage == AttackType.INSIDER_DRIFT:
            # Employee accesses resources completely outside their role/department baseline
            self._inject_suspicious_actions(
                session, stage, ResourceType.ADMIN_CONSOLE, EventType.RESOURCE_ACCESS,
                {"action": "unauthorized_read"}
            )
            self._inject_suspicious_actions(
                session, stage, ResourceType.SERVERS, EventType.RESOURCE_ACCESS,
                {"action": "port_scan"}
            )

        elif stage == AttackType.PRIVILEGE_ESCALATION:
            self._inject_suspicious_actions(
                session, stage, ResourceType.TERMINAL, EventType.PROCESS_START,
                {"process_name": "sudo su", "command_line": "sudo su -root", "user": "root"}
            )

        elif stage == AttackType.DATA_EXFILTRATION:
            self._inject_suspicious_actions(
                session, stage, ResourceType.FILE_SERVER, EventType.FILE_DOWNLOAD,
                {"file_name": self.rng.choice(ATTACK_PAYLOADS["suspicious_files"]), "bytes_transferred": 850 * 1024 * 1024}
            )
            self._inject_suspicious_actions(
                session, stage, ResourceType.GITHUB, EventType.FILE_UPLOAD,
                {"destination": self.rng.choice(ATTACK_PAYLOADS["c2_domains"]), "bytes_transferred": 850 * 1024 * 1024}
            )

        elif stage == AttackType.INSIDER_THREAT:
            self._inject_suspicious_actions(
                session, stage, ResourceType.FINANCE_DB, EventType.DATABASE_QUERY,
                {"query": "SELECT * FROM corporate_ledgers", "rows_returned": 50000}
            )

        elif stage == AttackType.OFF_HOURS_ACCESS:
            target_hour = self.rng.randint(1, 3)
            diff_hours = target_hour - session.start_time.hour
            offset = timedelta(hours=diff_hours)
            
            session.start_time += offset
            session.end_time += offset
            session.day_of_week = session.start_time.strftime("%A")
            for ev in session.events:
                ev.timestamp += offset
                ev.attack_type = stage

        elif stage == AttackType.LATERAL_MOVEMENT:
            self._inject_suspicious_actions(
                session, stage, ResourceType.SERVERS, EventType.RESOURCE_ACCESS,
                {"target_server": "DEV-DB-SRV04", "port": 445}
            )
            self._inject_suspicious_actions(
                session, stage, ResourceType.SERVERS, EventType.RESOURCE_ACCESS,
                {"target_server": "PROD-DC-SRV01", "port": 139}
            )

        elif stage == AttackType.USB_DATA_THEFT:
            self._inject_suspicious_actions(
                session, stage, ResourceType.FILE_SERVER, EventType.FILE_WRITE,
                {
                    "file_name": self.rng.choice(ATTACK_PAYLOADS["suspicious_files"]),
                    "usb_connected": True,
                    "usb_volume_name": "BACKUP",
                    "bytes_transferred": 350 * 1024 * 1024
                }
            )

        elif stage == AttackType.MALWARE_EXECUTION:
            self._inject_suspicious_actions(
                session, stage, ResourceType.TERMINAL, EventType.PROCESS_START,
                {
                    "process_name": self.rng.choice(ATTACK_PAYLOADS["malware_filenames"]),
                    "command_line": "./revshell"
                }
            )

        elif stage == AttackType.SUSPICIOUS_POWERSHELL:
            self._inject_suspicious_actions(
                session, stage, ResourceType.TERMINAL, EventType.PROCESS_START,
                {
                    "process_name": "powershell.exe",
                    "command_line": self.rng.choice(ATTACK_PAYLOADS["powershell_commands"])
                }
            )

        elif stage == AttackType.BEACONING_C2:
            self._inject_beaconing(session, stage)

    def _inject_failed_logins(self, session: Session, stage: AttackType, same_user: bool) -> None:
        """Prepends a sequence of failed logins to the session."""
        num_failures = self.rng.randint(5, 10)
        gap_seconds = 2
        total_time_needed = num_failures * gap_seconds

        original_start = session.start_time
        session.start_time = original_start - timedelta(seconds=total_time_needed)
        session.duration_seconds += total_time_needed
        session.day_of_week = session.start_time.strftime("%A")

        failed_events: List[Event] = []
        ip_addr = session.events[0].ip_address if session.events else "10.10.1.1"
        emp = self.company.get_employee(session.employee_id)
        entity_type = emp.entity_type if emp else "user"

        for i in range(num_failures):
            ev_ts = session.start_time + timedelta(seconds=i * gap_seconds)
            
            if same_user:
                emp_id = session.employee_id
            else:
                emp_list = list(self.company.employees.keys())
                if session.employee_id in emp_list:
                    emp_list.remove(session.employee_id)
                emp_id = self.rng.choice(emp_list) if emp_list else session.employee_id

            ev_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
            failed_ev = Event(
                event_uuid=ev_uuid,
                session_id=session.session_id,
                timestamp=ev_ts,
                event_type=EventType.LOGIN,
                employee_id=emp_id,
                device_id=session.device_id,
                resource=None,
                status=EventStatus.FAILED,
                attack_type=stage,
                risk_score=0.0,
                ip_address=ip_addr,
                metadata={
                    "geo_location": session.office_location,
                    "auth_method": "Password",
                    "session_duration": session.duration_seconds,
                    "command_sequence": []
                },
                entity_type=entity_type
            )
            failed_events.append(failed_ev)

        session.events = failed_events + session.events

    def _inject_suspicious_actions(
        self, session: Session, stage: AttackType, resource: ResourceType, 
        event_type: EventType, metadata: dict
    ) -> None:
        """Inserts an anomalous action event inside the session timeline."""
        if len(session.events) < 3:
            return

        insert_idx = self.rng.randint(2, len(session.events) - 2)
        prev_ts = session.events[insert_idx - 1].timestamp
        ev_ts = prev_ts + timedelta(seconds=1)

        emp = self.company.get_employee(session.employee_id)
        entity_type = emp.entity_type if emp else "user"

        # Construct Honeywell metadata
        combined_metadata = {
            "geo_location": session.office_location,
            "auth_method": "VPN - MFA" if session.remote_session else "Active Directory",
            "session_duration": session.duration_seconds,
            "command_sequence": metadata.get("command_line", "").split() if "command_line" in metadata else []
        }
        combined_metadata.update(metadata)

        ev_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        suspicious_event = Event(
            event_uuid=ev_uuid,
            session_id=session.session_id,
            timestamp=ev_ts,
            event_type=event_type,
            employee_id=session.employee_id,
            device_id=session.device_id,
            resource=resource,
            status=EventStatus.SUCCESS,
            attack_type=stage,
            risk_score=0.0,
            ip_address=session.events[0].ip_address,
            metadata=combined_metadata,
            entity_type=entity_type
        )
        session.events.insert(insert_idx, suspicious_event)

    def _inject_beaconing(self, session: Session, stage: AttackType) -> None:
        """Injects periodic C2 beaconing API calls into the session."""
        if len(session.events) < 3:
            return

        interval = self.rng.choice(ATTACK_PAYLOADS["beacon_intervals"])
        c2_domain = self.rng.choice(ATTACK_PAYLOADS["c2_domains"])
        
        start_ts = session.events[1].timestamp + timedelta(seconds=5)
        end_ts = session.events[-2].timestamp - timedelta(seconds=5)
        
        current_ts = start_ts
        beacons: List[Event] = []
        emp = self.company.get_employee(session.employee_id)
        entity_type = emp.entity_type if emp else "user"

        while current_ts < end_ts:
            ev_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
            beacon_ev = Event(
                event_uuid=ev_uuid,
                session_id=session.session_id,
                timestamp=current_ts,
                event_type=EventType.API_CALL,
                employee_id=session.employee_id,
                device_id=session.device_id,
                resource=ResourceType.SERVERS,
                status=EventStatus.SUCCESS,
                attack_type=stage,
                risk_score=0.0,
                ip_address=session.events[0].ip_address,
                metadata={
                    "geo_location": session.office_location,
                    "auth_method": "VPN - MFA" if session.remote_session else "Active Directory",
                    "session_duration": session.duration_seconds,
                    "command_sequence": ["curl -X POST " + c2_domain],
                    "destination_domain": c2_domain,
                    "direction": "outbound",
                    "bytes_sent": 128
                },
                entity_type=entity_type
            )
            beacons.append(beacon_ev)
            current_ts += timedelta(seconds=interval)

        merged_events = session.events + beacons
        merged_events.sort(key=lambda x: x.timestamp)
        session.events = merged_events

    def _compute_session_risk(self, session: Session) -> float:
        """Calculates rule-based risk score for a session based on base severity and modifiers."""
        severities = ANOMALY_CONFIG.get("attack_severities", {})
        modifiers = ANOMALY_CONFIG.get("risk_modifiers", {})
        
        score = severities.get(session.attack_type, 0.0)

        for ev in session.events:
            if ev.event_type == EventType.LOGIN and ev.status == EventStatus.FAILED:
                score += modifiers.get("failed_login", 5.0)

            employee = self.company.get_employee(ev.employee_id)
            if employee and employee.department != DepartmentName.IT:
                if ev.resource in [ResourceType.ADMIN_CONSOLE, ResourceType.SERVERS, ResourceType.FIREWALL]:
                    score += modifiers.get("unauthorized_admin", 15.0)

            bytes_transferred = ev.metadata.get("bytes_transferred", 0)
            if bytes_transferred > 100 * 1024 * 1024:
                score += modifiers.get("large_transfer", 20.0)

            if ev.event_type == EventType.PROCESS_START:
                cmd = ev.metadata.get("command_line", "").lower()
                proc = ev.metadata.get("process_name", "").lower()
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
                    score += modifiers.get("suspicious_cli", 20.0)

        return min(100.0, max(0.0, score))
