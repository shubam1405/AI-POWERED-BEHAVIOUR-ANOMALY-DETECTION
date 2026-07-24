import math
from datetime import datetime
from typing import List, Dict, Any, Set

from models.session import Session
from models.event import Event
from models.enums import (
    Browser, OperatingSystem, DepartmentName, EventType, ResourceType, EventStatus
)
from config.config import ATTACK_PAYLOADS, RESOURCES_CONFIG

# Mappings for categorical columns to numerical values
BROWSER_MAP = {b: float(idx) for idx, b in enumerate(Browser)}
OS_MAP = {o: float(idx) for idx, o in enumerate(OperatingSystem)}
DEPT_MAP = {d: float(idx) for idx, d in enumerate(DepartmentName)}
EVENT_TYPE_MAP = {e: float(idx) for idx, e in enumerate(EventType)}
RESOURCE_TYPE_MAP = {r: float(idx) for idx, r in enumerate(ResourceType)}
DAY_MAP = {
    "Monday": 0.0, "Tuesday": 1.0, "Wednesday": 2.0,
    "Thursday": 3.0, "Friday": 4.0, "Saturday": 5.0, "Sunday": 6.0
}

class SessionFeatureExtractor:
    """Extracts fundamental session attributes."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        day_num = DAY_MAP.get(session.day_of_week, -1.0)
        is_weekend = 1.0 if day_num >= 5.0 else 0.0
        
        return {
            "session_duration": float(session.duration_seconds),
            "login_hour": float(session.start_time.hour),
            "logout_hour": float(session.end_time.hour),
            "day_of_week": day_num,
            "weekend_flag": is_weekend,
            "remote_session": 1.0 if session.remote_session else 0.0,
            "vpn_used": 1.0 if session.login_method.value == "VPN" else 0.0,
            "browser": BROWSER_MAP.get(session.browser, -1.0),
            "operating_system": OS_MAP.get(session.operating_system, -1.0),
            "department": DEPT_MAP.get(session.department, -1.0)
        }

class AuthenticationFeatureExtractor:
    """Extracts login attempts, failures, and ratios."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        failed_logins = sum(1 for e in events if e.event_type == EventType.LOGIN and e.status == EventStatus.FAILED)
        successful_logins = sum(1 for e in events if e.event_type == EventType.LOGIN and e.status == EventStatus.SUCCESS)
        total_attempts = failed_logins + successful_logins
        failed_ratio = failed_logins / total_attempts if total_attempts > 0 else 0.0

        return {
            "failed_login_count": float(failed_logins),
            "successful_login_count": float(successful_logins),
            "login_method": 1.0 if session.login_method.value == "VPN" else 0.0,
            "authentication_attempts": float(total_attempts),
            "failed_login_ratio": failed_ratio
        }

class ResourceFeatureExtractor:
    """Extracts counts and types of resources accessed."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        resource_access_events = [e for e in events if e.resource is not None]
        total_accesses = len(resource_access_events)
        unique_resources = len(set(e.resource for e in resource_access_events))

        # Admin resource access: Admin Console, Servers, Firewall
        admin_resources = {ResourceType.ADMIN_CONSOLE, ResourceType.SERVERS, ResourceType.FIREWALL}
        admin_accesses = sum(1 for e in resource_access_events if e.resource in admin_resources)

        # Sensitive resources (restricted or confidential sensitivity level)
        sensitive_names = {
            r["name"] for r in RESOURCES_CONFIG 
            if r["sensitivity"].value in ["Restricted", "Confidential"]
        }
        sensitive_accesses = sum(1 for e in resource_access_events if e.resource in sensitive_names)

        # Individual resource counts
        github_count = sum(1 for e in events if e.resource == ResourceType.GITHUB)
        db_count = sum(1 for e in events if e.event_type == EventType.DATABASE_QUERY)
        confluence_count = sum(1 for e in events if e.resource == ResourceType.CONFLUENCE)
        hrms_count = sum(1 for e in events if e.resource == ResourceType.HRMS)
        erp_count = sum(1 for e in events if e.resource == ResourceType.ERP)

        return {
            "total_resource_accesses": float(total_accesses),
            "unique_resources": float(unique_resources),
            "admin_resource_accesses": float(admin_accesses),
            "sensitive_resource_accesses": float(sensitive_accesses),
            "github_access_count": float(github_count),
            "database_query_count": float(db_count),
            "confluence_access_count": float(confluence_count),
            "hrms_access_count": float(hrms_count),
            "erp_access_count": float(erp_count)
        }

class FileActivityFeatureExtractor:
    """Extracts counts, bytes, and ratios of file activity."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        f_read = sum(1 for e in events if e.event_type == EventType.FILE_READ)
        f_write = sum(1 for e in events if e.event_type == EventType.FILE_WRITE)
        f_upload = sum(1 for e in events if e.event_type == EventType.FILE_UPLOAD)
        f_download = sum(1 for e in events if e.event_type == EventType.FILE_DOWNLOAD)

        # Extract bytes from metadata
        upload_bytes = 0.0
        download_bytes = 0.0
        for e in events:
            if e.event_type == EventType.FILE_UPLOAD:
                upload_bytes += float(e.metadata.get("bytes_transferred", 0.0))
            elif e.event_type == EventType.FILE_DOWNLOAD:
                download_bytes += float(e.metadata.get("bytes_transferred", 0.0))
            elif e.event_type == EventType.FILE_WRITE:
                upload_bytes += float(e.metadata.get("bytes_written", 0.0))

        ratio = upload_bytes / download_bytes if download_bytes > 0.0 else 0.0

        return {
            "files_read": float(f_read),
            "files_written": float(f_write),
            "files_uploaded": float(f_upload),
            "files_downloaded": float(f_download),
            "total_upload_bytes": upload_bytes,
            "total_download_bytes": download_bytes,
            "upload_download_ratio": ratio
        }

class ProcessFeatureExtractor:
    """Extracts process executions, suspicious commands, and malware counts."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        p_start = sum(1 for e in events if e.event_type == EventType.PROCESS_START)
        p_stop = sum(1 for e in events if e.event_type == EventType.PROCESS_STOP)

        powershell_execs = 0
        suspicious_procs = 0
        malware_execs = 0

        malware_names = {name.lower() for name in ATTACK_PAYLOADS.get("malware_filenames", [])}

        for e in events:
            if e.event_type == EventType.PROCESS_START or e.command_sequence:
                # Standardize command check by joining Honeywell command sequence list
                cmd = " ".join(e.command_sequence).lower()
                proc = e.command_sequence[0].lower() if e.command_sequence else ""

                if not cmd:
                    # Fallback to old metadata attributes if command_sequence was empty
                    cmd = e.metadata.get("command_line", "").lower()
                    proc = e.metadata.get("process_name", "").lower()

                if not cmd:
                    continue

                # Powershell detection
                if "powershell" in cmd or "powershell" in proc:
                    powershell_execs += 1

                # Malware execution detection
                is_malware = False
                for m in malware_names:
                    if m in cmd or m in proc:
                        is_malware = True
                        break
                if is_malware:
                    malware_execs += 1
                    suspicious_procs += 1

                # General suspicious flags
                is_susp_flags = any(flag in cmd for flag in ["bypass", "hidden", "encodedcommand"])
                if is_susp_flags and not is_malware:
                    suspicious_procs += 1

        return {
            "processes_started": float(p_start),
            "processes_stopped": float(p_stop),
            "powershell_executions": float(powershell_execs),
            "suspicious_process_count": float(suspicious_procs),
            "malware_execution_count": float(malware_execs)
        }

class NetworkFeatureExtractor:
    """Extracts network/VPN connections and destination/source subnets."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        net_conns = sum(1 for e in events if e.event_type in [EventType.NETWORK_CONNECT, EventType.NETWORK_DISCONNECT])
        vpn_conns = sum(1 for e in events if e.event_type == EventType.VPN_CONNECT)
        vpn_disconns = sum(1 for e in events if e.event_type == EventType.VPN_DISCONNECT)

        # Unique destination domains/IPs from metadata
        dest_ips = set()
        src_ips = set(e.ip_address for e in events if e.ip_address)
        external_conns = 0
        beacon_events = 0

        c2_domains = {d.lower() for d in ATTACK_PAYLOADS.get("c2_domains", [])}

        for e in events:
            dest = e.metadata.get("destination_domain", "").lower()
            target = e.metadata.get("destination", "").lower()
            
            # Check command sequence for curl target
            cmd = " ".join(e.command_sequence).lower()
            for d in c2_domains:
                if d in cmd:
                    dest.add(d) if hasattr(dest, "add") else None
                    external_conns += 1
                    if e.event_type == EventType.API_CALL:
                        beacon_events += 1

            d_val = dest or target
            if d_val:
                dest_ips.add(d_val)
                if d_val in c2_domains:
                    external_conns += 1
                    if e.event_type == EventType.API_CALL:
                        beacon_events += 1

        return {
            "network_connections": float(net_conns),
            "vpn_connections": float(vpn_conns),
            "vpn_disconnections": float(vpn_disconns),
            "unique_destination_ips": float(len(dest_ips)),
            "unique_source_ips": float(len(src_ips)),
            "external_connections": float(external_conns),
            "beaconing_event_count": float(beacon_events)
        }

class TemporalFeatureExtractor:
    """Extracts event counts, timing intervals, and off-hours properties."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        total_evs = len(events)
        duration_min = max(1.0, session.duration_seconds / 60.0)
        evs_per_min = total_evs / duration_min

        # Calculate idle times between consecutive events
        times = sorted(e.timestamp for e in events)
        intervals = []
        for i in range(len(times) - 1):
            intervals.append((times[i+1] - times[i]).total_seconds())

        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        max_idle = max(intervals) if intervals else 0.0
        min_idle = min(intervals) if intervals else 0.0

        is_off_hours = 0.0
        login_hour = session.start_time.hour
        if login_hour < 7 or login_hour > 20:
            is_off_hours = 1.0

        return {
            "total_events": float(total_evs),
            "events_per_minute": float(evs_per_min),
            "average_time_between_events": float(avg_interval),
            "maximum_idle_time": float(max_idle),
            "minimum_idle_time": float(min_idle),
            "off_hours_access": is_off_hours,
            "session_start_hour": float(session.start_time.hour)
        }

class StatisticalFeatureExtractor:
    """Extracts ratios of distinct event categories relative to total counts."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        total = len(events)
        if total == 0:
            return {
                "unique_event_types": 0.0,
                "login_event_ratio": 0.0,
                "file_event_ratio": 0.0,
                "process_event_ratio": 0.0,
                "network_event_ratio": 0.0,
                "resource_event_ratio": 0.0
            }

        unique_types = len(set(e.event_type for e in events))
        
        login_evs = sum(1 for e in events if e.event_type in [EventType.LOGIN, EventType.LOGOUT])
        file_evs = sum(1 for e in events if e.event_type.value.startswith("File"))
        proc_evs = sum(1 for e in events if e.event_type in [EventType.PROCESS_START, EventType.PROCESS_STOP])
        net_evs = sum(1 for e in events if "Connect" in e.event_type.value or "Disconnect" in e.event_type.value)
        res_evs = sum(1 for e in events if e.event_type == EventType.RESOURCE_ACCESS)

        return {
            "unique_event_types": float(unique_types),
            "login_event_ratio": float(login_evs / total),
            "file_event_ratio": float(file_evs / total),
            "process_event_ratio": float(proc_evs / total),
            "network_event_ratio": float(net_evs / total),
            "resource_event_ratio": float(res_evs / total)
        }

class SequenceFeatureExtractor:
    """Extracts order and position indicators from the event sequence."""
    def extract(self, session: Session, events: List[Event]) -> Dict[str, float]:
        first_ev = events[0].event_type if events else EventType.LOGIN
        last_ev = events[-1].event_type if events else EventType.LOGOUT
        seq_len = len(events)

        login_time = events[0].timestamp if events else session.start_time
        first_resource_time = -1.0
        for ev in events:
            if ev.event_type == EventType.RESOURCE_ACCESS:
                first_resource_time = (ev.timestamp - login_time).total_seconds()
                break

        # Relative positions
        admin_pos = -1.0
        download_pos = -1.0
        proc_pos = -1.0

        admin_resources = {ResourceType.ADMIN_CONSOLE, ResourceType.SERVERS, ResourceType.FIREWALL}

        for idx, ev in enumerate(events):
            if ev.resource in admin_resources and admin_pos == -1.0:
                admin_pos = idx / seq_len if seq_len > 0 else 0.0
            if ev.event_type == EventType.FILE_DOWNLOAD and download_pos == -1.0:
                download_pos = idx / seq_len if seq_len > 0 else 0.0
            if ev.event_type == EventType.PROCESS_START and proc_pos == -1.0:
                proc_pos = idx / seq_len if seq_len > 0 else 0.0

        return {
            "first_event_type": EVENT_TYPE_MAP.get(first_ev, -1.0),
            "last_event_type": EVENT_TYPE_MAP.get(last_ev, -1.0),
            "event_sequence_length": float(seq_len),
            "login_to_first_resource_time": first_resource_time,
            "admin_access_position": admin_pos,
            "file_download_position": download_pos,
            "process_execution_position": proc_pos
        }

class BehavioralFeatureExtractor:
    """Computes deviations from an employee's historical baseline."""
    def extract_with_history(self, session: Session, events: List[Event], history: Dict[str, Any]) -> Dict[str, float]:
        if not history.get("sessions_seen", 0):
            return {
                "location_deviation": 0.0,
                "device_deviation": 0.0,
                "browser_deviation": 0.0,
                "operating_system_deviation": 0.0,
                "working_hours_deviation": 0.0,
                "resource_access_deviation": 0.0
            }

        loc_dev = 1.0 if session.office_location not in history["locations"] else 0.0
        dev_dev = 1.0 if session.device_id not in history["devices"] else 0.0
        browser_dev = 1.0 if session.browser not in history["browsers"] else 0.0
        os_dev = 1.0 if session.operating_system not in history["operating_systems"] else 0.0

        hours = history["login_hours"]
        work_hours_dev = 0.0
        if len(hours) >= 3:
            mean = sum(hours) / len(hours)
            var = sum((h - mean) ** 2 for h in history["login_hours"]) / len(hours)
            std = math.sqrt(var)
            if std > 0.1:
                if abs(session.start_time.hour - mean) > 2.0 * std:
                    work_hours_dev = 1.0

        res_dev = 0.0
        for ev in events:
            if ev.resource is not None:
                r_val = ev.resource.value if hasattr(ev.resource, "value") else str(ev.resource)
                if r_val not in history["resources"]:
                    res_dev = 1.0
                    break

        return {
            "location_deviation": loc_dev,
            "device_deviation": dev_dev,
            "browser_deviation": browser_dev,
            "operating_system_deviation": os_dev,
            "working_hours_deviation": work_hours_dev,
            "resource_access_deviation": res_dev
        }

