import os
import re
import csv
import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import pandas as pd

from models.enums import (
    Browser, OperatingSystem, DepartmentName, EventType, ResourceType, EventStatus, LoginMethod, AttackType, SessionStatus
)
from models.session import Session
from models.event import Event
from attack_classification.dataset_loader import AttackDatasetLoader

logger = logging.getLogger("UniversalDatasetAdapter")

@dataclass
class InternalEventRecord:
    """Layer 4: Internal Event Model - Decoupled intermediate event before session building."""
    event_uuid: str
    employee_id: str
    session_id: Optional[str]
    timestamp: datetime
    event_type: EventType
    device_id: str
    source_ip: str
    destination_ip: Optional[str]
    resource: Optional[str]
    status: EventStatus
    geo_location: str
    auth_method: str
    bytes_transferred: float
    process_name: Optional[str]
    command_line: Optional[str]
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Validation report containing schema detection, compatibility score, feature checks, and warnings."""
    total_rows: int = 0
    mapped_rows: int = 0
    skipped_rows: int = 0
    sessions_built: int = 0
    unique_employees: int = 0
    mapped_columns: Dict[str, Tuple[str, float]] = field(default_factory=dict) # canonical -> (raw_col, confidence)
    missing_columns: List[str] = field(default_factory=list)
    unused_columns: List[str] = field(default_factory=list)
    inferred_columns: List[str] = field(default_factory=list)
    compatibility_score: float = 0.0 # 0 to 100%
    compatibility_level: str = "UNKNOWN" # FULL, PARTIAL, DEGRADED
    pipeline_status: str = "PENDING"
    feature_compatibility_pass: bool = False
    feature_name_alignment_pass: bool = False
    warnings: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "CYBER CAGE UNIVERSAL DATASET ADAPTER REPORT",
            "=" * 60,
            f"Total Rows Processed       : {self.total_rows:,}",
            f"Mapped Rows                : {self.mapped_rows:,}",
            f"Skipped Rows               : {self.skipped_rows:,}",
            f"Sessions Reconstructed    : {self.sessions_built:,}",
            f"Unique Employees           : {self.unique_employees:,}",
            f"Overall Mapping Confidence : {self.compatibility_score:.1f}% ({self.compatibility_level})",
            f"Pipeline Status            : {self.pipeline_status}",
            "-" * 60,
            "Mapped Columns & Confidence:",
        ]
        for canonical, (raw_col, conf) in self.mapped_columns.items():
            lines.append(f"  - {canonical:<20} -> {raw_col:<20} ({conf * 100:.0f}%)")
        if self.unused_columns:
            lines.append(f"Unused External Columns   : {self.unused_columns}")
        if self.missing_columns:
            lines.append(f"Missing Optional Columns   : {self.missing_columns} (defaults applied)")
        if self.inferred_columns:
            lines.append(f"Inferred Columns          : {self.inferred_columns}")
        lines.extend([
            "-" * 60,
            f"Feature Check (71 Count)   : {'PASS' if self.feature_compatibility_pass else 'FAIL'}",
            f"Feature Order Alignment   : {'PASS' if self.feature_name_alignment_pass else 'FAIL'}",
            "=" * 60
        ])
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        return "\n".join(lines)


class UniversalDatasetAdapter:
    """
    Universal Dataset Adapter for Cyber Cage.
    Decouples external log formats from internal ML pipeline via a multi-stage layered architecture.
    """
    def __init__(self, config_path: Optional[str] = None, inactivity_threshold_seconds: int = 1800):
        self.inactivity_threshold_seconds = inactivity_threshold_seconds
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "dataset_mapping.json")
            if not os.path.exists(config_path):
                config_path = os.path.join("config", "dataset_mapping.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

        self.synonyms = self.config.get("synonyms", {})
        self.regex_patterns = self.config.get("event_type_regex", {})
        self.defaults = self.config.get("defaults", {})
        self.core_fields = set(self.config.get("core_fields", ["employee_id", "event_timestamp", "event_type"]))

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "synonyms": {
                "employee_id": ["username", "user", "user_id", "employee_id", "emp_id", "account", "subject", "actor"],
                "session_id": ["session_id", "sess_id", "session", "sid"],
                "event_timestamp": ["timestamp", "time", "datetime", "date_time", "created_at", "@timestamp", "ts"],
                "event_type": ["action", "event", "event_type", "activity", "operation", "command"],
                "device_id": ["device", "host", "hostname", "device_id", "machine", "workstation"],
                "source_ip": ["src_ip", "source_ip", "ip", "client_ip", "ip_address"],
                "destination_ip": ["dst_ip", "destination_ip", "target_ip", "dest_ip"],
                "resource": ["resource", "resource_accessed", "target", "file", "url", "db", "uri"],
                "status": ["status", "result", "outcome", "success"],
                "location": ["geo_location", "office_location", "location", "city"],
                "auth_method": ["auth_method", "login_method", "authentication"],
                "bytes_transferred": ["bytes", "bytes_transferred", "size", "data_size"],
                "process_name": ["process_name", "process", "exe", "image"],
                "command_line": ["command_line", "cmd", "cmdline", "command_sequence"]
            },
            "event_type_regex": {
                r"(?i).*(login|signin|logon|auth).*": "LOGIN",
                r"(?i).*(logout|signout|logoff).*": "LOGOUT",
                r"(?i).*(download|fetch|pull).*": "FILE_DOWNLOAD",
                r"(?i).*(upload|push|post).*": "FILE_UPLOAD",
                r"(?i).*(read|open).*": "FILE_READ",
                r"(?i).*(write|save|modify).*": "FILE_WRITE",
                r"(?i).*(process|exec|cmd|run).*": "PROCESS_START",
                r"(?i).*(stop|kill).*": "PROCESS_STOP",
                r"(?i).*(sql|query|select).*": "DATABASE_QUERY",
                r"(?i).*(vpn).*": "VPN_CONNECT"
            },
            "defaults": {
                "employee_id": "EMP-UNKNOWN",
                "device_id": "DEV-UNKNOWN",
                "source_ip": "127.0.0.1",
                "geo_location": "Office",
                "department": "IT",
                "role": "Staff",
                "status": "SUCCESS",
                "auth_method": "Password"
            },
            "core_fields": ["employee_id", "event_timestamp", "event_type"]
        }

    def detect_schema(self, columns: List[str]) -> Tuple[Dict[str, Tuple[str, float]], List[str], List[str], List[str], float, str]:
        """
        Layer 1: Schema Detection & Column Mapping.
        Inspects input columns, computes fuzzy match confidence, and categorizes columns.
        """
        mapped_columns: Dict[str, Tuple[str, float]] = {}
        used_raw_cols: Set[str] = set()
        inferred_columns: List[str] = []

        # Exact and fuzzy matching against synonyms
        for canonical_field, synonym_list in self.synonyms.items():
            best_match: Optional[str] = None
            best_conf: float = 0.0

            for raw_col in columns:
                if raw_col in used_raw_cols:
                    continue
                
                clean_raw = raw_col.strip().lower()
                clean_canonical = canonical_field.strip().lower()

                # Exact match
                if clean_raw == clean_canonical:
                    best_match = raw_col
                    best_conf = 1.0
                    break
                
                # Direct synonym match
                for idx, syn in enumerate(synonym_list):
                    if clean_raw == syn.lower():
                        conf = 1.0 if idx == 0 else 0.95
                        if conf > best_conf:
                            best_match = raw_col
                            best_conf = conf
                            break
                    elif syn.lower() in clean_raw or clean_raw in syn.lower():
                        conf = 0.85
                        if conf > best_conf:
                            best_match = raw_col
                            best_conf = conf

            if best_match and best_conf >= 0.70:
                mapped_columns[canonical_field] = (best_match, best_conf)
                used_raw_cols.add(best_match)
            else:
                inferred_columns.append(canonical_field)

        unused_columns = [col for col in columns if col not in used_raw_cols]
        missing_columns = [f for f in self.synonyms.keys() if f not in mapped_columns]

        # Calculate Overall Compatibility Score
        core_mapped_count = sum(1 for f in self.core_fields if f in mapped_columns)
        core_ratio = core_mapped_count / max(1, len(self.core_fields))
        total_fields_mapped_ratio = len(mapped_columns) / max(1, len(self.synonyms))
        
        overall_score = round((core_ratio * 0.7 + total_fields_mapped_ratio * 0.3) * 100.0, 1)

        if overall_score >= 90.0:
            level = "FULL (100%)" if overall_score == 100.0 else f"FULL ({overall_score:.0f}%)"
        elif overall_score >= 70.0:
            level = f"PARTIAL ({overall_score:.0f}%)"
        else:
            level = f"DEGRADED ({overall_score:.0f}%)"

        return mapped_columns, missing_columns, unused_columns, inferred_columns, overall_score, level

    def normalize_timestamp(self, val: Any) -> datetime:
        """Layer 2: Timestamp Normalisation (ISO 8601, Epoch, custom formats)."""
        if isinstance(val, datetime):
            return val
        if pd.isna(val) or val is None or str(val).strip() == "":
            return datetime.now()

        val_str = str(val).strip()

        # Check for Unix epoch (seconds or ms)
        if val_str.isdigit():
            epoch_num = int(val_str)
            if epoch_num > 1e11: # Milliseconds
                return datetime.fromtimestamp(epoch_num / 1000.0)
            else:
                return datetime.fromtimestamp(epoch_num)

        # Standard date formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
            "%Y/%m/%d %H:%M:%S",
            "%d-%b-%Y %H:%M:%S",
            "%Y-%m-%d"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(val_str, fmt)
            except ValueError:
                pass
        
        # Fallback to dateutil / ISO
        try:
            return datetime.fromisoformat(val_str)
        except ValueError:
            return datetime.now()

    def normalize_event_type(self, raw_action: Any, resource: Optional[str] = None, cmd: Optional[str] = None) -> EventType:
        """Layer 3: Event Normalisation via Regex and Synonym Rules."""
        action_str = str(raw_action).strip() if raw_action is not None else ""
        
        # Try Regex matching
        for pattern, et_name in self.regex_patterns.items():
            if re.match(pattern, action_str):
                try:
                    return EventType[et_name]
                except KeyError:
                    pass

        # Contextual inference if action text is vague
        if cmd and any(kw in cmd.lower() for kw in ["cmd", "powershell", "exe", "exec"]):
            return EventType.PROCESS_START
        if resource and any(kw in resource.lower() for kw in ["db", "sql", "table"]):
            return EventType.DATABASE_QUERY
        if resource and any(kw in resource.lower() for kw in [".csv", ".pdf", ".zip", "file"]):
            return EventType.FILE_READ

        return EventType.RESOURCE_ACCESS

    def process_row_to_internal_record(
        self, row: Dict[str, Any], mapped_cols: Dict[str, Tuple[str, float]], idx: int
    ) -> InternalEventRecord:
        """Layer 4: Convert single raw row to InternalEventRecord."""
        def get_val(canonical: str, default_val: Any = None) -> Any:
            if canonical in mapped_cols:
                raw_col = mapped_cols[canonical][0]
                val = row.get(raw_col)
                if val is not None and not pd.isna(val):
                    return val
            return default_val

        employee_id = str(get_val("employee_id", self.defaults.get("employee_id", f"EMP-{idx:04d}")))
        session_id_val = get_val("session_id")
        session_id = str(session_id_val) if session_id_val else None

        raw_ts = get_val("event_timestamp")
        timestamp = self.normalize_timestamp(raw_ts)

        raw_action = get_val("event_type", "RESOURCE_ACCESS")
        resource = get_val("resource")
        resource_str = str(resource) if resource else None
        cmd = get_val("command_line")
        cmd_str = str(cmd) if cmd else None
        
        event_type = self.normalize_event_type(raw_action, resource_str, cmd_str)

        device_id = str(get_val("device_id", self.defaults.get("device_id", "DEV-UNKNOWN")))
        source_ip = str(get_val("source_ip", self.defaults.get("source_ip", "127.0.0.1")))
        destination_ip = get_val("destination_ip")
        destination_ip_str = str(destination_ip) if destination_ip else None

        raw_status = get_val("status", "SUCCESS")
        status = EventStatus.FAILED if str(raw_status).upper() in ["FAILED", "FAILURE", "0", "FALSE", "DENIED"] else EventStatus.SUCCESS

        geo_location = str(get_val("location", self.defaults.get("geo_location", "Office")))
        auth_method = str(get_val("auth_method", self.defaults.get("auth_method", "Password")))
        
        bytes_val = get_val("bytes_transferred", 0.0)
        try:
            bytes_transferred = float(bytes_val)
        except (ValueError, TypeError):
            bytes_transferred = 0.0

        proc = get_val("process_name")
        process_name = str(proc) if proc else None

        # Build raw metadata from unused fields
        raw_meta = {k: v for k, v in row.items() if not pd.isna(v)}

        return InternalEventRecord(
            event_uuid=str(uuid.uuid4()),
            employee_id=employee_id,
            session_id=session_id,
            timestamp=timestamp,
            event_type=event_type,
            device_id=device_id,
            source_ip=source_ip,
            destination_ip=destination_ip_str,
            resource=resource_str,
            status=status,
            geo_location=geo_location,
            auth_method=auth_method,
            bytes_transferred=bytes_transferred,
            process_name=process_name,
            command_line=cmd_str,
            raw_metadata=raw_meta
        )

    def build_sessions_from_records(self, records: List[InternalEventRecord], employee_lookup: Optional[Dict[str, Any]] = None) -> List[Session]:
        """
        Layer 5: Smart Session Builder.
        If session_id is provided, group by session_id! Otherwise group by (employee_id, device_id) and split on inactivity gap.
        """
        if not records:
            return []

        has_session_ids = any(r.session_id is not None for r in records)
        grouped_records: Dict[str, List[InternalEventRecord]] = {}

        if has_session_ids:
            # Group directly by session_id
            for rec in records:
                sid = rec.session_id or f"SES-GEN-{rec.employee_id}"
                grouped_records.setdefault(sid, []).append(rec)
        else:
            # Group by employee_id + device_id
            for rec in records:
                key = f"{rec.employee_id}::{rec.device_id}"
                grouped_records.setdefault(key, []).append(rec)

        sessions: List[Session] = []
        session_counter = 1

        for group_key, group_evs in grouped_records.items():
            group_evs.sort(key=lambda r: r.timestamp)

            if has_session_ids:
                sessions.append(self._construct_session_object(group_evs, group_key, employee_lookup))
            else:
                # Gap-based sessionization
                curr_events: List[InternalEventRecord] = []
                for rec in group_evs:
                    if not curr_events:
                        curr_events.append(rec)
                    else:
                        gap = (rec.timestamp - curr_events[-1].timestamp).total_seconds()
                        if gap > self.inactivity_threshold_seconds:
                            sid = f"SES-AD-{session_counter:06d}"
                            sessions.append(self._construct_session_object(curr_events, sid, employee_lookup))
                            session_counter += 1
                            curr_events = [rec]
                        else:
                            curr_events.append(rec)
                if curr_events:
                    sid = f"SES-AD-{session_counter:06d}"
                    sessions.append(self._construct_session_object(curr_events, sid, employee_lookup))
                    session_counter += 1

        return sessions

    def _construct_session_object(
        self, records: List[InternalEventRecord], session_id: str, employee_lookup: Optional[Dict[str, Any]] = None
    ) -> Session:
        first_r = records[0]
        last_r = records[-1]
        
        emp_id = first_r.employee_id
        department = DepartmentName.IT
        role = "Staff"
        browser = Browser.CHROME
        operating_system = OperatingSystem.WINDOWS

        if employee_lookup and emp_id in employee_lookup:
            emp = employee_lookup[emp_id]
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

        login_method = LoginMethod.VPN if "vpn" in first_r.auth_method.lower() else LoginMethod.OFFICE
        remote_session = (login_method == LoginMethod.VPN)
        duration_sec = max(1.0, (last_r.timestamp - first_r.timestamp).total_seconds())

        session = Session(
            session_uuid=str(uuid.uuid4()),
            session_id=session_id,
            employee_uuid=str(uuid.uuid4()),
            employee_id=emp_id,
            department=department,
            role=role,
            device_id=first_r.device_id,
            browser=browser,
            operating_system=operating_system,
            office_location=first_r.geo_location,
            login_method=login_method,
            remote_session=remote_session,
            day_of_week=first_r.timestamp.strftime("%A"),
            start_time=first_r.timestamp,
            end_time=last_r.timestamp,
            duration_seconds=duration_sec,
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=[]
        )

        for rec in records:
            res_enum = ResourceType(rec.resource) if rec.resource and rec.resource in [r.value for r in ResourceType] else None
            ev_obj = Event(
                event_uuid=rec.event_uuid,
                session_id=session_id,
                timestamp=rec.timestamp,
                event_type=rec.event_type,
                employee_id=rec.employee_id,
                device_id=rec.device_id,
                resource=res_enum,
                status=rec.status,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address=rec.source_ip,
                is_anomalous=False,
                metadata={
                    "geo_location": rec.geo_location or "Office",
                    "auth_method": rec.auth_method or "Password",
                    "bytes_transferred": rec.bytes_transferred or 0.0,
                    "process_name": rec.process_name or "",
                    "command_line": rec.command_line or "",
                    "destination_ip": rec.destination_ip or ""
                },
                entity_type="user"
            )
            session.add_event(ev_obj)

        return session

    def validate_features(self, sample_session: Session) -> Tuple[bool, bool, List[str]]:
        """
        Layer 6: Feature Compatibility & Order Alignment Validator.
        Validates exact 71-feature count AND canonical feature name order alignment.
        """
        from features.feature_engineer import FeatureEngineer
        from attack_classification.dataset_loader import AttackDatasetLoader
        
        loader = AttackDatasetLoader().load()
        expected_features = loader.feature_names
        
        fe = FeatureEngineer()
        raw_list, _, _ = fe.run(sessions=[sample_session])
        raw_dict = raw_list[0] if raw_list else {}
        
        # Populate GRU score placeholders if missing
        raw_dict.setdefault("reconstruction_error", 0.0)
        raw_dict.setdefault("anomaly_score", 0.0)
        
        extracted = {k: raw_dict[k] for k in expected_features if k in raw_dict}
        
        count_pass = (len(extracted) == len(expected_features))
        name_order_pass = (list(extracted.keys()) == expected_features)
        
        warnings = []
        if not count_pass:
            warnings.append(f"Feature count mismatch: expected {len(expected_features)}, got {len(extracted)}")
        if not name_order_pass:
            warnings.append("Feature column order mismatch with trained XGBoost model!")

        return count_pass, name_order_pass, warnings

    def process(
        self, data_input: Union[str, pd.DataFrame, List[Dict[str, Any]]], employee_lookup: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Session], ValidationReport]:
        """
        End-to-End Adapter Processing:
        Ingests external file/DataFrame/dict list, executes layers 1-5, runs feature validation, returns Sessions + ValidationReport.
        """
        report = ValidationReport()

        # Load raw data into DataFrame
        if isinstance(data_input, str):
            if not os.path.exists(data_input):
                raise FileNotFoundError(f"Dataset path not found: {data_input}")
            if data_input.endswith(".json") or data_input.endswith(".jsonl"):
                df = pd.read_json(data_input)
            else:
                df = pd.read_csv(data_input)
        elif isinstance(data_input, pd.DataFrame):
            df = data_input.copy()
        elif isinstance(data_input, list):
            df = pd.DataFrame(data_input)
        else:
            raise ValueError("Unsupported data input type. Pass CSV path, DataFrame, or list of dicts.")

        report.total_rows = len(df)
        columns = list(df.columns)

        # Layer 1: Schema Detection & Column Mapping
        mapped_cols, missing_cols, unused_cols, inferred_cols, score, level = self.detect_schema(columns)
        report.mapped_columns = mapped_cols
        report.missing_columns = missing_cols
        report.unused_columns = unused_cols
        report.inferred_columns = inferred_cols
        report.compatibility_score = score
        report.compatibility_level = level

        if score < 70.0:
            report.warnings.append("Low schema mapping confidence (<70%). Pipeline running in DEGRADED compatibility mode.")

        # Check core requirements
        has_user = "employee_id" in mapped_cols
        has_time = "event_timestamp" in mapped_cols
        has_type = "event_type" in mapped_cols
        
        if not (has_user and has_time and has_type):
            report.warnings.append("One or more core fields missing (employee_id, timestamp, event_type). Default fallbacks applied.")

        # Layers 2-4: Conversion to Internal Event Records
        records: List[InternalEventRecord] = []
        rows_dict = df.to_dict(orient="records")
        for idx, row in enumerate(rows_dict):
            rec = self.process_row_to_internal_record(row, mapped_cols, idx)
            records.append(rec)

        report.mapped_rows = len(records)
        report.skipped_rows = report.total_rows - report.mapped_rows

        # Layer 5: Smart Session Building
        sessions = self.build_sessions_from_records(records, employee_lookup)
        report.sessions_built = len(sessions)
        report.unique_employees = len(set(s.employee_id for s in sessions))

        # Layer 6: Feature Compatibility & Order Alignment Check
        if sessions:
            c_pass, o_pass, feat_warns = self.validate_features(sessions[0])
            report.feature_compatibility_pass = c_pass
            report.feature_name_alignment_pass = o_pass
            report.warnings.extend(feat_warns)
            if c_pass and o_pass:
                report.pipeline_status = "READY (Proceed with full/reduced feature set)"
            else:
                report.pipeline_status = "WARNING (Feature alignment issue)"
        else:
            report.pipeline_status = "EMPTY (No sessions reconstructed)"

        return sessions, report
