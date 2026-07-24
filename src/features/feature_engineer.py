import csv
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

from models.enums import Browser, OperatingSystem, DepartmentName, EventType, ResourceType, EventStatus, LoginMethod, AttackType, SessionStatus
from models.session import Session
from models.event import Event
from features.data_adapter import DataAdapter
from features.behavior_knowledge_base import ColdStartEngine, DriftMonitor
from features.feature_extractors import (
    SessionFeatureExtractor,
    AuthenticationFeatureExtractor,
    ResourceFeatureExtractor,
    FileActivityFeatureExtractor,
    ProcessFeatureExtractor,
    NetworkFeatureExtractor,
    TemporalFeatureExtractor,
    StatisticalFeatureExtractor,
    SequenceFeatureExtractor
)
from features.scaler import StandardScaler
from config.config import RISK_WEIGHTS, ANOMALY_CONFIG, ENABLE_DRIFT_MONITORING
from simulator.company import Company

logger = logging.getLogger("FeatureEngineer")

# Feature category mappings for metadata
FEATURE_METADATA = {
    # Session Features
    "session_duration": {"category": "Temporal Features", "description": "Session duration in seconds", "fields": ["session_duration"]},
    "login_hour": {"category": "Temporal Features", "description": "Hour of login", "fields": ["timestamp"]},
    "logout_hour": {"category": "Temporal Features", "description": "Hour of logout", "fields": ["timestamp"]},
    "day_of_week": {"category": "Temporal Features", "description": "Day of the week (0-6)", "fields": ["timestamp"]},
    "weekend_flag": {"category": "Temporal Features", "description": "1.0 if session is on weekend, else 0.0", "fields": ["timestamp"]},
    "remote_session": {"category": "Location Features", "description": "1.0 if remote connection, else 0.0", "fields": ["auth_method", "source_ip"]},
    "vpn_used": {"category": "Network Features", "description": "1.0 if VPN authentication was used, else 0.0", "fields": ["auth_method"]},
    "browser": {"category": "Device Features", "description": "Numerical code for the browser used", "fields": ["device_fingerprint"]},
    "operating_system": {"category": "Device Features", "description": "Numerical code for the operating system used", "fields": ["device_fingerprint"]},
    "department": {"category": "Behavior Features", "description": "Numerical code for the entity department", "fields": ["entity_id"]},

    # Authentication Features
    "failed_login_count": {"category": "Authentication Features", "description": "Number of failed login attempts", "fields": ["auth_method", "label"]},
    "successful_login_count": {"category": "Authentication Features", "description": "Number of successful logins", "fields": ["auth_method", "label"]},
    "login_method": {"category": "Authentication Features", "description": "Authentication method type", "fields": ["auth_method"]},
    "authentication_attempts": {"category": "Authentication Features", "description": "Total authentication attempts", "fields": ["auth_method"]},
    "failed_login_ratio": {"category": "Authentication Features", "description": "Ratio of failed logins over total attempts", "fields": ["auth_method"]},

    # Resource Access Features
    "total_resource_accesses": {"category": "Resource Features", "description": "Total resource access counts", "fields": ["resource_accessed"]},
    "unique_resources": {"category": "Resource Features", "description": "Count of unique resources accessed", "fields": ["resource_accessed"]},
    "admin_resource_accesses": {"category": "Resource Features", "description": "Count of admin-level resource accesses", "fields": ["resource_accessed"]},
    "sensitive_resource_accesses": {"category": "Resource Features", "description": "Count of sensitive database/server accesses", "fields": ["resource_accessed"]},
    "github_access_count": {"category": "Resource Features", "description": "Access counts to GitHub", "fields": ["resource_accessed"]},
    "database_query_count": {"category": "Resource Features", "description": "Count of database queries performed", "fields": ["resource_accessed"]},
    "confluence_access_count": {"category": "Resource Features", "description": "Access counts to Confluence", "fields": ["resource_accessed"]},
    "hrms_access_count": {"category": "Resource Features", "description": "Access counts to HRMS system", "fields": ["resource_accessed"]},
    "erp_access_count": {"category": "Resource Features", "description": "Access counts to ERP system", "fields": ["resource_accessed"]},

    # File Activity Features
    "files_read": {"category": "Resource Features", "description": "Count of file read events", "fields": ["resource_accessed"]},
    "files_written": {"category": "Resource Features", "description": "Count of file write events", "fields": ["resource_accessed"]},
    "files_uploaded": {"category": "Resource Features", "description": "Count of file upload events", "fields": ["resource_accessed"]},
    "files_downloaded": {"category": "Resource Features", "description": "Count of file download events", "fields": ["resource_accessed"]},
    "total_upload_bytes": {"category": "Resource Features", "description": "Bytes transferred in upload events", "fields": ["resource_accessed"]},
    "total_download_bytes": {"category": "Resource Features", "description": "Bytes transferred in download events", "fields": ["resource_accessed"]},
    "upload_download_ratio": {"category": "Resource Features", "description": "Ratio of uploaded bytes over downloaded bytes", "fields": ["resource_accessed"]},

    # Process Features
    "processes_started": {"category": "Command Features", "description": "Count of process executions started", "fields": ["command_sequence"]},
    "processes_stopped": {"category": "Command Features", "description": "Count of process executions stopped", "fields": ["command_sequence"]},
    "powershell_executions": {"category": "Command Features", "description": "Count of PowerShell script executions", "fields": ["command_sequence"]},
    "suspicious_process_count": {"category": "Command Features", "description": "Count of administrative or privilege escalation tools run", "fields": ["command_sequence"]},
    "malware_execution_count": {"category": "Command Features", "description": "Count of recognized malware signatures executed", "fields": ["command_sequence"]},

    # Network Features
    "network_connections": {"category": "Network Features", "description": "Total network connections", "fields": ["source_ip"]},
    "vpn_connections": {"category": "Network Features", "description": "VPN connection count", "fields": ["auth_method"]},
    "vpn_disconnections": {"category": "Network Features", "description": "VPN disconnection count", "fields": ["auth_method"]},
    "unique_destination_ips": {"category": "Network Features", "description": "Count of unique destination servers/domains", "fields": ["command_sequence"]},
    "unique_source_ips": {"category": "Network Features", "description": "Count of unique source IPs used", "fields": ["source_ip"]},
    "external_connections": {"category": "Network Features", "description": "Count of connections to external domains", "fields": ["command_sequence"]},
    "beaconing_event_count": {"category": "Network Features", "description": "Count of periodic outbound C2 beacons", "fields": ["command_sequence"]},

    # Temporal Features
    "total_events": {"category": "Temporal Features", "description": "Total count of events in session", "fields": ["timestamp"]},
    "events_per_minute": {"category": "Temporal Features", "description": "Average number of events per minute", "fields": ["timestamp", "session_duration"]},
    "average_time_between_events": {"category": "Temporal Features", "description": "Average gap between consecutive events in seconds", "fields": ["timestamp"]},
    "maximum_idle_time": {"category": "Temporal Features", "description": "Maximum idle time between events in seconds", "fields": ["timestamp"]},
    "minimum_idle_time": {"category": "Temporal Features", "description": "Minimum idle time between events in seconds", "fields": ["timestamp"]},
    "off_hours_access": {"category": "Temporal Features", "description": "1.0 if session started during off-hours, else 0.0", "fields": ["timestamp"]},
    "session_start_hour": {"category": "Temporal Features", "description": "Session starting hour (0-23)", "fields": ["timestamp"]},

    # Sequence Features
    "first_event_type": {"category": "Sequence Features", "description": "Type of the first event in the session", "fields": ["timestamp"]},
    "last_event_type": {"category": "Sequence Features", "description": "Type of the last event in the session", "fields": ["timestamp"]},
    "event_sequence_length": {"category": "Sequence Features", "description": "Event sequence length", "fields": ["timestamp"]},
    "login_to_first_resource_time": {"category": "Sequence Features", "description": "Inactivity delay between login and first resource access", "fields": ["timestamp"]},
    "admin_access_position": {"category": "Sequence Features", "description": "Relative timeline position of first administrative access", "fields": ["timestamp"]},
    "file_download_position": {"category": "Sequence Features", "description": "Relative timeline position of first file download", "fields": ["timestamp"]},
    "process_execution_position": {"category": "Sequence Features", "description": "Relative timeline position of first process start", "fields": ["timestamp"]},

    # Behavioral Deviations (Behavior Features)
    "location_deviation": {"category": "Behavior Features", "description": "1.0 if geolocation differs from baseline, else 0.0", "fields": ["geo_location"]},
    "device_deviation": {"category": "Behavior Features", "description": "1.0 if device fingerprint differs from baseline, else 0.0", "fields": ["device_fingerprint"]},
    "browser_deviation": {"category": "Behavior Features", "description": "1.0 if browser differs from baseline, else 0.0", "fields": ["device_fingerprint"]},
    "operating_system_deviation": {"category": "Behavior Features", "description": "1.0 if operating system differs from baseline, else 0.0", "fields": ["device_fingerprint"]},
    "working_hours_deviation": {"category": "Behavior Features", "description": "1.0 if login hour is > 2 std dev outside average working hours, else 0.0", "fields": ["timestamp"]},
    "resource_access_deviation": {"category": "Behavior Features", "description": "1.0 if accessing resource never accessed historically, else 0.0", "fields": ["resource_accessed"]}
}

class FeatureEngineer:
    """
    Orchestrates raw data standardized sessionization, 
    Behavior Knowledge Base maintenance (Cold Start + Drift Monitor),
    and split feature vector generation (Tabular vs Sequential).
    """
    def __init__(self, data_dir: str = "data/raw", company: Company = None):
        self.data_dir = data_dir
        self.sessions_file = os.path.join(data_dir, "sessions_anomalous.csv")
        self.events_file = os.path.join(data_dir, "events_anomalous.csv")
        self.employees_file = os.path.join(data_dir, "employees.csv")
        self.company = company if company else Company()

    def run(self, sessions: List[Session] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Processes logs, extracts split features, computes risk scores, and outputs metadata context."""
        logger.info("Executing Intelligence Fusion & Feature Engineering Pipeline...")
        
        if sessions is None:
            # 1. Parse and standardize inputs via DataAdapter
            raw_events = self._read_raw_csv_events(self.events_file)
            
            # Read employees profile mapping from employees.csv to assist sessionization
            employees_lookup = self._load_employee_lookup()
            
            adapter = DataAdapter()
            sessions = adapter.standardize_and_sessionize(raw_events, employees_lookup)
        
        # Sort chronologically to preserve baseline integrity
        sessions.sort(key=lambda s: s.start_time)

        # 2. Initialize baselines and extractor services
        cold_start = ColdStartEngine(self.company)
        drift_monitor = DriftMonitor()

        sess_extractor = SessionFeatureExtractor()
        auth_extractor = AuthenticationFeatureExtractor()
        res_extractor = ResourceFeatureExtractor()
        file_extractor = FileActivityFeatureExtractor()
        proc_extractor = ProcessFeatureExtractor()
        net_extractor = NetworkFeatureExtractor()
        temp_extractor = TemporalFeatureExtractor()
        stat_extractor = StatisticalFeatureExtractor()
        seq_extractor = SequenceFeatureExtractor()

        raw_tabular_features: List[Dict[str, Any]] = []
        copilot_context_records: Dict[str, Any] = {}

        # Running histories map: {entity_id: List[Session]}
        entity_session_histories: Dict[str, List[Session]] = {}

        for session in sessions:
            sid = session.session_id
            entity_id = session.employee_id
            
            # Fetch history sessions prior to this session
            history = entity_session_histories.get(entity_id, [])
            
            # Retrieve effective baseline profile
            effective_baseline = cold_start.get_effective_baseline(entity_id, history)
            
            # Temporal, resource, device, location, network, and command extraction
            f_sess = sess_extractor.extract(session, session.events)
            f_auth = auth_extractor.extract(session, session.events)
            f_res = res_extractor.extract(session, session.events)
            f_file = file_extractor.extract(session, session.events)
            f_proc = proc_extractor.extract(session, session.events)
            f_net = net_extractor.extract(session, session.events)
            f_temp = temp_extractor.extract(session, session.events)
            f_stat = stat_extractor.extract(session, session.events)
            f_seq = seq_extractor.extract(session, session.events)

            # Compute behavioral deviations and behavior score using DriftMonitor service
            f_beh, behavior_score = drift_monitor.process_and_update(session, history)

            # Re-update baseline running history
            entity_session_histories.setdefault(entity_id, []).append(session)

            # -------------------------------------------------------------
            # Future-Proof Risk Score & Intelligence Fusion
            # -------------------------------------------------------------
            w_behav = RISK_WEIGHTS.get("behavior_deviation", 0.30)
            w_severity = RISK_WEIGHTS.get("attack_severity", 0.40)
            w_recon = RISK_WEIGHTS.get("reconstruction_error", 0.15)
            w_class = RISK_WEIGHTS.get("classifier_confidence", 0.15)

            # Placeholders for future Autoencoder / Classifier outputs
            recon_error = 0.0
            classifier_confidence = 0.0

            # Map attack severity (if anomalous)
            severity = 0.0
            if session.is_anomalous:
                severities = ANOMALY_CONFIG.get("attack_severities", {})
                severity = severities.get(session.attack_type, 0.0)

            # Compute integrated risk score
            final_risk = w_behav * behavior_score + w_severity * severity + w_recon * recon_error + w_class * classifier_confidence
            final_risk = min(100.0, max(0.0, final_risk))

            # Store tabular row
            tabular_row = {}
            tabular_row["session_id"] = session.session_id
            tabular_row["employee_id"] = session.employee_id
            tabular_row["is_anomalous"] = 1.0 if session.is_anomalous else 0.0
            tabular_row["attack_type"] = session.attack_type.value
            tabular_row["risk_score"] = float(final_risk)

            # Merge features
            tabular_row.update(f_sess)
            tabular_row.update(f_auth)
            tabular_row.update(f_res)
            tabular_row.update(f_file)
            tabular_row.update(f_proc)
            tabular_row.update(f_net)
            tabular_row.update(f_temp)
            tabular_row.update(f_stat)
            tabular_row.update(f_seq)
            tabular_row.update(f_beh)

            raw_tabular_features.append(tabular_row)

            # Expose raw event mappings for SHAP explainability
            event_mappings = [ev.event_uuid for ev in session.events]

            # Collect lightweight copilot context parameters
            copilot_context_records[session.session_id] = {
                "session_context": {
                    "entity_id": session.employee_id,
                    "entity_type": session.events[0].entity_type if session.events else "user",
                    "office_location": session.office_location,
                    "device_fingerprint": session.device_id,
                    "duration_seconds": session.duration_seconds,
                    "start_time": session.start_time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "risk_metadata": {
                    "integrated_risk_score": final_risk,
                    "attack_type": session.attack_type.value,
                    "is_anomalous": session.is_anomalous,
                    "behavior_score": behavior_score,
                    "severity": severity
                },
                "behavior_metadata": {
                    "deviations": f_beh
                },
                "shap_event_mapping": event_mappings
            }

        # 3. Fit StandardScaler and scale tabular features
        ignore_cols = ["session_id", "employee_id", "is_anomalous", "attack_type", "risk_score"]
        scaler = StandardScaler()
        scaler.fit(raw_tabular_features, ignore_cols)
        
        os.makedirs("data/processed", exist_ok=True)
        scaler.save("data/processed/scaler.pkl")

        scaled_tabular_features = scaler.transform(raw_tabular_features, ignore_cols)

        # 4. Save metadata files
        with open("data/processed/feature_metadata.json", "w") as f:
            json.dump(FEATURE_METADATA, f, indent=4)
        
        with open("data/processed/copilot_context.json", "w") as f:
            json.dump(copilot_context_records, f, indent=4)

        logger.info("Successfully finished Feature Engineering extraction, Risk score fusion, and Scaled exports.")
        return raw_tabular_features, scaled_tabular_features, copilot_context_records

    def _read_raw_csv_events(self, file_path: str) -> List[Dict[str, Any]]:
        """Utility to parse flat events CSV file into raw dictionaries."""
        events_raw = []
        if not os.path.exists(file_path):
            logger.warning(f"Events file {file_path} not found.")
            return []
        
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events_raw.append(row)
        return events_raw

    def _load_employee_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Utility to load employee profiles from employees.csv for baseline mapping."""
        lookup = {}
        if os.path.exists(self.employees_file):
            with open(self.employees_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lookup[row["employee_id"]] = row
        return lookup
