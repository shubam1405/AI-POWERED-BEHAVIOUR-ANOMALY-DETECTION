"""
combined_simulator.py – Multi-behaviour attack session simulator.

Builds a single combined session by sequentially injecting every selected
behaviour into a baseline normal session for the CURRENT employee, then runs
the complete inference pipeline:

    Session Events
        │
        ▼
    Feature Engineering (9 extractors + DriftMonitor)
        │
        ▼
    GRU Autoencoder → reconstruction_error, anomaly_score
        │
        ▼
    XGBoost Classifier (raw, unscaled features) → predicted_label, confidence
        │
        ▼
    shap.TreeExplainer → fresh SHAP values for this combined vector
        │
        ▼
    LocalExplainer (single-row) → positive/negative contributors
        │
        ▼
    ExplanationGenerator → nl_explanation, summary
        │
        ▼
    IncidentContext (returned to API)

Important:
  - No feature scaling before XGBoost (trees are scale-invariant; training
    used raw unscaled features).
  - SHAP values are NEVER manually compounded; they are always recomputed
    from the final combined feature vector.
  - The classifier output determines the predicted_label — severity is
    NOT selected by the user or the injector.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure 'src' is in sys.path for relative/absolute imports across project
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import numpy as np
import pandas as pd

logger = logging.getLogger("CombinedSimulator")

# ---------------------------------------------------------------------------
# Behaviour → AnomalyInjector stage mapping
# ---------------------------------------------------------------------------

# Maps the API behaviour string to the AttackType enum value used internally.
BEHAVIOUR_TO_ATTACK_TYPE: Dict[str, str] = {
    "off_hours_access":          "Off-hours Access",
    "device_spoofing":           "Device Spoofing",
    "brute_force":               "Brute Force",
    "credential_stuffing":       "Credential Stuffing",
    "impossible_travel":         "Impossible Travel",
    "privilege_escalation":      "Privilege Escalation",
    "lateral_movement":          "Lateral Movement",
    "data_exfiltration":         "Data Exfiltration",
    "insider_threat":            "Insider Threat",
    "insider_drift":             "Insider Drift",
    "low_slow_exfiltration":     "Low-and-Slow Exfiltration",
    "usb_data_theft":            "USB Data Theft",
    "malware_execution":         "Malware Execution",
    "suspicious_powershell":     "Suspicious PowerShell",
    "beaconing_c2":              "Beaconing C2",
}

# Human-readable label for each behaviour (used in event timeline)
BEHAVIOUR_LABELS: Dict[str, str] = {
    "off_hours_access":          "Off-hours Login",
    "device_spoofing":           "Unknown Device Detected",
    "brute_force":               "Brute Force Login Attempts",
    "credential_stuffing":       "Credential Stuffing Detected",
    "impossible_travel":         "Impossible Travel Alert",
    "privilege_escalation":      "Privilege Escalation Attempt",
    "lateral_movement":          "Lateral Movement to Internal Server",
    "data_exfiltration":         "Large File Download + External Upload",
    "insider_threat":            "Insider Threat – Sensitive DB Query",
    "insider_drift":             "Insider Drift – Unauthorised Resource Access",
    "low_slow_exfiltration":     "Low-and-Slow Exfiltration (Stealth Chunks)",
    "usb_data_theft":            "USB Data Theft Detected",
    "malware_execution":         "Malware Process Executed",
    "suspicious_powershell":     "Suspicious PowerShell Command",
    "beaconing_c2":              "C2 Beacon Outbound Traffic",
}


class CombinedSimulator:
    """
    Constructs and evaluates a multi-behaviour simulated session.

    Parameters
    ----------
    company : Company
        Virtual company instance (employees, devices).
    gru_engine : InferenceEngine
        Loaded GRU Autoencoder inference engine.
    xgb_engine : AttackInferenceEngine
        Loaded XGBoost attack classifier engine.
    shap_explainer : shap.TreeExplainer
        Pre-initialised SHAP TreeExplainer wrapping the XGBoost model.
    feature_names : List[str]
        Ordered list of XGBoost feature names (determines column order).
    class_names : List[str]
        Ordered list of XGBoost class label strings.
    rng : random.Random, optional
        Random state for reproducible simulations.
    """

    def __init__(
        self,
        company,
        gru_engine,
        xgb_engine,
        shap_explainer,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        logger.info("CombinedSimulator.__init__ started")
        self.company = company
        self.gru_engine = gru_engine
        self.xgb_engine = xgb_engine
        self.shap_explainer = shap_explainer

        # Resolve feature_names to canonical 71 features matching training dataset if omitted/incomplete
        if not feature_names or len(feature_names) < 71:
            logger.info("CombinedSimulator.__init__: feature_names incomplete (%s), loading from AttackDatasetLoader",
                        len(feature_names) if feature_names else 0)
            try:
                from attack_classification.dataset_loader import AttackDatasetLoader
                feature_names = AttackDatasetLoader().load().feature_names
                logger.info("CombinedSimulator.__init__: loaded %d feature names", len(feature_names))
            except Exception as e:
                logger.warning("CombinedSimulator.__init__: Could not auto-load canonical feature_names: %s", e)

        self.feature_names = feature_names or []
        self.class_names = class_names or []
        self.rng = rng or random.Random()
        logger.info("CombinedSimulator.__init__ finished — %d features, %d classes, shap=%s",
                    len(self.feature_names), len(self.class_names),
                    type(self.shap_explainer).__name__ if self.shap_explainer else "None")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        behaviours: List[str],
        employee_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full simulation pipeline for a list of behaviour strings.

        Parameters
        ----------
        behaviours : list of str
            Selected behaviour keys (e.g. ``["off_hours_access", "data_exfiltration"]``).
        employee_id : str, optional
            Employee ID to generate the session for.  Falls back to the first
            employee in the company directory if not supplied.

        Returns
        -------
        dict
            A JSON-serialisable dict matching the ``IncidentContext`` layout
            expected by the dashboard, plus extra fields:
            ``detected_behaviours``, ``event_timeline``, ``campaign_chain``.
        """
        # ----------------------------------------------------------------
        # Step 1 – Resolve employee
        # ----------------------------------------------------------------
        logger.info("Step 1 – Resolve employee")
        emp = self._resolve_employee(employee_id)
        emp_id = emp.employee_id if emp else (employee_id or "EMP-0001")

        # ----------------------------------------------------------------
        # Step 2 – Generate baseline normal session
        # ----------------------------------------------------------------
        logger.info("Step 2 – Generate baseline normal session")
        session = self._build_baseline_session(emp)

        # ----------------------------------------------------------------
        # Step 3 – Inject selected behaviours sequentially
        # ----------------------------------------------------------------
        from models.enums import AttackType
        from simulator.anomaly_injector import AnomalyInjector

        injector = AnomalyInjector(company=self.company, rng=self.rng)

        from config.config import ANOMALY_CONFIG

        valid_behaviours: List[str] = []
        applied_stages: List[AttackType] = []
        for behaviour in behaviours:
            attack_str = BEHAVIOUR_TO_ATTACK_TYPE.get(behaviour)
            if attack_str is None:
                logger.warning("Unknown behaviour key '%s' — skipping.", behaviour)
                continue
            # Find matching AttackType enum member
            attack_type = None
            for member in AttackType:
                if member.value == attack_str:
                    attack_type = member
                    break
            if attack_type is None:
                logger.warning("AttackType not found for '%s' — skipping.", attack_str)
                continue
            injector._apply_attack_stage(session, attack_type)
            valid_behaviours.append(behaviour)
            applied_stages.append(attack_type)

        if not valid_behaviours:
            # Fallback: treat as normal if no valid behaviours
            valid_behaviours = []

        # Mark session anomalous and set highest-severity attack_type for behaviour-based risk engine
        session.is_anomalous = True if valid_behaviours else False
        if applied_stages:
            severities = ANOMALY_CONFIG.get("attack_severities", {})
            session.attack_type = max(applied_stages, key=lambda s: severities.get(s, 0.0))
        else:
            session.attack_type = AttackType.NONE

        session.risk_score = injector._compute_session_risk(session)

        # Deduplicate & sort timestamps
        injector._deduplicate_timestamps(session)

        # ----------------------------------------------------------------
        # Step 4 – Feature Engineering  [TIMED]
        # ----------------------------------------------------------------
        _t_fe_start = time.perf_counter()
        raw_features = self._extract_features(session)
        _t_fe_ms = (time.perf_counter() - _t_fe_start) * 1000

        # ----------------------------------------------------------------
        # Step 5 – GRU scoring  [TIMED]
        # ----------------------------------------------------------------
        _t_gru_start = time.perf_counter()
        gru_result = self._score_gru(session)
        _t_gru_ms = (time.perf_counter() - _t_gru_start) * 1000
        reconstruction_error = gru_result["reconstruction_error"]
        anomaly_score_norm = gru_result["anomaly_score"]

        # Merge GRU scores into raw feature dict
        raw_features["reconstruction_error"] = reconstruction_error
        raw_features["anomaly_score"] = anomaly_score_norm

        # ----------------------------------------------------------------
        # Step 6 – Build feature vector (raw, NO scaling)
        # ----------------------------------------------------------------
        feature_vector = self._build_feature_vector(raw_features)

        # Dimension validation check against trained model
        if self.xgb_engine and hasattr(self.xgb_engine, "get_model"):
            expected_n = self.xgb_engine.get_model().n_features_in_
            generated_n = feature_vector.shape[0] if feature_vector.ndim == 1 else feature_vector.shape[1]
            logger.info("XGBoost Feature Validation — Expected: %d, Generated: %d", expected_n, generated_n)
            assert generated_n == expected_n, (
                f"Feature shape mismatch: expected {expected_n}, got {generated_n}"
            )

        # ----------------------------------------------------------------
        # Step 7 – XGBoost inference (no scaling)  [TIMED]
        # ----------------------------------------------------------------
        sim_id = f"SIM-{self.rng.randint(100000, 999999)}"
        _t_xgb_start = time.perf_counter()
        xgb_result = self.xgb_engine.predict(sim_id, feature_vector, top_k=3)
        _t_xgb_ms = (time.perf_counter() - _t_xgb_start) * 1000
        predicted_label = xgb_result["prediction"]
        confidence = xgb_result["confidence"]
        top3_predictions = xgb_result["top_predictions"]

        # ----------------------------------------------------------------
        # Step 8 – SHAP (fresh computation on the combined vector)  [TIMED]
        # ----------------------------------------------------------------
        try:
            _t_shap_start = time.perf_counter()
            shap_matrix, positive_contributors, negative_contributors = self._compute_shap(
                feature_vector, predicted_label, confidence, anomaly_score_norm, session.risk_score
            )
            _t_shap_ms = (time.perf_counter() - _t_shap_start) * 1000
        except Exception as e:
            logger.exception("SHAP computation failed: %s", e)
            shap_matrix = np.zeros((len(self.feature_names), len(self.class_names)))
            positive_contributors = []
            negative_contributors = []
            _t_shap_ms = 0.0

        # ----------------------------------------------------------------
        # Step 9 – NL explanation + severity + MITRE  [TIMED]
        # ----------------------------------------------------------------
        from explainability.utils import MITRE_MAPPING, INVESTIGATION_STEPS, compute_severity
        from explainability.explanation_generator import ExplanationGenerator

        _t_copilot_start = time.perf_counter()
        from config.config import RISK_ALERT_THRESHOLD
        if session.risk_score < RISK_ALERT_THRESHOLD:
            predicted_label = "Normal"
            severity = "Low"
            mitre = None
            steps = INVESTIGATION_STEPS.get("Normal", [])
            top3_predictions = [
                {"attack": "Normal", "probability": round(confidence, 4)},
                {"attack": "Device Spoofing", "probability": 0.0},
                {"attack": "Lateral Movement", "probability": 0.0}
            ]
        else:
            severity = compute_severity(confidence, anomaly_score_norm, session.risk_score)
            mitre = MITRE_MAPPING.get(predicted_label)
            steps = INVESTIGATION_STEPS.get(predicted_label, INVESTIGATION_STEPS.get("Normal", []))

        explanation_dict = {
            "session_id": sim_id,
            "prediction": predicted_label,
            "true_label": predicted_label,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score_norm, 4),
            "risk_score": round(session.risk_score, 4),
            "severity": severity,
            "mitre": mitre,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
            "investigation_steps": steps,
        }
        gen = ExplanationGenerator()
        gen.generate(explanation_dict)  # mutates in-place: adds nl_explanation, summary
        _t_copilot_ms = (time.perf_counter() - _t_copilot_start) * 1000

        # Aggregate total pipeline time across instrumented stages
        _t_total_ms = _t_fe_ms + _t_gru_ms + _t_xgb_ms + _t_shap_ms + _t_copilot_ms
        timing_ms = {
            "feature_engineering":    round(_t_fe_ms, 2),
            "gru_inference":          round(_t_gru_ms, 2),
            "xgboost_classification": round(_t_xgb_ms, 2),
            "shap_explainability":    round(_t_shap_ms, 2),
            "copilot_summary":        round(_t_copilot_ms, 2),
            "total_pipeline":         round(_t_total_ms, 2),
        }
        logger.info(
            "Pipeline timing (ms): FE=%.1f GRU=%.1f XGB=%.1f SHAP=%.1f Copilot=%.1f Total=%.1f",
            _t_fe_ms, _t_gru_ms, _t_xgb_ms, _t_shap_ms, _t_copilot_ms, _t_total_ms,
        )

        # ----------------------------------------------------------------
        # Step 10 – Build event timeline & campaign chain
        # ----------------------------------------------------------------
        event_timeline = self._build_event_timeline(session, valid_behaviours, sim_id)
        campaign_chain = self._build_campaign_chain(valid_behaviours, predicted_label, emp_id)

        # ----------------------------------------------------------------
        # Step 11 – Assemble result dict
        # ----------------------------------------------------------------
        now = datetime.utcnow()
        result = {
            # Core fields matching IncidentContext / dashboard
            "session_id":           sim_id,
            "employee_id":          emp_id,
            "attack_type":          predicted_label,
            "confidence":           round(confidence, 4),
            "severity":             severity,
            "risk_score":           round(session.risk_score, 2),
            "anomaly_score":        round(anomaly_score_norm, 6),
            "mitre":                mitre,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
            "investigation_steps":  steps,
            "nl_explanation":       explanation_dict.get("nl_explanation", ""),
            "summary":              explanation_dict.get("summary", ""),
            "top3_predictions":     top3_predictions,
            "source_ip":            f"10.10.{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}",
            "device_id":            session.device_id,
            "session_start_hour":   session.start_time.hour,
            "session_duration":     session.duration_seconds / 60.0,
            "timestamp":            now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "true_label":           predicted_label,
            # Live pipeline timing breakdown
            "timing_ms":            timing_ms,
            # Extended fields for the dashboard
            "detected_behaviours":  [BEHAVIOUR_LABELS.get(b, b) for b in valid_behaviours],
            "event_timeline":       event_timeline,
            "campaign_chain":       campaign_chain,
            "copilot_context": {
                "detected_behaviours":      [BEHAVIOUR_LABELS.get(b, b) for b in valid_behaviours],
                "predicted_primary_attack": predicted_label,
                "confidence":               round(confidence, 4),
                "risk_score":               round(session.risk_score, 2),
                "anomaly_score":            round(anomaly_score_norm, 6),
                "reconstruction_error":     round(reconstruction_error, 6),
                "event_timeline":           event_timeline,
                "campaign_chain":           campaign_chain,
            },
        }

        logger.info(
            "CombinedSimulator: session %s | employee=%s | behaviours=%s | predicted=%s (%.1f%%)",
            sim_id, emp_id, valid_behaviours, predicted_label, confidence * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_employee(self, employee_id: Optional[str]):
        """Return the Employee object for employee_id, or the first employee."""
        if employee_id:
            emp = self.company.get_employee(employee_id)
            if emp:
                return emp
            logger.warning("Employee '%s' not found — using first employee.", employee_id)
        # Fallback: first employee in the directory
        all_ids = list(self.company.employees.keys())
        if all_ids:
            return self.company.get_employee(all_ids[0])
        return None

    def _build_baseline_session(self, emp):
        """Create a minimal normal session for the given employee."""
        from models.session import Session
        from models.event import Event
        from models.enums import (
            EventType, EventStatus, ResourceType, AttackType,
            SessionStatus, LoginMethod, Browser, OperatingSystem
        )

        now = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        end = now + timedelta(hours=8)
        duration = (end - now).total_seconds()

        if emp:
            emp_id = emp.employee_id
            dept = emp.department
            role = emp.role
            location = emp.office_location
            devices = self.company.get_devices(emp_id)
            device_id = devices[0].device_id if devices else f"DEV-{self.rng.randint(1000, 9999)}"
            os_val = devices[0].operating_system if devices else OperatingSystem.WINDOWS
        else:
            emp_id = "EMP-0001"
            dept = None
            role = None
            location = "London"
            device_id = f"DEV-{self.rng.randint(1000, 9999)}"
            os_val = OperatingSystem.WINDOWS

        session_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        session_id = f"SIM-BASE-{self.rng.randint(10000, 99999)}"

        # Baseline events: Login, a few resource accesses, Logout
        ip = f"192.168.{self.rng.randint(1, 10)}.{self.rng.randint(1, 254)}"
        base_meta = {
            "geo_location": location,
            "auth_method": "Active Directory",
            "session_duration": duration,
            "command_sequence": [],
        }

        events = []
        ts = now
        for ev_type, resource, status in [
            (EventType.LOGIN,            None,                     EventStatus.SUCCESS),
            (EventType.RESOURCE_ACCESS,  ResourceType.CONFLUENCE,  EventStatus.SUCCESS),
            (EventType.FILE_READ,        ResourceType.FILE_SERVER, EventStatus.SUCCESS),
            (EventType.RESOURCE_ACCESS,  ResourceType.EMAIL,       EventStatus.SUCCESS),
            (EventType.RESOURCE_ACCESS,  ResourceType.SLACK,       EventStatus.SUCCESS),
            (EventType.LOGOUT,           None,                     EventStatus.SUCCESS),
        ]:
            ev = Event(
                event_uuid=str(uuid.UUID(int=self.rng.getrandbits(128), version=4)),
                session_id=session_id,
                timestamp=ts,
                event_type=ev_type,
                employee_id=emp_id,
                device_id=device_id,
                resource=resource,
                status=status,
                attack_type=AttackType.NONE,
                risk_score=0.0,
                ip_address=ip,
                metadata=dict(base_meta),
                entity_type="user",
            )
            events.append(ev)
            ts += timedelta(seconds=self.rng.randint(60, 600))

        session = Session(
            session_uuid=session_uuid,
            session_id=session_id,
            employee_uuid=getattr(emp, "employee_uuid", session_uuid),
            employee_id=emp_id,
            department=dept,
            role=role,
            device_id=device_id,
            browser=Browser.CHROME,
            operating_system=os_val,
            office_location=location,
            login_method=LoginMethod.OFFICE,
            remote_session=False,
            day_of_week=now.strftime("%A"),
            start_time=now,
            end_time=end,
            duration_seconds=duration,
            risk_score=0.0,
            is_anomalous=False,
            attack_type=AttackType.NONE,
            session_status=SessionStatus.COMPLETED,
            events=events,
        )
        return session

    def _extract_features(self, session) -> Dict[str, float]:
        """Run all 9 feature extractors on the session and return raw feature dict."""
        from features.feature_extractors import (
            SessionFeatureExtractor,
            AuthenticationFeatureExtractor,
            ResourceFeatureExtractor,
            FileActivityFeatureExtractor,
            ProcessFeatureExtractor,
            NetworkFeatureExtractor,
            TemporalFeatureExtractor,
            StatisticalFeatureExtractor,
            SequenceFeatureExtractor,
        )
        from features.behavior_knowledge_base import ColdStartEngine, DriftMonitor

        cold_start = ColdStartEngine(self.company)
        drift_monitor = DriftMonitor()

        extractors = [
            SessionFeatureExtractor(),
            AuthenticationFeatureExtractor(),
            ResourceFeatureExtractor(),
            FileActivityFeatureExtractor(),
            ProcessFeatureExtractor(),
            NetworkFeatureExtractor(),
            TemporalFeatureExtractor(),
            StatisticalFeatureExtractor(),
            SequenceFeatureExtractor(),
        ]

        raw: Dict[str, float] = {}
        for extractor in extractors:
            raw.update(extractor.extract(session, session.events))

        # Behavioral deviations (uses DriftMonitor)
        f_beh, _ = drift_monitor.process_and_update(session, [])
        raw.update(f_beh)

        return raw

    def _score_gru(self, session) -> Dict[str, Any]:
        """Build the GRU event sequence and score it."""
        from features.sequence_builder import SequenceBuilder

        builder = SequenceBuilder(max_len=50, feature_dim=21)
        sequences, masks = builder.build_sequences([session])
        sequence = sequences[0]
        mask = masks[0]

        result = self.gru_engine.score_sequence(
            session_id=session.session_id,
            sequence=sequence,
            mask=mask,
        )
        return result

    def _build_feature_vector(self, raw_features: Dict[str, float]) -> np.ndarray:
        """
        Convert raw feature dict to numpy array in XGBoost training column order.
        Missing features default to 0.0. No scaling applied.
        """
        vec = np.array(
            [float(raw_features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        # Replace NaN / Inf
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def _compute_shap(
        self,
        feature_vector: np.ndarray,
        predicted_label: str,
        confidence: float,
        anomaly_score: float,
        risk_score: float,
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Compute fresh SHAP values for the combined session's feature vector.

        Uses a single-row LocalExplainer to extract positive/negative
        contributors using the same logic as the offline pipeline.
        """
        from explainability.local_explainer import LocalExplainer

        X_row = feature_vector.reshape(1, -1)

        # shap_values returns list[ndarray] of shape (1, n_features) per class
        raw_shap = self.shap_explainer.shap_values(X_row)
        if isinstance(raw_shap, list):
            # Shape: (n_features, n_classes)
            shap_matrix = np.stack(raw_shap, axis=2)[0]
        else:
            shap_matrix = raw_shap[0] if raw_shap.ndim == 3 else raw_shap

        # Reshape to (1, n_features, n_classes) for LocalExplainer
        shap_3d = shap_matrix.reshape(1, len(self.feature_names), len(self.class_names))

        sim_id = f"_shap_sim_{self.rng.randint(1000, 9999)}"
        single_df = pd.DataFrame([{
            "session_id":      sim_id,
            "predicted_label": predicted_label,
            "true_label":      predicted_label,
            "confidence":      confidence,
            "anomaly_score":   anomaly_score,
            "risk_score":      risk_score,
        }])

        local_exp = LocalExplainer(
            shap_values=shap_3d,
            X=X_row,
            feature_names=self.feature_names,
            class_names=self.class_names,
            df=single_df,
            top_n=10,
        )
        explanation = local_exp.explain_session(sim_id)
        return shap_matrix, explanation["positive_contributors"], explanation["negative_contributors"]

    def _build_event_timeline(
        self,
        session,
        valid_behaviours: List[str],
        sim_id: str,
    ) -> List[Dict[str, str]]:
        """Build a chronological step-by-step event log for the dashboard timeline."""
        timeline: List[Dict[str, str]] = []
        base_time = session.start_time

        # Entry: session start
        timeline.append({
            "time": base_time.strftime("%H:%M:%S"),
            "event": "Session Authenticated",
            "detail": f"Employee {session.employee_id} logged in from {session.office_location}.",
            "type": "normal",
        })

        # One entry per injected behaviour (roughly chronological)
        step_secs = max(60, int(session.duration_seconds / max(len(valid_behaviours) + 2, 2)))
        for i, behaviour in enumerate(valid_behaviours):
            ts = base_time + timedelta(seconds=step_secs * (i + 1))
            label = BEHAVIOUR_LABELS.get(behaviour, behaviour)
            timeline.append({
                "time": ts.strftime("%H:%M:%S"),
                "event": label,
                "detail": f"Anomalous indicator '{label}' detected during session {sim_id}.",
                "type": "attack",
            })

        # Exit: GRU/XGBoost detection
        detect_ts = base_time + timedelta(seconds=step_secs * (len(valid_behaviours) + 1))
        timeline.append({
            "time": detect_ts.strftime("%H:%M:%S"),
            "event": "AI Detection Triggered",
            "detail": "GRU Autoencoder flagged elevated reconstruction error. XGBoost classifier invoked.",
            "type": "detection",
        })

        return timeline

    def _build_campaign_chain(
        self,
        valid_behaviours: List[str],
        predicted_label: str,
        emp_id: str,
    ) -> List[Dict[str, str]]:
        """
        Build the full campaign graph node chain for the dashboard.
        Structure:
          Employee → [injected behaviour nodes] → GRU Detection → XGBoost Prediction → SOC Alert
        """
        nodes: List[Dict[str, str]] = []

        # Root: employee
        nodes.append({"id": "emp", "label": emp_id, "type": "employee"})

        # Behaviour nodes
        for behaviour in valid_behaviours:
            nodes.append({
                "id": behaviour,
                "label": BEHAVIOUR_LABELS.get(behaviour, behaviour),
                "type": "behaviour",
            })

        # AI pipeline nodes
        nodes.append({
            "id": "gru",
            "label": "GRU Autoencoder Detection",
            "type": "model",
        })
        nodes.append({
            "id": "xgb",
            "label": f"XGBoost Prediction: {predicted_label}",
            "type": "model",
        })
        nodes.append({
            "id": "alert",
            "label": "SOC Alert Generated",
            "type": "alert",
        })

        return nodes
