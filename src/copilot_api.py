"""
copilot_api.py – FastAPI REST server for the Cyber Cage AI Security Copilot.

Enables on-demand report generation, Q&A chat, dashboard feeds, and campaign correlation
queries directly via REST endpoints. Also hosts a simulation engine for live SOC demonstrations
and serves the React production bundle from frontend/dist if available.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    print("FastAPI dependencies missing. Run: pip install fastapi uvicorn")
    sys.exit(1)

from copilot.context_builder import ContextBuilder
from copilot.llm_client import create_client
from copilot.report_generator import ReportGenerator
from copilot.incident_summarizer import IncidentSummarizer
from copilot.recommendations import RecommendationEngine
from copilot.analyst_chat import AnalystChat
from copilot.correlation import IncidentCorrelator, Campaign
from copilot.exporter import CopilotExporter
from copilot.utils import IncidentContext

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberCage.CopilotAPI")

# Initialize FastAPI App
app = FastAPI(
    title="Cyber Cage AI Security Copilot API",
    description="REST backend for UEBA anomaly triage, playbooks, campaigns, and chatbot.",
    version="1.0.0",
)

# CORS Middleware (Enable dashboard UI access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engines and indexes once at startup
logger.info("Initializing Copilot engines ...")
builder = ContextBuilder().load()
llm = create_client()
generator = ReportGenerator(llm)
summarizer = IncidentSummarizer(llm)
rec_engine = RecommendationEngine()
chat = AnalystChat(llm)
correlator = IncidentCorrelator()

# Precompute campaigns for instant lookup
all_contexts = builder.build_all()
campaigns_list: List[Campaign] = correlator.correlate(all_contexts)
campaign_index: Dict[str, Campaign] = {c.campaign_id: c for c in campaigns_list}

# Live simulation cache (keeps simulated session contexts in memory)
SIMULATION_CACHE: Dict[str, IncidentContext] = {}

# ---------------------------------------------------------------------------
# Load combined simulation engine at startup
# ---------------------------------------------------------------------------
_combined_sim = None
try:
    import shap
    from simulator.company import Company
    from anomaly_detection.inference import InferenceEngine
    from attack_classification.inference import AttackInferenceEngine
    from attack_classification.utils import ATTACK_LABELS, META_COLUMNS
    from simulator.combined_simulator import CombinedSimulator
    import pandas as pd

    _company = Company()

    # Load GRU engine
    _gru_engine = InferenceEngine(
        model_path="models/gru_autoencoder.pt",
        threshold=0.05,
    )
    # Calibrate with approximate training-set min/max
    _gru_engine.calibrate(err_min=0.0, err_max=1.0)

    # Load XGBoost engine and retrieve feature names
    _xgb_model_path = "models/xgboost_attack_classifier.pkl"
    _xgb_engine = AttackInferenceEngine(
        model_path=_xgb_model_path,
        class_names=ATTACK_LABELS,
    )

    # Retrieve canonical 71 feature names matching training dataset (tabular + GRU scores)
    _feature_names: List[str] = []
    try:
        from attack_classification.dataset_loader import AttackDatasetLoader
        _loader = AttackDatasetLoader().load()
        _feature_names = _loader.feature_names
        logger.info("Retrieved %d canonical feature names from AttackDatasetLoader.", len(_feature_names))
    except Exception as _dl_err:
        logger.warning("Could not load feature names via AttackDatasetLoader: %s", _dl_err)
        _tabular_csv = "data/processed/tabular_features.csv"
        if os.path.exists(_tabular_csv):
            import csv
            with open(_tabular_csv, "r", encoding="utf-8") as _f:
                _reader = csv.DictReader(_f)
                _all_cols = _reader.fieldnames or []
            _feature_names = [c for c in _all_cols if c not in META_COLUMNS]
            if "reconstruction_error" not in _feature_names:
                _feature_names.append("reconstruction_error")
            if "anomaly_score" not in _feature_names:
                _feature_names.append("anomaly_score")

    _xgb_engine.feature_names = _feature_names

    # Build SHAP TreeExplainer (cached at startup — fast for tree models)
    logger.info("Initialising SHAP TreeExplainer ...")
    _shap_explainer = shap.TreeExplainer(
        _xgb_engine.get_model(),
        feature_perturbation="interventional",
        model_output="raw",
    )

    _combined_sim = CombinedSimulator(
        company=_company,
        gru_engine=_gru_engine,
        xgb_engine=_xgb_engine,
        shap_explainer=_shap_explainer,
        feature_names=_feature_names,
        class_names=ATTACK_LABELS,
    )
    logger.info("CombinedSimulator ready (%d features, %d classes).", len(_feature_names), len(ATTACK_LABELS))
except Exception as _sim_init_err:
    logger.warning("CombinedSimulator could not be initialised: %s", _sim_init_err)

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    question: str

class SimulationRequest(BaseModel):
    behaviours: List[str]
    employee_id: Optional[str] = None

# Simulation mapping
ATTACK_MAPPING = {
    "normal_user": "Normal",
    "normal": "Normal",
    "brute_force": "Brute Force",
    "beaconing_c2": "Beaconing C2",
    "credential_stuffing": "Credential Stuffing",
    "device_spoofing": "Device Spoofing",
    "impossible_travel": "Impossible Travel",
    "insider_drift": "Insider Drift",
    "lateral_movement": "Lateral Movement",
    "privilege_escalation": "Privilege Escalation",
    "suspicious_powershell": "Suspicious PowerShell",
    "data_exfiltration": "Data Exfiltration",
    "low_slow_exfiltration": "Low-and-Slow Exfiltration",
    "usb_data_theft": "USB Data Theft",
    "off_hours_access": "Off-hours Access",
    "malware_execution": "Malware Execution"
}


# Helper to lookup context (checks simulation cache first)
def _get_context(session_id: str) -> IncidentContext:
    if session_id in SIMULATION_CACHE:
        return SIMULATION_CACHE[session_id]
    return builder.build(session_id)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "provider": llm.provider_name,
        "indexed_sessions": builder.total_sessions,
        "active_campaigns": len(campaigns_list),
        "simulated_sessions_count": len(SIMULATION_CACHE)
    }


@app.get("/sessions")
def list_sessions(
    severity: Optional[str] = Query(None, description="Filter by severity: Low/Medium/High/Critical"),
    anomalous_only: bool = Query(True, description="Filter out normal sessions")
):
    try:
        # Merge precomputed and simulated sessions
        sessions = list(SIMULATION_CACHE.values())
        
        if anomalous_only:
            sessions += [s for s in builder.get_anomalous()]
        else:
            sessions += [s for s in builder.build_all()]

        if severity:
            sessions = [s for s in sessions if s.severity.lower() == severity.lower()]

        # Remove duplicates by session ID
        seen = set()
        unique_sessions = []
        for s in sessions:
            if s.session_id not in seen:
                seen.add(s.session_id)
                unique_sessions.append(s)

        return [
            {
                "session_id": s.session_id,
                "employee_id": s.employee_id,
                "attack_type": s.attack_type,
                "severity": s.severity,
                "confidence": s.confidence,
                "risk_score": s.risk_score,
            }
            for s in unique_sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
def get_session(session_id: str):
    try:
        ctx = _get_context(session_id)
        return ctx.__dict__
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{session_id}")
def get_report(session_id: str):
    try:
        ctx = _get_context(session_id)
        return generator.generate_report(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{session_id}/markdown")
def get_report_markdown(session_id: str):
    try:
        ctx = _get_context(session_id)
        report = generator.generate_report(ctx)
        return report["report_text_markdown"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{session_id}")
def get_summary(session_id: str):
    try:
        ctx = _get_context(session_id)
        return summarizer.generate_all(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{session_id}")
def get_recommendations(session_id: str):
    try:
        ctx = _get_context(session_id)
        recs = rec_engine.generate(ctx)
        return recs.__dict__
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{session_id}")
def post_chat(session_id: str, request: ChatRequest):
    try:
        ctx = _get_context(session_id)
        response = chat.ask(ctx, request.question)
        return {
            "session_id": session_id,
            "question": request.question,
            "answer": response,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
def get_dashboard_cards(high_severity_only: bool = False):
    try:
        sessions = list(SIMULATION_CACHE.values())
        if high_severity_only:
            sessions += builder.get_high_critical()
        else:
            sessions += builder.get_anomalous()

        # Remove duplicates
        seen = set()
        unique_sessions = []
        for s in sessions:
            if s.session_id not in seen:
                seen.add(s.session_id)
                unique_sessions.append(s)

        cards = []
        for s in unique_sessions:
            recs = rec_engine.generate(s)
            sums = summarizer.generate_all(s)
            cards.append({
                "session_id": s.session_id,
                "employee": s.employee_id,
                "attack_type": s.attack_type,
                "severity": s.severity,
                "confidence": round(s.confidence, 4),
                "risk_score": round(s.risk_score, 2),
                "summary": sums["dashboard"],
                "recommended_action": recs.priority_action,
                "mitre": {
                    "tactic": s.mitre_tactic,
                    "technique": s.mitre_technique
                }
            })
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/campaigns")
def list_campaigns():
    return list(campaign_index.values())


@app.get("/campaign/{campaign_id}")
def get_campaign(campaign_id: str):
    camp = campaign_index.get(campaign_id.upper())
    if not camp:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found.")
    return camp


@app.post("/simulate")
def post_simulate_multi(request: SimulationRequest):
    """Generate a combined multi-behaviour session using the full inference pipeline."""
    if _combined_sim is None:
        raise HTTPException(
            status_code=503,
            detail="CombinedSimulator not available. Check server startup logs."
        )
    try:
        result = _combined_sim.simulate(
            behaviours=request.behaviours,
            employee_id=request.employee_id,
        )

        # Cache as IncidentContext so downstream endpoints (/chat, /report) work
        sim_id = result["session_id"]
        sim_ctx = IncidentContext(
            session_id=sim_id,
            employee_id=result["employee_id"],
            attack_type=result["attack_type"],
            confidence=result["confidence"],
            severity=result["severity"],
            risk_score=result["risk_score"],
            anomaly_score=result["anomaly_score"],
            mitre=result.get("mitre"),
            positive_contributors=result["positive_contributors"],
            negative_contributors=result["negative_contributors"],
            investigation_steps=result["investigation_steps"],
            nl_explanation=result["nl_explanation"],
            summary=result["summary"],
            copilot_context=result["copilot_context"],
            top3_predictions=result["top3_predictions"],
            session_start_hour=result.get("session_start_hour"),
            session_duration=result.get("session_duration"),
            source_ip=result.get("source_ip"),
            device_id=result.get("device_id"),
            timestamp=result.get("timestamp"),
            true_label=result.get("true_label", result["attack_type"]),
        )
        SIMULATION_CACHE[sim_id] = sim_ctx

        # Generate report and recommendations
        report = generator.generate_report(sim_ctx)
        recs = rec_engine.generate(sim_ctx)
        result["report"] = report
        result["recommendations"] = recs.__dict__

        logger.info("Multi-behaviour simulation %s → %s complete.", sim_id, result["attack_type"])
        return result
    except Exception as e:
        logger.error("Multi-behaviour simulation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/{attack_type}")
def post_simulate(attack_type: str):
    """Generate one synthetic session matching the attack type using model output cache."""
    try:
        mapped_class = ATTACK_MAPPING.get(attack_type.lower())
        if not mapped_class:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown attack type: {attack_type}. Supported: {list(ATTACK_MAPPING.keys())}"
            )

        # Filter the explanations list to find sessions matching mapped_class
        candidates = [s for s in all_contexts if s.attack_type == mapped_class]
        if not candidates:
            # Fall back to a random anomalous session if no direct matches
            candidates = [s for s in all_contexts if s.is_anomalous]

        base_ctx = random.choice(candidates)

        # Generate unique simulated session keys
        sim_id = f"SIM-{random.randint(100000, 999999)}"
        sim_emp = f"EMP-{random.randint(1000, 9999)}"
        
        # Build fresh timestamp
        now = datetime.utcnow()
        timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Set up a campaign relationship if simulated attack is not normal
        camp_id = None
        if mapped_class != "Normal":
            # Link to one of the active campaigns to show correlation
            camp_id = random.choice(list(campaign_index.keys()))
            # Append this session to that campaign's internal list
            camp = campaign_index[camp_id]
            if sim_id not in camp.sessions:
                camp.sessions.append(sim_id)
                camp.session_count += 1
                if sim_emp not in camp.affected_employees:
                    camp.affected_employees.append(sim_emp)

        # Add minor random jitter to the scores to simulate fresh model outputs
        jitter_factor = random.uniform(0.95, 1.05)
        new_risk = min(100.0, max(0.0, base_ctx.risk_score * jitter_factor))
        new_anomaly = min(2.0, max(0.0, base_ctx.anomaly_score * jitter_factor))
        new_confidence = min(1.0, max(0.1, base_ctx.confidence * jitter_factor))

        # Clone context with risk threshold checks
        from config.config import RISK_ALERT_THRESHOLD
        
        sim_attack = base_ctx.attack_type
        sim_severity = base_ctx.severity
        sim_mitre = base_ctx.mitre
        sim_steps = base_ctx.investigation_steps
        sim_top3 = base_ctx.top3_predictions
        sim_nl = base_ctx.nl_explanation
        sim_summary = base_ctx.summary
        sim_copilot = {
            **base_ctx.copilot_context,
            "campaign_id": camp_id,
            "risk_score": new_risk,
            "anomaly_score": new_anomaly,
            "confidence": new_confidence
        }
        
        if new_risk < RISK_ALERT_THRESHOLD:
            sim_attack = "Normal"
            sim_severity = "Low"
            sim_mitre = None
            sim_steps = []
            sim_nl = "No significant anomalies were detected. The session conforms to the user's established baseline."
            sim_summary = "No significant anomalies were detected. The session conforms to the user's established baseline."
            sim_copilot = {
                "detected_behaviours": [],
                "predicted_primary_attack": "Normal",
                "confidence": round(new_confidence, 4),
                "risk_score": new_risk,
                "anomaly_score": new_anomaly,
                "reconstruction_error": new_anomaly,
                "severity": "Low",
            }
            sim_top3 = [
                {"attack": "Normal", "probability": round(new_confidence, 4)},
                {"attack": "Device Spoofing", "probability": 0.0},
                {"attack": "Lateral Movement", "probability": 0.0}
            ]

        sim_ctx = IncidentContext(
            session_id=sim_id,
            employee_id=sim_emp,
            attack_type=sim_attack,
            confidence=new_confidence,
            severity=sim_severity,
            risk_score=new_risk,
            anomaly_score=new_anomaly,
            mitre=sim_mitre,
            positive_contributors=base_ctx.positive_contributors,
            negative_contributors=base_ctx.negative_contributors,
            investigation_steps=sim_steps,
            nl_explanation=sim_nl,
            summary=sim_summary,
            copilot_context=sim_copilot,
            top3_predictions=sim_top3,
            session_start_hour=now.hour,
            session_duration=base_ctx.session_duration,
            source_ip=f"10.10.{random.randint(1, 254)}.{random.randint(1, 254)}",
            device_id=f"DEV-{random.randint(1000, 9999)}",
            timestamp=timestamp_str
        )

        # Save to memory cache
        SIMULATION_CACHE[sim_id] = sim_ctx

        # Generate fresh report and playbooks
        report = generator.generate_report(sim_ctx)
        recs = rec_engine.generate(sim_ctx)

        # Attach reports back to dict representation for API return
        result = sim_ctx.__dict__.copy()
        result["report"] = report
        result["recommendations"] = recs.__dict__

        logger.info("Simulated session %s (%s) created.", sim_id, mapped_class)
        return result

    except Exception as e:
        logger.error("Simulation endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Static Files & React Frontend Server Routes
# ---------------------------------------------------------------------------

dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if os.path.exists(dist_path):
    logger.info("Mounting React production assets from: %s", dist_path)
    
    # Mount build assets
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(dist_path, "index.html"))

    @app.get("/{full_path:path}")
    def catch_all(full_path: str):
        # Allow client-side routing fallback to index.html
        file_path = os.path.join(dist_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_path, "index.html"))
else:
    logger.warning("Vite dist folder not found at %s. Serve app locally via Vite: 'npm run dev' inside frontend/", dist_path)

    @app.get("/")
    def serve_fallback():
        return HTMLResponse(
            "<html>"
            "<body style='font-family: sans-serif; background: #0b0f19; color: #f8fafc; text-align: center; padding-top: 100px;'>"
            "<h1>🛡️ Cyber Cage XDR AI Dashboard</h1>"
            "<p style='color: #94a3b8;'>React production build ('frontend/dist/') not found.</p>"
            "<p style='color: #34d399;'>Start development mode by running:</p>"
            "<pre style='background: #0f172a; padding: 15px; border-radius: 8px; width: max-content; margin: 10px auto; color: #a7f3d0;'>cd frontend; npm run dev</pre>"
            "<p style='color: #94a3b8;'>Or generate a production build with:</p>"
            "<pre style='background: #0f172a; padding: 15px; border-radius: 8px; width: max-content; margin: 10px auto; color: #a7f3d0;'>cd frontend; npm run build</pre>"
            "</body>"
            "</html>"
        )


# Entry point for runner
def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("copilot_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server(port=8000)
