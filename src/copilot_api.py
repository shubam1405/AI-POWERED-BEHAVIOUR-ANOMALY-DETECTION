"""
copilot_api.py – FastAPI REST server for the Cyber Cage AI Security Copilot.

Enables on-demand report generation, Q&A chat, dashboard feeds, and campaign correlation
queries directly via REST endpoints. Also hosts a simulation engine for live SOC demonstrations.

Startup is intentionally lightweight: all heavy models and indexes are loaded lazily on
first use, keeping memory well below Render's 512 MiB free-tier limit.
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

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberCage.CopilotAPI")

# Initialize FastAPI App
app = FastAPI(
    title="Cyber Cage AI Security Copilot API",
    description="REST backend for UEBA anomaly triage, playbooks, campaigns, and chatbot.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loading globals (all None at startup)
# ---------------------------------------------------------------------------
_builder = None
_llm = None
_generator = None
_summarizer = None
_rec_engine = None
_chat = None
_correlator = None
_all_contexts = None
_campaigns_list = None
_campaign_index = None
_combined_sim = None

# Live simulation cache
SIMULATION_CACHE: Dict = {}

# Simulation behaviour → class mapping
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
    "malware_execution": "Malware Execution",
}


# ---------------------------------------------------------------------------
# Lazy getter functions — create on first call, cache globally
# ---------------------------------------------------------------------------

def get_builder():
    global _builder
    if _builder is None:
        from copilot.context_builder import ContextBuilder
        logger.info("Lazy-loading ContextBuilder ...")
        _builder = ContextBuilder().load()
    return _builder


def get_llm():
    global _llm
    if _llm is None:
        from copilot.llm_client import create_client
        logger.info("Lazy-loading LLM client ...")
        _llm = create_client()
    return _llm


def get_generator():
    global _generator
    if _generator is None:
        from copilot.report_generator import ReportGenerator
        logger.info("Lazy-loading ReportGenerator ...")
        _generator = ReportGenerator(get_llm())
    return _generator


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from copilot.incident_summarizer import IncidentSummarizer
        logger.info("Lazy-loading IncidentSummarizer ...")
        _summarizer = IncidentSummarizer(get_llm())
    return _summarizer


def get_rec_engine():
    global _rec_engine
    if _rec_engine is None:
        from copilot.recommendations import RecommendationEngine
        logger.info("Lazy-loading RecommendationEngine ...")
        _rec_engine = RecommendationEngine()
    return _rec_engine


def get_chat():
    global _chat
    if _chat is None:
        from copilot.analyst_chat import AnalystChat
        logger.info("Lazy-loading AnalystChat ...")
        _chat = AnalystChat(get_llm())
    return _chat


def get_correlator():
    global _correlator
    if _correlator is None:
        from copilot.correlation import IncidentCorrelator
        logger.info("Lazy-loading IncidentCorrelator ...")
        _correlator = IncidentCorrelator()
    return _correlator


def get_all_contexts():
    global _all_contexts
    if _all_contexts is None:
        logger.info("Lazy-loading all IncidentContexts via build_all() ...")
        _all_contexts = get_builder().build_all()
    return _all_contexts


def get_campaign_index():
    """Lazily build + correlate campaigns. Returns {campaign_id: Campaign}."""
    global _campaigns_list, _campaign_index
    if _campaign_index is None:
        logger.info("Lazy-loading campaign correlation ...")
        _campaigns_list = get_correlator().correlate(get_all_contexts())
        _campaign_index = {c.campaign_id: c for c in _campaigns_list}
    return _campaign_index


def get_combined_sim():
    """Lazily build the full inference pipeline (GRU + XGBoost + SHAP)."""
    global _combined_sim
    if _combined_sim is not None:
        return _combined_sim

    try:
        import shap
        import numpy as np
        from attack_classification.utils import ATTACK_LABELS, META_COLUMNS
        from simulator.company import Company
        from anomaly_detection.inference import InferenceEngine
        from attack_classification.inference import AttackInferenceEngine
        from simulator.combined_simulator import CombinedSimulator

        logger.info("Lazy-loading CombinedSimulator (GRU + XGBoost + SHAP) ...")

        company = Company()

        gru_engine = InferenceEngine(
            model_path="models/gru_autoencoder.pt",
            threshold=0.05,
        )
        gru_engine.calibrate(err_min=0.0, err_max=1.0)

        xgb_engine = AttackInferenceEngine(
            model_path="models/xgboost_attack_classifier.pkl",
            class_names=ATTACK_LABELS,
        )

        # Retrieve canonical feature names
        feature_names: List[str] = []
        try:
            from attack_classification.dataset_loader import AttackDatasetLoader
            loader = AttackDatasetLoader().load()
            feature_names = loader.feature_names
            logger.info("Retrieved %d feature names from AttackDatasetLoader.", len(feature_names))
        except Exception as dl_err:
            logger.warning("Could not load feature names via AttackDatasetLoader: %s", dl_err)
            tabular_csv = "data/processed/tabular_features.csv"
            if os.path.exists(tabular_csv):
                import csv
                with open(tabular_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    all_cols = reader.fieldnames or []
                feature_names = [c for c in all_cols if c not in META_COLUMNS]
                if "reconstruction_error" not in feature_names:
                    feature_names.append("reconstruction_error")
                if "anomaly_score" not in feature_names:
                    feature_names.append("anomaly_score")

        xgb_engine.feature_names = feature_names

        logger.info("Initialising SHAP TreeExplainer ...")
        shap_explainer = shap.TreeExplainer(
            xgb_engine.get_model(),
            feature_perturbation="interventional",
            model_output="raw",
        )

        _combined_sim = CombinedSimulator(
            company=company,
            gru_engine=gru_engine,
            xgb_engine=xgb_engine,
            shap_explainer=shap_explainer,
            feature_names=feature_names,
            class_names=ATTACK_LABELS,
        )
        logger.info(
            "CombinedSimulator ready (%d features, %d classes).",
            len(feature_names), len(ATTACK_LABELS),
        )
    except Exception as err:
        logger.warning("CombinedSimulator could not be initialised: %s", err)
        _combined_sim = None  # stay None so endpoint returns 503

    return _combined_sim


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class SimulationRequest(BaseModel):
    behaviours: List[str]
    employee_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_context(session_id: str):
    from copilot.utils import IncidentContext
    if session_id in SIMULATION_CACHE:
        return SIMULATION_CACHE[session_id]
    return get_builder().build(session_id)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Lightweight health check — does NOT initialise any heavy component."""
    return {
        "status": "healthy",
        "simulation_cache_size": len(SIMULATION_CACHE),
        "components_loaded": {
            "builder": _builder is not None,
            "llm": _llm is not None,
            "combined_sim": _combined_sim is not None,
            "campaign_index": _campaign_index is not None,
        },
    }


@app.get("/sessions")
def list_sessions(
    severity: Optional[str] = Query(None, description="Filter by severity: Low/Medium/High/Critical"),
    anomalous_only: bool = Query(True, description="Filter out normal sessions"),
):
    try:
        sessions = list(SIMULATION_CACHE.values())

        if anomalous_only:
            sessions += [s for s in get_builder().get_anomalous()]
        else:
            sessions += [s for s in get_builder().build_all()]

        if severity:
            sessions = [s for s in sessions if s.severity.lower() == severity.lower()]

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
        return get_generator().generate_report(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{session_id}/markdown")
def get_report_markdown(session_id: str):
    try:
        ctx = _get_context(session_id)
        report = get_generator().generate_report(ctx)
        return report["report_text_markdown"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{session_id}")
def get_summary(session_id: str):
    try:
        ctx = _get_context(session_id)
        return get_summarizer().generate_all(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{session_id}")
def get_recommendations(session_id: str):
    try:
        ctx = _get_context(session_id)
        recs = get_rec_engine().generate(ctx)
        return recs.__dict__
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{session_id}")
def post_chat(session_id: str, request: ChatRequest):
    try:
        ctx = _get_context(session_id)
        response = get_chat().ask(ctx, request.question)
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
            sessions += get_builder().get_high_critical()
        else:
            sessions += get_builder().get_anomalous()

        seen = set()
        unique_sessions = []
        for s in sessions:
            if s.session_id not in seen:
                seen.add(s.session_id)
                unique_sessions.append(s)

        cards = []
        for s in unique_sessions:
            recs = get_rec_engine().generate(s)
            sums = get_summarizer().generate_all(s)
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
                    "technique": s.mitre_technique,
                },
            })
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/campaigns")
def list_campaigns():
    return list(get_campaign_index().values())


@app.get("/campaign/{campaign_id}")
def get_campaign(campaign_id: str):
    camp = get_campaign_index().get(campaign_id.upper())
    if not camp:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found.")
    return camp


@app.post("/simulate")
def post_simulate_multi(request: SimulationRequest):
    """Generate a combined multi-behaviour session using the full inference pipeline."""
    from copilot.utils import IncidentContext

    sim = get_combined_sim()
    if sim is None:
        raise HTTPException(
            status_code=503,
            detail="CombinedSimulator not available. Check server startup logs.",
        )
    try:
        result = sim.simulate(
            behaviours=request.behaviours,
            employee_id=request.employee_id,
        )

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

        report = get_generator().generate_report(sim_ctx)
        recs = get_rec_engine().generate(sim_ctx)
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
    from copilot.utils import IncidentContext
    from config.config import RISK_ALERT_THRESHOLD

    try:
        mapped_class = ATTACK_MAPPING.get(attack_type.lower())
        if not mapped_class:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown attack type: {attack_type}. Supported: {list(ATTACK_MAPPING.keys())}",
            )

        all_ctx = get_all_contexts()
        candidates = [s for s in all_ctx if s.attack_type == mapped_class]
        if not candidates:
            candidates = [s for s in all_ctx if s.is_anomalous]

        base_ctx = random.choice(candidates)

        sim_id = f"SIM-{random.randint(100000, 999999)}"
        sim_emp = f"EMP-{random.randint(1000, 9999)}"
        now = datetime.utcnow()
        timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        camp_id = None
        ci = get_campaign_index()
        if mapped_class != "Normal" and ci:
            camp_id = random.choice(list(ci.keys()))
            camp = ci[camp_id]
            if sim_id not in camp.sessions:
                camp.sessions.append(sim_id)
                camp.session_count += 1
                if sim_emp not in camp.affected_employees:
                    camp.affected_employees.append(sim_emp)

        jitter_factor = random.uniform(0.95, 1.05)
        new_risk = min(100.0, max(0.0, base_ctx.risk_score * jitter_factor))
        new_anomaly = min(2.0, max(0.0, base_ctx.anomaly_score * jitter_factor))
        new_confidence = min(1.0, max(0.1, base_ctx.confidence * jitter_factor))

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
            "confidence": new_confidence,
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
                {"attack": "Lateral Movement", "probability": 0.0},
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
            timestamp=timestamp_str,
        )

        SIMULATION_CACHE[sim_id] = sim_ctx

        report = get_generator().generate_report(sim_ctx)
        recs = get_rec_engine().generate(sim_ctx)

        result = sim_ctx.__dict__.copy()
        result["report"] = report
        result["recommendations"] = recs.__dict__

        logger.info("Simulated session %s (%s) created.", sim_id, mapped_class)
        return result

    except Exception as e:
        logger.error("Simulation endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Static Files & React Frontend (optional — skip on Render if using Vercel)
# ---------------------------------------------------------------------------

_SERVE_FRONTEND = os.environ.get("SERVE_FRONTEND", "true").lower() != "false"
dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if _SERVE_FRONTEND and os.path.exists(dist_path):
    logger.info("Mounting React production assets from: %s", dist_path)
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(dist_path, "index.html"))

    @app.get("/{full_path:path}")
    def catch_all(full_path: str):
        file_path = os.path.join(dist_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_path, "index.html"))

else:
    if not _SERVE_FRONTEND:
        logger.info("Frontend serving disabled (SERVE_FRONTEND=false). API-only mode.")
    else:
        logger.warning(
            "Vite dist folder not found at %s. Serve app locally via: cd frontend && npm run dev",
            dist_path,
        )

    @app.get("/")
    def serve_fallback():
        return HTMLResponse(
            "<html>"
            "<body style='font-family: sans-serif; background: #0b0f19; color: #f8fafc; text-align: center; padding-top: 100px;'>"
            "<h1>🛡️ Cyber Cage XDR AI Dashboard</h1>"
            "<p style='color: #94a3b8;'>API is running. Frontend is served separately.</p>"
            "</body>"
            "</html>"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("copilot_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server(port=int(os.environ.get("PORT", 8000)))
