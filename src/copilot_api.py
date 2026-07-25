"""
copilot_api.py – FastAPI REST server for the Cyber Cage AI Security Copilot.

Enables on-demand report generation, Q&A chat, dashboard feeds, and campaign correlation
queries directly via REST endpoints.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
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

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberCage.CopilotAPI")

# Initialize FastAPI App
app = FastAPI(
    title="Cyber Cage AI Security Copilot API",
    description="REST backend for UEBA anomaly triage, playbooks, campaigns, and chatbot.",
    version="1.0.0",
)

# CORS Middleware (Enable dashboard dashboard UI access)
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

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "provider": llm.provider_name,
        "indexed_sessions": builder.total_sessions,
        "active_campaigns": len(campaigns_list),
    }


@app.get("/sessions")
def list_sessions(
    severity: Optional[str] = Query(None, description="Filter by severity: Low/Medium/High/Critical"),
    anomalous_only: bool = Query(True, description="Filter out normal sessions")
):
    try:
        if anomalous_only:
            sessions = builder.get_anomalous()
        else:
            sessions = builder.build_all()

        if severity:
            sessions = [s for s in sessions if s.severity.lower() == severity.lower()]

        return [
            {
                "session_id": s.session_id,
                "employee_id": s.employee_id,
                "attack_type": s.attack_type,
                "severity": s.severity,
                "confidence": s.confidence,
                "risk_score": s.risk_score,
            }
            for s in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
def get_session(session_id: str):
    try:
        ctx = builder.build(session_id)
        # Convert dataclass to dict
        return ctx.__dict__
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{session_id}")
def get_report(session_id: str):
    try:
        ctx = builder.build(session_id)
        return generator.generate_report(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/{session_id}/markdown")
def get_report_markdown(session_id: str):
    try:
        ctx = builder.build(session_id)
        report = generator.generate_report(ctx)
        return report["report_text_markdown"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary/{session_id}")
def get_summary(session_id: str):
    try:
        ctx = builder.build(session_id)
        return summarizer.generate_all(ctx)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{session_id}")
def get_recommendations(session_id: str):
    try:
        ctx = builder.build(session_id)
        recs = rec_engine.generate(ctx)
        return recs.__dict__
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{session_id}")
def post_chat(session_id: str, request: ChatRequest):
    try:
        ctx = builder.build(session_id)
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
        if high_severity_only:
            sessions = builder.get_high_critical()
        else:
            sessions = builder.get_anomalous()

        cards = []
        for s in sessions:
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


# Entry point for runner
def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("copilot_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server(port=8000)
