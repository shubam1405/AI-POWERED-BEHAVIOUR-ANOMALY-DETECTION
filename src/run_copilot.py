"""
run_copilot.py – Orchestrator for the Cyber Cage AI Security Copilot (Phase 8).

Processes anomalous session events, executes the LLM client or fallback template engine,
produces incident playbooks, runs graph correlation campaigns, and exports reports.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CyberCage.CopilotOrchestrator")

# ---------------------------------------------------------------------------
# Imports (after sys.path fix)
# ---------------------------------------------------------------------------
from copilot.context_builder import ContextBuilder
from copilot.llm_client import create_client
from copilot.report_generator import ReportGenerator
from copilot.incident_summarizer import IncidentSummarizer
from copilot.recommendations import RecommendationEngine
from copilot.analyst_chat import AnalystChat
from copilot.correlation import IncidentCorrelator
from copilot.exporter import CopilotExporter
from copilot.utils import ensure_dir, elapsed_str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cyber Cage Phase 8 — AI Security Copilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--explanations", default="outputs/session_explanations.json",
                   help="Path to Phase 7 session explanations JSON")
    p.add_argument("--predictions",  default="outputs/attack_predictions.csv",
                   help="Path to Phase 6 attack predictions CSV")
    p.add_argument("--dataset",      default="outputs/merged_dataset.csv",
                   help="Path to Phase 6 merged dataset CSV")
    p.add_argument("--output-dir",   default="outputs", dest="output_dir",
                   help="Base output directory")
    
    p.add_argument("--session",      default=None,
                   help="Process a single session ID specifically")
    p.add_argument("--severity",     default="High", choices=["Low", "Medium", "High", "Critical", "All"],
                   help="Filter sessions of this severity or higher (for batch reports)")
    p.add_argument("--all",          action="store_true",
                   help="Generate reports for ALL 13,015 sessions (extremely slow if calling remote LLM)")
    p.add_argument("--no-reports",   action="store_true", dest="no_reports",
                   help="Skip writing individual markdown/json report files (summaries/dashboard card updates only)")
    p.add_argument("--correlate-only", action="store_true", dest="correlate_only",
                   help="Run ONLY the campaign correlation modules, skip report writing")
    p.add_argument("--chat",         action="store_true",
                   help="Start interactive console chat UI (must specify --session)")
    p.add_argument("--provider",     default=None,
                   help="Force LLM provider: openai / azure / ollama / template")
    p.add_argument("--model",        default=None,
                   help="Override LLM model name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_total = time.time()

    ensure_dir(args.output_dir)

    logger.info("=" * 64)
    logger.info("Cyber Cage UEBA — AI Security Copilot Engine (Phase 8)")
    logger.info("=" * 64)

    # 1. Build unified contexts
    logger.info("[Phase 1] Building incident context indexes …")
    builder = ContextBuilder(
        explanations_path=args.explanations,
        predictions_path=args.predictions,
        dataset_path=args.dataset,
    ).load()

    # 2. Select and initialize LLM provider
    logger.info("[Phase 2] Initializing LLM client …")
    if args.provider:
        os.environ["COPILOT_PROVIDER"] = args.provider
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
        os.environ["OLLAMA_MODEL"] = args.model
    llm = create_client()
    logger.info("  Active provider: %s", llm.provider_name)

    # 3. Handle single-session mode
    if args.session:
        _run_single_session(args, builder, llm)
        return

    # 4. Handle correlation campaigns (always run on all anomalous sessions)
    logger.info("\n[Phase 3] Running XDR campaign correlation engine …")
    correlator = IncidentCorrelator()
    all_contexts = builder.build_all()
    campaigns = correlator.correlate(all_contexts)

    # Export campaigns
    exporter = CopilotExporter(args.output_dir)
    exporter.save_campaigns(campaigns)

    if args.correlate_only:
        logger.info("Correlation-only run requested. Exiting.")
        return

    # 5. Determine batch session target list
    logger.info("\n[Phase 4] Filtering session incident targets …")
    anomalous_sessions = builder.get_anomalous()
    logger.info("  Total anomalous sessions found: %d / %d", len(anomalous_sessions), len(all_contexts))

    targets = []
    if args.all:
        targets = anomalous_sessions
        logger.info("  --all specified. Targeting all %d anomalous sessions.", len(targets))
    else:
        # Filter by severity
        sev_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        limit_rank = sev_rank.get(args.severity, 3)  # default to High
        
        if args.severity == "All":
            targets = anomalous_sessions
        else:
            targets = [c for c in anomalous_sessions if sev_rank.get(c.severity, 1) >= limit_rank]
            logger.info("  Targeting sessions with severity >= %s: %d sessions", args.severity, len(targets))

    # 6. Generate incident playbooks, summaries, and reports
    logger.info("\n[Phase 5] Executing Copilot generators …")
    generator = ReportGenerator(llm)
    summarizer = IncidentSummarizer(llm)
    rec_engine = RecommendationEngine()

    generated_reports = []
    exec_summaries = []
    dashboard_cards = []

    t_gen = time.time()
    for idx, ctx in enumerate(targets):
        t_item = time.time()
        
        # Immediate Playbooks
        recs = rec_engine.generate(ctx)
        
        # Multi-audience summaries
        sums = summarizer.generate_all(ctx)
        
        # Save summary row
        exec_summaries.append({
            "session_id": ctx.session_id,
            "employee_id": ctx.employee_id,
            "attack_type": ctx.attack_type,
            "severity": ctx.severity,
            "confidence": ctx.confidence,
            "summary": sums["executive"],
        })

        # Save Dashboard card
        dashboard_cards.append({
            "session_id": ctx.session_id,
            "employee": ctx.employee_id,
            "attack_type": ctx.attack_type,
            "severity": ctx.severity,
            "confidence": round(ctx.confidence, 4),
            "risk_score": round(ctx.risk_score, 2),
            "summary": sums["dashboard"],
            "recommended_action": recs.priority_action,
            "mitre": {
                "tactic": ctx.mitre_tactic,
                "technique": ctx.mitre_technique
            },
            "top_features": ctx.top_positive_features,
            "report_path": f"outputs/incident_reports/incident_{ctx.session_id}.md"
        })

        # Report files (if not --no-reports)
        if not args.no_reports:
            logger.info("  [%d/%d] Generating report for %s (%s, risk: %.2f) ...", 
                        idx + 1, len(targets), ctx.session_id, ctx.attack_type, ctx.risk_score)
            
            report = generator.generate_report(ctx)
            generated_reports.append(report)
            
            # Export reports to filesystem
            exporter.save_session_report(report)
            
        else:
            if (idx + 1) % 100 == 0 or idx == len(targets) - 1:
                logger.info("  Processed summaries for %d / %d sessions ...", idx + 1, len(targets))

    # 7. Final exports
    logger.info("\n[Phase 6] Exporting aggregated artifacts …")
    exporter.save_executive_summaries_csv(exec_summaries)
    exporter.save_dashboard_cards(dashboard_cards)
    if generated_reports:
        exporter.save_copilot_reports(generated_reports)

    # 8. Completed summary logs
    logger.info("\n" + "=" * 64)
    logger.info("COPILOT ORCHESTRATION COMPLETE")
    logger.info("=" * 64)
    logger.info("Total processing time : %s", elapsed_str(t_total))
    logger.info("Incidents evaluated   : %d", len(all_contexts))
    logger.info("Campaigns detected    : %d", len(campaigns))
    logger.info("Reports written       : %d", len(generated_reports))
    logger.info("Dashboard cards saved : %d", len(dashboard_cards))
    logger.info("Outputs directory     : %s", args.output_dir)
    logger.info("=" * 64)


def _run_single_session(args: argparse.Namespace, builder: ContextBuilder, llm: Any) -> None:
    """Run interactive or single-session report/chat generation."""
    session_id = args.session
    logger.info("Single-session mode — targeting session: %s", session_id)

    try:
        ctx = builder.build(session_id)
    except Exception:
        logger.error("Session ID %s not found in merged indexes.", session_id)
        return

    # Chat Mode
    if args.chat:
        _interactive_chat_loop(ctx, llm)
        return

    # Standard report run
    logger.info("Generating reports & playbooks for session %s...", session_id)
    generator = ReportGenerator(llm)
    summarizer = IncidentSummarizer(llm)
    rec_engine = RecommendationEngine()
    exporter = CopilotExporter(args.output_dir)

    recs = rec_engine.generate(ctx)
    sums = summarizer.generate_all(ctx)
    report = generator.generate_report(ctx)

    # Save to disk
    exporter.save_session_report(report)
    
    # Save dashboard card sample
    card = {
        "session_id": ctx.session_id,
        "employee": ctx.employee_id,
        "attack_type": ctx.attack_type,
        "severity": ctx.severity,
        "confidence": ctx.confidence,
        "summary": sums["dashboard"],
        "recommended_action": recs.priority_action,
        "mitre": {
            "tactic": ctx.mitre_tactic,
            "technique": ctx.mitre_technique
        }
    }
    exporter.save_dashboard_cards([card])

    # Console display
    print("\n" + "=" * 64)
    print(f"SOC REPORT — {session_id}")
    print("=" * 64)
    print(report["report_text_markdown"])
    print("\n" + "=" * 64)
    print("AUDIENCE SUMMARIES")
    print("=" * 64)
    print(f"Dashboard Summary:\n  {sums['dashboard']}\n")
    print(f"Executive (150w):\n  {sums['executive']}\n")
    print(f"Email Alert Template:\n{sums['email']}\n")
    print("=" * 64)


def _interactive_chat_loop(ctx: IncidentContext, llm: Any) -> None:
    """Analyst console chat UI."""
    chat = AnalystChat(llm)
    
    print("\n" + "=" * 64)
    print(f"CYBER CAGE AI COPILOT — ANALYST CHAT")
    print(f"Grounded Context: {ctx.session_id} ({ctx.attack_type}, Severity: {ctx.severity})")
    print(f"Type 'exit', 'quit', or 'reset' to manage the chat session.")
    print("=" * 64)

    # Prompt analyst
    while True:
        try:
            query = input("\nAnalyst > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting chat session.")
                break
            
            t0 = time.time()
            response = chat.ask(ctx, query)
            duration = time.time() - t0
            
            print(f"\nCopilot [{duration:.2f}s] > {response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    main()
