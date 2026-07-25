"""
explain_xgboost.py – Orchestrator for the Cyber Cage XAI Engine (Phase 7).

Executes 7 phases:
    Phase 1  Load model + data, compute (or load cached) SHAP values
    Phase 2  Global explainability — mean |SHAP|, per-class importance
    Phase 3  Local explainability — per-session structured dicts
    Phase 4  Natural language generation — nl_explanation + summary
    Phase 5  Attach copilot_context (Phase 8 ready)
    Phase 6  Visualizations — 6 SHAP plot types
    Phase 7  Export — JSON, CSV, global CSVs

Usage
-----
From the project root::

    python src/explain_xgboost.py [options]

Options
-------
--model          PATH  xgboost_attack_classifier.pkl
--dataset        PATH  merged_dataset.csv
--predictions    PATH  attack_predictions.csv
--output-dir     PATH  outputs/
--plots-dir      PATH  outputs/plots/
--top-n          INT   Top N contributors per session (default: 10)
--no-plots            Skip visualization phase
--force-recompute     Ignore SHAP cache and recompute
--session        STR   Explain only this session_id (interactive mode)
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
logger = logging.getLogger("CyberCage.ExplainXGBoost")

# ---------------------------------------------------------------------------
# Imports (after sys.path fix)
# ---------------------------------------------------------------------------
from explainability.shap_engine          import SHAPEngine
from explainability.global_explainer     import GlobalExplainer
from explainability.local_explainer      import LocalExplainer
from explainability.explanation_generator import ExplanationGenerator
from explainability.visualization        import ExplainabilityVisualizer
from explainability.exporter             import AttackExplainExporter
from explainability.utils                import ensure_dir, elapsed_str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cyber Cage Phase 7 — SHAP Explainability Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",           default="models/xgboost_attack_classifier.pkl")
    p.add_argument("--dataset",         default="outputs/merged_dataset.csv")
    p.add_argument("--predictions",     default="outputs/attack_predictions.csv")
    p.add_argument("--output-dir",      default="outputs",        dest="output_dir")
    p.add_argument("--plots-dir",       default="outputs/plots",  dest="plots_dir")
    p.add_argument("--top-n",           type=int, default=10,     dest="top_n")
    p.add_argument("--no-plots",        action="store_true",      dest="no_plots")
    p.add_argument("--force-recompute", action="store_true",      dest="force_recompute")
    p.add_argument("--session",         default=None,
                   help="Explain a single session_id (interactive mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    t_total = time.time()

    ensure_dir(args.output_dir)
    ensure_dir(args.plots_dir)

    logger.info("=" * 64)
    logger.info("Cyber Cage UEBA — SHAP Explainability Engine (Phase 7)")
    logger.info("=" * 64)

    # ---------------------------------------------------------------
    # Phase 1 — Load model + data, compute SHAP
    # ---------------------------------------------------------------
    logger.info("\n[Phase 1] Loading model and computing SHAP values …")
    engine = SHAPEngine(
        model_path=args.model,
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        cache_path=os.path.join(args.output_dir, "shap_values.npz"),
        force_recompute=args.force_recompute,
    )
    engine.load().compute_shap()

    shap_values   = engine.shap_values
    X             = engine.X
    feature_names = engine.feature_names
    class_names   = engine.class_names
    df            = engine.df

    logger.info(
        "  SHAP tensor shape: %s  |  Features: %d  |  Classes: %d",
        shap_values.shape, len(feature_names), len(class_names),
    )

    # ---------------------------------------------------------------
    # Interactive single-session mode
    # ---------------------------------------------------------------
    if args.session:
        _interactive_mode(args, engine, feature_names, class_names, df, shap_values, X)
        return

    # ---------------------------------------------------------------
    # Phase 2 — Global explainability
    # ---------------------------------------------------------------
    logger.info("\n[Phase 2] Computing global feature importance …")
    global_exp = GlobalExplainer(
        shap_values=shap_values,
        feature_names=feature_names,
        class_names=class_names,
        output_dir=args.output_dir,
    )
    global_exp.compute()
    global_exp.save()
    global_exp.log_top_features(n=20)

    # ---------------------------------------------------------------
    # Phase 3 — Local explainability
    # ---------------------------------------------------------------
    logger.info("\n[Phase 3] Generating per-session explanations …")
    local_exp = LocalExplainer(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        class_names=class_names,
        df=df,
        top_n=args.top_n,
    )
    explanations = local_exp.explain_all(log_every=1000)

    # ---------------------------------------------------------------
    # Phase 4 — Natural language generation
    # ---------------------------------------------------------------
    logger.info("\n[Phase 4] Generating natural language explanations …")
    nl_gen = ExplanationGenerator()
    nl_gen.generate_batch(explanations)

    # ---------------------------------------------------------------
    # Phase 5 — Attach copilot_context
    # ---------------------------------------------------------------
    logger.info("\n[Phase 5] Attaching copilot_context for Phase 8 …")
    exporter = AttackExplainExporter(output_dir=args.output_dir)
    exporter.attach_copilot_context(explanations)

    # ---------------------------------------------------------------
    # Phase 6 — Visualizations
    # ---------------------------------------------------------------
    if not args.no_plots:
        logger.info("\n[Phase 6] Generating SHAP visualizations …")
        viz = ExplainabilityVisualizer(
            shap_values=shap_values,
            X=X,
            feature_names=feature_names,
            class_names=class_names,
            output_dir=args.plots_dir,
            explainer=engine.explainer,
        )
        plot_paths = viz.plot_all(top_n_features=20)
        logger.info("  %d plots generated.", len(plot_paths))
    else:
        logger.info("\n[Phase 6] Visualization SKIPPED (--no-plots).")

    # ---------------------------------------------------------------
    # Phase 7 — Export
    # ---------------------------------------------------------------
    logger.info("\n[Phase 7] Exporting explanations …")
    json_path = exporter.save_json(explanations)
    csv_path  = exporter.save_csv(explanations)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    top1 = global_exp.global_importance.iloc[0]
    severity_counts = {}
    for exp in explanations:
        sev = exp.get("severity", "Unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    logger.info("\n" + "=" * 64)
    logger.info("EXPLANATION COMPLETE")
    logger.info("=" * 64)
    logger.info("Total time            : %s", elapsed_str(t_total))
    logger.info("Sessions explained    : %d", len(explanations))
    logger.info("Features used         : %d", len(feature_names))
    logger.info("Classes               : %d", len(class_names))
    logger.info("Top global feature    : %s (mean|SHAP|=%.4f)",
                top1["feature"], top1["mean_abs_shap"])
    logger.info("Severity breakdown    : %s", severity_counts)
    logger.info("JSON output           : %s", json_path)
    logger.info("CSV output            : %s", csv_path)
    logger.info("Global importance     : %s/global_feature_importance.csv", args.output_dir)
    logger.info("Per-class importance  : %s/per_class_feature_importance.csv", args.output_dir)
    if not args.no_plots:
        logger.info("Plots                 : %s/", args.plots_dir)
    logger.info("=" * 64)


def _interactive_mode(
    args, engine, feature_names, class_names, df, shap_values, X
) -> None:
    """Explain a single session and print a formatted report."""
    import json

    session_id = args.session
    logger.info("Interactive mode — explaining session: %s", session_id)

    local_exp = LocalExplainer(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        class_names=class_names,
        df=df,
        top_n=args.top_n,
    )
    exp = local_exp.explain_session(session_id)

    nl_gen = ExplanationGenerator()
    nl_gen.generate(exp)

    exporter = AttackExplainExporter(output_dir=args.output_dir)
    exporter.attach_copilot_context([exp])

    # Pretty print
    print("\n" + "=" * 64)
    print(f"SESSION EXPLANATION — {session_id}")
    print("=" * 64)
    print(f"Prediction    : {exp['prediction']}")
    print(f"Confidence    : {exp['confidence']:.0%}")
    print(f"Severity      : {exp['severity']}")
    print(f"Anomaly Score : {exp['anomaly_score']:.4f}")
    print(f"Risk Score    : {exp['risk_score']:.2f}")
    mitre = exp.get("mitre")
    if mitre:
        print(f"MITRE         : {mitre['tactic']} — {mitre['technique_id']} {mitre['technique']}")
    print()
    print("─── Positive Contributors ───────────────────────────────")
    for c in exp.get("positive_contributors", [])[:5]:
        print(f"  {c['feature']:<35} value={c['value']:<10} impact={c['impact']}")
    print()
    print("─── Negative Contributors ───────────────────────────────")
    for c in exp.get("negative_contributors", [])[:3]:
        print(f"  {c['feature']:<35} shap={c['shap_value']:<10} impact={c['impact']}")
    print()
    print("─── Explanation ─────────────────────────────────────────")
    print(exp.get("nl_explanation", ""))
    print()
    print("─── Investigation Steps ─────────────────────────────────")
    for step in exp.get("investigation_steps", []):
        print(f"  • {step}")
    print()
    print("─── Copilot Context (JSON) ──────────────────────────────")
    print(json.dumps(exp.get("copilot_context", {}), indent=2))
    print("=" * 64)


if __name__ == "__main__":
    main()
