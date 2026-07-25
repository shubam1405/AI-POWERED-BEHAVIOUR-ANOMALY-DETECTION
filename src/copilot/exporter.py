"""
exporter.py – Generates all output artifacts: JSON reports, Markdown reports,
executive summaries, dashboard cards, consolidated copilot reports, and campaigns.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from copilot.utils import ensure_dir, sanitize_filename
from copilot.correlation import Campaign

logger = logging.getLogger("Copilot.Exporter")


class CopilotExporter:
    """Manages Phase 8 output file creation and dashboard card exports."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.out_dir = Path(output_dir)
        self.reports_dir = self.out_dir / "incident_reports"
        ensure_dir(self.out_dir)
        ensure_dir(self.reports_dir)

    def save_session_report(self, report: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save a single incident report in both JSON and Markdown formats.

        Parameters
        ----------
        report : Dict[str, Any]

        Returns
        -------
        Tuple[Path, Path]  (json_path, md_path)
        """
        session_id = report["session_id"]
        safe_id = sanitize_filename(session_id)
        
        json_path = self.reports_dir / f"incident_{safe_id}.json"
        md_path = self.reports_dir / f"incident_{safe_id}.md"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report["report_text_markdown"])

        return json_path, md_path

    def save_executive_summaries_csv(self, summaries: List[Dict[str, Any]]) -> Path:
        """Export flat CSV containing session summaries for rapid parsing."""
        csv_path = self.out_dir / "executive_summaries.csv"
        
        # Define fields
        fields = ["session_id", "employee_id", "attack_type", "severity", "confidence", "summary"]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summaries:
                writer.writerow({k: row.get(k, "") for k in fields})
                
        logger.info("Saved executive summaries CSV: %s", csv_path)
        return csv_path

    def save_dashboard_cards(self, cards: List[Dict[str, Any]]) -> Path:
        """Save JSON array of dashboard-ready widgets."""
        cards_path = self.out_dir / "dashboard_cards.json"
        with open(cards_path, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)
        logger.info("Saved dashboard cards JSON: %s", cards_path)
        return cards_path

    def save_copilot_reports(self, reports: List[Dict[str, Any]]) -> Path:
        """Save all generated incident reports into a single consolidated file."""
        reports_path = self.out_dir / "copilot_reports.json"
        with open(reports_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        logger.info("Saved consolidated copilot reports: %s", reports_path)
        return reports_path

    def save_campaigns(self, campaigns: List[Campaign]) -> Tuple[Path, Path]:
        """Save campaigns JSON and flat CSV summaries."""
        json_path = self.out_dir / "campaigns.json"
        csv_path = self.out_dir / "campaign_summary.csv"

        # Serialize Campaign objects
        camp_list = []
        for c in campaigns:
            camp_list.append({
                "campaign_id": c.campaign_id,
                "detection_basis": c.detection_basis,
                "attack_types": c.attack_types,
                "overall_severity": c.overall_severity,
                "session_count": c.session_count,
                "affected_employees": c.affected_employees,
                "common_source_ips": c.common_source_ips,
                "common_devices": c.common_devices,
                "sessions": c.sessions,
                "mitre_tactics": c.mitre_tactics,
                "is_attack_chain": c.is_attack_chain,
                "chain_description": c.chain_description,
                "summary": c.summary,
                "recommended_actions": c.recommended_actions,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
            })

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(camp_list, f, indent=2)

        # Save flat CSV summary
        fields = ["campaign_id", "overall_severity", "session_count", "affected_employees", "attack_types", "is_attack_chain"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for c in camp_list:
                writer.writerow({
                    "campaign_id": c["campaign_id"],
                    "overall_severity": c["overall_severity"],
                    "session_count": c["session_count"],
                    "affected_employees": ", ".join(c["affected_employees"]),
                    "attack_types": ", ".join(c["attack_types"]),
                    "is_attack_chain": c["is_attack_chain"],
                })

        logger.info("Saved campaigns files: %s and %s", json_path, csv_path)
        return json_path, csv_path
