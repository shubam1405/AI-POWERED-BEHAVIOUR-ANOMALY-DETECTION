import csv
import os
import json
import logging
from typing import List

from models.session import Session
from models.event import Event

logger = logging.getLogger("AnomalyExporter")

class AnomalyExporter:
    """
    Exports anomalous sessions and events data to CSV files.
    """
    def __init__(self, sessions: List[Session], events: List[Event], output_dir: str = "data/raw"):
        self.sessions = sessions
        self.events = events
        self.output_dir = output_dir

    def export(self) -> None:
        """Writes anomalous sessions and events to their respective CSV files."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._export_sessions()
        self._export_events()

    def _export_sessions(self) -> None:
        filepath = os.path.join(self.output_dir, "sessions_anomalous.csv")
        logger.info(f"Exporting anomalous sessions to {filepath}...")

        headers = [
            "session_uuid",
            "session_id",
            "employee_uuid",
            "employee_id",
            "department",
            "role",
            "device_id",
            "browser",
            "operating_system",
            "office_location",
            "login_method",
            "remote_session",
            "day_of_week",
            "start_time",
            "end_time",
            "duration_seconds",
            "risk_score",
            "is_anomalous",
            "attack_type"
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for session in self.sessions:
                row = {
                    "session_uuid": session.session_uuid,
                    "session_id": session.session_id,
                    "employee_uuid": session.employee_uuid,
                    "employee_id": session.employee_id,
                    "department": session.department.value,
                    "role": session.role,
                    "device_id": session.device_id,
                    "browser": session.browser.value,
                    "operating_system": session.operating_system.value,
                    "office_location": session.office_location,
                    "login_method": session.login_method.value,
                    "remote_session": str(session.remote_session),
                    "day_of_week": session.day_of_week,
                    "start_time": session.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": session.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": f"{session.duration_seconds:.2f}",
                    "risk_score": f"{session.risk_score:.2f}",
                    "is_anomalous": str(session.is_anomalous),
                    "attack_type": session.attack_type.value
                }
                writer.writerow(row)

        logger.info(f"Sessions export completed. File size: {os.path.getsize(filepath)} bytes.")

    def _export_events(self) -> None:
        filepath = os.path.join(self.output_dir, "events_anomalous.csv")
        logger.info(f"Exporting anomalous events to {filepath}...")

        headers = [
            "entity_id",
            "entity_type",
            "timestamp",
            "source_ip",
            "geo_location",
            "resource_accessed",
            "auth_method",
            "session_duration",
            "command_sequence",
            "device_fingerprint",
            "label"
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for event in self.events:
                row = {
                    "entity_id": event.entity_id,
                    "entity_type": event.entity_type,
                    "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_ip": event.source_ip,
                    "geo_location": event.geo_location,
                    "resource_accessed": event.resource_accessed,
                    "auth_method": event.auth_method,
                    "session_duration": f"{event.session_duration:.2f}",
                    "command_sequence": json.dumps(event.command_sequence),
                    "device_fingerprint": event.device_fingerprint,
                    "label": event.label
                }
                writer.writerow(row)

        logger.info(f"Events export completed. File size: {os.path.getsize(filepath)} bytes.")
