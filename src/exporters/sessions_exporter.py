import csv
import os
import logging
from typing import List

from models.session import Session

logger = logging.getLogger("SessionsExporter")

class SessionsExporter:
    """
    Serializes and writes simulated Session objects to a CSV file.
    """
    def __init__(self, sessions: List[Session], output_dir: str = "data/raw"):
        self.sessions = sessions
        self.output_dir = output_dir

    def export(self) -> None:
        """Writes all sessions to sessions.csv in the configured output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, "sessions.csv")
        logger.info(f"Exporting {len(self.sessions)} sessions to {filepath}...")

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

        logger.info(f"Export completed. File size: {os.path.getsize(filepath)} bytes.")
