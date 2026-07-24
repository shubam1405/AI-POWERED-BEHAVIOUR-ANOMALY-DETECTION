import csv
import os
import json
import logging
from typing import List

from models.event import Event

logger = logging.getLogger("EventsExporter")

class EventsExporter:
    """
    Serializes and writes simulated Event objects to a CSV file following the Honeywell schema.
    """
    def __init__(self, events: List[Event], output_dir: str = "data/raw"):
        self.events = events
        self.output_dir = output_dir

    def export(self) -> None:
        """Writes all events to events.csv in the configured output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, "events.csv")
        logger.info(f"Exporting {len(self.events)} events to {filepath}...")

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

        logger.info(f"Export completed. File size: {os.path.getsize(filepath)} bytes.")
