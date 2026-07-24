import csv
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("FeatureExporter")

class FeatureExporter:
    """
    Exports raw and scaled feature datasets to CSV files in the data/processed directory.
    """
    def __init__(self, raw_features: List[Dict[str, Any]], scaled_features: List[Dict[str, Any]], output_dir: str = "data/processed"):
        self.raw_features = raw_features
        self.scaled_features = scaled_features
        self.output_dir = output_dir

    def export(self) -> None:
        """Writes both raw and scaled tabular features to CSV files."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._export_csv(self.raw_features, "tabular_features.csv")
        self._export_csv(self.scaled_features, "tabular_features_scaled.csv")

    def _export_csv(self, data: List[Dict[str, Any]], filename: str) -> None:
        if not data:
            logger.warning(f"No data to export for {filename}")
            return

        filepath = os.path.join(self.output_dir, filename)
        logger.info(f"Exporting features to {filepath}...")

        # Ensure metadata and labels are first
        metadata_cols = ["session_id", "employee_id"]
        label_cols = ["is_anomalous", "attack_type", "risk_score"]
        
        all_keys = list(data[0].keys())
        feature_cols = [k for k in all_keys if k not in metadata_cols and k not in label_cols]
        
        headers = metadata_cols + label_cols + feature_cols

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

        logger.info(f"Export of {filename} completed. File size: {os.path.getsize(filepath)} bytes.")
