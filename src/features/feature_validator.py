import math
import logging
from typing import List, Dict, Any

logger = logging.getLogger("FeatureValidator")

class FeatureValidationError(Exception):
    """Exception raised when feature validation fails."""
    def __init__(self, errors: List[str]):
        super().__init__("Feature validation failed with the following errors:\n" + "\n".join(errors))
        self.errors = errors

class FeatureValidator:
    """
    Validates generated feature datasets to ensure no missing values,
    correct row counts, numeric consistency, and label preservation.
    """
    def validate(self, raw_features: List[Dict[str, Any]], expected_session_count: int) -> bool:
        """Validates the raw feature dataset list. Raises FeatureValidationError on failure."""
        errors: List[str] = []

        # 1. Row count verification
        if len(raw_features) != expected_session_count:
            errors.append(f"Feature count mismatch: got {len(raw_features)} rows, expected {expected_session_count}.")

        seen_session_ids = set()
        ignore_cols = {"session_id", "employee_id", "attack_type"}

        for idx, row in enumerate(raw_features):
            sid = row.get("session_id", "UNKNOWN")
            
            # 2. Check unique session IDs
            if sid == "UNKNOWN":
                errors.append(f"Row #{idx}: Missing session_id.")
            elif sid in seen_session_ids:
                errors.append(f"Row #{idx}: Duplicate session_id: {sid}.")
            seen_session_ids.add(sid)

            # 3. Check for missing values
            for col, val in row.items():
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    errors.append(f"Session {sid}: Column '{col}' has missing/NaN value.")

                # 4. Check numeric type consistency
                if col not in ignore_cols:
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        errors.append(f"Session {sid}: Column '{col}' value '{val}' is not numeric.")

            # 5. Label preservation validation
            if "is_anomalous" not in row:
                errors.append(f"Session {sid}: Label 'is_anomalous' is missing.")
            if "attack_type" not in row:
                errors.append(f"Session {sid}: Label 'attack_type' is missing.")
            if "risk_score" not in row:
                errors.append(f"Session {sid}: Label 'risk_score' is missing.")

            # 6. Basic range validations
            duration = row.get("session_duration", 0.0)
            if duration < 0.0:
                errors.append(f"Session {sid}: session_duration is negative ({duration}).")
                
            failed_logins = row.get("failed_login_count", 0.0)
            if failed_logins < 0.0:
                errors.append(f"Session {sid}: failed_login_count is negative ({failed_logins}).")
                
            failed_ratio = row.get("failed_login_ratio", 0.0)
            if not (0.0 <= failed_ratio <= 1.0):
                errors.append(f"Session {sid}: failed_login_ratio {failed_ratio} is out of bounds [0, 1].")

            weekend_flag = row.get("weekend_flag", 0.0)
            if weekend_flag not in [0.0, 1.0]:
                errors.append(f"Session {sid}: weekend_flag is invalid ({weekend_flag}).")

            # Check sequence positions
            admin_pos = row.get("admin_access_position", -1.0)
            if admin_pos != -1.0 and not (0.0 <= admin_pos <= 1.0):
                errors.append(f"Session {sid}: admin_access_position {admin_pos} is invalid.")

            dl_pos = row.get("file_download_position", -1.0)
            if dl_pos != -1.0 and not (0.0 <= dl_pos <= 1.0):
                errors.append(f"Session {sid}: file_download_position {dl_pos} is invalid.")

        if errors:
            raise FeatureValidationError(errors)

        logger.info("Feature validation successful. All feature consistency constraints verified.")
        return True
