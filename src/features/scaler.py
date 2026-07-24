import pickle
import math
from typing import List, Dict

class StandardScaler:
    """
    StandardScaler scales feature lists to zero mean and unit variance.
    Implements a pure Python standard scaler without external library dependencies.
    """
    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, data: List[Dict[str, float]], ignore_cols: List[str]) -> None:
        """Calculates column-wise mean and standard deviation, ignoring label columns."""
        if not data:
            return

        cols = [c for c in data[0].keys() if c not in ignore_cols]
        n = len(data)

        for col in cols:
            # Coerce values to floats
            vals = []
            for row in data:
                try:
                    vals.append(float(row.get(col, 0.0)))
                except (ValueError, TypeError):
                    vals.append(0.0)

            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / n
            std = math.sqrt(var)

            self.means[col] = mean
            # Guard against zero variance division
            self.stds[col] = std if std > 1e-9 else 1.0

    def transform(self, data: List[Dict[str, float]], ignore_cols: List[str]) -> List[Dict[str, float]]:
        """Scales the input data using the fitted means and standard deviations."""
        scaled_data = []
        for row in data:
            new_row = {}
            for col, val in row.items():
                if col in ignore_cols:
                    new_row[col] = val
                else:
                    try:
                        fval = float(val)
                    except (ValueError, TypeError):
                        fval = 0.0
                    mean = self.means.get(col, 0.0)
                    std = self.stds.get(col, 1.0)
                    new_row[col] = (fval - mean) / std
            scaled_data.append(new_row)
        return scaled_data

    def save(self, filepath: str) -> None:
        """Serializes the scaler state to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "StandardScaler":
        """Loads a serialized scaler state from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
