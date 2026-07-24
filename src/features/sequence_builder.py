import json
import os
import logging
from typing import List, Dict, Any, Tuple

from models.session import Session
from models.event import Event
from models.enums import EventType, ResourceType, EventStatus, SensitivityLevel
from config.config import RESOURCES_CONFIG, ATTACK_PAYLOADS

logger = logging.getLogger("SequenceBuilder")

EVENT_TYPE_INDEX = {
    EventType.LOGIN: 0,
    EventType.LOGOUT: 1,
    EventType.RESOURCE_ACCESS: 2,
    EventType.FILE_READ: 3,
    EventType.FILE_WRITE: 4,
    EventType.FILE_DOWNLOAD: 5,
    EventType.FILE_UPLOAD: 6,
    EventType.PROCESS_START: 7,
    EventType.PROCESS_STOP: 8,
    EventType.API_CALL: 9,
    EventType.DATABASE_QUERY: 10,
    EventType.VPN_CONNECT: 11,
    EventType.VPN_DISCONNECT: 12,
    EventType.NETWORK_CONNECT: 13,
    EventType.NETWORK_DISCONNECT: 14
}

class SequenceBuilder:
    """
    Transforms session event logs into numerical sequences of shape (num_sessions, max_len, feature_dim)
    with padding and attention masks, ready for a GRU/LSTM Autoencoder.
    """
    def __init__(self, max_len: int = 50, feature_dim: int = 21):
        self.max_len = max_len
        self.feature_dim = feature_dim

    def build_sequences(self, sessions: List[Session]) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """Converts lists of Session events to padded sequences and attention masks."""
        logger.info(f"Building chronological sequences for {len(sessions)} sessions (max_len={self.max_len})...")
        
        sequences: List[List[List[float]]] = []
        masks: List[List[float]] = []

        # Resources sensitivity map
        res_sensitivity = {}
        for r in RESOURCES_CONFIG:
            s_val = r["sensitivity"].value
            # Map sensitivity level to float index
            idx = 0.0
            if s_val == "Internal":
                idx = 1.0
            elif s_val == "Confidential":
                idx = 2.0
            elif s_val == "Restricted":
                idx = 3.0
            res_sensitivity[r["name"]] = idx

        malware_names = {name.lower() for name in ATTACK_PAYLOADS.get("malware_filenames", [])}

        for session in sessions:
            events = sorted(session.events, key=lambda e: e.timestamp)
            seq: List[List[float]] = []
            mask: List[float] = []

            last_ts = session.start_time

            for idx in range(self.max_len):
                if idx < len(events):
                    ev = events[idx]
                    
                    # 1. One-hot encode Event Type (15 dimensions)
                    ev_vec = [0.0] * 15
                    ev_idx = EVENT_TYPE_INDEX.get(ev.event_type, 2) # default to RESOURCE_ACCESS
                    ev_vec[ev_idx] = 1.0

                    # 2. Resource Sensitivity Level (1 dimension)
                    sensitivity = res_sensitivity.get(ev.resource, 0.0)

                    # 3. Admin Resource Flag (1 dimension)
                    is_admin = 0.0
                    if ev.resource in [ResourceType.ADMIN_CONSOLE, ResourceType.FIREWALL, ResourceType.SERVERS]:
                        is_admin = 1.0

                    # 4. Event Success Status (1 dimension)
                    success = 1.0 if ev.status == EventStatus.SUCCESS else 0.0

                    # 5. Normalized Time Delta (1 dimension)
                    time_delta = (ev.timestamp - last_ts).total_seconds()
                    norm_delta = min(1.0, time_delta / 300.0) # normalise: clamp to 5 minutes max
                    last_ts = ev.timestamp

                    # 6. Command flags (2 dimensions)
                    cmd = " ".join(ev.command_sequence).lower()
                    is_powershell = 1.0 if "powershell" in cmd else 0.0
                    
                    is_malware = 0.0
                    for m in malware_names:
                        if m in cmd:
                            is_malware = 1.0
                            break

                    # Construct final event feature vector (size 21)
                    event_vector = ev_vec + [sensitivity, is_admin, success, norm_delta, is_powershell, is_malware]
                    seq.append(event_vector)
                    mask.append(1.0)
                else:
                    # Padding: Zero vector
                    seq.append([0.0] * self.feature_dim)
                    mask.append(0.0)

            sequences.append(seq)
            masks.append(mask)

        return sequences, masks

    def save(self, sequences: List[List[List[float]]], masks: List[List[float]], output_dir: str = "data/processed") -> None:
        """Saves the built sequences and attention masks to sequential_features.json."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "sequential_features.json")
        logger.info(f"Saving sequential features to {filepath}...")

        output_data = {
            "sequences": sequences,
            "masks": masks,
            "max_len": self.max_len,
            "feature_dim": self.feature_dim
        }

        with open(filepath, "w") as f:
            json.dump(output_data, f)
        
        logger.info(f"Sequential features save complete. File size: {os.path.getsize(filepath)} bytes.")
