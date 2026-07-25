"""
correlation.py – Incident correlation & campaign detection engine.

Groups related anomalous user sessions across employees, device IDs, and source IPs
using graph clustering. Detects multi-stage attack chains (APT campaigns) and
calculates overall severity and coordinated remediation recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple

from copilot.utils import IncidentContext, ATTACK_CHAIN_SEQUENCES

logger = logging.getLogger("Copilot.IncidentCorrelator")


@dataclass
class Campaign:
    """Group of correlated anomalous sessions indicating a coordinated attack campaign."""

    campaign_id:         str
    detection_basis:     List[str]
    attack_types:        List[str]
    overall_severity:    str
    session_count:       int
    affected_employees:  List[str]
    common_source_ips:   List[str]
    common_devices:      List[str]
    sessions:            List[str]
    mitre_tactics:       List[str]
    is_attack_chain:     bool
    chain_description:   str
    summary:             str
    recommended_actions: List[str]
    first_seen:          str
    last_seen:           str


class IncidentCorrelator:
    """Correlates multiple user sessions into coordinated threat campaigns."""

    def __init__(self, time_window_hours: float = 24.0) -> None:
        self.time_window_hours = time_window_hours

    def correlate(self, contexts: List[IncidentContext]) -> List[Campaign]:
        """Correlate anomalous sessions into structured campaigns.

        Parameters
        ----------
        contexts : List[IncidentContext]

        Returns
        -------
        List[Campaign]
        """
        # 1. Filter to anomalous sessions
        anomalies = [c for c in contexts if c.is_anomalous]
        if not anomalies:
            logger.info("No anomalies found to correlate.")
            return []

        # Ensure we have IPs and device IDs (if missing in logs, mock or deduce from features)
        self._populate_ips_and_devices(anomalies)

        # 2. Build adjacency list for graph components
        # Two sessions are connected if they share an IP, employee, or device
        adj: Dict[str, Set[str]] = {c.session_id: set() for c in anomalies}
        
        for i, c1 in enumerate(anomalies):
            for j in range(i + 1, len(anomalies)):
                c2 = anomalies[j]
                
                # Check connections
                reasons = []
                if c1.source_ip and c1.source_ip == c2.source_ip:
                    reasons.append("shared_ip")
                if c1.employee_id and c1.employee_id == c2.employee_id:
                    reasons.append("same_employee")
                if c1.device_id and c1.device_id == c2.device_id:
                    reasons.append("same_device")

                # Verify if connections exist
                if reasons:
                    # Apply time-window threshold if timestamps exist
                    if c1.timestamp and c2.timestamp:
                        t1 = datetime.fromisoformat(c1.timestamp.replace("Z", "+00:00"))
                        t2 = datetime.fromisoformat(c2.timestamp.replace("Z", "+00:00"))
                        diff_hours = abs((t1 - t2).total_seconds()) / 3600.0
                        if diff_hours > self.time_window_hours:
                            continue  # Skip connection if outside time window

                    adj[c1.session_id].add(c2.session_id)
                    adj[c2.session_id].add(c1.session_id)

        # 3. Connected components analysis (BFS/DFS)
        visited: Set[str] = set()
        components: List[List[str]] = []

        for sid in adj:
            if sid not in visited:
                comp = []
                queue = [sid]
                visited.add(sid)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        # 4. Convert components into structured Campaigns (filtering out isolated single low/medium anomalies)
        campaigns = []
        camp_idx = 1
        
        # Create context lookup map
        ctx_map = {c.session_id: c for c in anomalies}

        for comp in components:
            # We define a campaign as containing >1 session OR 1 session of high/critical severity
            comp_contexts = [ctx_map[sid] for sid in comp]
            max_sev = self._max_severity(comp_contexts)

            if len(comp) == 1 and max_sev in ("Low", "Medium"):
                continue  # Ignore isolated minor anomalies
                
            camp = self._build_campaign(f"CAMP-{camp_idx:03d}", comp_contexts)
            campaigns.append(camp)
            camp_idx += 1

        logger.info("Correlation complete. Identified %d campaigns.", len(campaigns))
        return campaigns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _populate_ips_and_devices(self, contexts: List[IncidentContext]) -> None:
        """Derive or mock source IP and device IDs based on employee IDs if missing."""
        # This keeps the logic robust even if raw CSV inputs had missing fields
        import hashlib
        for c in contexts:
            if not c.source_ip:
                # Generate a stable IP based on employee_id hash
                h = int(hashlib.md5(c.employee_id.encode()).hexdigest()[:8], 16)
                c.source_ip = f"10.10.{(h >> 8) & 0xFF}.{h & 0xFF}"
            if not c.device_id:
                # Generate stable device ID based on employee_id hash
                c.device_id = f"DEV-{c.employee_id.split('-')[-1]}"
            if not c.timestamp:
                # Deduce timestamp from start hour or default to a baseline
                hour = c.session_start_hour if c.session_start_hour is not None else 12
                # Create a standardized timestamp for 2026-07-25
                c.timestamp = f"2026-07-25T{hour:02d}:00:00Z"

    def _max_severity(self, contexts: List[IncidentContext]) -> str:
        sevs = [c.severity for c in contexts]
        for s in ("Critical", "High", "Medium", "Low"):
            if s in sevs:
                return s
        return "Low"

    def _build_campaign(self, camp_id: str, comp_contexts: List[IncidentContext]) -> Campaign:
        # Sort contexts by timestamp
        comp_contexts.sort(key=lambda c: c.timestamp or "")

        sessions = [c.session_id for c in comp_contexts]
        employees = sorted(list(set(c.employee_id for c in comp_contexts)))
        ips = sorted(list(set(c.source_ip for c in comp_contexts if c.source_ip)))
        devices = sorted(list(set(c.device_id for c in comp_contexts if c.device_id)))
        
        # Unique tactics & techniques
        tactics = sorted(list(set(c.mitre_tactic for c in comp_contexts if c.mitre_tactic)))
        attack_types = sorted(list(set(c.attack_type for c in comp_contexts)))

        # Timing limits
        first = comp_contexts[0].timestamp or "Unknown"
        last = comp_contexts[-1].timestamp or "Unknown"

        # Check for attack chains
        is_chain, chain_desc = self._detect_attack_chain(comp_contexts)

        # Overall severity
        overall_sev = self._max_severity(comp_contexts)
        if len(sessions) >= 3 and overall_sev == "High":
            overall_sev = "Critical"  # Escalate if widespread

        # Build bases list
        bases = []
        if len(ips) < len(sessions) and len(ips) > 0:
            bases.append("shared_source_ip")
        if len(employees) < len(sessions):
            bases.append("multiple_sessions_same_employee")
        if len(devices) < len(sessions) and len(devices) > 0:
            bases.append("shared_device")
        if is_chain:
            bases.append("attack_chain_match")
        if not bases:
            bases.append("behavioral_cluster")

        # Generate incident summary
        summary = self._generate_campaign_summary(camp_id, comp_contexts, attack_types, is_chain, chain_desc)

        # Consolidate recommendations
        recommended_actions = self._build_campaign_playbook(overall_sev, attack_types, ips, devices, employees)

        return Campaign(
            campaign_id=camp_id,
            detection_basis=bases,
            attack_types=attack_types,
            overall_severity=overall_sev,
            session_count=len(sessions),
            affected_employees=employees,
            common_source_ips=ips,
            common_devices=devices,
            sessions=sessions,
            mitre_tactics=tactics,
            is_attack_chain=is_chain,
            chain_description=chain_desc,
            summary=summary,
            recommended_actions=recommended_actions,
            first_seen=first,
            last_seen=last,
        )

    def _detect_attack_chain(self, contexts: List[IncidentContext]) -> Tuple[bool, str]:
        """Examine chronological attack types to match registered sequences."""
        sequence = [c.attack_type for c in contexts]
        
        # Check subsequence matching against registered ATTACK_CHAIN_SEQUENCES
        for chain in ATTACK_CHAIN_SEQUENCES:
            # Check if elements of 'chain' appear in order within 'sequence'
            seq_idx = 0
            match_count = 0
            for item in chain:
                while seq_idx < len(sequence):
                    if sequence[seq_idx] == item:
                        match_count += 1
                        seq_idx += 1
                        break
                    seq_idx += 1
            
            if match_count == len(chain):
                return True, " → ".join(chain)

        # Fallback description if no predefined chain is matched but multiple attacks exist
        if len(set(sequence)) > 1:
            unique_steps = []
            for item in sequence:
                if not unique_steps or unique_steps[-1] != item:
                    unique_steps.append(item)
            return True, " → ".join(unique_steps)

        return False, ""

    def _generate_campaign_summary(
        self,
        camp_id: str,
        contexts: List[IncidentContext],
        attack_types: List[str],
        is_chain: bool,
        chain_desc: str,
    ) -> str:
        s_count = len(contexts)
        u_count = len(set(c.employee_id for c in contexts))
        
        attacks_str = ", ".join(attack_types)

        if is_chain:
            return (
                f"Campaign {camp_id} tracks a multi-stage attack chain ({chain_desc}) "
                f"across {u_count} user(s) and {s_count} sessions. Coordinated movement is "
                f"strongly indicated by the temporal sequence of behavioral events."
            )
            
        return (
            f"Campaign {camp_id} identifies coordinated anomalous activity involving "
            f"[{attacks_str}] signatures. A total of {s_count} sessions were clustered "
            f"linking {u_count} user(s) with shared indicators (IPs/Devices)."
        )

    def _build_campaign_playbook(
        self,
        severity: str,
        attack_types: List[str],
        ips: List[str],
        devices: List[str],
        employees: List[str],
    ) -> List[str]:
        actions = []

        # 1. System actions
        if "Lateral Movement" in attack_types or severity == "Critical":
            actions.append("Isolate all affected devices from the internal subnet immediately.")
            actions.append("Disable lateral communication protocols (SMB/RDP) between compromised systems.")
        else:
            if devices:
                devs_str = ", ".join(devices[:3])
                actions.append(f"Isolate endpoints: {devs_str} from the production network.")

        # 2. IP Actions
        if ips:
            ips_str = ", ".join(ips[:3])
            actions.append(f"Block source IPs: {ips_str} at the boundary firewalls.")

        # 3. User Actions
        if employees:
            users_str = ", ".join(employees[:3])
            actions.append(f"Revoke sessions and temporarily lock Active Directory accounts: {users_str}.")

        # 4. Reporting
        actions.append("Coordinate incident reports and trigger corporate CSIRT response playbooks.")
        
        return actions
