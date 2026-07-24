import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from models.session import Session
from models.event import Event
from models.enums import (
    EventType,
    ResourceType,
    EventStatus,
    AttackType,
    DepartmentName,
)
from simulator.company import Company
from config.config import RESOURCES_CONFIG

logger = logging.getLogger("EventGenerator")

class EventGenerator:
    """
    Generates realistic, chronological lists of events for each user session
    based on department workflows, employee behavior profiles, and network status.
    """
    WORKFLOW_TEMPLATES = {
        DepartmentName.ENGINEERING: [ResourceType.GITHUB, ResourceType.JIRA, ResourceType.CONFLUENCE, ResourceType.TERMINAL],
        DepartmentName.HR: [ResourceType.HRMS, ResourceType.PAYROLL, ResourceType.EMPLOYEE_RECORDS],
        DepartmentName.FINANCE: [ResourceType.ERP, ResourceType.ACCOUNTING, ResourceType.INVOICES],
        DepartmentName.SALES: [ResourceType.CRM, ResourceType.EMAIL, ResourceType.MEETINGS],
        DepartmentName.MARKETING: [ResourceType.ANALYTICS, ResourceType.CAMPAIGN_DASHBOARD, ResourceType.SLACK],
        DepartmentName.IT: [ResourceType.ADMIN_CONSOLE, ResourceType.MONITORING, ResourceType.FIREWALL],
    }

    def __init__(self, company: Company, rng: random.Random):
        self.company = company
        self.rng = rng

    def _generate_ip(self, remote_session: bool) -> str:
        """Generates a realistic IP address based on office or VPN subnet configuration."""
        if remote_session:
            # VPN Subnet: 172.16.x.y
            x = self.rng.randint(0, 31)
            y = self.rng.randint(1, 254)
            return f"172.16.{x}.{y}"
        else:
            # Office Subnet: 10.10.x.y
            x = self.rng.randint(0, 255)
            y = self.rng.randint(1, 254)
            return f"10.10.{x}.{y}"

    def generate_events(self, sessions: List[Session]) -> List[Event]:
        """
        Generates and assigns events for each session in place.
        Returns a flat list of all generated Event objects.
        """
        logger.info(f"Generating events for {len(sessions)} sessions...")
        all_events: List[Event] = []

        for session in sessions:
            employee = self.company.get_employee(session.employee_id)
            if not employee:
                continue

            bp = employee.behavior_profile
            duration_seconds = session.duration_seconds
            session_minutes = duration_seconds / 60.0

            # 1. Determine number of intermediate resource actions
            raw_actions = int(self.rng.normalvariate(bp.average_actions, bp.actions_stddev))
            # Clamp action counts: at least 2 intermediate actions, at most 1 action per 2 minutes
            max_allowed = max(3, int(session_minutes / 2.0))
            num_actions = max(2, min(raw_actions, max_allowed))

            # 2. Sample resources based on BehaviorProfile probabilities
            sampled_resources = []
            if bp.resource_probabilities:
                res_list = list(bp.resource_probabilities.keys())
                res_weights = list(bp.resource_probabilities.values())
                if sum(res_weights) > 0:
                    sampled_resources = self.rng.choices(res_list, weights=res_weights, k=num_actions)

            # Fallback if no resources could be sampled
            if not sampled_resources:
                dept_allowed = [
                    r["name"] for r in RESOURCES_CONFIG 
                    if session.department in r["allowed_departments"]
                ]
                if dept_allowed:
                    sampled_resources = self.rng.choices(dept_allowed, k=num_actions)
                else:
                    sampled_resources = [ResourceType.EMAIL] * num_actions

            # 3. Sort resources according to department workflow template ordering
            template = self.WORKFLOW_TEMPLATES.get(session.department, [])
            def get_sort_key(res: ResourceType) -> int:
                try:
                    return template.index(res)
                except ValueError:
                    return len(template)

            sampled_resources.sort(key=get_sort_key)

            # 4. Generate actual action sequences
            action_flow: List[Dict[str, Any]] = []
            for res in sampled_resources:
                # Add Resource Access event
                action_flow.append({"event_type": EventType.RESOURCE_ACCESS, "resource": res})

                # Intersperse sub-actions based on the resource type
                sub_actions = []
                if res == ResourceType.GITHUB:
                    sub_actions = self.rng.choices(
                        [EventType.FILE_READ, EventType.FILE_WRITE, EventType.FILE_DOWNLOAD, EventType.FILE_UPLOAD],
                        k=self.rng.randint(1, 2)
                    )
                elif res == ResourceType.TERMINAL:
                    sub_actions = self.rng.choices(
                        [EventType.PROCESS_START, EventType.PROCESS_STOP, EventType.API_CALL],
                        k=self.rng.randint(1, 2)
                    )
                elif res == ResourceType.FILE_SERVER:
                    sub_actions = self.rng.choices(
                        [EventType.FILE_READ, EventType.FILE_WRITE, EventType.FILE_DOWNLOAD, EventType.FILE_UPLOAD],
                        k=self.rng.randint(1, 2)
                    )
                elif res in [ResourceType.FINANCE_DB, ResourceType.EMPLOYEE_RECORDS]:
                    sub_actions = self.rng.choices(
                        [EventType.DATABASE_QUERY, EventType.FILE_READ],
                        k=self.rng.randint(1, 2)
                    )
                elif res in [
                    ResourceType.ERP, ResourceType.HRMS, ResourceType.CRM, 
                    ResourceType.ACCOUNTING, ResourceType.ADMIN_CONSOLE, 
                    ResourceType.CAMPAIGN_DASHBOARD, ResourceType.MONITORING, 
                    ResourceType.CONFLUENCE
                ]:
                    sub_actions = self.rng.choices(
                        [EventType.API_CALL, EventType.FILE_READ, EventType.FILE_WRITE],
                        k=self.rng.randint(1, 2)
                    )
                elif res in [ResourceType.EMAIL, ResourceType.SLACK]:
                    sub_actions = self.rng.choices(
                        [EventType.FILE_READ, EventType.FILE_WRITE],
                        k=self.rng.randint(1, 2)
                    )
                elif res == ResourceType.FIREWALL:
                    sub_actions = self.rng.choices(
                        [EventType.API_CALL, EventType.PROCESS_START],
                        k=self.rng.randint(1, 2)
                    )
                else:
                    sub_actions = [EventType.API_CALL]

                for act in sub_actions:
                    action_flow.append({"event_type": act, "resource": res})

            # 5. Build full sequence: LOGIN -> CONNECT -> ACTIONS -> DISCONNECT -> LOGOUT
            full_sequence: List[Dict[str, Any]] = []
            
            # Start
            full_sequence.append({"event_type": EventType.LOGIN, "resource": None})
            if session.remote_session:
                full_sequence.append({"event_type": EventType.VPN_CONNECT, "resource": ResourceType.VPN})
            else:
                full_sequence.append({"event_type": EventType.NETWORK_CONNECT, "resource": None})

            # Intermediate activities
            full_sequence.extend(action_flow)

            # End
            if session.remote_session:
                full_sequence.append({"event_type": EventType.VPN_DISCONNECT, "resource": ResourceType.VPN})
            else:
                full_sequence.append({"event_type": EventType.NETWORK_DISCONNECT, "resource": None})
            full_sequence.append({"event_type": EventType.LOGOUT, "resource": None})

            total_events = len(full_sequence)

            # 6. Generate strictly increasing, non-duplicate timestamps
            start_ts = session.start_time
            end_ts = session.end_time
            
            try:
                # Attempt to sample unique integer second offsets
                sec_offsets = sorted(self.rng.sample(range(1, int(duration_seconds)), total_events - 2))
                event_times = [start_ts] + [start_ts + timedelta(seconds=sec) for sec in sec_offsets] + [end_ts]
            except ValueError:
                # Fallback to float percentage offsets if duration is too short for second increments
                raw_offsets = sorted(self.rng.random() for _ in range(total_events - 2))
                event_times = [start_ts]
                last_t = start_ts
                for offset in raw_offsets:
                    t = start_ts + timedelta(seconds=offset * duration_seconds)
                    if t <= last_t:
                        t = last_t + timedelta(microseconds=1)
                    event_times.append(t)
                    last_t = t
                if end_ts <= last_t:
                    end_ts = last_t + timedelta(microseconds=1)
                event_times.append(end_ts)

            # 7. Create Event objects and assign them
            ip_address = self._generate_ip(session.remote_session)
            session.events = [] # Clear any previous events

            entity_type = employee.entity_type if hasattr(employee, "entity_type") else "user"
            
            # Auth Method
            if entity_type == "service account":
                auth_method = "Token"
            elif entity_type == "device":
                auth_method = "Certificate"
            else:
                auth_method = "VPN - MFA" if session.remote_session else "Active Directory"

            for idx, item in enumerate(full_sequence):
                ev_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
                
                # Determine command sequence
                commands = []
                if item["event_type"] == EventType.PROCESS_START:
                    if entity_type == "service account":
                        commands = ["pg_dump -U postgres", "tar -czf backup.tar.gz"]
                    elif entity_type == "device":
                        commands = ["/usr/bin/telemetry-agent"]
                    else:
                        commands = self.rng.choice([["git pull", "git checkout -b feature"], ["ls", "cd src", "python main.py"]])
                
                event = Event(
                    event_uuid=ev_uuid,
                    session_id=session.session_id,
                    timestamp=event_times[idx],
                    event_type=item["event_type"],
                    employee_id=session.employee_id,
                    device_id=session.device_id,
                    resource=item["resource"],
                    status=EventStatus.SUCCESS,
                    attack_type=AttackType.NONE,
                    risk_score=0.0,
                    ip_address=ip_address,
                    metadata={
                        "geo_location": session.office_location,
                        "auth_method": auth_method,
                        "session_duration": session.duration_seconds,
                        "command_sequence": commands
                    },
                    entity_type=entity_type
                )
                session.add_event(event)
                all_events.append(event)

        logger.info(f"Successfully generated {len(all_events)} events across {len(sessions)} sessions.")
        return all_events
