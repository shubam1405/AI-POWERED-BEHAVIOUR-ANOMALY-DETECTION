import random
import uuid
from typing import List, Dict, Tuple
from faker import Faker
from models.enums import DepartmentName, AccessLevel, OperatingSystem, Browser, DeviceType, ResourceType
from models.employee import Employee
from models.behavior_profile import (
    BehaviorProfile, BehaviorKnowledgeBase, TemporalBehavior, AuthenticationBehavior,
    ResourceBehavior, DeviceBehavior, LocationBehavior, CommandBehavior, SessionBehavior
)
from simulator.company import Company
from config.config import (
    DEPARTMENTS,
    OFFICE_LOCATIONS,
    ROLES_BY_DEPARTMENT,
    ROLE_ACCESS_MAPPING,
    WORK_HOURS_CONFIG,
    DEPARTMENT_OS_DISTRIBUTION,
    OS_BROWSER_DISTRIBUTION,
    DEPARTMENT_RESOURCE_PREFERENCES,
    RESOURCES_CONFIG,
    VPN_PROBABILITIES,
    SIMULATION_CONFIG,
    BEHAVIOR_PROFILE_DEFAULTS
)

class EmployeeGenerator:
    """Generates 200 employees with Faker-based identities and probability-driven BehaviorProfiles."""
    def __init__(self, company: Company, rng: random.Random):
        self.company = company
        self.rng = rng
        self.fake = Faker()
        self.fake.seed_instance(rng.randint(0, 1000000))
        self.employee_counter = 0

    def _generate_employee_id(self) -> str:
        self.employee_counter += 1
        return f"EMP-{self.employee_counter:04d}"

    def _choose_from_distribution(self, dist: dict):
        """Helper to sample from a distribution dictionary."""
        items = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(items, weights=weights, k=1)[0]

    def generate_all(self, count: int = 200) -> List[Employee]:
        """Generates a list of employees with assigned departments, roles, and profiles."""
        employees: List[Employee] = []

        # Reserve space for 10 service accounts and 10 device agents
        user_count = count - 20
        dept_allocation = []
        dept_counts = {
            DepartmentName.ENGINEERING: int(user_count * 0.25),
            DepartmentName.SALES: int(user_count * 0.25),
            DepartmentName.FINANCE: int(user_count * 0.15),
            DepartmentName.IT: int(user_count * 0.15),
            DepartmentName.MARKETING: int(user_count * 0.10),
            DepartmentName.HR: int(user_count * 0.10)
        }
        
        # Adjust potential rounding issues
        total_allocated = sum(dept_counts.values())
        if total_allocated < user_count:
            dept_counts[DepartmentName.ENGINEERING] += (user_count - total_allocated)

        for dept, c in dept_counts.items():
            dept_allocation.extend([dept] * c)

        # Shuffle the allocations to randomize employee generation sequence
        self.rng.shuffle(dept_allocation)

        for dep_name in dept_allocation:
            employee_id = self._generate_employee_id()
            employee_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
            name = self.fake.name()
            
            # Choose role
            roles = ROLES_BY_DEPARTMENT[dep_name]
            role = self.rng.choice(roles)

            # Choose office location
            office = self.rng.choice(OFFICE_LOCATIONS)

            # Access Level mapping
            access_level = AccessLevel.LOW
            for key, val in ROLE_ACCESS_MAPPING.items():
                if key in role:
                    access_level = val
                    break

            # Working Hours
            base_hours = WORK_HOURS_CONFIG[dep_name]
            start_hour = base_hours[0] + self.rng.choice([-1, 0, 1])
            duration = base_hours[1] - base_hours[0]
            end_hour = start_hour + duration

            preferred_os = self._choose_from_distribution(DEPARTMENT_OS_DISTRIBUTION.get(dep_name, {OperatingSystem.WINDOWS: 1.0}))
            preferred_browser = self._choose_from_distribution(OS_BROWSER_DISTRIBUTION.get(preferred_os, {Browser.CHROME: 1.0}))
            vpn_prob = VPN_PROBABILITIES.get(dep_name, 0.10)

            behavior_profile = self._generate_behavior_profile(
                dep_name, (start_hour, end_hour), preferred_browser, preferred_os
            )

            employee = Employee(
                employee_uuid=employee_uuid,
                employee_id=employee_id,
                name=name,
                department=dep_name,
                role=role,
                office_location=office,
                allowed_devices=[],
                preferred_browser=preferred_browser,
                preferred_os=preferred_os,
                vpn_probability=vpn_prob,
                access_level=access_level,
                behavior_profile=behavior_profile,
                entity_type="user"
            )

            self.company.add_employee(employee)
            employees.append(employee)

        # Generate 10 Service Accounts (entity_type = "service account")
        for i in range(1, 11):
            svc_id = f"SVC-{i:04d}"
            svc_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
            
            # Automated service account profile: runs database queries at 2 AM
            login_dist = {2: 1.0}
            resource_probs = {ResourceType.FINANCE_DB.value: 0.90, ResourceType.SERVERS.value: 0.10}
            
            bp = BehaviorKnowledgeBase(
                temporal=TemporalBehavior(working_hours=(2, 3), login_hour_distribution=login_dist),
                authentication=AuthenticationBehavior(success_rate=1.0, preferred_methods={"Token": 1.0}),
                resource=ResourceBehavior(resource_probabilities=resource_probs, admin_access_probability=0.20),
                device=DeviceBehavior(device_probabilities={"SERVER-0001": 1.0}, primary_device_id="SERVER-0001"),
                location=LocationBehavior(office_locations={"Munich": 1.0}, remote_work_probability=0.0),
                command=CommandBehavior(habitual_commands=["pg_dump -U postgres", "tar -czf backup.tar.gz"], powershell_probability=0.0),
                session=SessionBehavior(average_session_length=15.0, session_length_stddev=2.0, average_actions=3, actions_stddev=1)
            )
            
            employee = Employee(
                employee_uuid=svc_uuid,
                employee_id=svc_id,
                name=f"Backup DB Agent {i}",
                department=DepartmentName.IT,
                role="Service Account",
                office_location="Munich",
                allowed_devices=["SERVER-0001"],
                preferred_browser=Browser.CHROME,
                preferred_os=OperatingSystem.LINUX,
                vpn_probability=0.0,
                access_level=AccessLevel.HIGH,
                behavior_profile=bp,
                entity_type="service account"
            )
            self.company.add_employee(employee)
            employees.append(employee)

        # Generate 10 Device Entities (entity_type = "device")
        for i in range(1, 11):
            dev_id = f"DEV-AGENT-{i:04d}"
            dev_uuid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
            
            # Periodic network logging: runs telemetry scripts
            login_dist = {h: 1.0/24 for h in range(24)}
            resource_probs = {ResourceType.MONITORING.value: 1.0}
            
            bp = BehaviorKnowledgeBase(
                temporal=TemporalBehavior(working_hours=(0, 24), login_hour_distribution=login_dist),
                authentication=AuthenticationBehavior(success_rate=0.99, preferred_methods={"Certificate": 1.0}),
                resource=ResourceBehavior(resource_probabilities=resource_probs),
                device=DeviceBehavior(device_probabilities={dev_id: 1.0}, primary_device_id=dev_id),
                location=LocationBehavior(office_locations={"New York": 1.0}, remote_work_probability=0.0),
                command=CommandBehavior(habitual_commands=["/usr/bin/telemetry-agent", "ping -c 5 srv"], powershell_probability=0.0),
                session=SessionBehavior(average_session_length=5.0, session_length_stddev=1.0, average_actions=2, actions_stddev=1)
            )
            
            employee = Employee(
                employee_uuid=dev_uuid,
                employee_id=dev_id,
                name=f"Network Switch Sensor {i}",
                department=DepartmentName.IT,
                role="IoT Sensor Agent",
                office_location="New York",
                allowed_devices=[dev_id],
                preferred_browser=Browser.CHROME,
                preferred_os=OperatingSystem.LINUX,
                vpn_probability=0.0,
                access_level=AccessLevel.LOW,
                behavior_profile=bp,
                entity_type="device"
            )
            self.company.add_employee(employee)
            employees.append(employee)

        return employees

    def _generate_behavior_profile(
        self,
        dept: DepartmentName,
        working_hours: Tuple[int, int],
        preferred_browser: Browser,
        preferred_os: OperatingSystem
    ) -> BehaviorProfile:
        """Generates a probability-based BehaviorProfile for the employee."""
        # 1. Login hour distribution
        login_dist = {}
        start_h, end_h = working_hours
        for h in range(24):
            if start_h <= h < end_h:
                login_dist[h] = round(self.rng.uniform(0.80, 0.98), 2)
            elif h == (start_h - 1) or h == end_h:
                login_dist[h] = round(self.rng.uniform(0.20, 0.45), 2)
            else:
                login_dist[h] = round(self.rng.uniform(0.01, 0.05), 2)

        # 2. Resource access probabilities
        resource_probs = {}
        prefs = DEPARTMENT_RESOURCE_PREFERENCES.get(dept, {"frequent": [], "rare": []})
        
        for res in RESOURCES_CONFIG:
            res_name = res["name"]
            allowed_depts = res["allowed_departments"]
            
            if dept not in allowed_depts:
                # Strictly 0 probability if not in allowed departments
                resource_probs[res_name] = 0.0
            elif res_name in prefs["frequent"]:
                resource_probs[res_name] = round(self.rng.uniform(0.70, 0.95), 2)
            elif res_name in prefs["rare"]:
                resource_probs[res_name] = round(self.rng.uniform(0.01, 0.05), 2)
            else:
                resource_probs[res_name] = round(self.rng.uniform(0.10, 0.35), 2)

        # 3. Browser probabilities
        browser_probs = {b: 0.0 for b in Browser}
        os_browsers = OS_BROWSER_DISTRIBUTION.get(preferred_os, {Browser.CHROME: 1.0})
        # Set values based on distribution, fallback to preferred browser with 1.0
        for b, weight in os_browsers.items():
            browser_probs[b] = round(weight, 2)

        # 4. Device probabilities (will be finalized/mapped later in Pass 3)
        device_probs = {}

        # 5. Length & Actions
        min_session = SIMULATION_CONFIG.get("min_session_minutes", 10)
        max_session = SIMULATION_CONFIG.get("max_session_minutes", 120)
        max_actions = SIMULATION_CONFIG.get("max_actions_per_session", 30)

        avg_session = round(self.rng.uniform(min_session, max_session), 1)
        avg_actions = self.rng.randint(5, max_actions)

        # Defaults from config
        session_std = BEHAVIOR_PROFILE_DEFAULTS.get("session_length_stddev", 8.0)
        actions_std = BEHAVIOR_PROFILE_DEFAULTS.get("actions_stddev", 3.0)
        sessions_per_day = BEHAVIOR_PROFILE_DEFAULTS.get("average_sessions_per_day", 3.0)
        remote_prob = BEHAVIOR_PROFILE_DEFAULTS.get("remote_work_probability", 0.25)

        return BehaviorProfile(
            working_hours=working_hours,
            login_hour_distribution=login_dist,
            resource_probabilities=resource_probs,
            browser_probabilities=browser_probs,
            device_probabilities=device_probs,
            average_session_length=avg_session,
            session_length_stddev=session_std,
            average_actions=avg_actions,
            actions_stddev=actions_std,
            average_sessions_per_day=sessions_per_day,
            remote_work_probability=remote_prob
        )
