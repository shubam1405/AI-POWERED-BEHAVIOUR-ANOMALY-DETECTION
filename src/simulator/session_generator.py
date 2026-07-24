import random
import uuid
import logging
from typing import List, Tuple
from datetime import datetime

from models.employee import Employee
from models.session import Session
from models.enums import (
    DeviceType,
    Browser,
    AttackType,
    SessionStatus,
    LoginMethod
)
from simulator.company import Company
from utils.id_generator import IdGenerator
from config.config import OS_BROWSER_DISTRIBUTION

logger = logging.getLogger("SessionGenerator")


class SessionGenerator:
    """
    Generates populated Session dataclass objects from scheduled slot boundaries
    using the employee's BehaviorProfile probabilities.
    """

    def __init__(
        self,
        company: Company,
        id_generator: IdGenerator,
        rng: random.Random
    ):
        self.company = company
        self.id_generator = id_generator
        self.rng = rng

    def generate_sessions(
        self,
        scheduled_slots: List[Tuple[Employee, datetime, datetime]]
    ) -> List[Session]:
        """
        Processes list of (Employee, start_dt, end_dt) and generates complete Session instances.
        """

        logger.info(
            f"Generating details for {len(scheduled_slots)} scheduled session slots..."
        )

        sessions: List[Session] = []

        for employee, start_dt, end_dt in scheduled_slots:

            session_id = self.id_generator.next_id()
            session_uuid = str(
                uuid.UUID(int=self.rng.getrandbits(128), version=4)
            )

            bp = employee.behavior_profile

            # -------------------------------------------------------------
            # 1. Device Selection
            # -------------------------------------------------------------

            emp_devices = self.company.get_devices(employee.employee_id)

            if bp.device_probabilities:
                device_types = list(bp.device_probabilities.keys())
                device_weights = list(bp.device_probabilities.values())

                chosen_type = self.rng.choices(
                    device_types,
                    weights=device_weights,
                    k=1
                )[0]
            else:
                chosen_type = DeviceType.LAPTOP

            matching_devices = [
                d for d in emp_devices
                if d.device_type == chosen_type
            ]

            if matching_devices:
                selected_device = self.rng.choice(matching_devices)
            elif emp_devices:
                selected_device = self.rng.choice(emp_devices)
            else:
                selected_device = None

            if selected_device:
                device_id = selected_device.device_id
                operating_system = selected_device.operating_system
            else:
                device_id = "UNKNOWN"
                operating_system = employee.preferred_os

            # -------------------------------------------------------------
            # 2. Browser Selection (OS + Employee Preference)
            # -------------------------------------------------------------

            os_distribution = OS_BROWSER_DISTRIBUTION.get(
                operating_system,
                {}
            )

            combined_distribution = {}

            if bp.browser_probabilities:

                for browser, os_weight in os_distribution.items():

                    profile_weight = bp.browser_probabilities.get(browser, 0.0)

                    combined_distribution[browser] = (
                        os_weight * profile_weight
                    )

                total = sum(combined_distribution.values())

                if total > 0:

                    combined_distribution = {
                        browser: weight / total
                        for browser, weight in combined_distribution.items()
                    }

                    browser = self.rng.choices(
                        population=list(combined_distribution.keys()),
                        weights=list(combined_distribution.values()),
                        k=1
                    )[0]

                elif os_distribution:

                    browser = self.rng.choices(
                        population=list(os_distribution.keys()),
                        weights=list(os_distribution.values()),
                        k=1
                    )[0]

                else:
                    browser = employee.preferred_browser

            else:

                if os_distribution:

                    browser = self.rng.choices(
                        population=list(os_distribution.keys()),
                        weights=list(os_distribution.values()),
                        k=1
                    )[0]

                else:

                    browser = employee.preferred_browser

            # -------------------------------------------------------------
            # 3. Remote / Office Login
            # -------------------------------------------------------------

            if self.rng.random() < bp.remote_work_probability:
                remote_session = True
                login_method = LoginMethod.VPN
            else:
                remote_session = False
                login_method = LoginMethod.OFFICE

            # -------------------------------------------------------------
            # Duration
            # -------------------------------------------------------------

            duration_seconds = (
                end_dt - start_dt
            ).total_seconds()

            # -------------------------------------------------------------
            # Build Session
            # -------------------------------------------------------------

            session = Session(
                session_uuid=session_uuid,
                session_id=session_id,
                employee_uuid=employee.employee_uuid,
                employee_id=employee.employee_id,
                department=employee.department,
                role=employee.role,
                device_id=device_id,
                browser=browser,
                operating_system=operating_system,
                office_location=employee.office_location,
                login_method=login_method,
                remote_session=remote_session,
                day_of_week=start_dt.strftime("%A"),
                start_time=start_dt,
                end_time=end_dt,
                duration_seconds=duration_seconds,
                risk_score=0.0,
                is_anomalous=False,
                attack_type=AttackType.NONE,
                session_status=SessionStatus.COMPLETED,
                events=[]
            )

            sessions.append(session)

        logger.info(
            f"Successfully generated {len(sessions)} Session objects."
        )

        return sessions