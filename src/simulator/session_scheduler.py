import math
import random
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

from models.employee import Employee
from simulator.company import Company
from config.config import (
    SIMULATION_CONFIG,
    WEEKEND_WORK_PROBABILITY,
    WORKING_HOUR_BUFFER_MINUTES,
)

logger = logging.getLogger("SessionScheduler")

class SessionScheduler:
    """
    Schedules user login session time slots over a simulation period.
    Ensures non-overlapping sessions, department-based weekend variations,
    and compliance with employee working hours.
    """
    def __init__(self, company: Company, start_date: datetime, days: int, rng: random.Random):
        self.company = company
        self.start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        self.days = days
        self.rng = rng
        self.buffer_minutes = WORKING_HOUR_BUFFER_MINUTES

    def _sample_poisson(self, lam: float) -> int:
        """Knuth's algorithm for Poisson random variable sampling."""
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _is_within_working_hours_with_buffer(
        self, start_dt: datetime, end_dt: datetime, working_hours: Tuple[int, int]
    ) -> bool:
        """Checks if the session is within employee working hours plus buffer."""
        start_h, end_h = working_hours
        buffer = timedelta(minutes=self.buffer_minutes)

        if start_h <= end_h:
            # Normal day shift on the same day
            work_start = start_dt.replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
            if end_h == 24:
                work_end = (start_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + buffer
            else:
                work_end = start_dt.replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
            return start_dt >= work_start and end_dt <= work_end
        else:
            # Overnight shift spanning midnight
            if start_dt.hour >= start_h:
                work_start = start_dt.replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
                work_end = (start_dt + timedelta(days=1)).replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
                return start_dt >= work_start and end_dt <= work_end
            elif start_dt.hour < end_h:
                work_start = (start_dt - timedelta(days=1)).replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
                work_end = start_dt.replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
                return start_dt >= work_start and end_dt <= work_end
            else:
                return False

    def schedule_sessions(self) -> List[Tuple[Employee, datetime, datetime]]:
        """
        Schedules session slots for all employees over the configured days.
        Returns sorted list of (Employee, start_time, end_time) tuples.
        """
        logger.info(f"Scheduling sessions for {len(self.company.employees)} employees over {self.days} days...")
        scheduled_slots: List[Tuple[Employee, datetime, datetime]] = []

        min_len = SIMULATION_CONFIG.get("min_session_minutes", 10)
        max_len = SIMULATION_CONFIG.get("max_session_minutes", 120)

        for day_offset in range(self.days):
            current_day = self.start_date + timedelta(days=day_offset)
            is_weekend = current_day.weekday() >= 5

            for employee in self.company.employees.values():
                bp = employee.behavior_profile
                logger.info(
                    f"{employee.employee_id} | "
                    f"Working Hours: {bp.working_hours} | "
                    f"Login Distribution: {bp.login_hour_distribution}"
                )
                            
                # Check weekend eligibility based on department probabilities
                if is_weekend:
                    weekend_prob = WEEKEND_WORK_PROBABILITY.get(employee.department, 0.05)
                    if self.rng.random() > weekend_prob:
                        continue # Does not work this weekend day

                # Determine expected sessions for this employee
                avg_sessions = bp.average_sessions_per_day
                num_sessions = self._sample_poisson(avg_sessions)
                if num_sessions <= 0:
                    continue

                employee_sessions: List[Tuple[datetime, datetime]] = []
                login_dist = bp.login_hour_distribution
                
                if not login_dist:
                    start_h, end_h = bp.working_hours
                    login_dist = {h: 1.0 for h in range(start_h, end_h)}

                hours = list(login_dist.keys())
                weights = list(login_dist.values())

                sessions_generated = 0
                attempts = 0

                # Attempt to generate slots up to a max retry limit to resolve overlaps
                while sessions_generated < num_sessions and attempts < 100:
                    attempts += 1

                    # Sample hour, minute, second
                    hour = self.rng.choices(hours, weights=weights, k=1)[0]
                    minute = self.rng.randint(0, 59)
                    second = self.rng.randint(0, 59)

                    start_dt = current_day.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    
                    # Sample duration
                    duration_min = self.rng.normalvariate(bp.average_session_length, bp.session_length_stddev)
                    duration_min = max(min_len, min(duration_min, max_len))
                    end_dt = start_dt + timedelta(minutes=duration_min)

                    # Validate working hour constraints + buffer
                    if not self._is_within_working_hours_with_buffer(start_dt, end_dt, bp.working_hours):
                        continue

                    # Check for overlaps with already scheduled sessions for this employee today
                    overlap = False
                    for s_start, s_end in employee_sessions:
                        if max(start_dt, s_start) < min(end_dt, s_end):
                            overlap = True
                            break

                    if not overlap:
                        employee_sessions.append((start_dt, end_dt))
                        sessions_generated += 1

                for s_start, s_end in employee_sessions:
                    scheduled_slots.append((employee, s_start, s_end))

        # Sort all sessions chronologically
        scheduled_slots.sort(key=lambda x: x[1])
        logger.info(f"Successfully scheduled {len(scheduled_slots)} session slots.")
        return scheduled_slots
