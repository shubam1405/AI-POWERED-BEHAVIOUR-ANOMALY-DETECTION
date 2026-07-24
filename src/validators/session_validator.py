import re
from typing import List, Dict
from datetime import datetime, timedelta
import logging

from models.session import Session
from simulator.company import Company
from models.enums import Browser, DepartmentName
from config.config import WORKING_HOUR_BUFFER_MINUTES

logger = logging.getLogger("SessionValidator")

class SessionValidationError(Exception):
    """Exception raised when session validation checks fail."""
    def __init__(self, errors: List[str]):
        super().__init__("Session validation failed with the following errors:\n" + "\n".join(errors))
        self.errors = errors

class SessionValidator:
    """
    Validates the generated simulated sessions to ensure logical consistency and constraint fulfillment.
    """
    @staticmethod
    def is_valid_uuid(uuid_str: str) -> bool:
        """Verifies if a string is a valid UUID4 format."""
        uuid_regex = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_regex.match(uuid_str))

    def validate(self, sessions: List[Session], company: Company) -> bool:
        """Runs validation checks on the supplied sessions list. Raises SessionValidationError on failure."""
        errors: List[str] = []
        seen_uuids = set()
        seen_ids = set()
        
        # Group sessions by employee to check chronological order and overlaps
        employee_sessions: Dict[str, List[Session]] = {}

        for session in sessions:
            sid = session.session_id

            # 1. Unique Session UUID and format
            if session.session_uuid in seen_uuids:
                errors.append(f"Session {sid}: Duplicate session_uuid: {session.session_uuid}")
            seen_uuids.add(session.session_uuid)
            if not self.is_valid_uuid(session.session_uuid):
                errors.append(f"Session {sid}: Invalid UUIDv4: {session.session_uuid}")

            # 2. Unique Session ID
            if session.session_id in seen_ids:
                errors.append(f"Session {sid}: Duplicate session_id: {session.session_id}")
            seen_ids.add(session.session_id)

            # 3. Employee and Device matching
            employee = company.get_employee(session.employee_id)
            if not employee:
                errors.append(f"Session {sid}: Employee {session.employee_id} does not exist in company.")
            else:
                if employee.employee_uuid != session.employee_uuid:
                    errors.append(f"Session {sid}: employee_uuid mismatch (got {session.employee_uuid}, expected {employee.employee_uuid})")

                # Verify device ownership
                if session.device_id != "UNKNOWN" and session.device_id not in employee.allowed_devices:
                    errors.append(f"Session {sid}: Device {session.device_id} is not owned by Employee {session.employee_id}")

                # Verify working hours boundaries (+ buffer configuration)
                bp = employee.behavior_profile
                start_h, end_h = bp.working_hours
                buffer = timedelta(minutes=WORKING_HOUR_BUFFER_MINUTES)
                start_time = session.start_time

                # Check window
                if start_h <= end_h:
                    work_start = start_time.replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
                    if end_h == 24:
                        work_end = (start_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + buffer
                    else:
                        work_end = start_time.replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
                    in_window = work_start <= start_time and session.end_time <= work_end
                else:
                    if start_time.hour >= start_h:
                        work_start = start_time.replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
                        work_end = (start_time + timedelta(days=1)).replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
                        in_window = work_start <= start_time and session.end_time <= work_end
                    elif start_time.hour < end_h:
                        work_start = (start_time - timedelta(days=1)).replace(hour=start_h, minute=0, second=0, microsecond=0) - buffer
                        work_end = start_time.replace(hour=end_h, minute=0, second=0, microsecond=0) + buffer
                        in_window = work_start <= start_time and session.end_time <= work_end
                    else:
                        in_window = False

                if not in_window:
                    errors.append(f"Session {sid}: Start time {session.start_time} / End time {session.end_time} violates working hours {start_h}-{end_h} with buffer")

            # 4. Strict duration verification
            expected_duration = (session.end_time - session.start_time).total_seconds()
            if abs(session.duration_seconds - expected_duration) > 1e-3:
                errors.append(f"Session {sid}: duration_seconds mismatch (got {session.duration_seconds}, expected {expected_duration})")
            
            if session.duration_seconds <= 0:
                errors.append(f"Session {sid}: duration_seconds must be positive.")

            # Validate day_of_week matches start_time day of week
            expected_day = session.start_time.strftime("%A")
            if session.day_of_week != expected_day:
                errors.append(f"Session {sid}: day_of_week mismatch (got {session.day_of_week}, expected {expected_day})")

            # 5. Enums verification
            if not isinstance(session.browser, Browser):
                errors.append(f"Session {sid}: Invalid browser enum: {session.browser}")
            if not isinstance(session.department, DepartmentName):
                errors.append(f"Session {sid}: Invalid department enum: {session.department}")

            employee_sessions.setdefault(session.employee_id, []).append(session)

        # 6. Chronological and Overlap checks per Employee
        for emp_id, emp_s_list in employee_sessions.items():
            for i in range(len(emp_s_list) - 1):
                s1 = emp_s_list[i]
                s2 = emp_s_list[i + 1]

                # Check chronological order
                if s1.start_time > s2.start_time:
                    errors.append(f"Employee {emp_id}: sessions are not in chronological order: {s1.session_id} starts after {s2.session_id}")

                # Check for session overlaps
                if s1.end_time > s2.start_time:
                    errors.append(f"Employee {emp_id}: overlapping sessions detected: {s1.session_id} (ends {s1.end_time}) overlaps with {s2.session_id} (starts {s2.start_time})")

        if errors:
            raise SessionValidationError(errors)

        logger.info("Session validation successful. All integrity constraints verified.")
        return True
