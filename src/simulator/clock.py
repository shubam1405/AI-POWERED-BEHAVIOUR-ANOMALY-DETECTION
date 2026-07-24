from datetime import datetime, timedelta

class SimulationClock:
    """A clock for the virtual simulation environment, decoupled from system clock."""
    def __init__(self, start_time: datetime):
        self._current_time: datetime = start_time

    def get_time(self) -> datetime:
        """Returns the current simulated time."""
        return self._current_time

    def tick(self, minutes: int = 1) -> None:
        """Ticks the clock forward by a specified number of minutes."""
        self._current_time += timedelta(minutes=minutes)

    def advance_to(self, target_time: datetime) -> None:
        """Force advance the clock to a specific time."""
        if target_time < self._current_time:
            raise ValueError("Cannot move simulated clock backwards.")
        self._current_time = target_time

    def __str__(self) -> str:
        return self._current_time.strftime("%Y-%m-%d %H:%M:%S")
