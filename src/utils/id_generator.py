class IdGenerator:
    """
    Centralized generator utility for producing sequenced human-readable IDs.
    Examples: SES-000001, SES-000002, etc.
    """
    def __init__(self, prefix: str, digits: int = 6, start: int = 1):
        self.prefix = prefix
        self.digits = digits
        self.counter = start

    def next_id(self) -> str:
        """Returns the next ID in the sequence and increments the counter."""
        current_val = self.counter
        self.counter += 1
        return f"{self.prefix}{current_val:0{self.digits}d}"
