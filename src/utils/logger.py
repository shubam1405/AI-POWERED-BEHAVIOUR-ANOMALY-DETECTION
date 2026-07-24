import logging
import sys

def setup_logger(name: str = "VirtualEnterprise") -> logging.Logger:
    """Configures and returns a standardized logger for the simulator application."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger is already configured, return existing
        return logger

    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Output to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    return logger
