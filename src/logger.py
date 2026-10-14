"""Configure application logging for console and file output."""

from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger()


def setup_logging():
    """
    One-time setup of the logging function
    """
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return
