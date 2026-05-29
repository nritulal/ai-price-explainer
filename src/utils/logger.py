"""Logging setup"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger with file and console handlers"""

    from src.utils.constants import LOGS_PATH

    if log_file is None:
        log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_path = LOGS_PATH / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)

    # Console handler with error handling for emojis
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter without emojis
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger