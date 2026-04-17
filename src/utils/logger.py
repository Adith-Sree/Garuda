from __future__ import annotations

"""
Logging Module

Centralised logging setup for Project Garuda with console + file handlers.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "garuda",
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to a log file.
        fmt: Log message format.
        datefmt: Date format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
