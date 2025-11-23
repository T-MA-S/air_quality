"""Logging configuration."""

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger
from src.config import settings


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up logger with JSON formatting if configured."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if settings.log_format == "json":
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

