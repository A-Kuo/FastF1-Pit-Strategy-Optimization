"""
Structured Logging Configuration for F1 Pit Strategy System

Provides JSON-structured logging suitable for production cloud environments.
Logs are easily searchable in CloudWatch, Cloud Logging, Stackdriver, etc.

Usage:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"race_id": 123})
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Optional
from config import settings


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Outputs logs as JSON for easy parsing in cloud logging platforms.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_obj.update(record.extra)

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """
    Simple text formatter for development/debugging.
    More readable than JSON for local development.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as plain text"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{timestamp}] {record.levelname:8s} "
            f"{record.name}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Uses settings.log_level if not provided.

    Returns:
        logging.Logger: Configured logger instance

    Example:
        from logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Application started")
        logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Determine log level
        if level is None:
            level = settings.log_level

        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Choose formatter based on environment
        if settings.log_format == "json":
            formatter = JSONFormatter()
        else:
            formatter = TextFormatter()

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def configure_root_logger(
    log_format: Optional[str] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Configure the root logger for the entire application.

    Args:
        log_format: "text" or "json". Uses settings.log_format if not provided.
        log_level: Log level. Uses settings.log_level if not provided.

    Example:
        from logging_config import configure_root_logger
        configure_root_logger(log_format="json", log_level="INFO")
    """
    if log_format is None:
        log_format = settings.log_format
    if log_level is None:
        log_level = settings.log_level

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


# Auto-configure on import
if __name__ != "__main__":
    configure_root_logger()
