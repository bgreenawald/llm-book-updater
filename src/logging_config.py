"""Centralized logging configuration for the application."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_name: str = "app") -> logger:
    """
    Configure logging for the application.

    Args:
        log_name: Base name for the log file

    Returns:
        logger: Configured logger instance
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Remove any existing handlers
    logger.remove()

    # Add file handler
    logger.add(
        sink=log_dir / f"{log_name}.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # Add stdout handler if not already added
    if not any(handler._sink == sys.stdout for handler in logger._core.handlers.values()):
        logger.add(
            sink=sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
        )

    return logger
