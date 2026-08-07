"""
utils/logger.py
===============

Centralised logging configuration for the OrbitDesk AI Support Agent.

All modules should obtain a logger via ``get_logger(__name__)`` rather than
creating their own handlers. This guarantees:

    * A single, consistent log format.
    * Logs are written to both the console and a rotating file.
    * The log level is configurable via ``config.Logging``.
    * No duplicate handlers are created when the logger is re-constructed.

The log file is stored under ``config.Paths.LOG_DIR`` and automatically
rotates when it reaches a configurable size, retaining a small number of
backups.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Final

from config import Logging, Paths


_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Maximum size of a single log file (2 MB) before rotation.
_MAX_BYTES: Final[int] = 2 * 1024 * 1024

_BACKUP_COUNT: Final[int] = 3


_configured_loggers: set[str] = set()


def _build_formatter() -> logging.Formatter:
    """Return a ``Formatter`` with the project's standard log format."""
    return logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _attach_handlers(logger: logging.Logger) -> None:
    """
    Attach console and rotating-file handlers to a logger.

    Each handler is created once per logger and guarded by the
    ``_configured_loggers`` set.

    Parameters
    ----------
    logger : logging.Logger
        The logger to configure.
    """
    logger_key = logger.name
    if logger_key in _configured_loggers:
        return

    formatter = _build_formatter()

    # --- Console handler ---------------------------------------------------
    if Logging.CONSOLE_ENABLED:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # --- Rotating file handler --------------------------------------------
    log_file = Paths.LOG_DIR / Logging.FILE_NAME
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    level = getattr(logging, Logging.LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate root-handler output.

    _configured_loggers.add(logger_key)


def get_logger(name: str) -> logging.Logger:
    """
    Return a fully configured logger for the calling module.

    Parameters
    ----------
    name : str
        Normally ``__name__`` of the importing module.

    Returns
    -------
    logging.Logger
        A logger with console + rotating-file handlers attached.
    """
    logger = logging.getLogger(name)
    _attach_handlers(logger)
    return logger


# Ensure the log directory exists before any handler tries to write to it.
Paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
