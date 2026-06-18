# FILE: kyc_system/utils/logger.py
"""
Structured logging utility for the KYC Verification System.
Provides a pre-configured logger that writes to both the console and a rotating
file at logs/kyc_errors.log.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Resolve log directory relative to this file's location
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "kyc_errors.log"

# ─── Formatter ────────────────────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger with the given name.

    The returned logger writes:
    - INFO and above → console (stdout)
    - WARNING and above → rotating file (max 5 MB × 3 backups)

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler — INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler — WARNING+
    fh = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
