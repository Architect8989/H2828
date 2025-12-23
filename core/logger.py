"""
EME-Core Logging Primitive
Truth-preserving append-only recorder - no interpretation, no analysis.
"""

import os
import datetime
from typing import NoReturn


def _write_log_entry(log_file: str, message: str) -> None:
    """
    Internal write operation with silent failure.
    No retries, no exceptions, no output.
    """
    try:
        # Build the complete line with timestamp
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec='milliseconds'
        ).replace('+00:00', 'Z')
        line = f"{timestamp} | {message}\n"
        
        # Append with immediate flush
        with open(log_file, 'a', buffering=1) as f:
            f.write(line)
    except Exception:
        # Fail silently - do nothing, no retries
        pass


def log_event(message: str) -> None:
    """
    Record a factual event in events.log.
    
    Args:
        message: Plain string describing the fact
    """
    _write_log_entry('logs/events.log', message)


def log_crash(message: str) -> None:
    """
    Record a crash event in crashes.log.
    
    Args:
        message: Plain string describing the crash
    """
    _write_log_entry('logs/crashes.log', message)
