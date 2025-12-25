import os
import json
import time
from pathlib import Path

LOG_DIR = Path("logs")
EXPERIMENT_LOG = LOG_DIR / "experiments.jsonl"
EVENTS_LOG = LOG_DIR / "events.log"
CRASH_LOG = LOG_DIR / "crashes.log"

def _ensure_dir(path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

class Logger:
    """
    Append-only structured experiment logger.
    One JSON object per line. No interpretation.
    """
    def __init__(self, path: Path = EXPERIMENT_LOG):
        self.path = path
        _ensure_dir(self.path)

    def record(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_event(message: str):
    try:
        _ensure_dir(EVENTS_LOG)
        ts = time.time()
        with open(EVENTS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {message}\n")
    except Exception:
        pass


def log_crash(message: str):
    try:
        _ensure_dir(CRASH_LOG)
        ts = time.time()
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {message}\n")
    except Exception:
        pass
