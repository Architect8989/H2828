
import os
import datetime
import json


def _ensure_dir(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass


def _write_text_log(log_file, message):
    try:
        _ensure_dir(log_file)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec='milliseconds'
        ).replace('+00:00', 'Z')
        line = f"{timestamp} | {message}\n"
        with open(log_file, 'a', buffering=1) as f:
            f.write(line)
    except Exception:
        pass


def _write_json_log(log_file, record):
    try:
        _ensure_dir(log_file)
        line = json.dumps(record) + "\n"
        with open(log_file, 'a', buffering=1) as f:
            f.write(line)
    except Exception:
        pass


def log_experiment(record):
    _write_json_log('logs/experiments.jsonl', record)


def log_event(message):
    _write_text_log('logs/events.log', message)


def log_crash(message):
    _write_text_log('logs/crashes.log', message)
