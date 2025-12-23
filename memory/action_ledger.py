"""
Action Ledger - Append-Only Causal Record Keeper
Records (action → time window → observed deltas) exactly as occurred.
"""

import json
import os
from datetime import datetime
from typing import Any


class ActionLedger:
    """
    Durable, append-only record of experiments.
    Records facts without interpretation.
    """
    
    def __init__(self, ledger_path: str = "memory/long_term/action_ledger.jsonl"):
        """
        Initialize ledger with storage path.
        
        Args:
            ledger_path: Path to append-only JSONL file
        """
        self.ledger_path = ledger_path
        self._ensure_directory()
    
    def _ensure_directory(self) -> None:
        """Create directory if it doesn't exist."""
        directory = os.path.dirname(self.ledger_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    def record(
        self,
        experiment_id: str,
        action: Any,
        time_window: tuple,
        deltas: list
    ) -> None:
        """
        Append one factual experiment record.
        
        Args:
            experiment_id: Unique experiment identifier
            action: Action object (type, parameters, fingerprint)
            time_window: (start, end) tuple of datetime objects
            deltas: List of Delta objects observed
        
        Raises:
            Any exception raised during file operations (fatal truth violation)
        """
        # Format timestamps as ISO strings
        start_time, end_time = time_window
        
        # Build record exactly as provided, no validation
        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "action": {
                "type": action.type.value if hasattr(action.type, 'value') else str(action.type),
                "parameters": action.parameters,
                "fingerprint": getattr(action, 'fingerprint', None)
            },
            "time_window": {
                "start": start_time.isoformat() + "Z",
                "end": end_time.isoformat() + "Z"
            },
            "deltas": [
                {
                    "timestamp": delta.timestamp.isoformat() + "Z",
                    "change_type": delta.change_type,
                    "confidence": delta.confidence
                }
                for delta in deltas
            ]
        }
        
        # Append-only write with immediate durability
        line = json.dumps(record, separators=(',', ':')) + '\n'
        
        # Write, flush, and sync for durability
        with open(self.ledger_path, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
