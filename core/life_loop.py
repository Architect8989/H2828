"""
EXECUTION PROTOCOL OF REALITY
Environment Mastery Engine - Life Loop Module (FROZEN INFRASTRUCTURE)

LifeLoop is infrastructure. Its job is:
1. Execute exactly one action
2. Measure exactly what happened
3. Refuse to lie about what happened

FROZEN: Do not modify this file. Any change requires full system re-audit.
"""

import time
from typing import Any

from core.logger import log_crash


class LifeLoop:
    """
    Infrastructure gatekeeper. No intelligence. No mutation. No lies.
    Dumb treadmill that produces factual records.
    """
    
    def __init__(self, action_executor, logger):
        """
        Enforce simple contracts at construction.
        """
        # Validate dependencies
        if not hasattr(action_executor, 'execute') or not callable(action_executor.execute):
            raise TypeError("action_executor must implement execute()")
        
        if not hasattr(logger, 'record') or not callable(logger.record):
            raise TypeError("logger must implement record()")
        
        self._action_executor = action_executor
        self._logger = logger
    
    def run_experiment(self, action: Any) -> dict:
        """
        Execute exactly one action. Measure exactly what happened.
        Returns raw factual record. No interpretation. No lies.
        """
        # Generate unique ID using monotonic time only
        experiment_id = self._generate_experiment_id()
        
        # 1. Capture start timestamp
        start_timestamp = time.perf_counter()
        
        # 2. Execute action exactly once
        raw_result = None
        raw_error = None
        
        try:
            raw_result = self._action_executor.execute(action, experiment_id=experiment_id)
        except Exception as e:
            raw_error = e  # Store the actual exception object
        
        # 3. Capture end timestamp
        end_timestamp = time.perf_counter()
        
        # 4. Create single factual record
        # Sanitize raw_result: store representation, not arbitrary object
        result_repr = None
        if raw_result is not None:
            # Convert to stable string representation
            try:
                result_repr = repr(raw_result)
                # Limit size to prevent log bloat
                if len(result_repr) > 1000:
                    result_repr = result_repr[:1000] + "...[truncated]"
            except:
                result_repr = "[unrepresentable_object]"
        
        record = {
            'experiment_id': experiment_id,
            'action_id': getattr(action, 'id', 'unknown'),
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'duration': end_timestamp - start_timestamp,
            'raw_result': result_repr,
            'error_occurred': raw_error is not None,
            'error_type': type(raw_error).__name__ if raw_error else None,
            'error_message': str(raw_error) if raw_error else None
        }
        
        # 5. Log exactly once
        try:
            self._logger.record(record)
        except Exception as e:
            # If logging fails, we cannot continue
            self._crash(f"LOGGING FAILED - EXPERIMENT {experiment_id}: {str(e)}")
            raise SystemError("Logging system failed - cannot guarantee truth")
        
        # 6. If there was an error, raise it after logging
        if raw_error is not None:
            raise raw_error
        
        # 7. Return record and terminate
        return record
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID using monotonic time only."""
        import hashlib
        monotonic_time = time.perf_counter_ns()
        return hashlib.sha256(f"{monotonic_time}".encode()).hexdigest()[:16]
    
    def _crash(self, message: str) -> None:
        """Log crash - must never raise."""
        try:
            log_crash(message)
        except Exception:
            # Swallow logging failures - we cannot raise here
            # but we also cannot continue, so SystemError was already raised
            pass


__all__ = ['LifeLoop']
