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
from typing import Any, Protocol, Tuple, runtime_checkable

from core.logger import log_crash
from execution.action_executor import Action


@runtime_checkable
class ChangeDetectorAdapterProtocol(Protocol):
    """1️⃣ Enforce adapter boundary — no raw sensors ever"""
    def compute(self, pre_snapshot: Any, post_snapshot: Any, 
                action_time_window: Tuple[float, float]) -> list[Any]:
        """Return list of Delta objects with canonical fields."""
        ...


class RealityViolation(Exception):
    def __init__(self, experiment_id: str, context: str):
        self.experiment_id = experiment_id
        self.context = context
        super().__init__(f"REALITY VIOLATION [{experiment_id}]: {context}")


class LifeLoop:
    """
    Infrastructure gatekeeper. No intelligence. No mutation. No lies.
    """
    
    def __init__(self, action_executor, perception_capturer, 
                 change_detector: ChangeDetectorAdapterProtocol,
                 delta_validator, action_ledger, env_graph_updater,
                 observation_delay: float = 0.5, delta_confidence_threshold: float = 0.9):
        """
        Enforce all contracts at construction. No compromises.
        """
        # 1️⃣ Enforce adapter boundary
        if not isinstance(change_detector, ChangeDetectorAdapterProtocol):
            raise TypeError("change_detector must implement ChangeDetectorAdapterProtocol")
        
        # Validate other dependencies
        required_contracts = [
            (action_executor, "execute"),
            (perception_capturer, "capture"),
            (delta_validator, "validate"),
            (action_ledger, "record"),
            (env_graph_updater, "update"),
        ]
        
        for obj, method in required_contracts:
            if not hasattr(obj, method) or not callable(getattr(obj, method)):
                raise TypeError(f"{obj.__class__.__name__} must implement {method}()")
        
        self._action_executor = action_executor
        self._perception_capturer = perception_capturer
        self._change_detector = change_detector
        self._delta_validator = delta_validator
        self._action_ledger = action_ledger
        self._env_graph_updater = env_graph_updater
        
        self.observation_delay = observation_delay
        self.delta_confidence_threshold = delta_confidence_threshold
        self._experiment_in_progress = False
    
    def run_experiment(self, action: Action) -> dict:
        """
        Execute exactly one action and measure exactly what happened.
        Returns raw factual record. No interpretation. No lies.
        """
        experiment_id = self._generate_experiment_id()
        
        # Prevent overlapping experiments
        if self._experiment_in_progress:
            self._crash(f"EXPERIMENT {experiment_id} - Overlap violation")
            raise RealityViolation(
                experiment_id,
                "Experiment already in progress. Overlapping experiments forbidden."
            )
        
        try:
            self._experiment_in_progress = True
            
            # Validate action
            try:
                action.validate()
            except Exception as e:
                self._crash(f"EXPERIMENT {experiment_id} - Invalid action: {str(e)}")
                raise RealityViolation(
                    experiment_id,
                    f"Action validation failed: {str(e)}"
                ) from e
            
            # 4️⃣ Capture pre-snapshot (three things: frame, timestamp, metadata)
            pre_snapshot = self._perception_capturer.capture()
            self._validate_perception_snapshot(pre_snapshot, experiment_id)
            
            # 5️⃣ Execute action with monotonic time only
            action_start = time.perf_counter()
            self._action_executor.execute(action, experiment_id=experiment_id)
            action_end = time.perf_counter()
            time_window = (action_start, action_end)
            
            # Fixed observation delay - 6️⃣ No intelligence
            time.sleep(self.observation_delay)
            
            # Capture post-snapshot
            post_snapshot = self._perception_capturer.capture()
            self._validate_perception_snapshot(post_snapshot, experiment_id)
            
            # Get deltas from adapter
            deltas = self._change_detector.compute(
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
                action_time_window=time_window
            )
            
            # 2️⃣ Never normalize deltas inside LifeLoop
            validated_deltas = []
            attribution_confident = True
            
            for delta in deltas:
                # Validate canonical fields only
                if not hasattr(delta, 'measurement_timestamp'):
                    self._crash(f"EXPERIMENT {experiment_id} - Delta missing measurement_timestamp")
                    raise RealityViolation(
                        experiment_id,
                        "Delta must have measurement_timestamp (canonical field)"
                    )
                
                if not hasattr(delta, 'measurement_reliability'):
                    self._crash(f"EXPERIMENT {experiment_id} - Delta missing measurement_reliability")
                    raise RealityViolation(
                        experiment_id,
                        "Delta must have measurement_reliability (canonical field)"
                    )
                
                # Type validation
                if not isinstance(delta.measurement_timestamp, float):
                    self._crash(f"EXPERIMENT {experiment_id} - Delta timestamp type violation")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta.measurement_timestamp must be float, got {type(delta.measurement_timestamp)}"
                    )
                
                if not isinstance(delta.measurement_reliability, (float, int)):
                    self._crash(f"EXPERIMENT {experiment_id} - Delta reliability type violation")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta.measurement_reliability must be numeric, got {type(delta.measurement_reliability)}"
                    )
                
                reliability = float(delta.measurement_reliability)
                if not 0.0 <= reliability <= 1.0:
                    self._crash(f"EXPERIMENT {experiment_id} - Delta reliability out of bounds")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta.measurement_reliability out of bounds: {reliability}"
                    )
                
                # 3️⃣ Delta validator must return bool
                is_valid = self._delta_validator.validate(delta, time_window)
                if not isinstance(is_valid, bool):
                    self._crash(f"EXPERIMENT {experiment_id} - Delta validator must return bool")
                    raise RealityViolation(
                        experiment_id,
                        "delta_validator.validate() must return bool"
                    )
                
                if not is_valid:
                    self._crash(f"EXPERIMENT {experiment_id} - Delta timestamp violation")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta.measurement_timestamp {delta.measurement_timestamp} outside action window {time_window}"
                    )
                
                if reliability < self.delta_confidence_threshold:
                    attribution_confident = False
                
                validated_deltas.append(delta)
            
            # 7️⃣ Ledger receives original deltas
            self._action_ledger.record(
                experiment_id=experiment_id,
                action=action,
                time_window=time_window,
                deltas=validated_deltas
            )
            
            # 8️⃣ Graph updater only runs when reliability clears threshold
            if attribution_confident:
                self._env_graph_updater.update(validated_deltas, experiment_id=experiment_id)
            
            # Return raw factual record
            return {
                'experiment_id': experiment_id,
                'action_id': action.id,
                'action_fingerprint': action.fingerprint,
                'time_window': time_window,
                'deltas': validated_deltas,
                'pre_snapshot_timestamp': pre_snapshot.timestamp,
                'post_snapshot_timestamp': post_snapshot.timestamp,
                'attribution_confident': attribution_confident,
                'delta_count': len(validated_deltas)
            }
            
        except Exception as e:
            if not isinstance(e, RealityViolation):
                self._crash(f"EXPERIMENT {experiment_id} - Unexpected failure: {str(e)}")
                raise RealityViolation(
                    experiment_id,
                    f"Unexpected failure: {str(e)}"
                ) from e
            else:
                raise
        finally:
            self._experiment_in_progress = False
    
    def _validate_perception_snapshot(self, snapshot: Any, experiment_id: str) -> None:
        """
        4️⃣ Snapshots must carry three things: frame, timestamp, metadata
        """
        required_attrs = ['frame', 'timestamp', 'metadata']
        for attr in required_attrs:
            if not hasattr(snapshot, attr):
                self._crash(f"EXPERIMENT {experiment_id} - Snapshot missing {attr}")
                raise RealityViolation(
                    experiment_id,
                    f"PerceptionSnapshot missing required attribute: {attr}"
                )
        
        import numpy as np
        
        frame = snapshot.frame
        if not isinstance(frame, np.ndarray):
            self._crash(f"EXPERIMENT {experiment_id} - Frame type violation")
            raise RealityViolation(
                experiment_id,
                f"PerceptionSnapshot.frame must be numpy.ndarray, got {type(frame)}"
            )
        
        # 5️⃣ Timestamp must be monotonic float
        if not isinstance(snapshot.timestamp, float):
            self._crash(f"EXPERIMENT {experiment_id} - Timestamp type violation")
            raise RealityViolation(
                experiment_id,
                f"PerceptionSnapshot.timestamp must be float, got {type(snapshot.timestamp)}"
            )
        
        if not isinstance(snapshot.metadata, dict):
            self._crash(f"EXPERIMENT {experiment_id} - Metadata type violation")
            raise RealityViolation(
                experiment_id,
                f"PerceptionSnapshot.metadata must be dict, got {type(snapshot.metadata)}"
            )
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID using monotonic time."""
        import hashlib
        monotonic_time = time.perf_counter_ns()
        monotonic_counter = time.perf_counter()
        return hashlib.sha256(
            f"{monotonic_time}:{monotonic_counter}".encode()
        ).hexdigest()[:16]
    
    def _crash(self, message: str) -> None:
        """Log crash - must never fail."""
        try:
            log_crash(message)
        except Exception:
            # If logging fails, we cannot continue
            raise SystemError("Logging system failed - cannot guarantee truth")


__all__ = ['LifeLoop']
