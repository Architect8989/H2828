"""
EXECUTION PROTOCOL OF REALITY
Environment Mastery Engine - Life Loop Module

One-sentence contract: Execute one action, measure what happened, record outcome.
Violation is fatal to truth.
"""

import time
from typing import Any

from core.logger import log_crash

# Import Action from execution layer - LifeLoop does NOT define domain objects
from execution.action_executor import Action


class RealityViolation(Exception):
    """Raised when truth cannot be guaranteed."""
    def __init__(self, experiment_id: str, context: str):
        self.experiment_id = experiment_id
        self.context = context
        super().__init__(f"REALITY VIOLATION [{experiment_id}]: {context}")


class LifeLoop:
    """
    Boring, strict, unforgiving gatekeeper.
    No intelligence. No agency. No optimization.
    
    PUBLIC API RULE: Only LifeLoop.run_experiment() exists.
    No module-level wrappers. No alternate entry points.
    """
    
    def __init__(self, action_executor, perception_capturer, change_detector, 
                 delta_validator, action_ledger, env_graph_updater,
                 observation_delay: float = 0.5, delta_confidence_threshold: float = 0.9):
        """
        Dependencies must be injected, not stubbed:
        - action_executor: Executes actions through OS backend
        - perception_capturer: Captures environment state
        - change_detector: Computes differences between states
        - delta_validator: Validates timing and attribution
        - action_ledger: Records causality chain
        - env_graph_updater: Updates navigation memory
        
        observation_delay: Explicit bounded wait after action (seconds)
        delta_confidence_threshold: Minimum confidence for attribution (0.0-1.0)
        """
        self._action_executor = action_executor
        self._perception_capturer = perception_capturer
        self._change_detector = change_detector
        self._delta_validator = delta_validator
        self._action_ledger = action_ledger
        self._env_graph_updater = env_graph_updater
        
        self.observation_delay = observation_delay
        self.delta_confidence_threshold = delta_confidence_threshold
        
        # Explicit state lock for anti-overlap
        self._experiment_in_progress = False
        
    def run_experiment(self, action: Action) -> dict:
        """
        Execute exactly one action and measure exactly what happened.
        
        This is the only public operation allowed by the module.
        Returns factual record of what occurred.
        """
        # Generate experiment ID early for crash logging
        experiment_id = self._generate_experiment_id()
        
        # --- STATE LOCK: Prevent overlapping experiments ---
        if self._experiment_in_progress:
            log_crash(f"EXPERIMENT {experiment_id} - Overlap violation")
            raise RealityViolation(
                experiment_id,
                "Experiment already in progress. Overlapping experiments forbidden."
            )
        
        try:
            self._experiment_in_progress = True
            
            # Step 1: Validate action before execution
            try:
                action.validate()
            except Exception as e:
                log_crash(f"EXPERIMENT {experiment_id} - Invalid action: {str(e)}")
                raise RealityViolation(
                    experiment_id,
                    f"Action validation failed: {str(e)}"
                ) from e
            
            # Step 2: Capture pre-snapshot at action start
            pre_snapshot = self._perception_capturer.capture()
            
            # ENFORCE PerceptionSnapshot contract
            self._validate_perception_snapshot(pre_snapshot)
            
            # Step 3: Execute exactly one action (monotonic time only)
            action_start = time.perf_counter()
            self._action_executor.execute(action)
            action_end = time.perf_counter()
            time_window = (action_start, action_end)
            
            # ENFORCE time window type (monotonic floats)
            if not isinstance(time_window[0], float) or not isinstance(time_window[1], float):
                log_crash(f"EXPERIMENT {experiment_id} - Time window type violation")
                raise RealityViolation(
                    experiment_id,
                    f"Time window must be monotonic floats, got {type(time_window[0])}, {type(time_window[1])}"
                )
            
            # Step 4: Explicit blind wait; no synchronization guarantees
            # This is a simple fixed delay for v1, will become adaptive later
            time.sleep(self.observation_delay)
            
            # Step 5: Capture post-snapshot
            post_snapshot = self._perception_capturer.capture()
            self._validate_perception_snapshot(post_snapshot)
            
            # Step 6: Measure changes using change detector with action time window
            deltas = self._change_detector.compute(
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
                action_time_window=time_window
            )
            
            # Step 7: Validate delta integrity
            validated_deltas = []
            attribution_confident = True
            
            for delta in deltas:
                # NORMALIZE delta schema at boundary
                normalized_delta = self._normalize_delta(delta)
                
                # Validate with monotonic time window
                is_valid = self._delta_validator.validate(normalized_delta, time_window)
                
                if not is_valid:
                    # Time attribution unclear → abort immediately
                    log_crash(f"EXPERIMENT {experiment_id} - Delta timestamp violation")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta timestamp {normalized_delta['timestamp']} outside action window {time_window}"
                    )
                
                # Use explicit threshold from constructor
                if normalized_delta['confidence'] < self.delta_confidence_threshold:
                    attribution_confident = False
                
                validated_deltas.append(normalized_delta)
            
            # Step 8: Attribution already performed during validation
            # Deltas are only included if timestamp falls in action window
            
            # Step 9: Record causality in action ledger
            self._action_ledger.record(
                experiment_id=experiment_id,
                action=action,
                time_window=time_window,
                deltas=validated_deltas  # Normalized dict deltas
            )
            
            # Step 10: Update navigation memory
            if attribution_confident:
                self._env_graph_updater.update(validated_deltas)  # Normalized dict deltas
            
            # Step 11: Return factual result (no success/failure judgement)
            # WARNING: Deltas may contain non-serializable objects (e.g., numpy arrays)
            # This return payload is for immediate consumption only
            # Persistence happens through the ledger
            result = {
                'experiment_id': experiment_id,
                'action_id': action.id,
                'action_fingerprint': action.fingerprint,
                'time_window': time_window,
                'deltas': validated_deltas,  # Contains normalized dict deltas only
                'attribution_confident': attribution_confident
            }
            
            return result
            
        except Exception as e:
            # Failure is loud, failure stops everything
            if not isinstance(e, RealityViolation):
                log_crash(f"EXPERIMENT {experiment_id} - Unexpected failure: {str(e)}")
                raise RealityViolation(
                    experiment_id,
                    f"Unexpected failure: {str(e)}"
                ) from e
            else:
                # Already logged above, just re-raise
                raise
        finally:
            # Always release the state lock, even on failure
            self._experiment_in_progress = False
    
    def _validate_perception_snapshot(self, snapshot: Any) -> None:
        """Fail fast if PerceptionSnapshot contract is violated."""
        if not hasattr(snapshot, 'frame'):
            raise RealityViolation(
                "snapshot_validation",
                "PerceptionSnapshot missing required attribute: frame"
            )
        if not hasattr(snapshot, 'timestamp'):
            raise RealityViolation(
                "snapshot_validation",
                "PerceptionSnapshot missing required attribute: timestamp"
            )
        # Timestamp must be monotonic float
        if not isinstance(snapshot.timestamp, float):
            raise RealityViolation(
                "snapshot_validation",
                f"PerceptionSnapshot.timestamp must be float (monotonic), got {type(snapshot.timestamp)}"
            )
    
    def _normalize_delta(self, delta: Any) -> dict:
        """
        Normalize delta to EME-core schema at LifeLoop boundary.
        
        Converts:
        - delta.timestamp OR delta['timestamp'] OR delta.measurement_timestamp → normalized['timestamp']
        - delta.confidence OR delta['confidence'] OR delta.measurement_reliability → normalized['confidence']
        
        Returns dict with keys: timestamp, confidence, _original
        """
        normalized = {}
        
        # Extract timestamp (monotonic float)
        if hasattr(delta, 'timestamp'):
            normalized['timestamp'] = delta.timestamp
        elif hasattr(delta, 'measurement_timestamp'):
            normalized['timestamp'] = delta.measurement_timestamp
        elif isinstance(delta, dict) and 'timestamp' in delta:
            normalized['timestamp'] = delta['timestamp']
        elif isinstance(delta, dict) and 'measurement_timestamp' in delta:
            normalized['timestamp'] = delta['measurement_timestamp']
        else:
            raise RealityViolation(
                "delta_normalization",
                f"Delta missing timestamp field: {type(delta)}"
            )
        
        # Ensure timestamp is monotonic float
        if not isinstance(normalized['timestamp'], float):
            raise RealityViolation(
                "delta_normalization",
                f"Delta timestamp must be float (monotonic), got {type(normalized['timestamp'])}"
            )
        
        # Extract confidence (float 0.0-1.0)
        if hasattr(delta, 'confidence'):
            normalized['confidence'] = delta.confidence
        elif hasattr(delta, 'measurement_reliability'):
            normalized['confidence'] = delta.measurement_reliability
        elif isinstance(delta, dict) and 'confidence' in delta:
            normalized['confidence'] = delta['confidence']
        elif isinstance(delta, dict) and 'measurement_reliability' in delta:
            normalized['confidence'] = delta['measurement_reliability']
        else:
            raise RealityViolation(
                "delta_normalization",
                f"Delta missing confidence field: {type(delta)}"
            )
        
        # Ensure confidence is float in valid range [0.0, 1.0]
        if not isinstance(normalized['confidence'], (float, int)):
            raise RealityViolation(
                "delta_normalization",
                f"Delta confidence must be numeric, got {type(normalized['confidence'])}"
            )
        
        normalized['confidence'] = float(normalized['confidence'])
        
        # ENFORCE confidence bounds
        if not 0.0 <= normalized['confidence'] <= 1.0:
            raise RealityViolation(
                "delta_normalization",
                f"Confidence out of bounds: {normalized['confidence']}"
            )
        
        # Preserve original delta for reference
        normalized['_original'] = delta
        
        return normalized
    
    def _generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID.
        
        Note: This ID is unique but not reproducible or deterministic.
        It is collision-safe and auditable for v1.
        """
        import hashlib
        monotonic_time = time.perf_counter_ns()
        monotonic_counter = time.perf_counter()
        return hashlib.sha256(
            f"{monotonic_time}:{monotonic_counter}".encode()
        ).hexdigest()[:16]


# --- PUBLIC INTERFACE ---
# Only one allowed public entity: the LifeLoop class
__all__ = ['LifeLoop']
