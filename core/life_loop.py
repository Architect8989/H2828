"""
EXECUTION PROTOCOL OF REALITY
Environment Mastery Engine - Life Loop Module

One-sentence contract: Execute one action, measure what happened, record outcome.
Violation is fatal to truth.
"""

import time
from datetime import datetime
import hashlib

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
            
            # Step 1 & 2: Capture pre-snapshot at action start
            pre_snapshot = self._perception_capturer.capture()
            action_start = datetime.now()
            
            # Step 3: Execute exactly one action
            self._action_executor.execute(action)
            
            # Step 4: Record action end
            action_end = datetime.now()
            time_window = (action_start, action_end)
            
            # Step 5: Wait bounded explicit observation delay
            time.sleep(self.observation_delay)
            
            # Step 6: Capture post-snapshot
            post_snapshot = self._perception_capturer.capture()
            
            # Step 7: Measure changes using change detector
            deltas = self._change_detector.compute(pre_snapshot, post_snapshot)
            
            # Step 8: Validate delta integrity
            validated_deltas = []
            attribution_confident = True
            
            for delta in deltas:
                is_valid = self._delta_validator.validate(delta, time_window)
                
                if not is_valid:
                    # Time attribution unclear → abort immediately
                    log_crash(f"EXPERIMENT {experiment_id} - Delta timestamp violation")
                    raise RealityViolation(
                        experiment_id,
                        f"Delta timestamp {delta.timestamp} outside action window {time_window}"
                    )
                
                # Use explicit threshold from constructor
                if delta.confidence < self.delta_confidence_threshold:
                    attribution_confident = False
                
                validated_deltas.append(delta)
            
            # Step 9: Attribution already performed during validation
            # Deltas are only included if timestamp falls in action window
            
            # Step 10: Record causality in action ledger
            self._action_ledger.record(
                experiment_id=experiment_id,
                action=action,
                time_window=time_window,
                deltas=validated_deltas
            )
            
            # Step 11: Update navigation memory
            if attribution_confident:
                self._env_graph_updater.update(validated_deltas)
            
            # Step 12: Return factual result (no success/failure judgement)
            result = {
                'experiment_id': experiment_id,
                'action_id': action.id,
                'action_fingerprint': action.fingerprint,
                'time_window': time_window,
                'deltas': validated_deltas,
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
    
    def _generate_experiment_id(self) -> str:
        """Generate unique, reproducible experiment ID."""
        timestamp = datetime.now().isoformat()
        nonce = str(time.perf_counter_ns())
        return hashlib.sha256(f"{timestamp}:{nonce}".encode()).hexdigest()[:16]


# --- PUBLIC INTERFACE ---
# Only one allowed public entity: the LifeLoop class
__all__ = ['LifeLoop']
