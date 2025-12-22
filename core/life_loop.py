"""
EXECUTION PROTOCOL OF REALITY
Environment Mastery Engine - Life Loop Module

One-sentence contract: Execute one action, measure what happened, record outcome.
Violation is fatal to truth.
"""

import time
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import hashlib

# Type definitions (provided by other modules)
@dataclass
class Action:
    """External action to execute."""
    id: str
    fingerprint: str  # Cryptographic hash of action definition
    executable_call: callable
    
@dataclass
class PerceptionSnapshot:
    """Raw sensory data from environment."""
    timestamp: datetime
    data: dict
    reliability_score: float

@dataclass 
class Delta:
    """Measured change in environment."""
    timestamp: datetime
    change_type: str
    before_state: dict
    after_state: dict
    confidence: float

@dataclass
class ExperimentResult:
    """Pure factual record of what occurred."""
    experiment_id: str
    action_id: str
    action_fingerprint: str
    time_window: tuple  # (start, end)
    deltas: List[Delta]
    attribution_confident: bool

# --- MODULE BOUNDARIES (imported dependencies) ---
def execute_action(action: Action) -> None:
    """External executor - calls OS/API."""
    pass

def capture_perception_snapshot() -> PerceptionSnapshot:
    """External perception system."""
    pass

def compute_changes(
    before: PerceptionSnapshot, 
    after: PerceptionSnapshot
) -> List[Delta]:
    """External change detector."""
    pass

def validate_delta_integrity(
    delta: Delta,
    action_window: tuple
) -> bool:
    """External validator."""
    pass

def record_to_action_ledger(
    experiment_id: str,
    action: Action,
    time_window: tuple,
    deltas: List[Delta]
) -> None:
    """External causal record keeper."""
    pass

def update_env_graph(deltas: List[Delta]) -> None:
    """External navigation memory."""
    pass
# --- END MODULE BOUNDARIES ---

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
    
    def __init__(self, observation_delay: float = 0.5, delta_confidence_threshold: float = 0.9):
        """
        observation_delay: Explicit bounded wait after action (seconds)
        delta_confidence_threshold: Minimum confidence for attribution (0.0-1.0)
                                   Changes below this threshold mark attribution_confident=False
        """
        self.observation_delay = observation_delay
        self.delta_confidence_threshold = delta_confidence_threshold
        
        # Explicit state lock for anti-overlap
        self._experiment_in_progress = False
        
    def run_experiment(self, action: Action) -> ExperimentResult:
        """
        Execute exactly one action and measure exactly what happened.
        
        This is the only public operation allowed by the module.
        """
        # --- STATE LOCK: Prevent overlapping experiments ---
        if self._experiment_in_progress:
            raise RealityViolation(
                "pre-start",
                "Experiment already in progress. Overlapping experiments forbidden."
            )
        
        experiment_id = self._generate_experiment_id()
        
        try:
            self._experiment_in_progress = True
            
            # Step 1 & 2: Capture pre-snapshot at action start
            pre_snapshot = capture_perception_snapshot()
            action_start = datetime.now()
            
            # Step 3: Execute exactly one action
            self._execute_single_action(action)
            
            # Step 4: Record action end
            action_end = datetime.now()
            time_window = (action_start, action_end)
            
            # Step 5: Wait bounded explicit observation delay
            time.sleep(self.observation_delay)
            
            # Step 6: Capture post-snapshot
            post_snapshot = capture_perception_snapshot()
            
            # Step 7: Measure changes using change detector
            deltas = compute_changes(pre_snapshot, post_snapshot)
            
            # Step 8: Validate delta integrity
            validated_deltas = []
            attribution_confident = True
            
            for delta in deltas:
                is_valid = validate_delta_integrity(delta, time_window)
                
                if not is_valid:
                    # Time attribution unclear → abort immediately
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
            record_to_action_ledger(
                experiment_id=experiment_id,
                action=action,
                time_window=time_window,
                deltas=validated_deltas
            )
            
            # Step 11: Update navigation memory
            if attribution_confident:
                update_env_graph(validated_deltas)
            
            # Step 12: Return factual result (no success/failure judgement)
            result = ExperimentResult(
                experiment_id=experiment_id,
                action_id=action.id,
                action_fingerprint=action.fingerprint,
                time_window=time_window,
                deltas=validated_deltas,
                attribution_confident=attribution_confident
            )
            
            return result
            
        except Exception as e:
            # Failure is loud, failure stops everything
            if not isinstance(e, RealityViolation):
                raise RealityViolation(
                    experiment_id,
                    f"Unexpected failure: {str(e)}"
                ) from e
            raise
        finally:
            # Always release the state lock, even on failure
            self._experiment_in_progress = False
    
    def _execute_single_action(self, action: Action) -> None:
        """
        Execute action with isolation guarantees.
        
        FINGERPRINT SEMANTICS:
        The fingerprint check is a GUARD against action corruption during transmission.
        It is NOT a full cryptographic identity of the action's effects.
        The authoritative action identity is maintained by action_ledger.
        """
        computed_fingerprint = self._compute_action_fingerprint(action)
        if computed_fingerprint != action.fingerprint:
            raise RealityViolation(
                "pre-execution",
                f"Action fingerprint mismatch: {computed_fingerprint} != {action.fingerprint}"
            )
        
        # Exactly one execution, no retries
        execute_action(action)
    
    def _generate_experiment_id(self) -> str:
        """Generate unique, reproducible experiment ID."""
        timestamp = datetime.now().isoformat()
        nonce = str(time.perf_counter_ns())
        return hashlib.sha256(f"{timestamp}:{nonce}".encode()).hexdigest()[:16]
    
    def _compute_action_fingerprint(self, action: Action) -> str:
        """
        Compute deterministic fingerprint of action.
        
        NOTE: This is a SIMPLE GUARD, not a full identity.
        It ensures basic action integrity but does NOT capture:
        - Action parameters
        - Environmental context  
        - Dynamic call arguments
        
        The action_ledger maintains authoritative action identity.
        """
        action_repr = f"{action.id}:{action.executable_call.__name__}"
        return hashlib.sha256(action_repr.encode()).hexdigest()[:16]

# --- PUBLIC INTERFACE ---
# Only one allowed public entity: the LifeLoop class
__all__ = ['LifeLoop']
