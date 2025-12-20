"""
Epistemic Governor - Environment Mastery Engine

Absolute epistemic constraint: No action may complete without producing
a valid, persistent update to the World Model.

This module is the execution boundary - the narrow waist of the system.
It defines what it means for EME to be real.

If removed, the system must not function.
"""

from typing import Any, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """Explicit action contract for EME execution."""
    type: str
    parameters: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate action structure on creation."""
        if not isinstance(self.type, str):
            raise ValueError(f"Action.type must be str, got {type(self.type)}")
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Action.parameters must be dict, got {type(self.parameters)}")


class EpistemicViolation(Exception):
    """Raised when Prime Directive is violated."""
    def __init__(self, violation: str, context: dict):
        super().__init__(f"EpistemicViolation: {violation}")
        self.violation = violation
        self.context = context


class LifeLoop:
    """
    Epistemic governor enforcing Action → Delta → World Model invariant.
    
    This module is:
    1. The execution boundary - all actions must pass through here
    2. The epistemic gatekeeper - validates deltas before state updates
    3. The failure mechanism - crashes loudly on violations
    
    This module is NOT:
    1. An intelligence layer (no planning, reasoning, interpretation)
    2. An orchestrator (no sequencing, retries, optimization)
    3. An agent (no goals, memory, learning)
    
    World Model Interface Contract:
    - apply_delta(delta) must raise an exception on failure
    - persist() must raise an exception on failure
    - Silent returns or success signals are forbidden
    
    Delta Interface Contract:
    - Must have is_valid() method returning bool
    - Must have is_empty() method returning bool
    - Must have changes attribute/property exposing immutable state
    """
    
    def __init__(
        self,
        world_model,  # Type hint removed due to circular import
        action_executor  # Type hint removed due to circular import
    ) -> None:
        """
        Initialize epistemic governor.
        
        Args:
            world_model: Authoritative World Model instance that must:
                         - Have apply_delta(delta) that raises on failure
                         - Have persist() that raises on failure
                         - Never return silent success signals
            action_executor: Constrained executor that returns deltas and
                             must have execute(action) method
                             
        Raises:
            EpistemicViolation: If world_model is missing
            TypeError: If interfaces are incorrect or lack required methods
        """
        # Require authoritative World Model (Hard Constraint #2)
        if world_model is None:
            raise EpistemicViolation(
                violation="MISSING_WORLD_MODEL",
                context={"message": "System requires authoritative World Model"}
            )
        
        # Validate World Model interface
        required_wm_methods = ['apply_delta', 'persist']
        for method in required_wm_methods:
            if not hasattr(world_model, method):
                raise TypeError(
                    f"world_model must have {method} method"
                )
        
        # Validate action executor interface
        if not hasattr(action_executor, 'execute'):
            raise TypeError(
                f"action_executor must have execute method, got {type(action_executor)}"
            )
        
        self._world_model = world_model
        self._action_executor = action_executor
    
    def _select_exploration_action(self) -> Action:
        """
        Placeholder action selection (non-intelligent).
        
        Returns:
            A single exploration action with explicit structure
            
        Note:
            This is placeholder logic only. Real action selection must be
            implemented externally and submitted to this loop.
        """
        # Explicit Action structure - no anonymous objects
        return Action(type="explore", parameters={})
    
    def _validate_delta(self, delta) -> None:
        """
        Validate delta structure and content using behavioral interface.
        
        Args:
            delta: Delta to validate (must satisfy Delta Interface Contract)
            
        Raises:
            EpistemicViolation: If delta violates interface contract
        """
        if delta is None:
            raise EpistemicViolation(
                violation="MISSING_DELTA",
                context={"message": "Action returned no delta"}
            )
        
        # Behavioral interface validation - no class name checks
        required_attributes = ['is_valid', 'is_empty']
        for attr in required_attributes:
            if not hasattr(delta, attr):
                raise EpistemicViolation(
                    violation="INVALID_DELTA_INTERFACE",
                    context={
                        "message": f"Delta missing required attribute: {attr}",
                        "delta_type": type(delta).__name__
                    }
                )
        
        # Method must be callable
        if not callable(delta.is_valid) or not callable(delta.is_empty):
            raise EpistemicViolation(
                violation="INVALID_DELTA_INTERFACE",
                context={"message": "Delta attributes must be callable methods"}
            )
        
        # Execute validation
        if not delta.is_valid():
            raise EpistemicViolation(
                violation="INVALID_DELTA_STRUCTURE",
                context={
                    "message": "Delta.is_valid() returned False",
                    "delta_type": type(delta).__name__
                }
            )
        
        if delta.is_empty():
            raise EpistemicViolation(
                violation="EMPTY_DELTA",
                context={
                    "message": "Delta.is_empty() returned True",
                    "delta_type": type(delta).__name__
                }
            )
    
    def run_iteration(self) -> None:
        """
        Execute one iteration of Action → Delta → World Model.
        
        Control Flow:
        1. Select single exploration action (placeholder)
        2. Execute action via constrained executor
        3. Validate presence and validity of delta
        4. Apply delta to World Model (must raise on failure)
        5. Persist World Model state (must raise on failure)
        
        Raises:
            EpistemicViolation: If Prime Directive is violated
            Exception: Any underlying exception (propagated without interpretation)
        """
        # 1. Select action (placeholder - real selection happens externally)
        action = self._select_exploration_action()
        
        # 2. Execute action and obtain delta (Hard Constraint #1, #3)
        delta = self._action_executor.execute(action)
        
        # 3. Validate delta (Hard Constraint #3)
        self._validate_delta(delta)
        
        # 4. Apply delta to World Model (contract: raises on failure)
        self._world_model.apply_delta(delta)
        
        # 5. Persist state (Hard Constraint: persistent update, raises on failure)
        self._world_model.persist()


# Export public interface
__all__ = ["LifeLoop", "EpistemicViolation", "Action"]            - No state maintained between calls
            - No logging or side effects
            - Exact one-pass execution only
        """
        # Normalize vision outputs without interpretation
        vision_output = self._vision_client.process_vision_output(
            ocr_output=ocr_output,
            ui_parser_output=ui_parser_output,
            vlm_output=vlm_output,
            object_detector_output=object_detector_output,
            screen_dimensions=screen_dimensions,
            screen_timestamp=screen_timestamp,
            input_hash=input_hash
        )
        
        # Infer action sequence without interpretation
        action_sequence = self._brain_client.infer(vision_output)
        
        # Execute sequence without interpretation
        self._action_executor.execute_sequence(action_sequence)
        
        return action_sequence


# Export public interface
__all__ = ["LifeLoop"]

