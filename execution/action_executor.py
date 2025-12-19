"""
Action Executor

Pure execution sequencer for ordered action lists.
This module strictly executes pre-specified actions in sequence without interpretation.
It contains zero intelligence about action validity, sequence optimization, or outcome assessment.

Absolute constraints:
- NO decision-making about which actions to run
- NO modification, reordering, or skipping of actions
- NO retry logic or fallback mechanisms
- NO success/failure inference or interpretation
- NO optimization of execution order or timing
- NO environment assumptions or domain knowledge
- NO hidden state between sequences
- NO logging or side effects
- NO timestamp generation (timestamps must be provided externally)
- NO policy decisions

UUIDs are generated for mechanical traceability only; no semantic meaning.
Sequence timing must be managed externally.

This module is a dumb sequencer only - boring, auditable, and replaceable.
"""

from typing import List, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from execution.body_interface import (
    BodyInterface,
    ConcreteAction,
    ExecutionResult,
    ExecutionBackendError
)


class ExecutionSequenceError(Exception):
    """Exception raised when sequence execution fails."""
    pass


class ActionSequence(BaseModel):
    """
    Ordered sequence of concrete actions to execute.
    
    The sequence is executed exactly as provided with no modifications.
    No validation of sequence logic, semantics, or effectiveness.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core identification
    sequence_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Generated for mechanical traceability only; no semantic meaning"
    )
    
    # Ordered actions (executed strictly in this order)
    actions: List[ConcreteAction] = Field(..., min_items=0)
    
    # Metadata (not used for execution decisions)
    source: Optional[str] = None  # Where this sequence came from (brain, test, etc.)


class ExecutionSequenceResult(BaseModel):
    """
    Mechanical execution results for a sequence.
    
    Contains only factual records of what was executed, in order.
    No interpretation of whether the sequence was "successful" or "correct".
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core identification
    sequence_id: str
    execution_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Generated for mechanical traceability only; no semantic meaning"
    )
    
    # Execution results in order
    action_results: List[ExecutionResult] = Field(default_factory=list)
    
    # Sequence-level mechanical status
    completed: bool = Field(...)  # Whether all actions were attempted (not whether they "worked")
    termination_reason: Optional[str] = None  # Why sequence stopped (error, completion, etc.)


class ActionExecutor:
    """
    Stateless sequencer for executing action sequences.
    
    This class:
    1. Validates sequence structure
    2. Executes actions strictly in order
    3. Stops only on hard exceptions
    4. Captures mechanical execution results
    
    This class does NOT:
    1. Decide which actions to run
    2. Modify, reorder, or skip actions
    3. Retry failed actions
    4. Infer success or failure
    5. Optimize execution
    6. Interpret results
    7. Maintain state between sequences
    8. Log or produce side effects
    9. Generate timestamps or policy decisions
    10. Measure execution timing
    """
    
    def __init__(self, body_interface: BodyInterface) -> None:
        """
        Initialize action executor with body interface.
        
        Args:
            body_interface: BodyInterface instance for executing individual actions
        
        Raises:
            TypeError: If body_interface is not a BodyInterface instance
        """
        if not isinstance(body_interface, BodyInterface):
            raise TypeError(
                f"body_interface must be BodyInterface instance, got {type(body_interface)}"
            )
        
        self._body_interface = body_interface
    
    def execute_sequence(self, sequence: ActionSequence) -> ExecutionSequenceResult:
        """
        Execute action sequence exactly as provided.
        
        Args:
            sequence: Ordered sequence of concrete actions
        
        Returns:
            ExecutionSequenceResult with mechanical execution details only
        
        Raises:
            ExecutionSequenceError: If sequence execution fails
        
        Note:
            - Actions executed strictly in provided order
            - No retry on any failure
            - No interpretation of results
            - Stop immediately on first hard exception
            - Zero intelligence about sequence validity
            - No logging or side effects
        """
        action_results: List[ExecutionResult] = []
        completed = False
        termination_reason = None
        
        try:
            # Execute each action in strict order
            # No optimization, reordering, or skipping
            for i, action in enumerate(sequence.actions):
                # Execute exactly once, no retry
                result = self._body_interface.execute_action(action)
                action_results.append(result)
                
                # Check if action was executed (mechanical fact, not success assessment)
                # Note: We do NOT stop if executed=False - that's a backend error we already handled
                # We only stop if body_interface raised an exception
                # The fact that we're here means no exception was raised
                
                # We continue to next action regardless of result content
                # No decision-making based on result
            
            # All actions executed without hard exception
            completed = True
            termination_reason = "sequence_completed"
            
        except Exception as e:
            # Hard exception from body_interface
            # Stop immediately, do not execute remaining actions
            completed = False
            termination_reason = f"execution_error: {type(e).__name__}"
            
            # Re-raise with typed exception preserving original context
            raise ExecutionSequenceError(
                f"Sequence execution terminated at action {len(action_results)}"
            ) from e
        
        # Create sequence result (mechanical facts only)
        sequence_result = ExecutionSequenceResult(
            sequence_id=sequence.sequence_id,
            action_results=action_results,
            completed=completed,
            termination_reason=termination_reason
        )
        
        return sequence_result


# Export public interface
__all__ = [
    "ActionExecutor",
    "ActionSequence",
    "ExecutionSequenceResult",
    "ExecutionSequenceError"
]