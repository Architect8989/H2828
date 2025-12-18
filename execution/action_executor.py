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

This module is a dumb sequencer only - boring, auditable, and replaceable.
"""

import logging
from datetime import datetime
from typing import List, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, ValidationError, Field, validator, ConfigDict

from execution.body_interface import (
    BodyInterface,
    ConcreteAction,
    ExecutionResult,
    EXECUTION_LOG_LEVEL
)

logger = logging.getLogger(__name__)


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
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Ordered actions (executed strictly in this order)
    actions: List[ConcreteAction] = Field(..., min_items=0)
    
    # Metadata (not used for execution decisions)
    source: Optional[str] = None  # Where this sequence came from (brain, test, etc.)
    
    @validator('actions')
    def validate_actions_are_concrete(cls, v: List[ConcreteAction]) -> List[ConcreteAction]:
        """Ensure all actions are validated ConcreteAction instances."""
        # Pydantic already validates each item as ConcreteAction
        # This validator exists only to document the requirement
        return v


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
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Timing (mechanical only)
    timestamp_start: datetime
    timestamp_end: datetime
    duration_ms: int = Field(..., ge=0)
    
    # Execution results in order
    action_results: List[ExecutionResult] = Field(default_factory=list)
    
    # Sequence-level mechanical status
    completed: bool = Field(...)  # Whether all actions were attempted (not whether they "worked")
    termination_reason: Optional[str] = None  # Why sequence stopped (error, completion, etc.)
    
    @validator('duration_ms')
    def validate_duration(cls, v: int) -> int:
        """Validate duration is non-negative."""
        if v < 0:
            raise ValueError(f"Duration must be non-negative, got {v}ms")
        return v


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
    """
    
    def __init__(
        self, 
        body_interface: BodyInterface,
        execution_log_level: int = EXECUTION_LOG_LEVEL
    ) -> None:
        """
        Initialize action executor with body interface.
        
        Args:
            body_interface: BodyInterface instance for executing individual actions
            execution_log_level: Logging level for execution audit
        
        Raises:
            TypeError: If body_interface is not a BodyInterface instance
        """
        if not isinstance(body_interface, BodyInterface):
            raise TypeError(
                f"body_interface must be BodyInterface instance, got {type(body_interface)}"
            )
        
        self._body_interface = body_interface
        self._logger = logging.getLogger(f"{__name__}.ActionExecutor")
        self._execution_log_level = execution_log_level
    
    def execute_sequence(self, sequence: ActionSequence) -> ExecutionSequenceResult:
        """
        Execute action sequence exactly as provided.
        
        Args:
            sequence: Ordered sequence of concrete actions
        
        Returns:
            ExecutionSequenceResult with mechanical execution details only
        
        Raises:
            ValidationError: If sequence fails schema validation
            RuntimeError: If body_interface raises exception (propagated immediately)
        
        Note:
            - Actions executed strictly in provided order
            - No retry on any failure
            - No interpretation of results
            - Stop immediately on first hard exception
            - Zero intelligence about sequence validity
        """
        # Log sequence receipt for auditability
        self._log_sequence_received(sequence)
        
        # Start sequence timing
        timestamp_start = datetime.utcnow()
        
        action_results: List[ExecutionResult] = []
        completed = False
        termination_reason = None
        
        try:
            # Execute each action in strict order
            # No optimization, reordering, or skipping
            for i, action in enumerate(sequence.actions):
                # Log action about to be executed (mechanical fact only)
                self._log_action_starting(sequence, action, i)
                
                # Execute exactly once, no retry
                result = self._body_interface.execute_action(action)
                action_results.append(result)
                
                # Log action completed (mechanical fact only)
                self._log_action_completed(sequence, action, result, i)
                
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
            
            # Log sequence termination due to error
            self._log_sequence_terminated(sequence, e, len(action_results))
            
            # Re-raise to let caller handle (we don't interpret)
            raise RuntimeError(
                f"Sequence execution terminated at action {len(action_results)}: {str(e)}"
            ) from e
        
        finally:
            # End sequence timing
            timestamp_end = datetime.utcnow()
        
        # Calculate duration
        duration_ms = int((timestamp_end - timestamp_start).total_seconds() * 1000)
        
        # Create sequence result (mechanical facts only)
        sequence_result = ExecutionSequenceResult(
            sequence_id=sequence.sequence_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_ms=duration_ms,
            action_results=action_results,
            completed=completed,
            termination_reason=termination_reason
        )
        
        # Log sequence completion for auditability
        self._log_sequence_completed(sequence, sequence_result)
        
        return sequence_result
    
    def _log_sequence_received(self, sequence: ActionSequence) -> None:
        """Log sequence receipt for auditability."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "sequence_received",
            "sequence_id": sequence.sequence_id,
            "action_count": len(sequence.actions),
            "source": sequence.source
        }
        
        self._logger.log(
            self._execution_log_level, 
            "Action sequence received for execution", 
            extra=log_data
        )
    
    def _log_action_starting(
        self, 
        sequence: ActionSequence, 
        action: ConcreteAction, 
        index: int
    ) -> None:
        """Log action about to be executed (mechanical fact only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "action_starting",
            "sequence_id": sequence.sequence_id,
            "action_index": index,
            "action_id": action.action_id,
            "action_type": action.action_type.value
        }
        
        self._logger.log(
            self._execution_log_level, 
            f"Executing action {index} of {len(sequence.actions)}", 
            extra=log_data
        )
    
    def _log_action_completed(
        self, 
        sequence: ActionSequence, 
        action: ConcreteAction, 
        result: ExecutionResult,
        index: int
    ) -> None:
        """Log action execution completion (mechanical fact only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "action_completed",
            "sequence_id": sequence.sequence_id,
            "action_index": index,
            "action_id": action.action_id,
            "execution_id": result.execution_id,
            "executed": result.executed,
            "duration_ms": result.duration_ms
        }
        
        self._logger.log(
            self._execution_log_level, 
            f"Action {index} execution completed", 
            extra=log_data
        )
    
    def _log_sequence_terminated(
        self, 
        sequence: ActionSequence, 
        error: Exception, 
        executed_count: int
    ) -> None:
        """Log sequence termination due to error (mechanical fact only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "sequence_terminated",
            "sequence_id": sequence.sequence_id,
            "executed_count": executed_count,
            "total_actions": len(sequence.actions),
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        self._logger.log(
            self._execution_log_level, 
            f"Sequence terminated after {executed_count} actions due to error", 
            extra=log_data
        )
    
    def _log_sequence_completed(
        self, 
        sequence: ActionSequence, 
        result: ExecutionSequenceResult
    ) -> None:
        """Log sequence completion for auditability (mechanical facts only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "sequence_completed",
            "sequence_id": sequence.sequence_id,
            "execution_id": result.execution_id,
            "completed": result.completed,
            "executed_count": len(result.action_results),
            "total_actions": len(sequence.actions),
            "duration_ms": result.duration_ms,
            "termination_reason": result.termination_reason
        }
        
        self._logger.log(
            self._execution_log_level, 
            "Action sequence execution completed", 
            extra=log_data
        )


# Export public interface
__all__ = [
    "ActionExecutor",
    "ActionSequence",
    "ExecutionSequenceResult"
]