"""
Body Interface

Boundary layer between cognitive action decisions and physical OS control.
This module strictly executes concrete, fully-specified actions without interpretation.
It contains zero intelligence about action correctness, UI state, or user intent.

Absolute constraints:
- NO decision-making about what action to take
- NO retry logic or fallback mechanisms
- NO success/failure inference or interpretation
- NO optimization of action sequences
- NO OS or application assumptions
- NO hidden state maintenance
- NO intent interpretation

This module is the muscle socket only - boring, auditable, and replaceable.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, ValidationError, Field, validator, ConfigDict

# Configure audit logging
logger = logging.getLogger(__name__)
EXECUTION_LOG_LEVEL = logging.INFO


class ActionType(str, Enum):
    """Enumeration of concrete, mechanical action types."""
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    KEY_PRESS = "key_press"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    SCROLL = "scroll"
    WAIT = "wait"


class MouseButton(str, Enum):
    """Mouse button identifiers without semantic interpretation."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    X1 = "x1"  # Back button
    X2 = "x2"  # Forward button


class CoordinateSystem(str, Enum):
    """Coordinate system for mouse positions."""
    ABSOLUTE = "absolute"    # Screen coordinates in pixels
    RELATIVE = "relative"    # Relative to current position


class ConcreteAction(BaseModel):
    """
    Fully specified, concrete action command.
    
    Every field must be explicitly provided - no defaults inferred from context.
    No interpretation of whether this action is correct, safe, or useful.
    """
    model_config = ConfigDict(
        extra='forbid',  # Reject unexpected fields
        frozen=True,     # Immutable after creation
        validate_assignment=True
    )
    
    # Core identification
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: ActionType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Position parameters (only for position-based actions)
    x: Optional[int] = Field(None, ge=0)
    y: Optional[int] = Field(None, ge=0)
    coordinate_system: CoordinateSystem = CoordinateSystem.ABSOLUTE
    
    # Mouse parameters (only for mouse actions)
    button: Optional[MouseButton] = None
    clicks: Optional[int] = Field(None, ge=1, le=10)  # Arbitrary but reasonable upper bound
    
    # Keyboard parameters (only for keyboard actions)
    key: Optional[str] = None
    
    # Scroll parameters (only for scroll actions)
    dx: Optional[int] = None  # Horizontal scroll units
    dy: Optional[int] = None  # Vertical scroll units
    
    # Wait parameters (only for wait actions)
    duration_ms: Optional[int] = Field(None, ge=0, le=300000)  # Max 5 minutes
    
    # Execution metadata (not interpreted)
    confidence: Optional[float] = Field(None, ge=0, le=1)  # From brain, not used for decisions
    
    @validator('action_type')
    def validate_required_fields_by_type(cls, v: ActionType, values: Dict[str, Any]) -> ActionType:
        """Validate that required fields are present for each action type."""
        errors = []
        
        if v in {ActionType.MOUSE_MOVE, ActionType.MOUSE_CLICK, ActionType.MOUSE_DOWN, ActionType.MOUSE_UP}:
            if values.get('x') is None:
                errors.append("Mouse actions require 'x' coordinate")
            if values.get('y') is None:
                errors.append("Mouse actions require 'y' coordinate")
        
        if v in {ActionType.MOUSE_CLICK, ActionType.MOUSE_DOWN, ActionType.MOUSE_UP}:
            if values.get('button') is None:
                errors.append(f"{v.value} actions require 'button'")
        
        if v == ActionType.MOUSE_CLICK:
            if values.get('clicks') is None:
                errors.append("Mouse click requires 'clicks' count")
        
        if v in {ActionType.KEY_PRESS, ActionType.KEY_DOWN, ActionType.KEY_UP}:
            if values.get('key') is None:
                errors.append(f"{v.value} actions require 'key'")
        
        if v == ActionType.SCROLL:
            if values.get('dx') is None and values.get('dy') is None:
                errors.append("Scroll requires at least one of 'dx' or 'dy'")
        
        if v == ActionType.WAIT:
            if values.get('duration_ms') is None:
                errors.append("Wait requires 'duration_ms'")
        
        if errors:
            raise ValueError(f"Action type {v.value} validation failed: {', '.join(errors)}")
        
        return v
    
    @validator('key')
    def validate_key_string(cls, v: Optional[str]) -> Optional[str]:
        """Ensure key is non-empty string if provided."""
        if v is not None and not v.strip():
            raise ValueError("Key must not be empty string")
        return v


class ExecutionResult(BaseModel):
    """
    Mechanical execution result only.
    
    Contains no interpretation of whether the action was "correct" or "successful"
    in a semantic sense. Only records what physically happened.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Identification
    action_id: str
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Timing (mechanical only)
    timestamp_start: datetime
    timestamp_end: datetime
    duration_ms: int = Field(..., ge=0)
    
    # Mechanical execution status
    executed: bool = Field(...)  # Whether the backend attempted execution
    backend_error: Optional[str] = None  # Raw error from backend, not interpreted
    
    # Backend-specific raw output (no interpretation)
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('duration_ms')
    def validate_duration(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate duration is non-negative and consistent with timestamps."""
        if v < 0:
            raise ValueError(f"Duration must be non-negative, got {v}ms")
        
        # Calculate actual duration from timestamps for consistency check
        if 'timestamp_start' in values and 'timestamp_end' in values:
            start = values['timestamp_start']
            end = values['timestamp_end']
            calculated_ms = int((end - start).total_seconds() * 1000)
            
            # Allow small rounding differences (up to 10ms)
            if abs(v - calculated_ms) > 10:
                logger.warning(
                    f"Duration mismatch: specified={v}ms, "
                    f"calculated={calculated_ms}ms from timestamps"
                )
        
        return v


class BodyBackend(ABC):
    """
    Abstract backend for OS control.
    
    Implementations (e.g., OpenCU, PyAutoGUI) must provide these concrete methods.
    No interpretation or decision-making in implementations.
    """
    
    @abstractmethod
    def move_mouse(self, x: int, y: int, coordinate_system: CoordinateSystem) -> Dict[str, Any]:
        """
        Move mouse to absolute coordinates or relative offset.
        
        Args:
            x: X coordinate or offset
            y: Y coordinate or offset
            coordinate_system: ABSOLUTE or RELATIVE
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def click_mouse(self, x: int, y: int, button: MouseButton, clicks: int) -> Dict[str, Any]:
        """
        Click mouse at position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Which mouse button
            clicks: Number of clicks
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def mouse_down(self, x: int, y: int, button: MouseButton) -> Dict[str, Any]:
        """
        Press mouse button down at position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Which mouse button
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def mouse_up(self, x: int, y: int, button: MouseButton) -> Dict[str, Any]:
        """
        Release mouse button at position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Which mouse button
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def press_key(self, key: str) -> Dict[str, Any]:
        """
        Press and release a key.
        
        Args:
            key: Key identifier (backend-specific)
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def key_down(self, key: str) -> Dict[str, Any]:
        """
        Press a key down (hold).
        
        Args:
            key: Key identifier (backend-specific)
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def key_up(self, key: str) -> Dict[str, Any]:
        """
        Release a key.
        
        Args:
            key: Key identifier (backend-specific)
        
        Returns:
            Raw backend output with no interpretation
        """
        pass
    
    @abstractmethod
    def scroll(self, dx: int, dy: int) -> Dict[str, Any]:
        """
        Scroll horizontally and/or vertically.
        
        Args:
            dx: Horizontal scroll units
            dy: Vertical scroll units
        
        Returns:
            Raw backend output with no interpretation
        """
        pass


class BodyInterface:
    """
    Stateless executor for concrete actions.
    
    This class:
    1. Validates action structure
    2. Executes exactly once via backend
    3. Captures mechanical execution results
    4. Logs for auditability
    
    This class does NOT:
    1. Decide what action to take
    2. Retry failed actions
    3. Optimize action sequences
    4. Infer success or failure
    5. Interpret intent or correctness
    6. Maintain state between calls
    """
    
    def __init__(self, backend: BodyBackend, execution_log_level: int = EXECUTION_LOG_LEVEL) -> None:
        """
        Initialize body interface with backend.
        
        Args:
            backend: Concrete backend implementation (e.g., OpenCU)
            execution_log_level: Logging level for execution audit
        """
        if not isinstance(backend, BodyBackend):
            raise TypeError(f"Backend must implement BodyBackend, got {type(backend)}")
        
        self._backend = backend
        self._logger = logging.getLogger(f"{__name__}.BodyInterface")
        self._execution_log_level = execution_log_level
    
    def execute_action(self, action: ConcreteAction) -> ExecutionResult:
        """
        Execute a concrete action exactly once.
        
        Args:
            action: Fully specified concrete action
        
        Returns:
            ExecutionResult with mechanical execution details only
        
        Raises:
            ValidationError: If action fails schema validation
            ValueError: If action cannot be executed due to structural issues
            RuntimeError: If backend execution fails (exact error preserved)
        
        Note:
            - No retry on failure
            - No interpretation of results
            - No state maintained between calls
            - Exact one-time execution only
        """
        # Log action for auditability
        self._log_action_received(action)
        
        # Start execution timing
        timestamp_start = datetime.utcnow()
        
        try:
            # Execute based on action type
            # Note: This branching is mechanical dispatch only, not decision-making
            raw_output = self._dispatch_to_backend(action)
            executed = True
            backend_error = None
            
        except Exception as e:
            # Capture backend error exactly, no interpretation
            raw_output = {}
            executed = False
            backend_error = f"{type(e).__name__}: {str(e)}"
            
            # Log execution failure (mechanical only)
            self._log_execution_failure(action, e)
            
            # Re-raise with original error context
            raise RuntimeError(f"Backend execution failed for action {action.action_id}: {str(e)}") from e
        
        finally:
            # End execution timing
            timestamp_end = datetime.utcnow()
        
        # Calculate duration
        duration_ms = int((timestamp_end - timestamp_start).total_seconds() * 1000)
        
        # Create execution result (mechanical facts only)
        result = ExecutionResult(
            action_id=action.action_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_ms=duration_ms,
            executed=executed,
            backend_error=backend_error,
            raw_output=raw_output
        )
        
        # Log execution result for auditability
        self._log_execution_result(action, result)
        
        return result
    
    def _dispatch_to_backend(self, action: ConcreteAction) -> Dict[str, Any]:
        """
        Dispatch action to backend based on type.
        
        Note: This is mechanical dispatch only, not decision-making.
        Each branch corresponds to a different physical operation.
        """
        if action.action_type == ActionType.MOUSE_MOVE:
            return self._backend.move_mouse(
                x=action.x,  # type: ignore (validated by schema)
                y=action.y,  # type: ignore
                coordinate_system=action.coordinate_system
            )
        
        elif action.action_type == ActionType.MOUSE_CLICK:
            return self._backend.click_mouse(
                x=action.x,  # type: ignore
                y=action.y,  # type: ignore
                button=action.button,  # type: ignore
                clicks=action.clicks  # type: ignore
            )
        
        elif action.action_type == ActionType.MOUSE_DOWN:
            return self._backend.mouse_down(
                x=action.x,  # type: ignore
                y=action.y,  # type: ignore
                button=action.button  # type: ignore
            )
        
        elif action.action_type == ActionType.MOUSE_UP:
            return self._backend.mouse_up(
                x=action.x,  # type: ignore
                y=action.y,  # type: ignore
                button=action.button  # type: ignore
            )
        
        elif action.action_type == ActionType.KEY_PRESS:
            return self._backend.press_key(
                key=action.key  # type: ignore
            )
        
        elif action.action_type == ActionType.KEY_DOWN:
            return self._backend.key_down(
                key=action.key  # type: ignore
            )
        
        elif action.action_type == ActionType.KEY_UP:
            return self._backend.key_up(
                key=action.key  # type: ignore
            )
        
        elif action.action_type == ActionType.SCROLL:
            return self._backend.scroll(
                dx=action.dx or 0,  # type: ignore
                dy=action.dy or 0   # type: ignore
            )
        
        elif action.action_type == ActionType.WAIT:
            # Wait is implemented here since it's backend-agnostic
            # But still treated as a mechanical operation
            time.sleep(action.duration_ms / 1000.0)  # type: ignore
            return {"wait_completed": True, "duration_ms": action.duration_ms}
        
        else:
            # This should never happen due to ActionType enum validation
            raise ValueError(f"Unsupported action type: {action.action_type}")
    
    def _log_action_received(self, action: ConcreteAction) -> None:
        """Log action receipt for auditability."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "action_received",
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "parameters": action.dict(exclude={"action_id", "action_type", "timestamp", "confidence"})
        }
        
        self._logger.log(self._execution_log_level, "Action received for execution", extra=log_data)
    
    def _log_execution_failure(self, action: ConcreteAction, error: Exception) -> None:
        """Log execution failure for auditability (mechanical facts only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "execution_failed",
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        self._logger.log(self._execution_log_level, "Action execution failed", extra=log_data)
    
    def _log_execution_result(self, action: ConcreteAction, result: ExecutionResult) -> None:
        """Log execution result for auditability (mechanical facts only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "execution_completed",
            "action_id": action.action_id,
            "execution_id": result.execution_id,
            "action_type": action.action_type.value,
            "executed": result.executed,
            "duration_ms": result.duration_ms,
            "backend_error": result.backend_error
        }
        
        self._logger.log(self._execution_log_level, "Action execution completed", extra=log_data)


# Export public interface
__all__ = [
    "BodyInterface",
    "BodyBackend",
    "ConcreteAction",
    "ExecutionResult",
    "ActionType",
    "MouseButton",
    "CoordinateSystem",
    "EXECUTION_LOG_LEVEL"
]