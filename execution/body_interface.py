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
- NO logging or side effects
- NO timestamp generation (timestamps must be provided externally)
- NO policy decisions

UUIDs are generated for mechanical traceability only; no semantic meaning.
WAIT is implemented as a mechanical timing operation, but introduces wall-clock coupling.
Execution timing must be managed externally.

This module is the muscle socket only - boring, auditable, and replaceable.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator, ConfigDict


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
    action_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Generated for mechanical traceability only; no semantic meaning"
    )
    action_type: ActionType
    timestamp: Optional[datetime] = None  # Must be provided by caller if needed
    
    # Position parameters (only for position-based actions)
    x: Optional[int] = Field(None, ge=0)
    y: Optional[int] = Field(None, ge=0)
    coordinate_system: CoordinateSystem = CoordinateSystem.ABSOLUTE
    
    # Mouse parameters (only for mouse actions)
    button: Optional[MouseButton] = None
    clicks: Optional[int] = Field(None, ge=1)  # Minimum 1, no upper bound
    
    # Keyboard parameters (only for keyboard actions)
    key: Optional[str] = None
    
    # Scroll parameters (only for scroll actions)
    dx: Optional[int] = None  # Horizontal scroll units
    dy: Optional[int] = None  # Vertical scroll units
    
    # Wait parameters (only for wait actions) - duration in milliseconds
    duration_ms: Optional[int] = Field(None, ge=0)  # Non-negative only
    
    # Execution metadata (not interpreted)
    confidence: Optional[float] = None  # Opaque metadata; numeric convertibility not validated
    
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


class ExecutionBackendError(RuntimeError):
    """Exception raised when backend execution fails."""
    pass


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
    execution_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Generated for mechanical traceability only; no semantic meaning"
    )
    
    # Mechanical execution status
    executed: bool = Field(...)  # Whether the backend attempted execution
    backend_error: Optional[str] = None  # Raw error from backend, not interpreted
    
    # Backend-specific raw output (no interpretation)
    raw_output: Dict[str, Any] = Field(default_factory=dict)


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
    
    @abstractmethod
    def wait(self, duration_ms: int) -> Dict[str, Any]:
        """
        Wait for specified duration.
        
        Note: This is a mechanical timing operation that introduces wall-clock coupling.
        
        Args:
            duration_ms: Duration to wait in milliseconds
        
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
    
    This class does NOT:
    1. Decide what action to take
    2. Retry failed actions
    3. Optimize action sequences
    4. Infer success or failure
    5. Interpret intent or correctness
    6. Maintain state between calls
    7. Log or produce side effects
    8. Generate timestamps or policy decisions
    9. Measure execution timing
    """
    
    def __init__(self, backend: BodyBackend) -> None:
        """
        Initialize body interface with backend.
        
        Args:
            backend: Concrete backend implementation (e.g., OpenCU)
        """
        if not isinstance(backend, BodyBackend):
            raise TypeError(f"Backend must implement BodyBackend, got {type(backend)}")
        
        self._backend = backend
    
    def execute_action(self, action: ConcreteAction) -> ExecutionResult:
        """
        Execute a concrete action exactly once.
        
        Args:
            action: Fully specified concrete action
        
        Returns:
            ExecutionResult with mechanical execution details only
        
        Raises:
            ValueError: If action cannot be executed due to structural issues
            ExecutionBackendError: If backend execution fails
        
        Note:
            - No retry on failure
            - No interpretation of results
            - No state maintained between calls
            - Exact one-time execution only
            - No logging or side effects
            - No timing measurements
        """
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
            
            # Raise typed backend error
            raise ExecutionBackendError(f"Backend execution failed: {str(e)}") from e
        
        # Create execution result (mechanical facts only)
        result = ExecutionResult(
            action_id=action.action_id,
            executed=executed,
            backend_error=backend_error,
            raw_output=raw_output
        )
        
        return result
    
    def _dispatch_to_backend(self, action: ConcreteAction) -> Dict[str, Any]:
        """
        Dispatch action to backend based on type.
        
        Note: This is mechanical dispatch only, not decision-making.
        Each branch corresponds to a different physical operation.
        WAIT is delegated to backend for consistency.
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
            return self._backend.wait(
                duration_ms=action.duration_ms  # type: ignore
            )
        
        else:
            # This should never happen due to ActionType enum validation
            raise ValueError(f"Unsupported action type: {action.action_type}")


# Export public interface
__all__ = [
    "BodyInterface",
    "BodyBackend",
    "ExecutionBackendError",
    "ConcreteAction",
    "ExecutionResult",
    "ActionType",
    "MouseButton",
    "CoordinateSystem"
]