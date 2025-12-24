"""Action Routing Layer

This module takes exactly one validated Action and invokes exactly one corresponding
OS-level actuation primitive through the backend.

One Action → one backend call. No transformation. No retries. No timing.
This module is a spinal cord: it transmits intent without understanding.

The only bridge between LifeLoop and the physical body.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, Final

from execution.backend import OSBackend, MouseButton, Key


# SCHEMA VERSIONING - Include in fingerprint for future-proofing
SCHEMA_VERSION: Final[int] = 1


class ActionType(Enum):
    """Supported action types. One-to-one mapping to backend primitives."""
    MOVE = "move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"


@dataclass(frozen=True)
class Action:
    """
    Canonical Action contract for Environment Mastery Engine.
    
    Immutable, deterministic, and cryptographically identified.
    """
    id: str
    type: ActionType
    parameters: Union[Tuple[int, int], MouseButton, Key]
    fingerprint: str
    _internal: bool = False  # Factory-only construction guard
    
    def __post_init__(self) -> None:
        """Enforce factory-only construction and validate invariants."""
        if not self._internal:
            raise RuntimeError("Action must be created via Action.create() factory method")
        
        if not self.fingerprint:
            raise ValueError("Action must have fingerprint")
    
    def __eq__(self, other: object) -> bool:
        """Identity equality only - prevent semantic equality bypass."""
        return self is other
    
    def __repr__(self) -> str:
        """Deterministic string representation for logs and ledgers."""
        params = self._canonical_parameters()
        return f"Action(id={self.id!r}, type={self.type}, parameters={params}, fingerprint={self.fingerprint[:8]}...)"
    
    def validate(self) -> None:
        """Verify Action invariants (read-only)."""
        if not self.fingerprint:
            raise ValueError("Invalid Action: missing fingerprint")
        
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Invalid Action: id must be non-empty string")
        
        if not isinstance(self.type, ActionType):
            raise ValueError("Invalid Action: type must be ActionType enum")
    
    @classmethod
    def create(cls, id: str, type: ActionType, parameters: Union[Tuple[int, int], MouseButton, Key]) -> "Action":
        """
        Canonical factory method for creating Actions.
        
        Args:
            id: Stable human-readable identifier
            type: Action type enum
            parameters: Backend-ready primitives matching the action type
            
        Returns:
            Immutable Action with computed cryptographic fingerprint
            
        Raises:
            ValueError: If parameters do not match ActionType
        """
        if not id or not isinstance(id, str):
            raise ValueError("Action.id must be non-empty string")
        
        if not isinstance(type, ActionType):
            raise ValueError("Action.type must be ActionType enum")
        
        cls._validate_parameters(type, parameters)
        
        # Compute deterministic SHA-256 fingerprint
        canonical_params = cls._canonicalize_parameters(type, parameters)
        data = f"{SCHEMA_VERSION}|{id}|{type.value}|{canonical_params}"
        fingerprint = hashlib.sha256(data.encode()).hexdigest()
        
        return cls(
            id=id,
            type=type,
            parameters=parameters,
            fingerprint=fingerprint,
            _internal=True
        )
    
    @staticmethod
    def _validate_parameters(type: ActionType, parameters: Union[Tuple[int, int], MouseButton, Key]) -> None:
        """Fail fast if parameters do not match ActionType."""
        if type == ActionType.MOVE:
            if not isinstance(parameters, tuple) or len(parameters) != 2:
                raise ValueError("MOVE action requires tuple (x, y)")
            x, y = parameters
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("MOVE coordinates must be integers")
        elif type in (ActionType.MOUSE_DOWN, ActionType.MOUSE_UP):
            if type(parameters) is not MouseButton:
                raise ValueError(f"{type} requires exact MouseButton enum")
        elif type in (ActionType.KEY_DOWN, ActionType.KEY_UP):
            if type(parameters) is not Key:
                raise ValueError(f"{type} requires exact Key enum")
        else:
            raise ValueError(f"Unsupported action type: {type}")
    
    @staticmethod
    def _canonicalize_parameters(type: ActionType, parameters: Union[Tuple[int, int], MouseButton, Key]) -> str:
        """Return deterministic string representation of parameters."""
        if type == ActionType.MOVE:
            x, y = parameters
            return f"move_{x}_{y}"
        elif type in (ActionType.MOUSE_DOWN, ActionType.MOUSE_UP):
            # MouseButton enum canonical mapping
            mouse_map = {
                MouseButton.LEFT: "left",
                MouseButton.RIGHT: "right",
                MouseButton.MIDDLE: "middle",
            }
            return mouse_map.get(parameters, "unknown")
        else:  # KEY_DOWN or KEY_UP
            # Key enum canonical mapping (extend as needed)
            key_map = {
                Key.ENTER: "enter",
                Key.ESCAPE: "escape",
                Key.SPACE: "space",
                Key.BACKSPACE: "backspace",
                Key.TAB: "tab",
                Key.SHIFT: "shift",
                Key.CONTROL: "control",
                Key.ALT: "alt",
            }
            return key_map.get(parameters, "unknown")
    
    def _canonical_parameters(self) -> str:
        """Helper method for __repr__."""
        return self._canonicalize_parameters(self.type, self.parameters)


class ActionExecutor:
    """
    Pure translator between Action and OSBackend.
    
    Routes each action type to exactly one backend call.
    Does not catch, retry, or interpret results.
    """
    
    # Required backend interface methods
    _REQUIRED_METHODS = (
        "move_pointer",
        "mouse_button_down", 
        "mouse_button_up",
        "key_down",
        "key_up",
    )
    
    def __init__(self, backend: OSBackend):
        """
        Initialize with backend interface.
        
        Args:
            backend: Concrete OSBackend implementation
            
        Raises:
            ValueError: If backend does not implement required methods
        """
        self._backend = backend
        self._assert_backend_contract()
    
    def _assert_backend_contract(self) -> None:
        """Fail fast if backend does not implement required methods."""
        for method in self._REQUIRED_METHODS:
            if not hasattr(self._backend, method):
                raise ValueError(f"Backend missing required method: {method}")
            if not callable(getattr(self._backend, method)):
                raise ValueError(f"Backend attribute {method} is not callable")
    
    def execute(self, action: Action) -> None:
        """
        Route one Action to one backend primitive.
        
        Args:
            action: Validated action to execute
            
        Raises:
            ValueError: If action is invalid
            Any exception raised by the backend
        """
        action.validate()  # Final invariant check before execution
        
        if action.type == ActionType.MOVE:
            self._execute_move(action.parameters)
        elif action.type == ActionType.MOUSE_DOWN:
            self._execute_mouse_down(action.parameters)
        elif action.type == ActionType.MOUSE_UP:
            self._execute_mouse_up(action.parameters)
        elif action.type == ActionType.KEY_DOWN:
            self._execute_key_down(action.parameters)
        elif action.type == ActionType.KEY_UP:
            self._execute_key_up(action.parameters)
        else:
            raise ValueError(f"Unsupported action type: {action.type}")
    
    def _execute_move(self, parameters: Tuple[int, int]) -> None:
        """Route move action to absolute pointer movement."""
        x, y = parameters
        self._backend.move_pointer(x, y)
    
    def _execute_mouse_down(self, parameters: MouseButton) -> None:
        """Route mouse_down action to button press."""
        self._backend.mouse_button_down(parameters)
    
    def _execute_mouse_up(self, parameters: MouseButton) -> None:
        """Route mouse_up action to button release."""
        self._backend.mouse_button_up(parameters)
    
    def _execute_key_down(self, parameters: Key) -> None:
        """Route key_down action to key press."""
        self._backend.key_down(parameters)
    
    def _execute_key_up(self, parameters: Key) -> None:
        """Route key_up action to key release."""
        self._backend.key_up(parameters)
