"""
Action Routing Layer

This module takes exactly one validated Action and invokes exactly one corresponding
OS-level actuation primitive through the backend.

One Action → one backend call. No transformation. No retries. No timing.
This module is a spinal cord: it transmits intent without understanding.

The only bridge between LifeLoop and the physical body.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple

from core.logger import log_event
from execution.backend import OSBackend, MouseButton, Key


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
    Validated action to execute.
    
    LifeLoop guarantees:
    - Exactly one action type
    - Parameters match the type
    - No invalid combinations
    - No compound actions
    
    This module assumes validation is complete.
    """
    type: ActionType
    parameters: Union[Tuple[int, int], MouseButton, Key]
    
    # [SUGGESTION] Add __str__ method for better logging
    # def __str__(self) -> str:
    #     if self.type == ActionType.MOVE:
    #         x, y = self.parameters
    #         return f"{self.type.name} ({x}, {y})"
    #     return f"{self.type.name} {self.parameters}"


class ActionExecutor:
    """
    Pure translator between Action and OSBackend.
    
    Routes each action type to exactly one backend call.
    Does not catch, retry, or interpret results.
    """
    
    def __init__(self, backend: OSBackend):
        """
        Initialize with backend interface.
        
        Args:
            backend: Concrete OSBackend implementation
        """
        self._backend = backend
    
    def execute(self, action: Action) -> None:
        """
        Route one Action to one backend primitive.
        
        Args:
            action: Validated action to execute
            
        Raises:
            ValueError: If action type is unsupported
            Any exception raised by the backend
        """
        # [SUGGESTION] Could improve parameter formatting for MOVE action
        # Current format: ACTION_ATTEMPT move (100, 100)
        # Alternative: ACTION_ATTEMPT MOVE x=100 y=100
        log_event(f"ACTION_ATTEMPT {action.type.value} {action.parameters}")
        
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
    
    # [SUGGESTION] Optional helper method for creating actions with validation
    # @staticmethod
    # def create_action(
    #     action_type: ActionType, 
    #     parameters: Union[Tuple[int, int], MouseButton, Key]
    # ) -> Action:
    #     """Create and validate an Action with appropriate parameter type checking."""
    #     if action_type == ActionType.MOVE:
    #         if not isinstance(parameters, tuple) or len(parameters) != 2:
    #             raise ValueError("MOVE action requires tuple (x, y)")
    #         x, y = parameters
    #         if not isinstance(x, int) or not isinstance(y, int):
    #             raise ValueError("MOVE coordinates must be integers")
    #     elif action_type in (ActionType.MOUSE_DOWN, ActionType.MOUSE_UP):
    #         if not isinstance(parameters, MouseButton):
    #             raise ValueError(f"{action_type} requires MouseButton parameter")
    #     elif action_type in (ActionType.KEY_DOWN, ActionType.KEY_UP):
    #         if not isinstance(parameters, Key):
    #             raise ValueError(f"{action_type} requires Key parameter")
    #     
    #     return Action(type=action_type, parameters=parameters)
