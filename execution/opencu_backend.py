"""
OpenCU Actuation Backend

This module implements the OS actuation contract using OpenCU, emitting exactly one
OS-level input event per method call, with no added behavior.

One call = one OpenCU primitive = one attempt.
No guarantees of success. No retries. No state. No timing.
If OpenCU raises → let it raise.

This file is a muscle: it has no nerves, no brain, no knowledge of why it moves.
"""

import opencu
from execution.backend import OSBackend, MouseButton, Key


class OpenCUBackend(OSBackend):
    """
    Concrete implementation of OSBackend using OpenCU primitives.
    
    Each method maps 1:1 to an OpenCU call with no transformation.
    No state. No retries. No logging. No guarantees.
    """
    
    def move_pointer(self, x: int, y: int) -> None:
        """
        Map to opencu.mouse_move(x, y, absolute=True).
        
        One call attempts one absolute movement. No bounds checking.
        """
        opencu.mouse_move(x, y, absolute=True)
    
    def mouse_button_down(self, button: MouseButton) -> None:
        """
        Map to opencu.mouse_down(button).
        
        One call attempts one press. No guarantee button was previously up.
        """
        opencu.mouse_down(button.value)
    
    def mouse_button_up(self, button: MouseButton) -> None:
        """
        Map to opencu.mouse_up(button).
        
        One call attempts one release. No guarantee button was previously down.
        """
        opencu.mouse_up(button.value)
    
    def key_down(self, key: Key) -> None:
        """
        Map to opencu.key_down(key).
        
        One call attempts one key press. No guarantee key was previously up.
        No character generation or modifier logic.
        """
        opencu.key_down(key.value)
    
    def key_up(self, key: Key) -> None:
        """
        Map to opencu.key_up(key).
        
        One call attempts one key release. No guarantee key was previously down.
        """
        opencu.key_up(key.value)
