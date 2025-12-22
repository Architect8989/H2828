"""
OS ACTUATION CONTRACT

This module defines the minimal, atomic, OS-level input capabilities required
for human-equivalent interaction, without policy, logic, timing, or intelligence.

One call = one OS-level input event. No guarantees of success. No assumptions
about outcome. No hidden side effects. No retries. No timing. No smoothing.
If the OS ignores the event, that is reality.

Architecture:
  LifeLoop → Action adapters → OSBackend (this contract) → Concrete backend → OS

This file is physics, not biology. It defines what motions exist, not how to use them.
Equivalent to: muscle actuation limits, joint movement primitives, nerve output signals.

THIS FILE IS ARCHITECTURALLY FROZEN.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from enum import Enum, IntEnum


class MouseButton(IntEnum):
    """Physical mouse buttons available for actuation."""
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    # Note: Side buttons, wheel clicks, etc. are physical variations
    # that may not exist on all hardware. Higher layers handle this.


class Key(str, Enum):
    """
    Physical keyboard keys available for actuation.
    
    Representation matches physical layout, not character generation.
    Shift+A produces 'A' but requires two physical keys.
    """
    # Alphanumeric
    A = 'a'  # Physical key, not character
    B = 'b'
    C = 'c'
    D = 'd'
    E = 'e'
    F = 'f'
    G = 'g'
    H = 'h'
    I = 'i'
    J = 'j'
    K = 'k'
    L = 'l'
    M = 'm'
    N = 'n'
    O = 'o'
    P = 'p'
    Q = 'q'
    R = 'r'
    S = 's'
    T = 't'
    U = 'u'
    V = 'v'
    W = 'w'
    X = 'x'
    Y = 'y'
    Z = 'z'
    
    ZERO = '0'
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    
    # Modifiers (physical position keys)
    SHIFT_LEFT = 'shift_left'
    SHIFT_RIGHT = 'shift_right'
    CTRL_LEFT = 'ctrl_left'
    CTRL_RIGHT = 'ctrl_right'
    ALT_LEFT = 'alt_left'
    ALT_RIGHT = 'alt_right'
    META_LEFT = 'meta_left'  # Windows/Command
    META_RIGHT = 'meta_right'
    
    # Navigation
    ENTER = 'enter'
    ESCAPE = 'escape'
    BACKSPACE = 'backspace'
    TAB = 'tab'
    SPACE = 'space'
    
    # Arrow keys
    ARROW_UP = 'arrow_up'
    ARROW_DOWN = 'arrow_down'
    ARROW_LEFT = 'arrow_left'
    ARROW_RIGHT = 'arrow_right'
    
    # Function keys
    F1 = 'f1'
    F2 = 'f2'
    F3 = 'f3'
    F4 = 'f4'
    F5 = 'f5'
    F6 = 'f6'
    F7 = 'f7'
    F8 = 'f8'
    F9 = 'f9'
    F10 = 'f10'
    F11 = 'f11'
    F12 = 'f12'
    
    # Special
    CAPS_LOCK = 'caps_lock'
    NUM_LOCK = 'num_lock'
    SCROLL_LOCK = 'scroll_lock'
    
    PRINT_SCREEN = 'print_screen'
    PAUSE = 'pause'
    INSERT = 'insert'
    DELETE = 'delete'
    HOME = 'home'
    END = 'end'
    PAGE_UP = 'page_up'
    PAGE_DOWN = 'page_down'
    
    # Numpad (physical separate keys)
    NUMPAD_0 = 'numpad_0'
    NUMPAD_1 = 'numpad_1'
    NUMPAD_2 = 'numpad_2'
    NUMPAD_3 = 'numpad_3'
    NUMPAD_4 = 'numpad_4'
    NUMPAD_5 = 'numpad_5'
    NUMPAD_6 = 'numpad_6'
    NUMPAD_7 = 'numpad_7'
    NUMPAD_8 = 'numpad_8'
    NUMPAD_9 = 'numpad_9'
    NUMPAD_ADD = 'numpad_add'
    NUMPAD_SUBTRACT = 'numpad_subtract'
    NUMPAD_MULTIPLY = 'numpad_multiply'
    NUMPAD_DIVIDE = 'numpad_divide'
    NUMPAD_DECIMAL = 'numpad_decimal'
    NUMPAD_ENTER = 'numpad_enter'
    
    # Punctuation (physical keys, not characters)
    COMMA = 'comma'
    PERIOD = 'period'
    SLASH = 'slash'
    SEMICOLON = 'semicolon'
    APOSTROPHE = 'apostrophe'
    BRACKET_LEFT = 'bracket_left'
    BRACKET_RIGHT = 'bracket_right'
    BACKSLASH = 'backslash'
    GRAVE = 'grave'  # Backtick/~
    MINUS = 'minus'
    EQUAL = 'equal'


class OSBackend(ABC):
    """
    Abstract interface defining OS input capabilities.
    
    GUARANTEES:
    - Each method represents one physical input primitive
    - One call attempts to produce one OS-level input event
    - Methods may raise exceptions if execution is impossible
    
    NO GUARANTEES:
    - Success or failure of the event
    - Timing or delay between call and effect
    - System state before or after
    - Event ordering or queuing
    - Availability of specific hardware features
    
    This is capability definition, not behavior prescription.
    """
    
    @abstractmethod
    def move_pointer(self, x: int, y: int) -> None:
        """
        Attempt absolute pointer movement to screen coordinates.
        
        Args:
            x: Horizontal coordinate in screen pixels
            y: Vertical coordinate in screen pixels
            
        Coordinates are absolute, not relative. The origin (0,0) is OS-defined
        (typically top-left). No bounds checking. No interpolation.
        
        Raises:
            Whatever the concrete backend raises if movement is impossible.
        """
        pass
    
    @abstractmethod
    def mouse_button_down(self, button: MouseButton) -> None:
        """
        Attempt to press a physical mouse button.
        
        Args:
            button: Physical button to press
            
        One call = one press attempt. No guarantee button was previously up.
        No double-click detection. No hold timing.
        
        Raises:
            Whatever the concrete backend raises if press is impossible.
        """
        pass
    
    @abstractmethod
    def mouse_button_up(self, button: MouseButton) -> None:
        """
        Attempt to release a physical mouse button.
        
        Args:
            button: Physical button to release
            
        One call = one release attempt. No guarantee button was previously down.
        
        Raises:
            Whatever the concrete backend raises if release is impossible.
        """
        pass
    
    @abstractmethod
    def key_down(self, key: Key) -> None:
        """
        Attempt to press a physical keyboard key.
        
        Args:
            key: Physical key to press
            
        One call = one key press attempt. No guarantee key was previously up.
        No character generation. No modifier combination logic.
        
        Raises:
            Whatever the concrete backend raises if press is impossible.
        """
        pass
    
    @abstractmethod
    def key_up(self, key: Key) -> None:
        """
        Attempt to release a physical keyboard key.
        
        Args:
            key: Physical key to release
            
        One call = one key release attempt. No guarantee key was previously down.
        
        Raises:
            Whatever the concrete backend raises if release is impossible.
        """
        pass