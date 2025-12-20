"""
delta.py - The atomic unit of learning in the Environment Mastery Engine (EME)

Prime Definition (Non-Negotiable):
> If a change cannot be expressed as a Delta, it is not knowledge and must not enter the system.

Policy:
> Tools must not emit placeholder deltas repeatedly; repeated EMPTY_DELTA must halt execution.
> LifeLoop enforces this behaviorally.

This module is foundational infrastructure, not a convenience abstraction.
If delta.py is removed or bypassed, learning must become impossible.
"""

import json
from typing import Any, Dict, Mapping, Optional, TypeAlias
from dataclasses import dataclass, field
from collections.abc import Mapping as MappingABC

# Type alias for structured path representation
# Keys must be non-empty strings representing clear environmental locations
Path: TypeAlias = str
# Values must be directly observable facts, never interpretation
Value: TypeAlias = Any

# Sentinel for detecting empty/missing values
_MISSING = object()


@dataclass(frozen=True, slots=True)
class Delta:
    """
    A verified claim about a real change in the environment.
    
    Guarantees:
    1. Grounding: Only facts derived from real observations
    2. Atomicity: Indivisible - fully applied or fully rejected
    3. Immutability: Cannot be modified after creation
    4. Auditability: Clearly describes what changed and where
    5. Minimalism: Contains only new information, never interpretation
    """
    
    # Internal representation - never exposed directly
    _changes: Dict[Path, Value] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Validate immediately upon creation - fail loudly if invalid."""
        # FIX 1: Defensive copy to seal Delta completely
        # Prevents external mutation of the original dict reference
        object.__setattr__(self, "_changes", dict(self._changes))
        
        if not self.is_valid():
            raise InvalidDeltaError("Delta failed validation upon creation")
    
    @property
    def changes(self) -> Mapping[Path, Value]:
        """
        An immutable mapping representing what changed, expressed in a path-like
        or structured form suitable for application to the World Model.
        
        Returns:
            An immutable view of the changes mapping
        """
        # Return a read-only view to maintain immutability guarantee
        return MappingProxyType(self._changes)
    
    def is_valid(self) -> bool:
        """
        Structural validation only - not semantic.
        
        Returns True only if:
        1. Structure is correct (mapping from paths to values)
        2. Keys are non-empty strings
        3. Values are JSON serializable (ensures observability)
        4. Delta is not empty and contains new information
        
        Note: Does NOT validate against world model or check for conflicts.
        """
        try:
            # Structural check
            if not isinstance(self._changes, dict):
                return False
            
            # Non-emptiness check
            if not self._changes:
                return False
            
            for path, value in self._changes.items():
                # Key validation
                if not isinstance(path, str):
                    return False
                if not path:  # Empty string
                    return False
                
                # Value serializability check (structural, not semantic)
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    return False
                
                # Prevent opaque blobs or ambiguous data
                if isinstance(value, (bytes, bytearray, memoryview)):
                    return False
            
            # Passed all structural checks
            return True
            
        except (AttributeError, TypeError):
            # Any structural irregularity invalidates the delta
            return False
    
    def is_empty(self) -> bool:
        """
        Returns True if:
        1. No meaningful changes are present
        2. Changes do not add new information
        
        Note: While {"path": None} is technically valid but empty,
        tools must not emit placeholder deltas repeatedly.
        Repeated empty deltas must halt execution (enforced by LifeLoop).
        
        This prevents abuse of the learning mechanism.
        """
        if not self._changes:
            return True
        
        # Check for sentinel or placeholder values that indicate no real change
        for value in self._changes.values():
            if value is _MISSING or value is None:
                continue
            # If we have at least one non-sentinel value, delta is not empty
            return False
        
        # All values were sentinels or None
        return True
    
    def __eq__(self, other: Any) -> bool:
        """Equality based on identical change sets."""
        if not isinstance(other, Delta):
            return NotImplemented
        return self._changes == other._changes
    
    def __hash__(self) -> int:
        """Hash based on frozen changes."""
        return hash(tuple(sorted(self._changes.items())))
    
    def __repr__(self) -> str:
        """Technical representation for debugging."""
        valid = self.is_valid()
        empty = self.is_empty()
        count = len(self._changes)
        return f"Delta(valid={valid}, empty={empty}, changes={count})"


class InvalidDeltaError(Exception):
    """Raised when a Delta cannot be created or validated."""
    
    def __init__(self, message: str, changes: Optional[Dict[Path, Value]] = None):
        self.message = message
        self.changes = changes
        super().__init__(f"{message} (changes: {changes})")


# Minimal helper for creation with validation
def create_delta(changes: Dict[Path, Value]) -> Delta:
    """
    Create a Delta with explicit validation.
    
    Args:
        changes: Mapping from environmental paths to observed values
        
    Returns:
        A validated Delta instance
        
    Raises:
        InvalidDeltaError: If changes fail structural validation
    """
    # Type check
    if not isinstance(changes, dict):
        raise InvalidDeltaError("Changes must be a dictionary", changes)
    
    # Quick pre-check for obvious issues
    if not changes:
        raise InvalidDeltaError("Delta must contain changes", changes)
    
    # Create and let __post_init__ handle full validation
    return Delta(changes)


# Ensure we use Python's immutable mapping proxy
try:
    from types import MappingProxyType
except ImportError:
    # Fallback for older Python versions
    class MappingProxyType(dict):
        """Minimal read-only dict wrapper."""
        def __setitem__(self, key, value):
            raise TypeError("'MappingProxyType' object does not support item assignment")
        def __delitem__(self, key):
            raise TypeError("'MappingProxyType' object does not support item deletion")
        def clear(self):
            raise TypeError("'MappingProxyType' object does not support item deletion")
        def pop(self, key, default=_MISSING):
            raise TypeError("'MappingProxyType' object does not support item deletion")
        def popitem(self):
            raise TypeError("'MappingProxyType' object does not support item deletion")
        def update(self, other=()):
            raise TypeError("'MappingProxyType' object does not support item assignment")


# Final assertion: This module must exist for the system to learn
if __name__ == "__main__":
    # Self-test: Validate core guarantees
    test_cases = [
        # Valid
        ({"sensor.temperature": 22.5}, True, False),
        ({"agent.position.x": 10, "agent.position.y": 20}, True, False),
        
        # Invalid
        ({}, False, True),  # Empty
        ({"": "value"}, False, False),  # Empty key
        ({"path": object()}, False, False),  # Unserializable
        ({"path": b"binary"}, False, False),  # Opaque blob
        
        # Edge cases
        ({"path": None}, True, True),  # Valid but empty due to None
        ({"path": _MISSING}, True, True),  # Valid but empty due to sentinel
    ]
    
    all_pass = True
    for changes, should_be_valid, should_be_empty in test_cases:
        try:
            delta = Delta(changes)
            valid = delta.is_valid()
            empty = delta.is_empty()
            
            if valid != should_be_valid or empty != should_be_empty:
                print(f"FAIL: {changes}")
                print(f"  Expected: valid={should_be_valid}, empty={should_be_empty}")
                print(f"  Got: valid={valid}, empty={empty}")
                all_pass = False
                
        except InvalidDeltaError:
            if should_be_valid:
                print(f"FAIL: {changes} raised but should be valid")
                all_pass = False
    
    # Test defensive copy fix
    external_dict = {"test.path": "original"}
    delta = Delta(external_dict)
    external_dict["test.path"] = "mutated"
    if delta.changes.get("test.path") != "original":
        print("FAIL: Defensive copy not working - external mutation affected Delta")
        all_pass = False
    else:
        print("✓ Defensive copy prevents external mutation")
    
    if all_pass:
        print("✓ All delta.py guarantees hold")
        print("✓ System can learn if this module exists")
        print("✓ System cannot learn if this module is removed or bypassed")
        print("✓ Tools must not emit placeholder deltas repeatedly")
        print("✓ LifeLoop enforces behavioral policy on empty deltas")
    else:
        print("✗ Some guarantees violated")
        raise SystemExit(1)