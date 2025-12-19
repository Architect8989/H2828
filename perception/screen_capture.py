"""
Screen Capture

Mechanical sensor for capturing raw visual input from the live operating system.
This module provides pixel data from the current screen frame without interpretation.

Absolute constraints:
- NO vision models, OCR, UI parsing, or semantic interpretation
- NO preprocessing: no resizing, cropping, normalization, filtering, or enhancement
- NO timestamps, logging, retries, caching, state retention, or side effects
- NO task knowledge, heuristics, or policy logic

This module is a mechanical sensor only - boring, auditable, and replaceable.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import mss


@dataclass(frozen=True)
class ScreenFrame:
    """
    Raw screen frame data without interpretation.
    
    Contains only mechanical capture results.
    No semantic meaning, no preprocessing, no enhancement.
    """
    pixels: bytes  # Raw pixel data in BGRA format
    width: int  # Screen width in pixels
    height: int  # Screen height in pixels
    monitor_index: int = 0  # Which monitor was captured


class ScreenCapture:
    """
    Stateless screen capture interface for Linux desktops.
    
    This class:
    1. Captures raw screen pixels from the primary display
    2. Returns exact pixel data without modification
    3. Provides screen dimensions for spatial context
    
    This class does NOT:
    1. Interpret screen contents
    2. Preprocess or enhance images
    3. Cache or retain state between captures
    4. Log, timestamp, or produce side effects
    5. Retry failed captures
    6. Make policy decisions about what to capture
    """
    
    def __init__(self) -> None:
        """
        Initialize screen capture interface.
        
        Note: No configuration, no state retention, no side effects.
        MSS backend is battle-tested and suitable for Linux desktops.
        """
        # MSS instance is created per-capture to ensure statelessness
        pass
    
    def capture(self, monitor_index: int = 0) -> ScreenFrame:
        """
        Capture current screen frame from specified monitor.
        
        Args:
            monitor_index: Monitor to capture (0 = primary)
        
        Returns:
            ScreenFrame with raw pixel data and dimensions
        
        Raises:
            mss.exception.ScreenShotError: If capture fails
            ValueError: If monitor_index is invalid
        
        Note:
            - No retry on failure
            - No preprocessing of captured data
            - No state maintained between calls
            - Exact one-time capture only
            - Pixel format is BGRA (Blue, Green, Red, Alpha)
        """
        if monitor_index < 0:
            raise ValueError(f"monitor_index must be non-negative, got {monitor_index}")
        
        # Create new MSS instance for each capture (stateless)
        with mss.mss() as sct:
            # Validate monitor exists
            if monitor_index >= len(sct.monitors):
                raise ValueError(
                    f"monitor_index {monitor_index} out of range. "
                    f"Available monitors: {len(sct.monitors) - 1}"
                )
            
            # Capture raw screen data
            # No preprocessing: grab() returns exact screen contents
            screenshot = sct.grab(sct.monitors[monitor_index])
            
            # Extract raw pixel data and dimensions
            # Note: screenshot.raw is BGRA format, exactly as captured
            frame = ScreenFrame(
                pixels=screenshot.raw,
                width=screenshot.width,
                height=screenshot.height,
                monitor_index=monitor_index
            )
            
            return frame
    
    def get_available_monitors(self) -> Tuple[int, ...]:
        """
        List available monitors without capturing.
        
        Returns:
            Tuple of monitor indices (0 = primary)
        
        Note:
            - No state retained
            - No side effects
            - Exact snapshot only
        """
        with mss.mss() as sct:
            # Return monitor count (excluding the all-in-one monitor at index 0 in MSS)
            # MSS stores monitors as: [{'left': 0, 'top': 0, 'width': total_width, 'height': total_height}, monitor1, monitor2, ...]
            # So available monitors are indices 1..N
            monitor_count = len(sct.monitors) - 1
            return tuple(range(1, monitor_count + 1)) if monitor_count > 0 else (0,)


# Export public interface
__all__ = ["ScreenCapture", "ScreenFrame"]