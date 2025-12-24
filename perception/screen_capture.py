"""
Screen Capture - Mechanical Visual Sensor
Environment Mastery Engine (EME) v1.0

Pure mechanical sensor for capturing raw screen state as measurable facts.
One call = one capture. No intelligence, no interpretation, no state retention.

Schema Contract:
- frame: np.ndarray shape (H, W, 3) in RGB format (uint8, 0-255)
- timestamp: monotonic float seconds (time.perf_counter()) at capture instant
- metadata: dict containing only mechanical facts (no interpretations)

Color Semantics:
- Source format: BGRA (Blue, Green, Red, Alpha) from screen buffer
- Explicit conversion: BGRA → RGB via channel reordering (B→2, G→1, R→0, A→discarded)
- No color space transformations, gamma corrections, or enhancements

Time Semantics:
- Monotonic clock (time.perf_counter()) for physical time measurement
- Captured immediately after screen buffer acquisition
- No relation to wall-clock time or system time zones

Integration Guarantee:
Output schema is stable and directly consumable by downstream change detection.
No adapters required. Version: ScreenCapture-1.0
"""

import numpy as np
import mss
import time
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class PerceptionSnapshot:
    """
    Mechanical screen capture snapshot.
    Contains only measurable facts - no interpretations.
    
    Attributes:
        frame: np.ndarray shape (H, W, 3) in RGB format, dtype=np.uint8
        timestamp: float seconds from monotonic clock at capture instant
        metadata: dict containing explicit mechanical facts:
            - width: int (pixels)
            - height: int (pixels)  
            - color_format: str ("RGB")
            - monitor_id: int (0=primary)
            - dtype: str (numpy dtype)
            - shape: tuple (H, W, 3)
    """
    frame: np.ndarray  # shape (H, W, 3), dtype=uint8, RGB
    timestamp: float   # monotonic seconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # mechanical facts only, optional


class ScreenCapture:
    """
    Stateless mechanical screen sensor.
    
    Responsibilities:
    1. Capture exact screen pixels from configured monitor
    2. Perform explicit BGRA → RGB conversion (no ambiguity)
    3. Record monotonic timestamp at capture instant
    4. Validate frame integrity (shape, dtype, bounds)
    5. Expose monitor selection semantics via constructor
    
    Non-responsibilities:
    1. No semantic interpretation (OCR, UI parsing, object detection)
    2. No preprocessing (resizing, cropping, filtering, enhancement)
    3. No state retention (caching, history, side effects)
    4. No retries or fallbacks (failure raises exception)
    5. No task knowledge or policy logic
    """
    
    # Color semantics
    COLOR_FORMAT: str = "RGB"  # Explicit output format
    DTYPE: np.dtype = np.dtype('uint8')  # Explicit pixel type
    
    def __init__(self, monitor_id: int = 0) -> None:
        """
        Initialize mechanical sensor for specific monitor.
        
        Args:
            monitor_id: Physical monitor identifier (0=primary)
            
        Note:
            - Monitor selection is configuration, not runtime policy
            - No state, no side effects
            - MSS backend is battle-tested for Linux desktop capture
        """
        self._monitor_id = monitor_id
    
    def capture(self) -> PerceptionSnapshot:
        """
        Perform one mechanical screen capture.
        
        Returns:
            PerceptionSnapshot with frame (RGB), monotonic timestamp, and mechanical metadata
            
        Raises:
            ValueError: monitor_id invalid or out of range
            mss.exception.ScreenShotError: Mechanical capture failure
            RuntimeError: Frame integrity violation (shape, dtype, bounds)
            
        Note:
            - One call = one capture (no retries)
            - Explicit BGRA → RGB conversion
            - Monotonic timestamp at capture instant
            - Full validation of mechanical output
            - No state retention between calls
        """
        monitor_id = self._monitor_id
        
        # Validate monitor selection
        if monitor_id < 0:
            raise ValueError(f"monitor_id must be non-negative, got {monitor_id}")
        
        # Create new MSS instance (stateless)
        with mss.mss() as sct:
            # Mechanical: list available physical monitors
            # MSS indexes: 0=virtual all-in-one, 1=primary, 2+=secondary
            physical_monitor_idx = monitor_id + 1 if monitor_id >= 0 else 1
            
            # Validate monitor exists
            if physical_monitor_idx >= len(sct.monitors):
                raise ValueError(
                    f"monitor_id {monitor_id} out of range. "
                    f"Available monitor IDs: {self._get_available_monitors(sct)}"
                )
            
            # Mechanical capture
            screenshot = sct.grab(sct.monitors[physical_monitor_idx])
            
            # Timestamp immediately after buffer acquisition
            timestamp = time.perf_counter()
            
            # Extract raw BGRA pixel data
            # No preprocessing: raw BGRA bytes from screen buffer
            bgra_bytes = screenshot.raw
            height = screenshot.height
            width = screenshot.width
            
            # Explicit BGRA → RGB conversion (mechanical channel reordering)
            # Reshape to (H, W, 4), extract BGR channels, reverse to RGB
            try:
                # Convert bytes to BGRA array
                bgra_array = np.frombuffer(bgra_bytes, dtype=self.DTYPE)
                bgra_array = bgra_array.reshape((height, width, 4))
                
                # Extract RGB channels (discard Alpha)
                # BGRA indices: 0=B, 1=G, 2=R, 3=A
                # RGB order: R=2, G=1, B=0
                rgb_array = bgra_array[..., [2, 1, 0]]  # Explicit reordering
                
            except (ValueError, IndexError) as e:
                raise RuntimeError(f"BGRA→RGB conversion failed: {e}")
            
            # Validate mechanical output integrity
            self._validate_frame(rgb_array, width, height)
            
            # Construct mechanical metadata (facts only)
            metadata = {
                "width": width,
                "height": height,
                "color_format": self.COLOR_FORMAT,
                "monitor_id": monitor_id,
                "dtype": str(rgb_array.dtype),
                "shape": rgb_array.shape
            }
            
            return PerceptionSnapshot(
                frame=rgb_array,
                timestamp=timestamp,
                metadata=metadata
            )
    
    def _validate_frame(self, frame: np.ndarray, expected_width: int, expected_height: int) -> None:
        """
        Validate mechanical frame integrity.
        
        Args:
            frame: np.ndarray to validate
            expected_width: Mechanical screen width
            expected_height: Mechanical screen height
            
        Raises:
            RuntimeError: If any mechanical validation fails
        """
        # Check dtype
        if frame.dtype != self.DTYPE:
            raise RuntimeError(
                f"Frame dtype mismatch: expected {self.DTYPE}, got {frame.dtype}"
            )
        
        # Check shape
        expected_shape = (expected_height, expected_width, 3)
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"Frame shape mismatch: expected {expected_shape}, got {frame.shape}"
            )
        
        # Check value bounds (0-255 for uint8 RGB)
        if frame.min() < 0 or frame.max() > 255:
            raise RuntimeError(
                f"Frame values out of bounds [0, 255]: min={frame.min()}, max={frame.max()}"
            )
        
        # Check for NaN or Inf
        if not np.isfinite(frame).all():
            raise RuntimeError("Frame contains NaN or infinite values")
    
    def _get_available_monitors(self, sct: mss.mss) -> tuple:
        """
        List available physical monitors for given MSS instance.
        
        Internal use only - for validation.
        """
        # MSS monitor indices: 0=virtual combined, 1+=physical
        # Convert to our semantics: 0=primary, 1=secondary, etc.
        physical_monitors = len(sct.monitors) - 1
        return tuple(range(physical_monitors))


# Public contract
__all__ = ["ScreenCapture", "PerceptionSnapshot"]
