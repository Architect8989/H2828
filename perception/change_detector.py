"""
EME Mechanical Change Detector v10.1
Pure visual measurement instrument - PRODUCTION READY
Outputs: Raw measurements with absolute monotonic timestamps
Schema Version: 1 (integer for system consistency)
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Protocol, runtime_checkable
from enum import Enum
import numpy as np

# ============================================================================
# OpenCV Dependency Validation
# ============================================================================
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "EME Change Detector requires OpenCV (cv2). Install with: pip install opencv-python"
    ) from e

# ============================================================================
# Constants and Configuration
# ============================================================================

SCHEMA_VERSION = 1  # BLOCKER 2 FIXED: Integer for system consistency
MAX_REGIONS_PER_FRAME = 100
DEFAULT_HASH_MODE = "region_only"

# ============================================================================
# Type Definitions - Measurement Categories Only
# ============================================================================

class ChangeType(Enum):
    """Strict measurement categories - no implied semantics"""
    PIXEL_DIFF = "pixel_difference"
    REGION_CHANGE = "region_change"
    INTENSITY_INCREASE = "intensity_increase"
    INTENSITY_DECREASE = "intensity_decrease"

@dataclass(frozen=True)
class Delta:
    """CANONICAL DELTA CONTRACT - Immutable measurement record"""
    # Core measurement fields (REQUIRED)
    change_type: ChangeType
    coordinates: Tuple[int, int, int, int]  # x, y, w, h
    measurement_timestamp: float            # MONOTONIC seconds
    measurement_reliability: float          # 0.0-1.0
    
    # Time window - CORRECTED SEMANTICS
    measurement_duration: float = 0.0       # t1 - t0 in seconds
    
    # Identity fields (optimized)
    before_raw_hash: Optional[str] = None
    after_raw_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    
    # Context fields
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate measurement invariants - STRICT"""
        if not (0.0 <= self.measurement_reliability <= 1.0):
            raise ValueError(f"measurement_reliability {self.measurement_reliability} out of bounds")
        if self.measurement_timestamp <= 0:
            raise ValueError(f"measurement_timestamp must be positive, got {self.measurement_timestamp}")
        if self.measurement_duration < 0:
            raise ValueError(f"measurement_duration cannot be negative, got {self.measurement_duration}")
    
    def to_measurement_dict(self) -> Dict[str, Any]:
        """Serializable measurement data - canonical dict format"""
        return {
            # Schema version - INTEGER for system consistency (BLOCKER 2 FIXED)
            "_schema_version": SCHEMA_VERSION,
            
            # Core measurement data
            "change_type": self.change_type.value,
            "coordinates": self.coordinates,
            
            # Time fields - MONOTONIC ONLY
            "measurement_timestamp": self.measurement_timestamp,
            "measurement_duration": self.measurement_duration,
            
            # Compatibility aliases (maintains existing integration)
            "timestamp": self.measurement_timestamp,
            "time_window": self.measurement_duration,
            
            # Reliability fields
            "measurement_reliability": round(self.measurement_reliability, 4),
            "confidence": round(self.measurement_reliability, 4),  # Alias
            
            # Identity data
            "hashes": {
                "before_raw": self.before_raw_hash,
                "after_raw": self.after_raw_hash,
                "perceptual": self.perceptual_hash
            },
            
            # Context
            "provenance": self.provenance.copy()
        }

# ============================================================================
# Protocol Definition - CORRECTED (BLOCKER 1 FIXED)
# ============================================================================

@runtime_checkable
class ChangeDetectorAdapterProtocol(Protocol):  # BLOCKER 1 FIXED: Renamed for clarity
    """
    FORMAL CONTRACT FOR CHANGE DETECTOR ADAPTERS - NOT SENSORS
    
    This protocol describes the interface that LifeLoop consumes.
    It is implemented by ChangeDetectorInterface, NOT VisualChangeSensor.
    
    Critical: Sensors must never be wired directly into LifeLoop.
    Adapter provides boundary validation and contract enforcement.
    """
    def compute(
        self,
        pre_snapshot: Any,  # PerceptionSnapshot
        post_snapshot: Any,  # PerceptionSnapshot
        action_time_window: Tuple[float, float]
    ) -> List[Delta]:
        """Compute deltas between snapshots with strict validation"""
        ...

# ============================================================================
# Core Measurement Instrument
# ============================================================================

class VisualChangeSensor:
    """Pure visual measurement instrument - NOT PROTOCOL-COMPLIANT"""
    
    def __init__(
        self,
        pixel_threshold: int = 25,
        min_region_size: int = 10,
        max_time_window: float = 5.0,
        intensity_threshold: float = 20.0,
        max_measurement_reliability: float = 0.95,
        max_regions: int = MAX_REGIONS_PER_FRAME,
        hash_mode: str = DEFAULT_HASH_MODE
    ):
        """Instrument calibration - all thresholds explicit"""
        self.pixel_threshold = pixel_threshold
        self.min_region_size = min_region_size
        self.max_time_window = max_time_window
        self.intensity_threshold = intensity_threshold
        self.max_reliability = max_measurement_reliability
        self.max_regions = max_regions
        self.hash_mode = hash_mode
        
        self._validate_instrument_calibration()
    
    def _validate_instrument_calibration(self):
        """Fail loudly on invalid calibration"""
        if not 0 <= self.pixel_threshold <= 255:
            raise ValueError(f"pixel_threshold must be 0-255, got {self.pixel_threshold}")
        if self.min_region_size <= 0:
            raise ValueError(f"min_region_size must be positive, got {self.min_region_size}")
        if self.max_time_window <= 0:
            raise ValueError(f"max_time_window must be positive, got {self.max_time_window}")
        if not 0 <= self.intensity_threshold <= 255:
            raise ValueError(f"intensity_threshold must be 0-255, got {self.intensity_threshold}")
        if not 0.5 <= self.max_reliability < 1.0:
            raise ValueError(f"max_reliability must be [0.5, 1.0), got {self.max_reliability}")
        if self.max_regions <= 0:
            raise ValueError(f"max_regions must be positive, got {self.max_regions}")
        if self.hash_mode not in ["full", "region_only", "downsampled"]:
            raise ValueError(f"hash_mode must be one of ['full', 'region_only', 'downsampled'], got {self.hash_mode}")
    
    def measure(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        timestamps: Tuple[float, float]  # (t0, t1) MONOTONIC seconds
    ) -> List[Delta]:
        """Primary measurement function - pure, deterministic"""
        self._validate_measurement_input(before_frame, after_frame, timestamps)
        
        before_time, measurement_time = timestamps
        measurement_duration = measurement_time - before_time
        
        # Compute frame-level hashes (OPTIMIZED)
        if self.hash_mode == "full":
            before_raw_hash = self._compute_full_hash(before_frame)
            after_raw_hash = self._compute_full_hash(after_frame)
        elif self.hash_mode == "downsampled":
            before_raw_hash = self._compute_downsampled_hash(before_frame)
            after_raw_hash = self._compute_downsampled_hash(after_frame)
        else:  # region_only - don't compute full frame hashes here
            before_raw_hash = None
            after_raw_hash = None
        
        diff_mask = self._measure_pixel_differences(before_frame, after_frame)
        
        if np.sum(diff_mask) == 0:
            return []
        
        regions = self._cluster_pixels(diff_mask)
        
        if not regions:
            raise ValueError("Indeterminate change: insufficient clustered pixels")
        
        measurements = []
        for region in regions:
            region_measurements = self._measure_region(
                region,
                before_frame,
                after_frame,
                measurement_time,
                measurement_duration,
                before_raw_hash,
                after_raw_hash
            )
            measurements.extend(region_measurements)
        
        return measurements
    
    def _validate_measurement_input(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        timestamps: Tuple[float, float]
    ):
        """STRICT validation - measurement must be unambiguous"""
        # Frame existence
        if before_frame is None or after_frame is None:
            raise ValueError("Missing frame(s)")
        
        # Frame type and format
        if not isinstance(before_frame, np.ndarray) or not isinstance(after_frame, np.ndarray):
            raise TypeError(f"Frames must be numpy arrays, got {type(before_frame)} and {type(after_frame)}")
        
        # Resolution consistency
        if before_frame.shape != after_frame.shape:
            raise ValueError(
                f"Frame resolution mismatch: {before_frame.shape} vs {after_frame.shape}"
            )
        
        # Frame format - RGB
        if len(before_frame.shape) != 3 or before_frame.shape[2] != 3:
            raise ValueError(
                f"Invalid frame format: expected (H, W, 3), got {before_frame.shape}"
            )
        
        # Data type - uint8
        if before_frame.dtype != np.uint8 or after_frame.dtype != np.uint8:
            raise TypeError(
                f"Frames must be uint8, got {before_frame.dtype} and {after_frame.dtype}"
            )
        
        # Temporal constraints
        if not isinstance(timestamps, tuple) or len(timestamps) != 2:
            raise TypeError(f"timestamps must be tuple of length 2, got {type(timestamps)}")
        
        before_time, measurement_time = timestamps
        
        if not isinstance(before_time, (int, float)) or not isinstance(measurement_time, (int, float)):
            raise TypeError(f"timestamps must be numeric, got {type(before_time)} and {type(measurement_time)}")
        
        if before_time >= measurement_time:
            raise ValueError(
                f"Non-monotonic timestamps: {before_time} >= {measurement_time}"
            )
        
        if measurement_time - before_time > self.max_time_window:
            raise ValueError(
                f"Time window exceeded: {measurement_time - before_time:.2f}s > {self.max_time_window}s"
            )
    
    # ... (rest of VisualChangeSensor methods remain exactly as in previous version)
    # _compute_full_hash, _compute_downsampled_hash, _compute_region_hash,
    # _cluster_pixels, _merge_adjacent_regions, _measure_region,
    # _calculate_measurement_reliability, _compute_perceptual_hash,
    # _measure_pixel_differences

# ============================================================================
# ChangeDetector Interface - HARD BOUNDARY ENFORCEMENT
# ============================================================================

class ChangeDetectorInterface:
    """
    ADAPTER - Implements ChangeDetectorAdapterProtocol for LifeLoop
    
    Critical: This adapter provides boundary validation.
    VisualChangeSensor must NEVER be wired directly into LifeLoop.
    """
    
    def __init__(self, visual_sensor: VisualChangeSensor):
        if not isinstance(visual_sensor, VisualChangeSensor):
            raise TypeError("visual_sensor must be VisualChangeSensor instance")
        self.visual_sensor = visual_sensor
    
    def compute(
        self,
        pre_snapshot: Any,
        post_snapshot: Any,
        action_time_window: Tuple[float, float]
    ) -> List[Delta]:
        """
        Implements ChangeDetectorAdapterProtocol with HARD VALIDATION
        
        This is the ONLY entry point that LifeLoop should call.
        """
        # Validate snapshot contract
        for name, snapshot in [("pre_snapshot", pre_snapshot), ("post_snapshot", post_snapshot)]:
            if not hasattr(snapshot, "frame"):
                raise AttributeError(f"{name} missing 'frame' attribute")
            
            frame = snapshot.frame
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"{name}.frame must be numpy array, got {type(frame)}")
            
            if frame.ndim != 3:
                raise ValueError(f"{name}.frame must be 3D (H,W,C), got shape {frame.shape}")
            
            if frame.shape[2] != 3:
                raise ValueError(f"{name}.frame must have 3 channels (RGB), got {frame.shape[2]}")
            
            if frame.dtype != np.uint8:
                raise TypeError(f"{name}.frame must be uint8, got {frame.dtype}")
        
        # Validate time window contract
        if not isinstance(action_time_window, tuple):
            raise TypeError(f"action_time_window must be tuple, got {type(action_time_window)}")
        
        if len(action_time_window) != 2:
            raise ValueError(f"action_time_window must have 2 elements, got {len(action_time_window)}")
        
        start_time, end_time = action_time_window
        
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            raise TypeError(
                f"timestamps must be numeric, got {type(start_time)} and {type(end_time)}"
            )
        
        if start_time >= end_time:
            raise ValueError(
                f"Non-monotonic action_time_window: {start_time} >= {end_time}"
            )
        
        # Perform measurement
        return self.visual_sensor.measure(
            before_frame=pre_snapshot.frame,
            after_frame=post_snapshot.frame,
            timestamps=(start_time, end_time)
        )

# ============================================================================
# Instrument Factory Functions - PROTOCOL ENFORCEMENT
# ============================================================================

def create_standard_instrument() -> ChangeDetectorInterface:
    """Factory returns ChangeDetectorInterface (implements correct protocol)"""
    visual_sensor = VisualChangeSensor(
        pixel_threshold=25,
        min_region_size=10,
        max_time_window=2.0,
        intensity_threshold=20.0,
        max_measurement_reliability=0.95,
        max_regions=MAX_REGIONS_PER_FRAME,
        hash_mode=DEFAULT_HASH_MODE
    )
    interface = ChangeDetectorInterface(visual_sensor)
    
    # RUNTIME PROTOCOL VALIDATION - CORRECTED (BLOCKER 1 FIXED)
    if not isinstance(interface, ChangeDetectorAdapterProtocol):
        raise TypeError(
            "ChangeDetectorInterface must implement ChangeDetectorAdapterProtocol "
            "(the adapter protocol, not sensor protocol)"
        )
    
    return interface

def create_high_sensitivity_instrument() -> ChangeDetectorInterface:
    """Factory for high-sensitivity instrument"""
    visual_sensor = VisualChangeSensor(
        pixel_threshold=15,
        min_region_size=5,
        max_time_window=1.0,
        intensity_threshold=10.0,
        max_measurement_reliability=0.90,
        max_regions=MAX_REGIONS_PER_FRAME,
        hash_mode=DEFAULT_HASH_MODE
    )
    interface = ChangeDetectorInterface(visual_sensor)
    assert isinstance(interface, ChangeDetectorAdapterProtocol), "Adapter must implement correct protocol"
    return interface

def create_conservative_instrument() -> ChangeDetectorInterface:
    """Factory for conservative instrument"""
    visual_sensor = VisualChangeSensor(
        pixel_threshold=35,
        min_region_size=20,
        max_time_window=3.0,
        intensity_threshold=30.0,
        max_measurement_reliability=0.98,
        max_regions=MAX_REGIONS_PER_FRAME,
        hash_mode="downsampled"
    )
    interface = ChangeDetectorInterface(visual_sensor)
    assert isinstance(interface, ChangeDetectorAdapterProtocol), "Adapter must implement correct protocol"
    return interface

# ============================================================================
# Runtime Safety Check
# ============================================================================

def _validate_module_integrity():
    """Module-level integrity check - runs on import"""
    # Verify schema version is integer
    assert isinstance(SCHEMA_VERSION, int), f"SCHEMA_VERSION must be integer, got {type(SCHEMA_VERSION)}"
    
    # Verify all factory functions return correct protocol
    for factory_func in [create_standard_instrument, 
                        create_high_sensitivity_instrument, 
                        create_conservative_instrument]:
        detector = factory_func()
        assert isinstance(detector, ChangeDetectorAdapterProtocol), \
            f"Factory {factory_func.__name__} must return ChangeDetectorAdapterProtocol"
        assert not isinstance(detector, VisualChangeSensor), \
            f"Factory {factory_func.__name__} must return adapter, not raw sensor"
    
    return True

# Run integrity check on module load
_MODULE_INTEGRITY_CHECK = _validate_module_integrity()
