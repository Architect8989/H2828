"""
EME Mechanical Change Detector v9.8
Pure visual measurement instrument
Outputs: Raw measurements only, calibrated thresholds, no hidden heuristics
"""

import hashlib
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import time

# ============================================================================
# Type Definitions - Measurement Categories Only
# ============================================================================

class ChangeType(Enum):
    """Strict measurement categories - no implied semantics"""
    PIXEL_DIFF = "pixel_difference"          # Reference-only, full-frame comparison
    REGION_CHANGE = "region_change"          # Measured region difference
    INTENSITY_INCREASE = "intensity_increase" # Luminance increase > threshold
    INTENSITY_DECREASE = "intensity_decrease" # Luminance decrease > threshold

@dataclass(frozen=True)
class Delta:
    """Immutable measurement record - pure data, no interpretation"""
    change_type: ChangeType
    coordinates: Tuple[int, int, int, int]  # x, y, w, h (pixel units)
    before_raw_hash: Optional[str]  # Exact SHA256 hash (16 chars)
    after_raw_hash: Optional[str]   # Exact SHA256 hash (16 chars)
    perceptual_hash: Optional[str] = None  # Noise-tolerant (low reliability)
    time_window: float = 0.0                # Measurement duration (seconds)
    measurement_reliability: float = 0.0    # 0.0-1.0, sensor quality only
    provenance: Dict[str, Any] = None       # Raw measurement context
    
    def __post_init__(self):
        """Validate measurement invariants"""
        if not (0.0 <= self.measurement_reliability <= 1.0):
            raise ValueError(f"measurement_reliability {self.measurement_reliability} out of bounds")
        if self.provenance is None:
            object.__setattr__(self, 'provenance', {})
    
    def to_measurement_dict(self) -> Dict[str, Any]:
        """Serializable measurement data - no interpretation"""
        return {
            "measurement_type": self.change_type.value,
            "coordinates": self.coordinates,
            "hashes": {
                "before_raw": self.before_raw_hash,
                "after_raw": self.after_raw_hash,
                "perceptual": self.perceptual_hash
            },
            "time_window_ms": int(self.time_window * 1000),
            "measurement_reliability": round(self.measurement_reliability, 4),
            "provenance": self.provenance.copy()
        }

# ============================================================================
# Core Measurement Instrument
# ============================================================================

class VisualChangeSensor:
    """
    Pure visual measurement instrument
    All thresholds explicit, all measurements calibrated
    """
    
    def __init__(self,
                 pixel_threshold: int = 25,
                 min_region_size: int = 10,
                 max_time_window: float = 5.0,
                 intensity_threshold: float = 20.0,
                 max_measurement_reliability: float = 0.95):
        """
        Instrument calibration - all thresholds explicit
        
        Args:
            pixel_threshold: RGB difference to register change (0-255)
            min_region_size: Minimum pixel cluster to report (pixels)
            max_time_window: Maximum allowed time between measurements (seconds)
            intensity_threshold: Luminance change to report intensity delta (0-255)
            max_measurement_reliability: Maximum reliability claim (<1.0)
        """
        self.pixel_threshold = pixel_threshold
        self.min_region_size = min_region_size
        self.max_time_window = max_time_window
        self.intensity_threshold = intensity_threshold
        self.max_reliability = max_measurement_reliability
        
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
    
    def measure(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        timestamps: Tuple[float, float]
    ) -> List[Delta]:
        """
        Primary measurement function - pure, deterministic
        
        Args:
            before_frame: RGB array (H, W, 3) - measurement t0
            after_frame: RGB array (H, W, 3) - measurement t1
            timestamps: (t0, t1) in seconds, monotonic
            
        Returns:
            List of Delta measurement records
            
        Raises:
            ValueError: On any measurement ambiguity or violation
        """
        # --------------------------------------------------------------------
        # 1. Input Validation - Fail Loud
        # --------------------------------------------------------------------
        self._validate_measurement_input(before_frame, after_frame, timestamps)
        
        before_time, after_time = timestamps
        time_window = after_time - before_time
        
        # --------------------------------------------------------------------
        # 2. Compute Exact Hashes - Ground Truth
        # --------------------------------------------------------------------
        before_raw_hash = self._compute_raw_hash(before_frame)
        after_raw_hash = self._compute_raw_hash(after_frame)
        
        # --------------------------------------------------------------------
        # 3. Pixel Difference Measurement
        # --------------------------------------------------------------------
        diff_mask = self._measure_pixel_differences(before_frame, after_frame)
        
        if np.sum(diff_mask) == 0:
            # No measurable change within threshold
            return []
        
        # --------------------------------------------------------------------
        # 4. Cluster Changed Pixels
        # --------------------------------------------------------------------
        regions = self._cluster_pixels(diff_mask)
        
        if not regions:
            raise ValueError("Indeterminate change: insufficient clustered pixels")
        
        # --------------------------------------------------------------------
        # 5. Measure Each Region
        # --------------------------------------------------------------------
        measurements = []
        for region in regions:
            region_measurements = self._measure_region(
                region, before_frame, after_frame, time_window
            )
            measurements.extend(region_measurements)
        
        return measurements
    
    def measure_full_frame(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        timestamps: Tuple[float, float]
    ) -> Delta:
        """
        Full-frame reference measurement
        Used for system calibration and reference only
        Not mixed with region measurements
        """
        before_time, after_time = timestamps
        time_window = after_time - before_time
        
        # Note: PIXEL_DIFF is reference-only, not emitted by measure()
        return Delta(
            change_type=ChangeType.PIXEL_DIFF,
            coordinates=(0, 0, before_frame.shape[1], before_frame.shape[0]),
            before_raw_hash=self._compute_raw_hash(before_frame),
            after_raw_hash=self._compute_raw_hash(after_frame),
            perceptual_hash=self._compute_perceptual_hash(after_frame),
            time_window=time_window,
            measurement_reliability=self.max_reliability,  # Capped by calibration
            provenance={
                "frame_shape": before_frame.shape,
                "measurement_type": "full_frame_reference",
                "pixel_count": before_frame.shape[0] * before_frame.shape[1],
                "calibration_note": "Reference measurement only"
            }
        )
    
    # ========================================================================
    # Core Measurement Algorithms
    # ========================================================================
    
    def _validate_measurement_input(
        self,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        timestamps: Tuple[float, float]
    ):
        """Strict validation - measurement must be unambiguous"""
        # Existence
        if before_frame is None or after_frame is None:
            raise ValueError("Missing frame(s)")
        
        # Resolution consistency
        if before_frame.shape != after_frame.shape:
            raise ValueError(
                f"Frame resolution mismatch: {before_frame.shape} vs {after_frame.shape}"
            )
        
        # Frame format
        if len(before_frame.shape) != 3 or before_frame.shape[2] != 3:
            raise ValueError(
                f"Invalid frame format: expected (H, W, 3), got {before_frame.shape}"
            )
        
        # Temporal constraints
        before_time, after_time = timestamps
        if before_time >= after_time:
            raise ValueError(
                f"Non-monotonic timestamps: {before_time} >= {after_time}"
            )
        
        if after_time - before_time > self.max_time_window:
            raise ValueError(
                f"Time window exceeded: {after_time - before_time:.2f}s > {self.max_time_window}s"
            )
    
    def _compute_raw_hash(self, image: np.ndarray) -> str:
        """
        Exact byte hash - deterministic identity
        Use for exact equality only
        """
        # Flatten and hash all bytes
        return hashlib.sha256(image.tobytes()).hexdigest()[:16]
    
    def _compute_perceptual_hash(self, image: np.ndarray) -> str:
        """
        Noise-tolerant perceptual hash
        Low reliability - for approximate matching only
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (16, 16))
        
        # Compute average
        avg = resized.mean()
        
        # Create hash bits
        hash_bits = []
        for i in range(16):
            for j in range(16):
                hash_bits.append('1' if resized[i, j] > avg else '0')
        
        # Convert to hex
        hash_int = int(''.join(hash_bits), 2)
        return f"{hash_int:064x}"
    
    def _measure_pixel_differences(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> np.ndarray:
        """
        Measure pixel-level changes with calibrated threshold
        Returns binary mask of changed pixels
        """
        # Grayscale for difference measurement
        before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        
        # Absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Apply calibrated threshold
        _, diff_mask = cv2.threshold(
            diff, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        
        return diff_mask.astype(np.uint8)
    
    def _cluster_pixels(self, diff_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Cluster contiguous changed pixels into bounding boxes
        Purely geometric operation
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            diff_mask, connectivity=8
        )
        
        regions = []
        for i in range(1, num_labels):  # Skip background (0)
            x, y, w, h, area = stats[i]
            
            # Filter by calibrated minimum size
            if area >= self.min_region_size:
                regions.append((x, y, w, h))
        
        # Merge overlapping regions
        if regions:
            return self._merge_adjacent_regions(regions)
        return []
    
    def _merge_adjacent_regions(
        self,
        regions: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge regions with overlap > 20% (calibrated geometric rule)"""
        if len(regions) <= 1:
            return regions
        
        regions = sorted(regions, key=lambda r: (r[0], r[1]))
        merged = []
        current = list(regions[0])
        
        for region in regions[1:]:
            x, y, w, h = region
            cx, cy, cw, ch = current
            
            # Calculate overlap
            overlap_x = max(0, min(cx + cw, x + w) - max(cx, x))
            overlap_y = max(0, min(cy + ch, y + h) - max(cy, y))
            overlap_area = overlap_x * overlap_y
            min_area = min(cw * ch, w * h)
            
            # Merge if significant overlap (calibrated threshold)
            if overlap_area > 0 and overlap_area / min_area > 0.2:
                new_x = min(cx, x)
                new_y = min(cy, y)
                new_w = max(cx + cw, x + w) - new_x
                new_h = max(cy + ch, y + h) - new_y
                current = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(current))
                current = list(region)
        
        merged.append(tuple(current))
        return merged
    
    def _measure_region(
        self,
        region: Tuple[int, int, int, int],
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        time_window: float
    ) -> List[Delta]:
        """
        Measure changes within a specific region
        Returns multiple measurement types if applicable
        """
        x, y, w, h = region
        measurements = []
        
        # Extract region data
        before_region = before_frame[y:y+h, x:x+w]
        after_region = after_frame[y:y+h, x:x+w]
        
        # Compute hashes
        before_raw = self._compute_raw_hash(before_region)
        after_raw = self._compute_raw_hash(after_region)
        perceptual_hash = self._compute_perceptual_hash(after_region)
        
        # --------------------------------------------------------------------
        # 1. Basic Region Change Measurement
        # --------------------------------------------------------------------
        reliability = self._calculate_measurement_reliability(before_region, after_region)
        
        region_delta = Delta(
            change_type=ChangeType.REGION_CHANGE,
            coordinates=region,
            before_raw_hash=before_raw,
            after_raw_hash=after_raw,
            perceptual_hash=perceptual_hash,
            time_window=time_window,
            measurement_reliability=min(reliability, self.max_reliability),
            provenance={
                "region_size_pixels": w * h,
                "measurement_method": "pixel_comparison",
                "pixel_threshold_used": self.pixel_threshold,
                "min_region_size_used": self.min_region_size
            }
        )
        measurements.append(region_delta)
        
        # --------------------------------------------------------------------
        # 2. Intensity Change Measurements (if above calibrated threshold)
        # --------------------------------------------------------------------
        before_mean = np.mean(before_region)
        after_mean = np.mean(after_region)
        intensity_change = after_mean - before_mean
        abs_change = abs(intensity_change)
        
        # Only report intensity changes if above calibrated threshold
        if abs_change > self.intensity_threshold:
            # Intensity measurements have lower reliability
            intensity_reliability = min(reliability * 0.8, self.max_reliability)
            
            if intensity_change > 0:
                intensity_delta = Delta(
                    change_type=ChangeType.INTENSITY_INCREASE,
                    coordinates=region,
                    before_raw_hash=before_raw,
                    after_raw_hash=after_raw,
                    perceptual_hash=perceptual_hash,
                    time_window=time_window,
                    measurement_reliability=intensity_reliability,
                    provenance={
                        "intensity_change": float(intensity_change),
                        "before_intensity": float(before_mean),
                        "after_intensity": float(after_mean),
                        "intensity_threshold_used": self.intensity_threshold,
                        "measurement_note": "Luminance increase only"
                    }
                )
                measurements.append(intensity_delta)
            else:
                intensity_delta = Delta(
                    change_type=ChangeType.INTENSITY_DECREASE,
                    coordinates=region,
                    before_raw_hash=before_raw,
                    after_raw_hash=after_raw,
                    perceptual_hash=perceptual_hash,
                    time_window=time_window,
                    measurement_reliability=intensity_reliability,
                    provenance={
                        "intensity_change": float(intensity_change),
                        "before_intensity": float(before_mean),
                        "after_intensity": float(after_mean),
                        "intensity_threshold_used": self.intensity_threshold,
                        "measurement_note": "Luminance decrease only"
                    }
                )
                measurements.append(intensity_delta)
        
        return measurements
    
    def _calculate_measurement_reliability(
        self,
        before_region: np.ndarray,
        after_region: np.ndarray
    ) -> float:
        """
        Calculate reliability of this measurement
        Based on signal strength and consistency ONLY
        Not confidence in meaning
        """
        # Calculate normalized difference magnitude
        diff = cv2.absdiff(before_region, after_region)
        mean_diff = np.mean(diff) / 255.0  # Normalize to [0, 1]
        
        # Calculate consistency (inverse of variance)
        diff_variance = np.var(diff) / (255.0 * 255.0)
        consistency = 1.0 - min(diff_variance * 5.0, 1.0)
        
        # Signal-to-noise ratio approximation
        signal_strength = mean_diff
        noise_level = max(0.001, diff_variance)
        snr = signal_strength / noise_level
        
        # Combine factors
        reliability = mean_diff * consistency * min(snr, 10.0) / 10.0
        
        # Clamp and return
        return float(np.clip(reliability, 0.0, 1.0))

# ============================================================================
# Measurement Validation (External Integration)
# ============================================================================
# Note: This function belongs in test harness or integration layer
# Included here for convenience only

def validate_measurement_output(deltas: List[Delta]) -> bool:
    """
    Validate output invariants - for integration testing only
    Returns True if all deltas meet measurement standards
    
    Location: External validation layer (not core measurement)
    """
    if not deltas:
        return True
    
    for delta in deltas:
        # Check reliability bounds
        if not (0.0 <= delta.measurement_reliability <= 1.0):
            return False
        
        # Check hash presence for relevant deltas
        if delta.change_type != ChangeType.PIXEL_DIFF:
            if delta.before_raw_hash is None or delta.after_raw_hash is None:
                return False
        
        # Check coordinates are valid
        x, y, w, h = delta.coordinates
        if w <= 0 or h <= 0:
            return False
    
    return True

# ============================================================================
# Instrument Factory - Calibrated Configurations
# ============================================================================

def create_standard_instrument() -> VisualChangeSensor:
    """Factory for standard visual measurement instrument"""
    return VisualChangeSensor(
        pixel_threshold=25,
        min_region_size=10,
        max_time_window=2.0,
        intensity_threshold=20.0,
        max_measurement_reliability=0.95
    )

def create_high_sensitivity_instrument() -> VisualChangeSensor:
    """Factory for high-sensitivity instrument (more false positives)"""
    return VisualChangeSensor(
        pixel_threshold=15,
        min_region_size=5,
        max_time_window=1.0,
        intensity_threshold=10.0,
        max_measurement_reliability=0.90
    )

def create_conservative_instrument() -> VisualChangeSensor:
    """Factory for conservative instrument (fewer false positives)"""
    return VisualChangeSensor(
        pixel_threshold=35,
        min_region_size=20,
        max_time_window=3.0,
        intensity_threshold=30.0,
        max_measurement_reliability=0.98
    )