"""
EME Affordance Extractor v9.8
Pure mechanical interaction detector - zero semantic interpretation
Outputs: Motor affordances only, no OS conventions, no meaning
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math
import hashlib

# ============================================================================
# Core Data Structures - Motor Actions Only
# ============================================================================

class InteractionType(Enum):
    """Strict motor affordances - no UI semantics"""
    CLICKABLE = "clickable"    # Region accepts pointer click/release
    TYPEABLE = "typeable"      # Region accepts keyboard input
    FOCUSABLE = "focusable"    # Region accepts focus navigation
    SCROLLABLE = "scrollable"  # Region accepts scroll/wheel
    DRAGGABLE = "draggable"    # Region accepts click-hold-move
    # Note: No "button", "menu", "textbox" - only motor actions

@dataclass(frozen=True)
class Affordance:
    """Immutable mechanical interaction measurement"""
    interaction_type: InteractionType
    coordinates: Tuple[int, int, int, int]  # x, y, w, h (pixel bounds)
    reliability: float                      # 0.0-1.0 (measurement repeatability)
    evidence: Dict[str, Any]                # Raw cues only, no interpretation
    
    def __post_init__(self):
        """Validate measurement invariants"""
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability {self.reliability} out of bounds")
        x, y, w, h = self.coordinates
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid coordinates: {self.coordinates}")
    
    def to_measurement_dict(self) -> Dict[str, Any]:
        """Serializable affordance data - motor actions only"""
        return {
            "motor_action": self.interaction_type.value,
            "coordinates": self.coordinates,
            "reliability": round(self.reliability, 4),
            "evidence": self.evidence.copy()
        }

# ============================================================================
# Mechanical Cue Detectors - No Semantics
# ============================================================================

class CueDetector:
    """Pure visual cue detection - no meaning extraction"""
    
    @staticmethod
    def detect_high_contrast_boundaries(
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Measure boundary contrast without interpreting purpose"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Edge density measurement
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # Boundary contrast measurement
        border_thickness = 2
        inner_rect = roi[border_thickness:-border_thickness, border_thickness:-border_thickness]
        outer_mean = np.mean(roi[:border_thickness])
        inner_mean = np.mean(inner_rect)
        contrast = abs(outer_mean - inner_mean) / 255.0
        
        return {
            "edge_density": float(edge_density),
            "boundary_contrast": float(contrast),
            "boundary_thickness_measured": border_thickness
        }
    
    @staticmethod
    def detect_rectangular_features(
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Measure rectangularity without inferring UI element"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"rectangularity": 0.0, "contour_count": 0}
        
        # Measure rectangularity of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        bounding_area = w * h
        
        # Convex hull approximation
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        rectangularity = contour_area / bounding_area if bounding_area > 0 else 0.0
        convexity = contour_area / hull_area if hull_area > 0 else 0.0
        
        return {
            "rectangularity": float(rectangularity),
            "convexity": float(convexity),
            "contour_count": len(contours),
            "area_ratio": float(contour_area / bounding_area) if bounding_area > 0 else 0.0
        }
    
    @staticmethod
    def detect_caret_cursor(
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        cursor_position: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Detect text input indicators without OCR"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Look for vertical line (caret)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Vertical line detection
        vertical_kernel = np.array([[1, -1, 1]], dtype=np.float32)
        vertical_response = cv2.filter2D(gray, -1, vertical_kernel)
        
        # Threshold for vertical lines
        _, vertical_mask = cv2.threshold(vertical_response, 50, 255, cv2.THRESH_BINARY)
        
        # Measure vertical line presence
        vertical_density = np.sum(vertical_mask > 0) / (w * h)
        
        # Cursor proximity (if available)
        cursor_proximity = 0.0
        cursor_over_region = False
        if cursor_position:
            cx, cy = cursor_position
            cursor_over_region = (x <= cx <= x + w and y <= cy <= y + h)
            cursor_proximity = 1.0 if cursor_over_region else 0.0
        
        return {
            "vertical_line_density": float(vertical_density),
            "cursor_proximity": cursor_proximity,
            "cursor_over_region": cursor_over_region,
            "measurement_method": "vertical_line_filter"
        }
    
    @staticmethod
    def detect_scroll_indicators(
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Detect scroll mechanics without content interpretation"""
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Look for vertical/horizontal bars
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Vertical bar detection (right edge)
        right_edge = roi[:, -5:] if w > 10 else roi
        edge_mean = np.mean(right_edge)
        center_mean = np.mean(roi[:, w//2-5:w//2+5]) if w > 10 else np.mean(roi)
        
        vertical_bar_contrast = abs(edge_mean - center_mean) / 255.0
        
        # Scroll thumb detection (small rectangle within bar)
        if w > 20:
            # Look for thumb in rightmost 15% of region
            thumb_region = roi[:, int(0.85*w):]
            thumb_rectangularity = CueDetector.detect_rectangular_features(
                frame, (x + int(0.85*w), y, w - int(0.85*w), h)
            )["rectangularity"]
        else:
            thumb_rectangularity = 0.0
        
        return {
            "vertical_bar_contrast": float(vertical_bar_contrast),
            "thumb_rectangularity": float(thumb_rectangularity),
            "edge_vs_center_contrast": float(abs(edge_mean - center_mean) / 255.0)
        }

# ============================================================================
# Pure Affordance Extractor - Zero Semantic Interpretation
# ============================================================================

class MechanicalAffordanceExtractor:
    """
    Pure mechanical interaction detector
    Input: Visual frame + optional motor state
    Output: Possible motor actions, no interpretation
    """
    
    def __init__(self,
                 min_region_size: int = 20,
                 max_region_size: int = 1000,
                 edge_density_threshold: float = 0.05,
                 boundary_contrast_threshold: float = 0.1,
                 min_reliability_threshold: float = 0.3,
                 intensity_change_threshold: float = 0.15,
                 content_clipping_threshold: float = 0.05):
        """
        Mechanical measurement thresholds only
        
        Args:
            min_region_size: Minimum pixel area for interaction
            max_region_size: Maximum pixel area for single interaction
            edge_density_threshold: Minimum edge density for boundaries
            boundary_contrast_threshold: Minimum contrast for boundaries
            min_reliability_threshold: Minimum reliability to report affordance
            intensity_change_threshold: Minimum intensity change for highlighting
            content_clipping_threshold: Edge density at border for clipping detection
        """
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        self.edge_threshold = edge_density_threshold
        self.contrast_threshold = boundary_contrast_threshold
        self.min_reliability = min_reliability_threshold
        self.intensity_threshold = intensity_change_threshold
        self.clipping_threshold = content_clipping_threshold
        
        self._validate_measurement_config()
    
    def _validate_measurement_config(self):
        """Fail loudly on invalid measurement parameters"""
        if self.min_region_size <= 0:
            raise ValueError("min_region_size must be positive")
        if self.max_region_size <= self.min_region_size:
            raise ValueError("max_region_size must be greater than min_region_size")
        if not 0 <= self.edge_threshold <= 1:
            raise ValueError("edge_density_threshold must be 0-1")
        if not 0 <= self.contrast_threshold <= 1:
            raise ValueError("boundary_contrast_threshold must be 0-1")
        if not 0 <= self.min_reliability <= 1:
            raise ValueError("min_reliability_threshold must be 0-1")
        if not 0 <= self.intensity_threshold <= 1:
            raise ValueError("intensity_change_threshold must be 0-1")
        if not 0 <= self.clipping_threshold <= 1:
            raise ValueError("content_clipping_threshold must be 0-1")
    
    def extract(
        self,
        frame: np.ndarray,
        auxiliary_signals: Optional[Dict[str, Any]] = None,
        include_candidate_confidence: bool = False
    ) -> List[Affordance]:
        """
        Extract mechanical affordances from current screen state
        
        Args:
            frame: RGB screen capture (H, W, 3)
            auxiliary_signals: Optional motor state signals (cursor, focus, etc.)
            include_candidate_confidence: Include candidate region confidence scores
            
        Returns:
            List of Affordance objects (motor actions only)
            
        Raises:
            ValueError: On invalid input or measurement ambiguity
        """
        # --------------------------------------------------------------------
        # 1. Input Validation - Fail Loud
        # --------------------------------------------------------------------
        self._validate_input(frame)
        
        # --------------------------------------------------------------------
        # 2. Extract Auxiliary Motor State (No Interpretation)
        # --------------------------------------------------------------------
        cursor_pos = auxiliary_signals.get("cursor_position") if auxiliary_signals else None
        cursor_shape = auxiliary_signals.get("cursor_shape") if auxiliary_signals else None
        focus_rect = auxiliary_signals.get("focus_rectangle") if auxiliary_signals else None
        
        # Hash cursor shape to remove semantic interpretation
        cursor_shape_hash = None
        if cursor_shape:
            # Hash the cursor shape to remove OS-specific semantics
            cursor_shape_hash = hashlib.sha256(cursor_shape.encode()).hexdigest()[:16]
        
        # --------------------------------------------------------------------
        # 3. Find Candidate Regions (Mechanical Only)
        # --------------------------------------------------------------------
        candidate_regions = self._find_candidate_regions(frame)
        
        if not candidate_regions:
            return []
        
        # --------------------------------------------------------------------
        # 4. Measure Affordances for Each Region
        # --------------------------------------------------------------------
        affordances = []
        
        for region in candidate_regions:
            region_affordances = self._measure_region_affordances(
                frame, region, cursor_pos, cursor_shape_hash, focus_rect
            )
            affordances.extend(region_affordances)
        
        # --------------------------------------------------------------------
        # 5. Filter by Reliability Threshold
        # --------------------------------------------------------------------
        reliable_affordances = [
            a for a in affordances 
            if a.reliability >= self.min_reliability
        ]
        
        # --------------------------------------------------------------------
        # 6. Add Candidate Confidence if requested
        # --------------------------------------------------------------------
        if include_candidate_confidence:
            for affordance in reliable_affordances:
                affordance.evidence["candidate_confidence"] = self._calculate_candidate_confidence(
                    frame, affordance.coordinates
                )
        
        return reliable_affordances
    
    def _validate_input(self, frame: np.ndarray):
        """Validate screen capture input"""
        if frame is None:
            raise ValueError("Missing frame")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame format: expected (H, W, 3), got {frame.shape}")
        
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            raise ValueError(f"Frame too small: {frame.shape}")
    
    def _find_candidate_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find regions that might afford interaction
        Based purely on visual salience, not semantics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for boundary finding
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours (potential region boundaries)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size constraints
            area = w * h
            if area < self.min_region_size or area > self.max_region_size:
                continue
            
            # Filter by aspect ratio (avoid extreme shapes)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            candidate_regions.append((x, y, w, h))
        
        return candidate_regions
    
    def _calculate_candidate_confidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate confidence that this region is a candidate
        Used for exploration bias, not semantic meaning
        """
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Edge density
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # Color variance (salience)
        color_variance = np.var(roi) / (255.0 * 255.0)
        
        # Combine factors (higher edge density and color variance = more salient)
        confidence = (edge_density * 0.6) + (color_variance * 0.4)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _measure_region_affordances(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        cursor_position: Optional[Tuple[int, int]],
        cursor_shape_hash: Optional[str],
        focus_rectangle: Optional[Tuple[int, int, int, int]]
    ) -> List[Affordance]:
        """
        Measure all possible motor affordances for a region
        Returns multiple affordances if evidence supports them
        """
        affordances = []
        
        # --------------------------------------------------------------------
        # 1. CLICKABLE Affordance Measurement
        # --------------------------------------------------------------------
        clickable_evidence = self._measure_clickable_evidence(
            frame, region, cursor_position, cursor_shape_hash
        )
        clickable_reliability = self._calculate_clickable_reliability(clickable_evidence)
        
        if clickable_reliability >= self.min_reliability:
            affordances.append(Affordance(
                interaction_type=InteractionType.CLICKABLE,
                coordinates=region,
                reliability=clickable_reliability,
                evidence=clickable_evidence
            ))
        
        # --------------------------------------------------------------------
        # 2. TYPEABLE Affordance Measurement
        # --------------------------------------------------------------------
        typeable_evidence = self._measure_typeable_evidence(
            frame, region, cursor_position, focus_rectangle
        )
        typeable_reliability = self._calculate_typeable_reliability(typeable_evidence)
        
        if typeable_reliability >= self.min_reliability:
            affordances.append(Affordance(
                interaction_type=InteractionType.TYPEABLE,
                coordinates=region,
                reliability=typeable_reliability,
                evidence=typeable_evidence
            ))
        
        # --------------------------------------------------------------------
        # 3. FOCUSABLE Affordance Measurement
        # --------------------------------------------------------------------
        focusable_evidence = self._measure_focusable_evidence(
            frame, region, focus_rectangle
        )
        focusable_reliability = self._calculate_focusable_reliability(focusable_evidence)
        
        if focusable_reliability >= self.min_reliability:
            affordances.append(Affordance(
                interaction_type=InteractionType.FOCUSABLE,
                coordinates=region,
                reliability=focusable_reliability,
                evidence=focusable_evidence
            ))
        
        # --------------------------------------------------------------------
        # 4. SCROLLABLE Affordance Measurement
        # --------------------------------------------------------------------
        scrollable_evidence = self._measure_scrollable_evidence(frame, region)
        scrollable_reliability = self._calculate_scrollable_reliability(scrollable_evidence)
        
        if scrollable_reliability >= self.min_reliability:
            affordances.append(Affordance(
                interaction_type=InteractionType.SCROLLABLE,
                coordinates=region,
                reliability=scrollable_reliability,
                evidence=scrollable_evidence
            ))
        
        # --------------------------------------------------------------------
        # 5. DRAGGABLE Affordance Measurement
        # --------------------------------------------------------------------
        draggable_evidence = self._measure_draggable_evidence(
            frame, region, cursor_shape_hash
        )
        draggable_reliability = self._calculate_draggable_reliability(draggable_evidence)
        
        if draggable_reliability >= self.min_reliability:
            affordances.append(Affordance(
                interaction_type=InteractionType.DRAGGABLE,
                coordinates=region,
                reliability=draggable_reliability,
                evidence=draggable_evidence
            ))
        
        return affordances
    
    # ========================================================================
    # CLICKABLE Affordance Measurement
    # ========================================================================
    
    def _measure_clickable_evidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        cursor_position: Optional[Tuple[int, int]],
        cursor_shape_hash: Optional[str]
    ) -> Dict[str, Any]:
        """Measure cues for clickable regions without inferring purpose"""
        evidence = {}
        
        # 1. Boundary contrast measurement
        boundary_cues = CueDetector.detect_high_contrast_boundaries(frame, region)
        evidence.update(boundary_cues)
        
        # 2. Rectangular feature measurement
        rect_cues = CueDetector.detect_rectangular_features(frame, region)
        evidence.update(rect_cues)
        
        # 3. Cursor proximity (if available) - pure measurement, no interpretation
        if cursor_position:
            cx, cy = cursor_position
            x, y, w, h = region
            if x <= cx <= x + w and y <= cy <= y + h:
                evidence["cursor_over_region"] = True
                evidence["cursor_distance"] = 0.0
            else:
                evidence["cursor_over_region"] = False
                # Calculate Euclidean distance to region center
                center_x, center_y = x + w//2, y + h//2
                distance = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                evidence["cursor_distance"] = float(distance)
        
        # 4. Cursor shape hash (if available) - no semantic interpretation
        if cursor_shape_hash:
            evidence["cursor_shape_hash"] = cursor_shape_hash
            # Note: We do NOT interpret the hash - it's just a unique identifier
        
        return evidence
    
    def _calculate_clickable_reliability(self, evidence: Dict[str, Any]) -> float:
        """Calculate reliability based on mechanical cues only"""
        reliability = 0.0
        
        # Boundary cues
        edge_density = evidence.get("edge_density", 0.0)
        boundary_contrast = evidence.get("boundary_contrast", 0.0)
        
        if edge_density > self.edge_threshold:
            reliability += 0.3
        if boundary_contrast > self.contrast_threshold:
            reliability += 0.3
        
        # Rectangular cues
        rectangularity = evidence.get("rectangularity", 0.0)
        if rectangularity > 0.7:
            reliability += 0.2
        
        # Cursor proximity (pure measurement, no semantic meaning)
        if evidence.get("cursor_over_region", False):
            reliability += 0.2  # Cursor in region suggests interactivity
        
        return min(reliability, 1.0)
    
    # ========================================================================
    # TYPEABLE Affordance Measurement
    # ========================================================================
    
    def _measure_typeable_evidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        cursor_position: Optional[Tuple[int, int]],
        focus_rectangle: Optional[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Measure cues for typeable regions without OCR"""
        evidence = {}
        
        # 1. Caret/cursor detection
        caret_cues = CueDetector.detect_caret_cursor(frame, region, cursor_position)
        evidence.update(caret_cues)
        
        # 2. Focus rectangle alignment (if available)
        if focus_rectangle:
            fx, fy, fw, fh = focus_rectangle
            x, y, w, h = region
            
            # Calculate overlap
            overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
            overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
            overlap_area = overlap_x * overlap_y
            region_area = w * h
            
            evidence["focus_overlap"] = overlap_area / region_area if region_area > 0 else 0.0
            evidence["focus_rectangle"] = focus_rectangle
        
        # 3. Text-like region characteristics
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Characteristic texture of text regions
        # (Without OCR - looking for high frequency content)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        texture_variance = np.var(gradient_magnitude)
        
        evidence["texture_variance"] = float(texture_variance)
        evidence["region_aspect_ratio"] = w / h if h > 0 else 0.0
        
        return evidence
    
    def _calculate_typeable_reliability(self, evidence: Dict[str, Any]) -> float:
        """Calculate reliability for typeable affordance"""
        reliability = 0.0
        
        # Caret evidence
        vertical_line_density = evidence.get("vertical_line_density", 0.0)
        if vertical_line_density > 0.01:  # At least 1% vertical lines
            reliability += 0.3
        
        # Focus evidence
        focus_overlap = evidence.get("focus_overlap", 0.0)
        if focus_overlap > 0.5:  # More than 50% overlap with focus
            reliability += 0.4
        
        # Texture evidence (text regions have high variance)
        texture_variance = evidence.get("texture_variance", 0.0)
        if texture_variance > 1000:  # Empirical threshold
            reliability += 0.2
        
        # Cursor proximity (if in region)
        if evidence.get("cursor_over_region", False):
            reliability += 0.1
        
        return min(reliability, 1.0)
    
    # ========================================================================
    # FOCUSABLE Affordance Measurement
    # ========================================================================
    
    def _measure_focusable_evidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        focus_rectangle: Optional[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Measure cues for focusable regions"""
        evidence = {}
        
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # 1. Focus ring detection (dotted or solid highlight)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Look for high-contrast border (focus ring often colored)
        # Convert to HSV for color-based border detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Common focus ring colors (blue, orange, etc.)
        # Without semantics - just looking for colored borders
        
        # Blue-ish borders (common for focus)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Check border pixels specifically
        border_mask = np.zeros_like(gray)
        border_thickness = 3
        border_mask[:border_thickness, :] = 255  # Top
        border_mask[-border_thickness:, :] = 255  # Bottom
        border_mask[:, :border_thickness] = 255   # Left
        border_mask[:, -border_thickness:] = 255  # Right
        
        colored_border_pixels = np.sum((blue_mask > 0) & (border_mask > 0))
        total_border_pixels = np.sum(border_mask > 0)
        
        evidence["colored_border_ratio"] = (
            colored_border_pixels / total_border_pixels if total_border_pixels > 0 else 0.0
        )
        
        # 2. Focus rectangle overlap (if available)
        if focus_rectangle:
            fx, fy, fw, fh = focus_rectangle
            overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
            overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
            overlap_area = overlap_x * overlap_y
            region_area = w * h
            
            evidence["focus_rectangle_overlap"] = (
                overlap_area / region_area if region_area > 0 else 0.0
            )
            evidence["has_focus_rectangle"] = True
        else:
            evidence["has_focus_rectangle"] = False
        
        # 3. Visual highlight detection
        # Focused regions often have different brightness
        border_mean = np.mean(roi[:border_thickness, :])
        center_mean = np.mean(roi[border_thickness:-border_thickness, border_thickness:-border_thickness])
        highlight_contrast = abs(border_mean - center_mean) / 255.0
        
        evidence["highlight_contrast"] = float(highlight_contrast)
        
        return evidence
    
    def _calculate_focusable_reliability(self, evidence: Dict[str, Any]) -> float:
        """Calculate reliability for focusable affordance"""
        reliability = 0.0
        
        # Colored border evidence
        colored_border_ratio = evidence.get("colored_border_ratio", 0.0)
        if colored_border_ratio > 0.1:  # At least 10% colored border
            reliability += 0.4
        
        # Focus rectangle evidence
        if evidence.get("has_focus_rectangle", False):
            overlap = evidence.get("focus_rectangle_overlap", 0.0)
            if overlap > 0.8:  # Strong overlap
                reliability += 0.5
            elif overlap > 0.3:  # Moderate overlap
                reliability += 0.2
        
        # Highlight contrast
        highlight_contrast = evidence.get("highlight_contrast", 0.0)
        if highlight_contrast > self.intensity_threshold:
            reliability += 0.2
        
        return min(reliability, 1.0)
    
    # ========================================================================
    # SCROLLABLE Affordance Measurement
    # ========================================================================
    
    def _measure_scrollable_evidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """Measure cues for scrollable regions"""
        evidence = CueDetector.detect_scroll_indicators(frame, region)
        
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # Additional: Content clipping detection
        # Scrollable regions often have content that extends beyond visible area
        
        # Edge detection for content boundaries
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if edges are cut off at region boundaries
        border_thickness = 5
        top_edges = np.sum(edges[:border_thickness, :] > 0)
        bottom_edges = np.sum(edges[-border_thickness:, :] > 0)
        left_edges = np.sum(edges[:, :border_thickness] > 0)
        right_edges = np.sum(edges[:, -border_thickness:] > 0)
        
        total_border_area = 2 * (w + h) * border_thickness
        edge_density_at_border = (top_edges + bottom_edges + left_edges + right_edges) / total_border_area
        
        evidence["edge_density_at_border"] = float(edge_density_at_border)
        evidence["content_clipping_indicator"] = edge_density_at_border > self.clipping_threshold
        
        return evidence
    
    def _calculate_scrollable_reliability(self, evidence: Dict[str, Any]) -> float:
        """Calculate reliability for scrollable affordance"""
        reliability = 0.0
        
        # Scroll bar evidence
        vertical_bar_contrast = evidence.get("vertical_bar_contrast", 0.0)
        if vertical_bar_contrast > 0.2:
            reliability += 0.4
        
        # Scroll thumb evidence
        thumb_rectangularity = evidence.get("thumb_rectangularity", 0.0)
        if thumb_rectangularity > 0.6:
            reliability += 0.3
        
        # Content clipping evidence
        if evidence.get("content_clipping_indicator", False):
            reliability += 0.2
        
        return min(reliability, 1.0)
    
    # ========================================================================
    # DRAGGABLE Affordance Measurement
    # ========================================================================
    
    def _measure_draggable_evidence(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        cursor_shape_hash: Optional[str]
    ) -> Dict[str, Any]:
        """Measure cues for draggable regions"""
        evidence = {}
        
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        # 1. Gripper pattern detection (dots, lines)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Look for dot patterns (common in draggable headers)
        # Use morphological operations to find small connected components
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find small blobs (dots)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        small_blobs = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2 < area < 50:  # Dot-sized blobs
                small_blobs += 1
        
        evidence["dot_pattern_count"] = small_blobs
        evidence["dot_pattern_density"] = small_blobs / (w * h) if w * h > 0 else 0.0
        
        # 2. Title bar-like region detection (top portion)
        top_portion_height = min(30, h // 3)
        top_portion = roi[:top_portion_height, :]
        
        # Check if top portion has different characteristics
        top_mean = np.mean(top_portion)
        bottom_mean = np.mean(roi[top_portion_height:, :])
        top_contrast = abs(top_mean - bottom_mean) / 255.0
        
        evidence["top_portion_contrast"] = float(top_contrast)
        evidence["top_portion_height"] = top_portion_height
        
        # 3. Cursor shape hash (if available) - no semantic interpretation
        if cursor_shape_hash:
            evidence["cursor_shape_hash"] = cursor_shape_hash
            # Note: We do NOT interpret the hash - it's just a unique identifier
        
        return evidence
    
    def _calculate_draggable_reliability(self, evidence: Dict[str, Any]) -> float:
        """Calculate reliability for draggable affordance"""
        reliability = 0.0
        
        # Gripper pattern evidence
        dot_pattern_density = evidence.get("dot_pattern_density", 0.0)
        if dot_pattern_density > 0.001:  # At least 0.1% density of dots
            reliability += 0.3
        
        # Top portion contrast (title bar-like)
        top_contrast = evidence.get("top_portion_contrast", 0.0)
        if top_contrast > self.intensity_threshold:
            reliability += 0.3
        
        # Cursor shape presence (without interpretation)
        if "cursor_shape_hash" in evidence:
            # Having a cursor shape (any shape) suggests interactivity
            reliability += 0.2
        
        return min(reliability, 1.0)

# ============================================================================
# Factory Functions
# ============================================================================

def create_standard_affordance_extractor() -> MechanicalAffordanceExtractor:
    """Factory for standard mechanical affordance extraction"""
    return MechanicalAffordanceExtractor(
        min_region_size=25,
        max_region_size=800,
        edge_density_threshold=0.03,
        boundary_contrast_threshold=0.08,
        min_reliability_threshold=0.3,
        intensity_change_threshold=0.15,
        content_clipping_threshold=0.05
    )

def create_high_sensitivity_extractor() -> MechanicalAffordanceExtractor:
    """Factory for high sensitivity (more false positives)"""
    return MechanicalAffordanceExtractor(
        min_region_size=15,
        max_region_size=1200,
        edge_density_threshold=0.02,
        boundary_contrast_threshold=0.05,
        min_reliability_threshold=0.2,
        intensity_change_threshold=0.1,
        content_clipping_threshold=0.03
    )

def create_conservative_extractor() -> MechanicalAffordanceExtractor:
    """Factory for conservative extraction (fewer false positives)"""
    return MechanicalAffordanceExtractor(
        min_region_size=40,
        max_region_size=500,
        edge_density_threshold=0.05,
        boundary_contrast_threshold=0.12,
        min_reliability_threshold=0.4,
        intensity_change_threshold=0.2,
        content_clipping_threshold=0.08
    )