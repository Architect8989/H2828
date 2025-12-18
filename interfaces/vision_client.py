"""
Vision Client Interface

Boundary layer between vision models and cognitive systems. This module strictly
normalizes and validates raw vision model outputs without interpretation.
It contains zero intelligence about screen contents, UI semantics, or user intent.

Absolute constraints:
- NO meaning, intent, or purpose inference
- NO semantic labeling (e.g., "button", "menu") unless provided by vision model
- NO ranking, filtering, prioritization, or cleaning
- NO importance/relevance decisions
- NO heuristics or domain knowledge injection
- NO hidden state maintenance
- NO action preparation or hinting

This module is infrastructure only - boring, auditable, and replaceable.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, ValidationError, Field, validator, ConfigDict

# Configure audit logging - INFO level ensures logs are captured in production
logger = logging.getLogger(__name__)
AUDIT_LEVEL = logging.INFO


class VisionModelType(str, Enum):
    """
    Enumeration of supported vision model types.
    
    WARNING: This enum is for structural identification only.
    Do NOT branch behavior based on model type outside this module.
    Doing so would violate model-agnostic principles.
    """
    OCR = "ocr"
    UI_PARSER = "ui_parser"
    VLM = "vlm"
    OBJECT_DETECTOR = "object_detector"


class CoordinateSystem(str, Enum):
    """Coordinate system used for spatial data."""
    NORMALIZED = "normalized"  # [0, 1] range relative to screen dimensions
    PIXELS = "pixels"         # Absolute pixel coordinates (non-negative numbers)
    RELATIVE = "relative"     # Relative to parent element (range depends on parent)


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box representation.
    
    Coordinates follow (x1, y1, x2, y2) convention:
    - (x1, y1): top-left corner
    - (x2, y2): bottom-right corner
    
    Coordinate validation is system-specific:
    - NORMALIZED: [0, 1] range
    - PIXELS: non-negative numbers (accepts both int and float)
    - RELATIVE: no constraints (depends on parent context)
    """
    x1: float
    y1: float
    x2: float
    y2: float
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED
    confidence: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate bounding box coordinates according to coordinate system."""
        # Basic ordering constraints (apply to all systems)
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(
                f"Invalid bounding box: x1({self.x1}) < x2({self.x2}) and "
                f"y1({self.y1}) < y2({self.y2}) required"
            )
        
        # System-specific validation
        if self.coordinate_system == CoordinateSystem.NORMALIZED:
            self._validate_normalized_coordinates()
        elif self.coordinate_system == CoordinateSystem.PIXELS:
            self._validate_pixel_coordinates()
        # RELATIVE coordinates have no constraints - they depend on parent context
        
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be in [0, 1] range. Got: {self.confidence}")
    
    def _validate_normalized_coordinates(self) -> None:
        """Validate normalized coordinates are in [0, 1] range."""
        if not (0 <= self.x1 <= 1 and 0 <= self.x2 <= 1 and 
                0 <= self.y1 <= 1 and 0 <= self.y2 <= 1):
            raise ValueError(
                f"Normalized coordinates must be in [0, 1] range. "
                f"Got: x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}"
            )
    
    def _validate_pixel_coordinates(self) -> None:
        """
        Validate pixel coordinates are non-negative.
        
        Note: Accepts both int and float for sub-pixel precision.
        Strings are not accepted - they must be converted before construction.
        """
        for coord_name, coord_value in [("x1", self.x1), ("y1", self.y1), 
                                        ("x2", self.x2), ("y2", self.y2)]:
            # Check that coordinate is a number (int or float)
            if not isinstance(coord_value, (int, float)):
                raise ValueError(
                    f"Pixel coordinate {coord_name} must be numeric (int or float). "
                    f"Got type: {type(coord_value)}"
                )
            
            # Check non-negative
            if coord_value < 0:
                raise ValueError(
                    f"Pixel coordinate {coord_name} must be non-negative. "
                    f"Got: {coord_value}"
                )


class VisionElement(BaseModel):
    """
    Base schema for vision model outputs. All fields are exactly as provided
    by the vision model with no semantic interpretation.
    
    WARNING: Do not add fields that imply understanding of content.
    This schema must remain purely descriptive, not interpretative.
    """
    model_config = ConfigDict(
        extra='forbid',  # Reject unexpected fields
        frozen=True,     # Immutable after creation
        validate_assignment=True
    )
    
    # Core identification
    element_id: str = Field(default_factory=lambda: str(uuid4()))
    source_model: VisionModelType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Spatial information (exactly as provided by model)
    bounding_box: Optional[BoundingBox] = None
    spatial_confidence: Optional[float] = Field(None, ge=0, le=1)
    
    # Raw model outputs (no cleaning or interpretation)
    raw_text: Optional[str] = None
    raw_labels: List[str] = Field(default_factory=list)
    raw_attributes: Dict[str, Any] = Field(default_factory=dict)
    
    # Relationships (exactly as provided by model)
    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)
    sibling_ids: List[str] = Field(default_factory=list)
    
    # Model-specific metadata
    model_confidence: Optional[float] = Field(None, ge=0, le=1)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('raw_text', pre=True)
    def preserve_text_exactly(cls, v: Optional[str]) -> Optional[str]:
        """Preserve text exactly as provided, including whitespace and casing."""
        return v if v is None else str(v)
    
    @validator('raw_labels', 'child_ids', 'sibling_ids', pre=True)
    def ensure_list(cls, v: Any) -> List:
        """Ensure list fields are always lists."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)
    
    @validator('source_model', pre=True)
    def validate_source_model(cls, v: Any) -> VisionModelType:
        """Convert string to VisionModelType while preserving original value."""
        if isinstance(v, VisionModelType):
            return v
        if isinstance(v, str):
            try:
                return VisionModelType(v)
            except ValueError:
                raise ValueError(
                    f"Invalid source_model value: {v}. "
                    f"Must be one of {[e.value for e in VisionModelType]}"
                )
        raise ValueError(f"Invalid source_model type: {type(v)}")


class VisionOutput(BaseModel):
    """
    Complete normalized output from vision processing.
    
    This is a pure structural container - no intelligence, no interpretation.
    The cognitive system must interpret these raw observations.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Identification
    processing_id: str = Field(default_factory=lambda: str(uuid4()))
    screen_timestamp: Optional[datetime] = None
    screen_dimensions: Optional[Tuple[int, int]] = None  # (width, height) in pixels
    
    # Raw elements from all vision models
    elements: List[VisionElement] = Field(default_factory=list)
    
    # Model outputs in their original structure
    raw_model_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit trail
    input_hash: Optional[str] = None  # Hash of raw input for reproducibility
    version: str = "1.0.0"
    
    @validator('screen_dimensions')
    def validate_screen_dimensions(cls, v: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Validate screen dimensions are positive integers."""
        if v is None:
            return None
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"Screen dimensions must be positive. Got: {width}x{height}")
        return v


class VisionClient:
    """
    Stateless client for processing raw vision model outputs.
    
    This class:
    1. Validates structure of raw vision outputs
    2. Normalizes into strict schema
    3. Preserves all data exactly as received
    4. Logs for auditability at INFO level
    
    This class does NOT:
    1. Interpret meaning of any element
    2. Decide importance or relevance
    3. Filter, clean, or prioritize outputs
    4. Maintain state between calls
    5. Prepare or suggest actions
    6. Branch behavior based on model type (except for structural extraction)
    """
    
    def __init__(self, audit_log_level: int = AUDIT_LEVEL) -> None:
        """
        Initialize stateless vision client.
        
        Args:
            audit_log_level: Logging level for audit trails. Defaults to INFO
                           to ensure auditability in production.
        """
        self._logger = logging.getLogger(f"{__name__}.VisionClient")
        self._audit_log_level = audit_log_level
    
    def process_vision_output(
        self,
        *,
        ocr_output: Optional[Dict[str, Any]] = None,
        ui_parser_output: Optional[Dict[str, Any]] = None,
        vlm_output: Optional[Dict[str, Any]] = None,
        object_detector_output: Optional[Dict[str, Any]] = None,
        screen_dimensions: Optional[Tuple[int, int]] = None,
        screen_timestamp: Optional[datetime] = None,
        input_hash: Optional[str] = None
    ) -> VisionOutput:
        """
        Process raw vision model outputs into normalized schema.
        
        Args:
            ocr_output: Raw OCR model output. Must contain 'text_elements' list.
            ui_parser_output: Raw UI parser output. Must contain 'ui_elements' list.
            vlm_output: Raw VLM output. Must contain 'descriptions' list.
            object_detector_output: Raw object detector output. Must contain 'detections' list.
            screen_dimensions: Optional (width, height) in pixels.
            screen_timestamp: Optional timestamp of screen capture.
            input_hash: Optional hash of input for reproducibility.
        
        Returns:
            Normalized VisionOutput containing all elements exactly as provided.
        
        Raises:
            ValidationError: If schema validation fails during model construction.
            ValueError: If any input fails structural validation or no outputs provided.
        
        Note:
            - All processing is deterministic and stateless
            - No interpretation of content occurs
            - All raw data is preserved exactly
            - Audit logs are written at configured level (default: INFO)
        """
        # Log raw inputs for auditability at configured level
        self._log_raw_inputs(
            ocr_output=ocr_output,
            ui_parser_output=ui_parser_output,
            vlm_output=vlm_output,
            object_detector_output=object_detector_output,
            screen_dimensions=screen_dimensions,
            input_hash=input_hash
        )
        
        # Validate at least one vision output provided
        if not any([ocr_output, ui_parser_output, vlm_output, object_detector_output]):
            raise ValueError("At least one vision output must be provided")
        
        # Collect all raw model outputs
        raw_outputs = {}
        elements: List[VisionElement] = []
        
        # Process each vision model output independently
        # NOTE: This branching is purely structural - we extract different
        # fields based on expected schema, but never interpret content
        if ocr_output is not None:
            ocr_elements = self._extract_ocr_elements(ocr_output)
            elements.extend(ocr_elements)
            raw_outputs["ocr"] = ocr_output
        
        if ui_parser_output is not None:
            ui_elements = self._extract_ui_elements(ui_parser_output)
            elements.extend(ui_elements)
            raw_outputs["ui_parser"] = ui_parser_output
        
        if vlm_output is not None:
            vlm_elements = self._extract_vlm_elements(vlm_output)
            elements.extend(vlm_elements)
            raw_outputs["vlm"] = vlm_output
        
        if object_detector_output is not None:
            detector_elements = self._extract_detector_elements(object_detector_output)
            elements.extend(detector_elements)
            raw_outputs["object_detector"] = object_detector_output
        
        # Create normalized output
        vision_output = VisionOutput(
            screen_dimensions=screen_dimensions,
            screen_timestamp=screen_timestamp,
            elements=elements,
            raw_model_outputs=raw_outputs,
            input_hash=input_hash
        )
        
        # Log normalized output for auditability
        self._log_normalized_output(vision_output)
        
        return vision_output
    
    def _extract_ocr_elements(self, ocr_output: Dict[str, Any]) -> List[VisionElement]:
        """
        Extract OCR elements from raw output.
        
        Note: Preserves all text exactly, including whitespace and casing.
        No text cleaning, correction, or interpretation.
        
        Raises:
            ValueError: If OCR output structure is invalid.
            ValidationError: Only from VisionElement/BoundingBox constructors.
        """
        try:
            elements = []
            text_elements = ocr_output.get("text_elements", [])
            
            for i, text_element in enumerate(text_elements):
                # Validate required fields - raise ValueError not ValidationError
                if "text" not in text_element:
                    raise ValueError(f"OCR element {i} missing 'text' field")
                
                # Extract bounding box if present
                bbox = None
                if "bounding_box" in text_element:
                    bbox_data = text_element["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"OCR element {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    # Convert coordinates to float for consistency
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"OCR element {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=CoordinateSystem(
                            text_element.get("coordinate_system", "normalized")
                        ),
                        confidence=text_element.get("bbox_confidence")
                    )
                
                # Create element - let VisionElement raise ValidationError if schema fails
                element = VisionElement(
                    source_model=VisionModelType.OCR,
                    raw_text=text_element["text"],
                    bounding_box=bbox,
                    spatial_confidence=text_element.get("spatial_confidence"),
                    model_confidence=text_element.get("confidence"),
                    raw_attributes={
                        k: v for k, v in text_element.items()
                        if k not in ["text", "bounding_box", "spatial_confidence", 
                                   "confidence", "coordinate_system", "bbox_confidence"]
                    },
                    model_metadata=ocr_output.get("metadata", {})
                )
                elements.append(element)
            
            return elements
        
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid OCR output structure: {str(e)}") from e
    
    def _extract_ui_elements(self, ui_output: Dict[str, Any]) -> List[VisionElement]:
        """
        Extract UI elements from raw output.
        
        Note: Preserves semantic labels ONLY if provided by model.
        Does NOT infer or assign semantic meaning.
        
        Raises:
            ValueError: If UI parser output structure is invalid.
            ValidationError: Only from VisionElement/BoundingBox constructors.
        """
        try:
            elements = []
            ui_elements = ui_output.get("ui_elements", [])
            
            for i, ui_element in enumerate(ui_elements):
                # Extract bounding box if present
                bbox = None
                if "bounding_box" in ui_element:
                    bbox_data = ui_element["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"UI element {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    # Convert coordinates to float for consistency
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"UI element {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=CoordinateSystem(
                            ui_element.get("coordinate_system", "normalized")
                        ),
                        confidence=ui_element.get("bbox_confidence")
                    )
                
                # Preserve labels exactly as provided (no interpretation)
                raw_labels = ui_element.get("labels", [])
                if isinstance(raw_labels, str):
                    raw_labels = [raw_labels]
                
                # Create element - let VisionElement raise ValidationError if schema fails
                element = VisionElement(
                    source_model=VisionModelType.UI_PARSER,
                    raw_text=ui_element.get("text"),
                    raw_labels=raw_labels,
                    bounding_box=bbox,
                    spatial_confidence=ui_element.get("spatial_confidence"),
                    model_confidence=ui_element.get("confidence"),
                    parent_id=ui_element.get("parent_id"),
                    child_ids=ui_element.get("children", []),
                    sibling_ids=ui_element.get("siblings", []),
                    raw_attributes={
                        k: v for k, v in ui_element.items()
                        if k not in ["text", "labels", "bounding_box", "spatial_confidence",
                                   "confidence", "parent_id", "children", "siblings",
                                   "coordinate_system", "bbox_confidence"]
                    },
                    model_metadata=ui_output.get("metadata", {})
                )
                elements.append(element)
            
            return elements
        
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid UI parser output structure: {str(e)}") from e
    
    def _extract_vlm_elements(self, vlm_output: Dict[str, Any]) -> List[VisionElement]:
        """
        Extract VLM elements from raw output.
        
        Note: Preserves descriptions exactly. No summarization, interpretation,
        or relevance filtering.
        
        Raises:
            ValueError: If VLM output structure is invalid.
            ValidationError: Only from VisionElement/BoundingBox constructors.
        """
        try:
            elements = []
            descriptions = vlm_output.get("descriptions", [])
            
            for i, description in enumerate(descriptions):
                # Extract bounding box if present
                bbox = None
                if "bounding_box" in description:
                    bbox_data = description["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"VLM description {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    # Convert coordinates to float for consistency
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"VLM description {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=CoordinateSystem(
                            description.get("coordinate_system", "normalized")
                        ),
                        confidence=description.get("bbox_confidence")
                    )
                
                # Create element - let VisionElement raise ValidationError if schema fails
                element = VisionElement(
                    source_model=VisionModelType.VLM,
                    raw_text=description.get("description"),
                    raw_labels=description.get("tags", []),
                    bounding_box=bbox,
                    spatial_confidence=description.get("spatial_confidence"),
                    model_confidence=description.get("confidence"),
                    raw_attributes={
                        k: v for k, v in description.items()
                        if k not in ["description", "tags", "bounding_box", 
                                   "spatial_confidence", "confidence",
                                   "coordinate_system", "bbox_confidence"]
                    },
                    model_metadata=vlm_output.get("metadata", {})
                )
                elements.append(element)
            
            return elements
        
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid VLM output structure: {str(e)}") from e
    
    def _extract_detector_elements(self, detector_output: Dict[str, Any]) -> List[VisionElement]:
        """
        Extract object detection elements from raw output.
        
        Note: Preserves class labels exactly. No label mapping, filtering,
        or confidence thresholding.
        
        Raises:
            ValueError: If object detector output structure is invalid.
            ValidationError: Only from VisionElement/BoundingBox constructors.
        """
        try:
            elements = []
            detections = detector_output.get("detections", [])
            
            for i, detection in enumerate(detections):
                # Validate required fields - raise ValueError not ValidationError
                if "bounding_box" not in detection:
                    raise ValueError(f"Detection {i} missing 'bounding_box' field")
                
                bbox_data = detection["bounding_box"]
                if len(bbox_data) != 4:
                    raise ValueError(
                        f"Detection {i} bounding_box must have 4 values. "
                        f"Got: {bbox_data}"
                    )
                
                # Convert coordinates to float for consistency
                try:
                    x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Detection {i} bounding_box coordinates must be numeric. "
                        f"Got: {bbox_data}. Error: {e}"
                    )
                
                bbox = BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    coordinate_system=CoordinateSystem(
                        detection.get("coordinate_system", "normalized")
                    ),
                    confidence=detection.get("bbox_confidence")
                )
                
                # Create element - let VisionElement raise ValidationError if schema fails
                element = VisionElement(
                    source_model=VisionModelType.OBJECT_DETECTOR,
                    raw_labels=[detection.get("class_label", "unknown")],
                    bounding_box=bbox,
                    spatial_confidence=detection.get("spatial_confidence"),
                    model_confidence=detection.get("confidence"),
                    raw_attributes={
                        k: v for k, v in detection.items()
                        if k not in ["class_label", "bounding_box", "spatial_confidence",
                                   "confidence", "coordinate_system", "bbox_confidence"]
                    },
                    model_metadata=detector_output.get("metadata", {})
                )
                elements.append(element)
            
            return elements
        
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid object detector output structure: {str(e)}") from e
    
    def _log_raw_inputs(self, **inputs: Any) -> None:
        """Log raw inputs for auditability at configured log level."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "vision_input",
            "inputs": {
                k: self._safe_serialize(v) for k, v in inputs.items()
                if v is not None
            }
        }
        
        # Use configured audit level (default: INFO for production auditability)
        self._logger.log(self._audit_log_level, "Raw vision inputs received", extra=log_data)
    
    def _log_normalized_output(self, output: VisionOutput) -> None:
        """Log normalized output for auditability at configured log level."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "vision_output",
            "output_id": output.processing_id,
            "element_count": len(output.elements),
            "screen_dimensions": output.screen_dimensions,
            "input_hash": output.input_hash,
            "model_types_used": list(output.raw_model_outputs.keys())
        }
        
        # Use configured audit level (default: INFO for production auditability)
        self._logger.log(self._audit_log_level, "Normalized vision output created", extra=log_data)
    
    def _safe_serialize(self, obj: Any) -> Any:
        """
        Safely serialize object for logging.
        
        Note: Does not modify or clean the data - only converts to loggable format.
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_serialize(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._safe_serialize(item) for item in obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)  # Fallback for objects
        else:
            try:
                return json.dumps(obj, default=str)
            except (TypeError, ValueError):
                return str(obj)


# Export public interface
__all__ = [
    "VisionClient",
    "VisionOutput",
    "VisionElement",
    "BoundingBox",
    "VisionModelType",
    "CoordinateSystem",
    "AUDIT_LEVEL"
]