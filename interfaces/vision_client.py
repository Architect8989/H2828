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
- NO logging or side effects
- NO timestamp generation (timestamps may be accepted if provided externally)
- NO policy decisions

Minimal structural normalization is performed to satisfy schema requirements.
UUIDs are generated only to ensure structural identity, not semantic meaning.

This module is infrastructure only - boring, auditable, and replaceable.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator, ConfigDict


class VisionModelType(str, Enum):
    """Enumeration of supported vision model types."""
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
    - (x1, y1): top-left corner (model may provide reversed)
    - (x2, y2): bottom-right corner (model may provide reversed)
    
    Coordinates are preserved exactly as provided by the model.
    No interpretation or validation of ordering or ranges.
    Confidence is validated for numeric convertibility only.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED
    confidence: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate confidence for numeric convertibility only."""
        if self.confidence is not None:
            try:
                _ = float(self.confidence)
            except (TypeError, ValueError):
                raise ValueError(f"Confidence must be convertible to float. Got: {self.confidence}")


class VisionElement(BaseModel):
    """Base schema for vision model outputs."""
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core identification
    element_id: str = Field(default_factory=lambda: str(uuid4()))
    source_model: VisionModelType
    timestamp: Optional[datetime] = None
    
    # Spatial information
    bounding_box: Optional[BoundingBox] = None
    spatial_confidence: Optional[float] = None  # No range validation
    
    # Raw model outputs
    raw_text: Optional[str] = None
    raw_labels: List[str] = Field(default_factory=list)
    raw_attributes: Dict[str, Any] = Field(default_factory=dict)
    
    # Relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)
    sibling_ids: List[str] = Field(default_factory=list)
    
    # Model-specific metadata
    model_confidence: Optional[float] = None  # No range validation
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('raw_text', pre=True)
    def preserve_text_exactly(cls, v: Optional[str]) -> Optional[str]:
        """Preserve text exactly as provided."""
        return v if v is None else str(v)
    
    @validator('raw_labels', 'child_ids', 'sibling_ids', pre=True)
    def ensure_list(cls, v: Any) -> List:
        """Ensure list fields are always lists (structural normalization)."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)
    
    @validator('source_model', pre=True)
    def validate_source_model(cls, v: Any) -> VisionModelType:
        """Convert string to VisionModelType."""
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
    """Complete normalized output from vision processing."""
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Identification - UUID generated for structural identity only
    processing_id: str = Field(default_factory=lambda: str(uuid4()))
    screen_timestamp: Optional[datetime] = None
    screen_dimensions: Optional[tuple[int, int]] = None
    
    # Raw elements from all vision models
    elements: List[VisionElement] = Field(default_factory=list)
    
    # Model outputs in their original structure
    raw_model_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit trail
    input_hash: Optional[str] = None
    version: str = "1.0.0"
    
    @validator('screen_dimensions')
    def validate_screen_dimensions(cls, v: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """Validate screen dimensions are positive integers."""
        if v is None:
            return None
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"Screen dimensions must be positive. Got: {width}x{height}")
        return v


class VisionClient:
    """Stateless client for processing raw vision model outputs."""
    
    def __init__(self) -> None:
        """Stateless client; no initialization required."""
        pass
    
    def process_vision_output(
        self,
        *,
        ocr_output: Optional[Dict[str, Any]] = None,
        ui_parser_output: Optional[Dict[str, Any]] = None,
        vlm_output: Optional[Dict[str, Any]] = None,
        object_detector_output: Optional[Dict[str, Any]] = None,
        screen_dimensions: Optional[tuple[int, int]] = None,
        screen_timestamp: Optional[datetime] = None,
        input_hash: Optional[str] = None
    ) -> VisionOutput:
        """
        Process raw vision model outputs into normalized schema.
        
        Returns:
            Normalized VisionOutput containing all elements exactly as provided.
        
        Raises:
            ValueError: If any input fails structural validation or no outputs provided.
        """
        if not any([ocr_output, ui_parser_output, vlm_output, object_detector_output]):
            raise ValueError("At least one vision output must be provided")
        
        raw_outputs = {}
        elements: List[VisionElement] = []
        
        if ocr_output is not None:
            ocr_elements = self._extract_ocr_elements(ocr_output, screen_timestamp)
            elements.extend(ocr_elements)
            raw_outputs["ocr"] = ocr_output
        
        if ui_parser_output is not None:
            ui_elements = self._extract_ui_elements(ui_parser_output, screen_timestamp)
            elements.extend(ui_elements)
            raw_outputs["ui_parser"] = ui_parser_output
        
        if vlm_output is not None:
            vlm_elements = self._extract_vlm_elements(vlm_output, screen_timestamp)
            elements.extend(vlm_elements)
            raw_outputs["vlm"] = vlm_output
        
        if object_detector_output is not None:
            detector_elements = self._extract_detector_elements(object_detector_output, screen_timestamp)
            elements.extend(detector_elements)
            raw_outputs["object_detector"] = object_detector_output
        
        return VisionOutput(
            screen_dimensions=screen_dimensions,
            screen_timestamp=screen_timestamp,
            elements=elements,
            raw_model_outputs=raw_outputs,
            input_hash=input_hash
        )
    
    # Helper methods perform minimal structural validation required to construct schemas; 
    # no semantic validation is performed.
    
    def _extract_ocr_elements(
        self, 
        ocr_output: Dict[str, Any], 
        timestamp: Optional[datetime]
    ) -> List[VisionElement]:
        """Extract OCR elements with structural validation only."""
        try:
            elements = []
            text_elements = ocr_output.get("text_elements", [])
            
            for i, text_element in enumerate(text_elements):
                if "text" not in text_element:
                    raise ValueError(f"OCR element {i} missing 'text' field")
                
                bbox = None
                if "bounding_box" in text_element:
                    bbox_data = text_element["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"OCR element {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"OCR element {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    # Validate coordinate system with clear error
                    cs_value = text_element.get("coordinate_system", "normalized")
                    try:
                        cs = CoordinateSystem(cs_value)
                    except ValueError:
                        raise ValueError(f"Invalid coordinate_system: {cs_value}")
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=cs,
                        confidence=text_element.get("bbox_confidence")
                    )
                
                element = VisionElement(
                    source_model=VisionModelType.OCR,
                    timestamp=timestamp,
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
    
    def _extract_ui_elements(
        self, 
        ui_output: Dict[str, Any], 
        timestamp: Optional[datetime]
    ) -> List[VisionElement]:
        """Extract UI elements with structural validation only."""
        try:
            elements = []
            ui_elements = ui_output.get("ui_elements", [])
            
            for i, ui_element in enumerate(ui_elements):
                bbox = None
                if "bounding_box" in ui_element:
                    bbox_data = ui_element["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"UI element {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"UI element {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    # Validate coordinate system with clear error
                    cs_value = ui_element.get("coordinate_system", "normalized")
                    try:
                        cs = CoordinateSystem(cs_value)
                    except ValueError:
                        raise ValueError(f"Invalid coordinate_system: {cs_value}")
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=cs,
                        confidence=ui_element.get("bbox_confidence")
                    )
                
                raw_labels = ui_element.get("labels", [])
                if isinstance(raw_labels, str):
                    raw_labels = [raw_labels]
                
                element = VisionElement(
                    source_model=VisionModelType.UI_PARSER,
                    timestamp=timestamp,
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
    
    def _extract_vlm_elements(
        self, 
        vlm_output: Dict[str, Any], 
        timestamp: Optional[datetime]
    ) -> List[VisionElement]:
        """Extract VLM elements with structural validation only."""
        try:
            elements = []
            descriptions = vlm_output.get("descriptions", [])
            
            for i, description in enumerate(descriptions):
                bbox = None
                if "bounding_box" in description:
                    bbox_data = description["bounding_box"]
                    if len(bbox_data) != 4:
                        raise ValueError(
                            f"VLM description {i} bounding_box must have 4 values. "
                            f"Got: {bbox_data}"
                        )
                    
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"VLM description {i} bounding_box coordinates must be numeric. "
                            f"Got: {bbox_data}. Error: {e}"
                        )
                    
                    # Validate coordinate system with clear error
                    cs_value = description.get("coordinate_system", "normalized")
                    try:
                        cs = CoordinateSystem(cs_value)
                    except ValueError:
                        raise ValueError(f"Invalid coordinate_system: {cs_value}")
                    
                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        coordinate_system=cs,
                        confidence=description.get("bbox_confidence")
                    )
                
                element = VisionElement(
                    source_model=VisionModelType.VLM,
                    timestamp=timestamp,
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
    
    def _extract_detector_elements(
        self, 
        detector_output: Dict[str, Any], 
        timestamp: Optional[datetime]
    ) -> List[VisionElement]:
        """Extract object detection elements with structural validation only."""
        try:
            elements = []
            detections = detector_output.get("detections", [])
            
            for i, detection in enumerate(detections):
                if "bounding_box" not in detection:
                    raise ValueError(f"Detection {i} missing 'bounding_box' field")
                
                bbox_data = detection["bounding_box"]
                if len(bbox_data) != 4:
                    raise ValueError(
                        f"Detection {i} bounding_box must have 4 values. "
                        f"Got: {bbox_data}"
                    )
                
                try:
                    x1, y1, x2, y2 = [float(coord) for coord in bbox_data]
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Detection {i} bounding_box coordinates must be numeric. "
                        f"Got: {bbox_data}. Error: {e}"
                    )
                
                # Validate coordinate system with clear error
                cs_value = detection.get("coordinate_system", "normalized")
                try:
                    cs = CoordinateSystem(cs_value)
                except ValueError:
                    raise ValueError(f"Invalid coordinate_system: {cs_value}")
                
                bbox = BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    coordinate_system=cs,
                    confidence=detection.get("bbox_confidence")
                )
                
                element = VisionElement(
                    source_model=VisionModelType.OBJECT_DETECTOR,
                    timestamp=timestamp,
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


__all__ = [
    "VisionClient",
    "VisionOutput",
    "VisionElement",
    "BoundingBox",
    "VisionModelType",
    "CoordinateSystem"
]