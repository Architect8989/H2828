"""
OCR Backend

Raw visual perception backend that captures screens and extracts verbatim text with geometry.
This module contains zero semantics, zero interpretation, and zero intelligence about screen contents.
It simply runs OCR engines on screen captures and returns raw detections.

Absolute constraints:
- NO description of what the screen "is"
- NO inference of UI elements, buttons, menus, or windows
- NO labeling of intent or affordances
- NO cleaning, normalizing, or correcting text
- NO deduplication of results
- NO ranking or scoring of importance
- NO merging outputs across engines
- NO dropping of low-confidence detections
- NO guessing of missing text

This module is pure mechanical perception - boring, auditable, and replaceable.
"""

import asyncio
import logging
import subprocess
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, ValidationError, validator, ConfigDict

# Import OCR engines (if available, fail fast if not)
try:
    import pytesseract
    from paddleocr import PaddleOCR
except ImportError as e:
    raise ImportError(
        f"OCR backend requires pytesseract and paddleocr: {str(e)}"
    ) from e

# Import screen capture (try multiple backends)
try:
    import mss
    import mss.tools
    SCREEN_CAPTURE_BACKEND = "mss"
except ImportError:
    try:
        from PIL import ImageGrab
        SCREEN_CAPTURE_BACKEND = "pil"
    except ImportError:
        raise ImportError(
            "OCR backend requires screen capture backend: install mss or pillow"
        )

logger = logging.getLogger(__name__)
OCR_LOG_LEVEL = logging.INFO


class OCREngine(str, Enum):
    """Enumeration of OCR engines used."""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"


class CoordinateSystem(str, Enum):
    """Coordinate system for bounding boxes."""
    NORMALIZED = "normalized"  # [0, 1] relative to screen
    PIXELS = "pixels"         # Absolute pixel coordinates


class OCRDetection(BaseModel):
    """
    Single OCR detection result.
    
    Contains raw detection exactly as provided by OCR engine.
    No cleaning, correction, or interpretation of text.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core detection data (exactly as provided by engine)
    text: str = Field(..., min_length=0)  # Empty string allowed if engine returns it
    bounding_box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    coordinate_system: CoordinateSystem = CoordinateSystem.PIXELS  # Store raw pixel coordinates
    
    # Engine metadata
    engine: OCREngine
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)  # Raw confidence, not interpreted
    
    # Engine-specific raw data (no interpretation)
    raw_engine_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text', pre=True)
    def preserve_text_exactly(cls, v: Any) -> str:
        """Preserve text exactly as provided, including whitespace."""
        return str(v) if v is not None else ""
    
    # Note: No bounding box validation - accept exactly as provided by engine
    # Malformed geometry is valuable information about OCR engine behavior


class ScreenCapture(BaseModel):
    """Mechanical screen capture data with no interpretation."""
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core capture data
    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    image_data: np.ndarray  # Raw image as numpy array
    
    # Capture metadata (mechanical only)
    capture_method: str
    color_mode: str = Field(default="RGB")
    
    @validator('image_data')
    def validate_image_data(cls, v: np.ndarray) -> np.ndarray:
        """Validate image data is non-empty numpy array."""
        if v.size == 0:
            raise ValueError("Image data must not be empty")
        if len(v.shape) not in (2, 3):
            raise ValueError(f"Image must be 2D (grayscale) or 3D (color), got shape {v.shape}")
        return v


class OCRResult(BaseModel):
    """
    Complete OCR result from multiple engines.
    
    Each engine's results are kept separate and unmodified.
    No merging, ranking, or conflict resolution.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Core identification
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Screen capture used for OCR
    screen_capture: ScreenCapture
    
    # Raw engine results (kept separate, no merging)
    paddleocr_detections: List[OCRDetection] = Field(default_factory=list)
    tesseract_detections: List[OCRDetection] = Field(default_factory=list)
    
    # Engine metadata (mechanical only)
    engine_versions: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: Dict[str, float] = Field(default_factory=dict)
    
    # Processing metadata (no interpretation)
    total_detections: int = Field(0, ge=0)
    
    @validator('total_detections')
    def calculate_total_detections(cls, v: int, values: Dict[str, Any]) -> int:
        """Calculate total detections across all engines."""
        paddleocr = values.get('paddleocr_detections', [])
        tesseract = values.get('tesseract_detections', [])
        return len(paddleocr) + len(tesseract)


class OCRBackend:
    """
    Stateless backend for screen capture and OCR processing.
    
    This class:
    1. Captures screen using available backend
    2. Runs PaddleOCR and Tesseract independently
    3. Returns raw detections from each engine
    4. Captures mechanical timing and metadata
    
    This class does NOT:
    1. Interpret screen contents or meaning
    2. Clean, correct, or normalize text
    3. Merge, rank, or filter detections
    4. Deduplicate or resolve conflicts
    5. Drop low-confidence results
    6. Cache or remember previous captures
    7. Optimize for "usefulness"
    """
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        paddleocr_lang: str = "en",
        ocr_log_level: int = OCR_LOG_LEVEL
    ) -> None:
        """
        Initialize OCR backend with engine configuration.
        
        Args:
            tesseract_path: Optional path to tesseract executable
            paddleocr_lang: Language for PaddleOCR (passed directly, no validation)
            ocr_log_level: Logging level for OCR operations
        
        Raises:
            RuntimeError: If OCR engines cannot be initialized
        """
        self._logger = logging.getLogger(f"{__name__}.OCRBackend")
        self._ocr_log_level = ocr_log_level
        
        # Configure Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize PaddleOCR (no interpretation of language parameter)
        self._paddleocr = PaddleOCR(
            use_angle_cls=False,  # No orientation correction (preserves raw)
            lang=paddleocr_lang,
            show_log=False,  # Suppress internal logs
            use_gpu=False,  # CPU only for deterministic behavior
        )
        
        # Capture engine versions (mechanical only)
        self._engine_versions = self._capture_engine_versions()
        
        # Log initialization
        self._log_initialization()
    
    async def capture_and_ocr(self) -> OCRResult:
        """
        Capture screen and run OCR engines independently.
        
        Returns:
            OCRResult with raw detections from both engines
        
        Raises:
            RuntimeError: If screen capture or OCR fails
            ValueError: If screen capture produces invalid data
        
        Note:
            - No retry on failure
            - No interpretation of results
            - No cleaning or filtering
            - Exactly one capture and OCR pass
        """
        capture_start = datetime.utcnow()
        
        try:
            # Step 1: Capture screen
            screen_capture = await self._capture_screen()
            
            # Step 2: Run OCR engines in parallel (no dependency between them)
            paddleocr_task = asyncio.create_task(
                self._run_paddleocr(screen_capture)
            )
            tesseract_task = asyncio.create_task(
                self._run_tesseract(screen_capture)
            )
            
            # Wait for both engines to complete
            paddleocr_detections, paddleocr_time = await paddleocr_task
            tesseract_detections, tesseract_time = await tesseract_task
            
            # Calculate total processing time
            total_time_ms = (datetime.utcnow() - capture_start).total_seconds() * 1000
            
            # Create result
            result = OCRResult(
                screen_capture=screen_capture,
                paddleocr_detections=paddleocr_detections,
                tesseract_detections=tesseract_detections,
                engine_versions=self._engine_versions,
                processing_time_ms={
                    "paddleocr": paddleocr_time,
                    "tesseract": tesseract_time,
                    "total": total_time_ms,
                },
            )
            
            # Log result metadata (no interpretation)
            self._log_ocr_result(result)
            
            return result
            
        except Exception as e:
            # Log failure and re-raise without handling
            self._log_ocr_failure(e, capture_start)
            raise
    
    async def _capture_screen(self) -> ScreenCapture:
        """Capture current screen using available backend."""
        capture_start = time.time()
        
        try:
            if SCREEN_CAPTURE_BACKEND == "mss":
                return await self._capture_with_mss()
            elif SCREEN_CAPTURE_BACKEND == "pil":
                return await self._capture_with_pil()
            else:
                raise RuntimeError(f"Unknown screen capture backend: {SCREEN_CAPTURE_BACKEND}")
                
        except Exception as e:
            raise RuntimeError(f"Screen capture failed: {str(e)}") from e
    
    async def _capture_with_mss(self) -> ScreenCapture:
        """Capture screen using mss (multi-platform)."""
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]  # Monitor 0 is "all monitors", 1 is primary
            
            # Capture screen
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array (no color conversion, preserve raw)
            img_array = np.array(sct_img)
            
            # MSS returns BGRA, convert to RGB for consistency
            if img_array.shape[2] == 4:  # BGRA
                img_array = img_array[..., :3]  # Drop alpha
                img_array = img_array[..., ::-1]  # BGR to RGB
            
            return ScreenCapture(
                width=img_array.shape[1],
                height=img_array.shape[0],
                image_data=img_array,
                capture_method="mss",
                color_mode="RGB",
            )
    
    async def _capture_with_pil(self) -> ScreenCapture:
        """Capture screen using PIL (fallback)."""
        from PIL import Image
        
        # Capture screen
        pil_img = ImageGrab.grab()
        
        # Convert to numpy array
        img_array = np.array(pil_img)
        
        return ScreenCapture(
            width=img_array.shape[1],
            height=img_array.shape[0],
            image_data=img_array,
            capture_method="pil",
            color_mode="RGB",
        )
    
    async def _run_paddleocr(self, capture: ScreenCapture) -> Tuple[List[OCRDetection], float]:
        """Run PaddleOCR on screen capture and return raw detections."""
        start_time = time.time()
        
        try:
            # Run PaddleOCR (results are exactly as returned)
            result = self._paddleocr.ocr(capture.image_data, cls=False)
            
            # Process raw results (no cleaning, filtering, or interpretation)
            detections = []
            
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        # Extract bounding box coordinates
                        bbox = line[0]
                        text_data = line[1]
                        
                        # Get text (preserve exactly)
                        text = text_data[0] if isinstance(text_data, (list, tuple)) else str(text_data)
                        
                        # Get confidence (if available)
                        confidence = None
                        if isinstance(text_data, (list, tuple)) and len(text_data) > 1:
                            try:
                                confidence = float(text_data[1])
                            except (ValueError, TypeError):
                                pass
                        
                        # Store raw pixel coordinates exactly as provided by PaddleOCR
                        # No normalization, no validation of geometry correctness
                        bbox_array = np.array(bbox)
                        x_coords = bbox_array[:, 0]
                        y_coords = bbox_array[:, 1]
                        
                        # Take min/max to form rectangle (but preserve any engine-provided anomalies)
                        x1 = float(np.min(x_coords))
                        y1 = float(np.min(y_coords))
                        x2 = float(np.max(x_coords))
                        y2 = float(np.max(y_coords))
                        
                        detection = OCRDetection(
                            text=text,
                            bounding_box=(x1, y1, x2, y2),
                            coordinate_system=CoordinateSystem.PIXELS,  # Raw pixel coordinates
                            engine=OCREngine.PADDLEOCR,
                            confidence=confidence,
                            raw_engine_data={"raw_bbox": bbox},
                        )
                        detections.append(detection)
            
            processing_time = (time.time() - start_time) * 1000
            return detections, processing_time
            
        except Exception as e:
            raise RuntimeError(f"PaddleOCR failed: {str(e)}") from e
    
    async def _run_tesseract(self, capture: ScreenCapture) -> Tuple[List[OCRDetection], float]:
        """Run Tesseract on screen capture and return raw detections."""
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image for Tesseract
            from PIL import Image
            
            pil_image = Image.fromarray(capture.image_data)
            
            # Run Tesseract with minimal configuration (no language model biases)
            # Output format: dict with bounding boxes and text
            data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config="--psm 3",  # Page segmentation: auto, no assumptions
            )
            
            # Process raw results (no cleaning, filtering, or interpretation)
            detections = []
            
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                # Extract data (preserve exactly)
                text = data['text'][i]
                conf = data['conf'][i]
                
                # No skipping of any detections - record everything exactly
                # Empty text with confidence -1 is still a valid observation
                
                # Get bounding box in pixels (no normalization)
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Store raw pixel coordinates without normalization
                x1 = float(x)
                y1 = float(y)
                x2 = float(x + w)
                y2 = float(y + h)
                
                # Convert confidence from percentage to [0, 1]
                confidence = None
                if conf >= 0:
                    confidence = conf / 100.0
                else:
                    # Negative confidence is preserved as None (raw data available in raw_engine_data)
                    confidence = None
                
                detection = OCRDetection(
                    text=text,
                    bounding_box=(x1, y1, x2, y2),
                    coordinate_system=CoordinateSystem.PIXELS,  # Raw pixel coordinates
                    engine=OCREngine.TESSERACT,
                    confidence=confidence,
                    raw_engine_data={
                        "level": data['level'][i],
                        "page_num": data['page_num'][i],
                        "block_num": data['block_num'][i],
                        "par_num": data['par_num'][i],
                        "line_num": data['line_num'][i],
                        "word_num": data['word_num'][i],
                        "raw_confidence": conf,  # Preserve original -1 values
                    },
                )
                detections.append(detection)
            
            processing_time = (time.time() - start_time) * 1000
            return detections, processing_time
            
        except Exception as e:
            raise RuntimeError(f"Tesseract failed: {str(e)}") from e
    
    def _capture_engine_versions(self) -> Dict[str, str]:
        """Capture OCR engine versions (mechanical only)."""
        versions = {}
        
        # Get Tesseract version
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            versions["tesseract"] = str(tesseract_version)
        except Exception as e:
            self._logger.warning(f"Could not get Tesseract version: {e}")
            versions["tesseract"] = "unknown"
        
        # Get PaddleOCR version (via package metadata)
        try:
            import paddleocr
            versions["paddleocr"] = getattr(paddleocr, "__version__", "unknown")
        except Exception as e:
            self._logger.warning(f"Could not get PaddleOCR version: {e}")
            versions["paddleocr"] = "unknown"
        
        # Get screen capture backend version
        try:
            if SCREEN_CAPTURE_BACKEND == "mss":
                import mss
                versions["mss"] = mss.__version__
            elif SCREEN_CAPTURE_BACKEND == "pil":
                from PIL import ImageGrab
                import PIL
                versions["pil"] = PIL.__version__
        except Exception as e:
            self._logger.warning(f"Could not get screen capture version: {e}")
            versions[SCREEN_CAPTURE_BACKEND] = "unknown"
        
        return versions
    
    def _log_initialization(self) -> None:
        """Log backend initialization (mechanical metadata only)."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "ocr_backend_initialized",
            "screen_capture_backend": SCREEN_CAPTURE_BACKEND,
            "engine_versions": self._engine_versions,
        }
        
        self._logger.log(
            self._ocr_log_level,
            "OCR backend initialized",
            extra=log_data
        )
    
    def _log_ocr_result(self, result: OCRResult) -> None:
        """Log OCR result metadata (no interpretation)."""
        log_data = {
            "timestamp": result.timestamp.isoformat(),
            "operation": "ocr_completed",
            "result_id": result.result_id,
            "capture_id": result.screen_capture.capture_id,
            "screen_resolution": f"{result.screen_capture.width}x{result.screen_capture.height}",
            "paddleocr_detections": len(result.paddleocr_detections),
            "tesseract_detections": len(result.tesseract_detections),
            "total_detections": result.total_detections,
            "processing_time_ms": result.processing_time_ms,
        }
        
        self._logger.log(
            self._ocr_log_level,
            "OCR processing completed",
            extra=log_data
        )
    
    def _log_ocr_failure(self, error: Exception, capture_start: datetime) -> None:
        """Log OCR failure without handling."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "ocr_failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "capture_start": capture_start.isoformat(),
        }
        
        self._logger.log(
            self._ocr_log_level,
            "OCR processing failed",
            extra=log_data
        )


# Export public interface
__all__ = [
    "OCRBackend",
    "OCRResult",
    "OCRDetection",
    "ScreenCapture",
    "OCREngine",
    "CoordinateSystem",
    "OCR_LOG_LEVEL",
]