"""
OCR Runner

Mechanical backend for converting raw screen pixels into OCR output.
This module executes OCR without interpreting or modifying the results.

Absolute constraints:
- NO semantic interpretation of text
- NO UI classification, ranking, filtering, or cleanup
- NO normalization to internal schemas (handled elsewhere)
- NO timestamps, logging, retries, caching, or state retention
- NO heuristics, task logic, or policy decisions
- NO augmentation of backend output with metadata

This module is a mechanical perception backend only - boring, auditable, and replaceable.
"""

from typing import Any, Dict
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

from perception.screen_capture import ScreenFrame


class OCRError(Exception):
    """Exception raised when OCR backend execution fails."""
    pass


class OCRRunner:
    """
    Stateless OCR backend for converting screen pixels to text detections.
    
    This class:
    1. Converts raw screen pixels to OCR input format
    2. Executes OCR backend on the image
    3. Returns raw detection results exactly as provided by backend
    
    This class does NOT:
    1. Interpret meaning or context of detected text
    2. Filter, rank, or clean OCR results
    3. Normalize coordinates, text formatting, or confidence values
    4. Cache results or maintain state between calls
    5. Log, timestamp, or produce side effects
    6. Retry failed OCR operations
    7. Make policy decisions about OCR configuration
    8. Augment backend output with external metadata
    
    Note: Engine state is maintained for performance but contains no semantic state.
    """
    
    def __init__(self, backend: str = "paddleocr", **backend_config) -> None:
        """
        Initialize OCR runner with specified backend.
        
        Args:
            backend: OCR backend to use ("paddleocr" or "tesseract")
            **backend_config: Backend-specific configuration passed directly to engine
        
        Raises:
            ValueError: If backend is not supported
        
        Note:
            - No state retention between calls
            - No side effects during initialization
            - Engine state is opaque and contains no semantic information
        """
        if backend not in ("paddleocr", "tesseract"):
            raise ValueError(f"Unsupported OCR backend: {backend}. Must be 'paddleocr' or 'tesseract'")
        
        self._backend_type = backend
        
        # Initialize backend engine with provided configuration
        # Engine state is mechanical cache only, no semantic state
        if backend == "paddleocr":
            self._engine = PaddleOCR(**backend_config)
        else:
            # Tesseract requires no initialization
            # Configuration is passed at runtime via pytesseract
            self._engine = None
            self._tesseract_config = backend_config
    
    def run(self, frame: ScreenFrame) -> Dict[str, Any]:
        """
        Execute OCR on raw screen frame.
        
        Args:
            frame: Raw screen frame with pixel data
        
        Returns:
            Raw OCR output exactly as provided by backend
        
        Raises:
            ValueError: If frame is invalid
            OCRError: If OCR execution fails
        
        Note:
            - No preprocessing beyond mechanical format conversion
            - No filtering of results (all detections returned)
            - No normalization of confidence or coordinates
            - No augmentation with external metadata
            - No retry on failure
            - Exact one-time execution only
        """
        if not isinstance(frame, ScreenFrame):
            raise ValueError(f"frame must be ScreenFrame, got {type(frame)}")
        
        if not frame.pixels or frame.width <= 0 or frame.height <= 0:
            raise ValueError(f"Invalid frame dimensions: {frame.width}x{frame.height}")
        
        # Convert raw BGRA pixels to RGB for OCR processing
        # This is mechanical format conversion only, no enhancement
        rgb_image = self._convert_bgra_to_rgb(frame)
        
        # Execute OCR using configured backend
        if self._backend_type == "paddleocr":
            raw_output = self._run_paddleocr(rgb_image)
        else:
            raw_output = self._run_tesseract(rgb_image)
        
        return raw_output
    
    def _convert_bgra_to_rgb(self, frame: ScreenFrame) -> np.ndarray:
        """
        Convert BGRA pixel data to RGB numpy array.
        
        Note: Mechanical format conversion only, no image enhancement.
        Uses NumPy only (no OpenCV dependency).
        """
        # Reshape raw bytes to image dimensions (BGRA format)
        bgra_array = np.frombuffer(frame.pixels, dtype=np.uint8)
        bgra_array = bgra_array.reshape((frame.height, frame.width, 4))
        
        # Extract B, G, R channels (drop Alpha)
        b = bgra_array[:, :, 0]
        g = bgra_array[:, :, 1]
        r = bgra_array[:, :, 2]
        
        # Stack as RGB
        rgb_array = np.stack([r, g, b], axis=2)
        
        return rgb_array
    
    def _run_paddleocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute OCR using PaddleOCR backend.
        
        Returns raw PaddleOCR results exactly as provided.
        Converts quadrilateral boxes to [x1,y1,x2,y2] format by taking min/max bounds.
        This is mechanical transformation only, no semantic interpretation.
        """
        try:
            # Execute OCR with current engine configuration
            ocr_result = self._engine.ocr(image, cls=False)
            
            # Format results without filtering or normalization
            text_elements = []
            
            if ocr_result is not None:
                for line in ocr_result:
                    if line is not None:
                        for detection in line:
                            if detection is not None and len(detection) >= 2:
                                # Extract bounding box and text exactly as provided
                                box_points, (text, confidence) = detection
                                
                                # Convert quadrilateral to axis-aligned bounding box
                                # This is mechanical transformation, not interpretation
                                x_coords = [point[0] for point in box_points]
                                y_coords = [point[1] for point in box_points]
                                flat_box = [
                                    min(x_coords),  # x1
                                    min(y_coords),  # y1
                                    max(x_coords),  # x2
                                    max(y_coords)   # y2
                                ]
                                
                                text_elements.append({
                                    "text": text,  # Preserve exactly
                                    "bounding_box": flat_box,
                                    "confidence": confidence,  # Preserve exactly
                                    "coordinate_system": "pixels",
                                    "original_quadrilateral": box_points  # Keep original data
                                })
            
            return {
                "text_elements": text_elements
            }
        
        except Exception as e:
            # Raise typed error preserving backend context
            raise OCRError(f"PaddleOCR execution failed: {str(e)}") from e
    
    def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute OCR using Tesseract backend.
        
        Returns raw Tesseract results exactly as provided.
        No filtering by confidence or text content.
        No normalization of confidence values.
        """
        try:
            # Get Tesseract data including all detections
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                **self._tesseract_config
            )
            
            # Format results without filtering or normalization
            text_elements = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # Preserve text exactly, including empty strings
                text = data['text'][i]
                
                # Preserve confidence exactly as provided by Tesseract
                conf = data['conf'][i]
                
                # Extract bounding box coordinates
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Format as [x1,y1,x2,y2]
                bounding_box = [x, y, x + w, y + h]
                
                text_elements.append({
                    "text": text,  # Preserve exactly
                    "bounding_box": bounding_box,
                    "confidence": conf,  # Preserve exactly (range: 0-100)
                    "coordinate_system": "pixels"
                })
            
            return {
                "text_elements": text_elements
            }
        
        except Exception as e:
            # Raise typed error preserving backend context
            raise OCRError(f"Tesseract execution failed: {str(e)}") from e


# Export public interface
__all__ = ["OCRRunner", "OCRError"]