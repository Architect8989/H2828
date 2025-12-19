"""
Life Loop

Pure orchestration loop for the Environment Mastery Engine.
This module wires existing interfaces together without adding intelligence.
It contains zero logic about what actions to take or how to interpret results.

Absolute constraints:
- NO decision-making about what actions to take
- NO retry logic or fallback mechanisms
- NO success/failure inference or interpretation
- NO optimization of sequences or timing
- NO environment assumptions or domain knowledge
- NO hidden state maintenance
- NO logging or side effects
- NO timestamp generation
- NO policy decisions
- NO learning or memory across iterations
- NO rate limiting or metrics collection

This module is a pure wiring harness - boring, auditable, and replaceable.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from interfaces.vision_client import VisionClient
from interfaces.brain_client import BrainClient
from execution.action_executor import ActionExecutor, ActionSequence


class LifeLoop:
    """
    Stateless orchestration loop for continuous environment interaction.
    
    This class:
    1. Wires vision → brain → action execution
    2. Runs a continuous interaction loop
    3. Propagates exceptions without interpretation
    
    This class does NOT:
    1. Make decisions about what actions to take
    2. Retry failed operations
    3. Optimize interaction timing
    4. Interpret success or failure
    5. Maintain state between iterations
    6. Log or produce side effects
    7. Generate timestamps or policy
    8. Assume environment semantics
    
    Note: Returned ActionSequence is for mechanical observability only; carries no semantic meaning.
    """
    
    def __init__(
        self,
        vision_client: VisionClient,
        brain_client: BrainClient,
        action_executor: ActionExecutor
    ) -> None:
        """
        Initialize orchestration loop with existing interfaces.
        
        Args:
            vision_client: Stateless vision output normalizer
            brain_client: Stateless action inference engine
            action_executor: Stateless action sequence executor
        
        Raises:
            TypeError: If any interface is of incorrect type
        """
        if not isinstance(vision_client, VisionClient):
            raise TypeError(
                f"vision_client must be VisionClient instance, got {type(vision_client)}"
            )
        if not isinstance(brain_client, BrainClient):
            raise TypeError(
                f"brain_client must be BrainClient instance, got {type(brain_client)}"
            )
        if not isinstance(action_executor, ActionExecutor):
            raise TypeError(
                f"action_executor must be ActionExecutor instance, got {type(action_executor)}"
            )
        
        self._vision_client = vision_client
        self._brain_client = brain_client
        self._action_executor = action_executor
    
    def run_iteration(
        self,
        ocr_output: Optional[Dict[str, Any]] = None,
        ui_parser_output: Optional[Dict[str, Any]] = None,
        vlm_output: Optional[Dict[str, Any]] = None,
        object_detector_output: Optional[Dict[str, Any]] = None,
        screen_dimensions: Optional[tuple[int, int]] = None,
        screen_timestamp: Optional[datetime] = None,
        input_hash: Optional[str] = None
    ) -> ActionSequence:
        """
        Execute one iteration of perception → cognition → action.
        
        Args:
            ocr_output: Raw OCR model output (must be processed externally)
            ui_parser_output: Raw UI parser output (must be processed externally)
            vlm_output: Raw VLM output (must be processed externally)
            object_detector_output: Raw object detector output (must be processed externally)
            screen_dimensions: (width, height) in pixels
            screen_timestamp: Optional timestamp of screen capture (must be provided externally)
            input_hash: Optional hash of input for reproducibility
        
        Returns:
            ActionSequence that was executed (mechanical observability only)
        
        Raises:
            Exception: If any component fails (propagated without interpretation)
        
        Note:
            - Vision model execution must happen outside this boundary
            - No retry on failure
            - No interpretation of results
            - No state maintained between calls
            - No logging or side effects
            - Exact one-pass execution only
        """
        # Normalize vision outputs without interpretation
        vision_output = self._vision_client.process_vision_output(
            ocr_output=ocr_output,
            ui_parser_output=ui_parser_output,
            vlm_output=vlm_output,
            object_detector_output=object_detector_output,
            screen_dimensions=screen_dimensions,
            screen_timestamp=screen_timestamp,
            input_hash=input_hash
        )
        
        # Infer action sequence without interpretation
        action_sequence = self._brain_client.infer(vision_output)
        
        # Execute sequence without interpretation
        self._action_executor.execute_sequence(action_sequence)
        
        return action_sequence


# Export public interface
__all__ = ["LifeLoop"]
