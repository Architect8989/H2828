"""
Brain client - Stateless reasoning interface for Environment Mastery Engine.

This module provides a deterministic, zero-intelligence socket between
perception systems and reasoning backends. It contains no cognition,
no goals, no planning, and no environment awareness.

Key invariants:
1. One call in → one call out
2. No memory of previous calls
3. No modification of reasoning content
4. Fail immediately on any misconfiguration
5. No silent behavior or hidden state
"""

import json
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass
import httpx

from pydantic import BaseModel, ValidationError, Field, ConfigDict

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class DeepSeekConfig:
    """Explicit configuration for DeepSeek backend.
    
    Frozen to prevent mutation.
    Contains only authentication and routing parameters.
    """
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"


class BackendType(str, Enum):
    """Explicit backend selection enum.
    
    Only supported backends are enumerated.
    No placeholders or future hooks.
    """
    DEEPSEEK = "deepseek"


@dataclass(frozen=True)
class BrainClientConfig:
    """Complete configuration for brain client.
    
    All fields are required to prevent ambiguous behavior.
    Frozen to enforce statelessness.
    """
    backend_type: BackendType
    deepseek_config: Optional[DeepSeekConfig] = None
    timeout_seconds: float = 30.0
    max_tokens: int = 4000
    temperature: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate configuration at construction.
        
        Raises:
            ValueError: If configuration is invalid or inconsistent
        """
        if self.backend_type == BackendType.DEEPSEEK and not self.deepseek_config:
            raise ValueError("DeepSeek backend requires deepseek_config")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")


# -------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------

class PerceivedState(BaseModel):
    """Current system state as perceived by upstream systems.
    
    Contains no validation that implies correct interpretation.
    """
    model_config = ConfigDict(extra='forbid')
    
    sensory_inputs: dict[str, Any] = Field(
        description="Raw sensory inputs from perception systems",
        min_items=0
    )
    processed_observations: dict[str, Any] = Field(
        description="Processed observations from perception pipeline",
        min_items=0
    )
    telemetry: dict[str, float] = Field(
        description="System telemetry readings",
        min_items=0
    )


class RecentAction(BaseModel):
    """Record of a recently executed action.
    
    Contains only execution record, no success/failure semantics.
    Timestamps are opaque data passed through from upstream systems.
    """
    model_config = ConfigDict(extra='forbid')
    
    action_type: str = Field(description="Type identifier for the action")
    parameters: dict[str, Any] = Field(
        description="Parameters used in action execution",
        min_items=0
    )
    timestamp: str = Field(description="Opaque timestamp from upstream systems")


class MemorySnapshot(BaseModel):
    """Short-term memory contents at current moment.
    
    Passively contains whatever memory system provides.
    No implication about relevance or importance.
    """
    model_config = ConfigDict(extra='forbid')
    
    active_recall: dict[str, Any] = Field(
        description="Currently active memory recalls",
        min_items=0
    )
    recent_patterns: dict[str, Any] = Field(
        description="Recently observed patterns",
        min_items=0
    )
    contextual_tags: dict[str, Any] = Field(
        description="Contextual tags applied by memory system",
        min_items=0
    )


class BrainInput(BaseModel):
    """Complete input structure sent to reasoning model.
    
    All fields are required to enforce explicit communication.
    No defaults imply behavior.
    """
    model_config = ConfigDict(extra='forbid')
    
    perceived_state: PerceivedState = Field(
        description="Current perceived state of the system"
    )
    recent_actions: list[RecentAction] = Field(
        description="Recently executed actions in chronological order",
        min_items=0
    )
    memory_snapshot: MemorySnapshot = Field(
        description="Current short-term memory contents"
    )
    request_id: str = Field(
        description="Unique identifier for this reasoning request",
        min_length=1
    )


# Type alias for clarity in backend interface
RawModelResponse = dict[str, Any]


class ModelResponse(BaseModel):
    """Expected response structure from reasoning model.
    
    Validates form only, not content correctness, safety, or appropriateness.
    No constraints are placed on content richness of untrusted model output.
    """
    model_config = ConfigDict(extra='forbid')
    
    reasoning_artifacts: dict[str, Any] = Field(
        description="Model's internal reasoning artifacts"
    )
    selected_actions: list[dict[str, Any]] = Field(
        description="Actions to execute, in proposed order",
        min_items=0
    )
    attention_focus: dict[str, float] = Field(
        description="Attention weights for next perception cycle"
    )
    memory_updates: dict[str, Any] = Field(
        description="Proposed updates to memory systems",
        min_items=0
    )
    self_reported_confidence: float = Field(
        description="Model's self-reported confidence score",
        ge=0.0,
        le=1.0
    )


# -------------------------------------------------------------------
# Backend Protocol and Implementations
# -------------------------------------------------------------------

class BackendError(Exception):
    """Base exception for backend failures.
    
    Raised immediately on any backend communication issue.
    """


class BackendTimeoutError(BackendError):
    """Raised when backend call exceeds timeout."""


class BackendValidationError(BackendError):
    """Raised when backend returns invalid response."""


class ModelBackend:
    """Abstract base class for model backends.
    
    Concrete implementations must not modify, filter, or interpret
    input/output content.
    
    Note: Timeout policy is owned by the caller; backend only applies 
    the provided timeout value.
    """
    
    def generate(
        self, 
        input_data: dict[str, Any], 
        timeout_seconds: float,
        max_tokens: int,
        temperature: float
    ) -> RawModelResponse:
        """Generate response from reasoning model.
        
        Args:
            input_data: Validated BrainInput as dictionary
            timeout_seconds: Maximum time allowed for the call (policy owned by caller)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for generation
            
        Returns:
            Raw model response as dictionary
            
        Raises:
            BackendError: On any communication or validation failure
            BackendTimeoutError: If call exceeds provided timeout
            
        Note:
            Must not apply retries, caching, or content modification.
        """
        raise NotImplementedError


class DeepSeekBackend(ModelBackend):
    """Concrete backend for DeepSeek API.
    
    Contains zero prompt engineering, zero content modification,
    and zero retry logic.
    
    Note: Timeout policy is owned by the caller; backend only applies 
    the provided timeout value.
    """
    
    def __init__(self, config: DeepSeekConfig):
        """Initialize DeepSeek backend.
        
        Args:
            config: Complete DeepSeek configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not config.api_key:
            raise ValueError("DeepSeek API key is required")
        
        self.config = config
    
    def generate(
        self, 
        input_data: dict[str, Any], 
        timeout_seconds: float,
        max_tokens: int,
        temperature: float
    ) -> RawModelResponse:
        """Call DeepSeek API with raw input.
        
        Args:
            input_data: Validated BrainInput as dictionary
            timeout_seconds: Maximum time allowed for the call (policy owned by caller)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for generation
            
        Returns:
            Raw API response as dictionary
            
        Raises:
            BackendError: On any API or network failure
            BackendTimeoutError: If call exceeds provided timeout
            BackendValidationError: If response format is invalid
        """
        # Create a new HTTP client for each request
        with httpx.Client(timeout=timeout_seconds) as client:
            try:
                # Convert input to JSON string - no modification, no sanitization
                request_body = {
                    "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(input_data)
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Single attempt only - no retries
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=request_body,
                    headers=headers
                )
                
                # Check HTTP status
                response.raise_for_status()
                
                # Parse and validate response structure
                result = response.json()
                
                # Defensive parsing of external API response
                try:
                    if not isinstance(result, dict):
                        raise BackendValidationError(
                            f"Backend response is not a dictionary: {type(result)}"
                        )
                    
                    if "choices" not in result:
                        raise BackendValidationError("Missing 'choices' in backend response")
                    
                    if not isinstance(result["choices"], list) or len(result["choices"]) == 0:
                        raise BackendValidationError(
                            f"'choices' must be non-empty list, got {type(result['choices'])}"
                        )
                    
                    first_choice = result["choices"][0]
                    if not isinstance(first_choice, dict):
                        raise BackendValidationError(
                            f"First choice is not a dictionary: {type(first_choice)}"
                        )
                    
                    if "message" not in first_choice:
                        raise BackendValidationError("Missing 'message' in choice")
                    
                    message = first_choice["message"]
                    if not isinstance(message, dict):
                        raise BackendValidationError(
                            f"Message is not a dictionary: {type(message)}"
                        )
                    
                    if "content" not in message:
                        raise BackendValidationError("Missing 'content' in message")
                    
                    content_str = message["content"]
                    if not isinstance(content_str, str):
                        raise BackendValidationError(
                            f"Content is not a string: {type(content_str)}"
                        )
                    
                    # Parse JSON with explicit error handling
                    try:
                        return json.loads(content_str)
                    except json.JSONDecodeError as e:
                        raise BackendValidationError(
                            f"Backend response content is not valid JSON: {str(e)}"
                        ) from e
                    
                except (KeyError, IndexError, TypeError) as e:
                    raise BackendValidationError(
                        f"Backend response structure validation failed: {str(e)}"
                    ) from e
                    
            except httpx.TimeoutException as e:
                raise BackendTimeoutError(
                    f"Backend timeout after {timeout_seconds}s"
                ) from e
            except httpx.HTTPStatusError as e:
                raise BackendError(
                    f"Backend HTTP error: {e.response.status_code}"
                ) from e
            except httpx.RequestError as e:
                raise BackendError(
                    f"Backend communication error: {str(e)}"
                ) from e


# -------------------------------------------------------------------
# Main Client
# -------------------------------------------------------------------

class BrainClient:
    """Stateless interface to reasoning model.
    
    This class contains:
    - Zero intelligence
    - Zero behavior
    - Zero assumptions
    - Zero hidden state
    
    All reasoning resides in the configured backend.
    """
    
    def __init__(self, config: BrainClientConfig):
        """Initialize brain client.
        
        Args:
            config: Complete brain client configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Trust BrainClientConfig validation
        if not isinstance(config, BrainClientConfig):
            raise ValueError(
                f"config must be BrainClientConfig, got {type(config)}"
            )
        
        self._config = config
        
        # Initialize backend
        self._backend = self._create_backend()
    
    def _create_backend(self) -> ModelBackend:
        """Create configured backend instance.
        
        Returns:
            Initialized model backend
            
        Raises:
            ValueError: If backend configuration is invalid
        """
        if self._config.backend_type == BackendType.DEEPSEEK:
            if not self._config.deepseek_config:
                raise ValueError("DeepSeek backend requires deepseek_config")
            return DeepSeekBackend(self._config.deepseek_config)
        
        raise ValueError(f"Unsupported backend: {self._config.backend_type}")
    
    def infer(self, input_data: BrainInput) -> ModelResponse:
        """Pass structured input to model and return structured output.
        
        Args:
            input_data: Fully structured input containing:
                - Current perceived state
                - Recent actions
                - Memory snapshot
                - Request ID
                
        Returns:
            ModelResponse: Raw, unmodified response from model
            
        Raises:
            ValidationError: If input fails schema validation
            BackendError: If backend fails mechanically
            BackendTimeoutError: If backend call times out
            BackendValidationError: If response fails structural validation
            RuntimeError: If any other mechanical failure occurs
            
        Note:
            - No retry logic
            - No optimization
            - No interpretation of content
            - No correction of reasoning
            - No defaults applied
            - No side effects (logging, state mutation)
        """
        # Input validation is already done by BrainInput type
        # Pydantic validation ensures schema compliance
        input_dict = input_data.model_dump()
        
        try:
            # Mechanical transfer to backend
            raw_response = self._backend.generate(
                input_dict, 
                self._config.timeout_seconds,
                self._config.max_tokens,
                self._config.temperature
            )
            
            # Validate response structure matches schema
            # This validates FORM only, not CONTENT
            response = ModelResponse(**raw_response)
            
            return response
            
        except (ValidationError, BackendError):
            # Re-raise expected exception types unchanged
            raise
        except Exception as e:
            # Any other failure is mechanical
            raise RuntimeError(f"Mechanical failure: {str(e)}") from e


# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------

__all__ = [
    "BrainClient",
    "BrainClientConfig",
    "DeepSeekConfig",
    "BackendType",
    "BrainInput",
    "ModelResponse",
    "PerceivedState",
    "RecentAction",
    "MemorySnapshot",
    "RawModelResponse",
    "BackendError",
    "BackendTimeoutError",
    "BackendValidationError"
]