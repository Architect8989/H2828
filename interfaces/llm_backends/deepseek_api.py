"""
DeepSeek API Backend Adapter

Production-grade LLM backend for DeepSeek API integration.
This module contains zero intelligence, zero decision-making, and zero interpretation.
It simply sends requests to DeepSeek API and returns raw responses.

Absolute constraints:
- NO decision-making about what actions to take
- NO interpretation, filtering, or modification of model outputs
- NO retry logic, backoff strategies, or fallback mechanisms
- NO safety logic, guardrails, or content filtering
- NO environment assumptions or domain knowledge
- NO memory storage or context management
- NO output validation or correction

This module is pure mechanical API integration - boring, auditable, and replaceable.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field, ValidationError, validator, ConfigDict

logger = logging.getLogger(__name__)
API_REQUEST_LOG_LEVEL = logging.INFO


class DeepSeekConfig(BaseModel):
    """
    Configuration for DeepSeek API backend.
    
    All fields are mechanical settings with no semantic interpretation.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # API configuration (mechanical only)
    api_key: str = Field(..., min_length=1)
    base_url: str = Field(default="https://api.deepseek.com")
    api_version: str = Field(default="v1")
    
    # Model parameters (passed directly to API, no interpretation)
    model_name: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    # HTTP configuration (mechanical only)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=0, ge=0)  # Note: Set to 0 per constraints
    
    @validator('api_key')
    def validate_api_key_not_placeholder(cls, v: str) -> str:
        """Ensure API key is not a placeholder value."""
        if v.strip() == "" or "your_api_key_here" in v.lower():
            raise ValueError("API key must be a valid DeepSeek API key")
        return v
    
    @validator('base_url')
    def validate_base_url_format(cls, v: str) -> str:
        """Ensure base URL has proper format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Base URL must start with http:// or https://. Got: {v}")
        return v.rstrip('/')


class Message(BaseModel):
    """
    Single message in chat conversation.
    
    This is a pure structural representation with no content interpretation.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    """
    Raw request structure for DeepSeek chat completion API.
    
    This exactly matches API expectations with no semantic additions.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Required fields
    model: str = Field(..., min_length=1)
    messages: List[Message] = Field(..., min_items=1)
    
    # Optional parameters (passed through exactly as provided)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stream: bool = Field(default=False)  # Always False for our use case
    
    @validator('messages')
    def validate_messages_non_empty_content(cls, v: List[Message]) -> List[Message]:
        """Ensure all messages have non-empty content."""
        for i, msg in enumerate(v):
            if not msg.content.strip():
                raise ValueError(f"Message {i} (role: {msg.role}) has empty content")
        return v


class TokenUsage(BaseModel):
    """
    Token usage information from API response.
    
    This is raw data only - no interpretation of what tokens mean.
    Values are accepted exactly as provided by API without validation.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    
    # Note: No validation of token consistency - API values accepted as-is
    # The brain client must interpret or validate token usage if needed


class ChatCompletionResponse(BaseModel):
    """
    Raw response structure from DeepSeek chat completion API.
    
    Contains exactly what the API returns with no cleaning or interpretation.
    No extraction or transformation of response content occurs here.
    """
    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        validate_assignment=True
    )
    
    # Required fields from API
    id: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    created: int = Field(..., ge=0)
    model: str = Field(..., min_length=1)
    choices: List[Dict[str, Any]] = Field(..., min_items=1)
    usage: TokenUsage
    
    # Metadata (not from API)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Note: No raw_text field - brain client must extract text from choices
    # No extraction or transformation of response content occurs here
    
    @validator('choices')
    def validate_choices_structure(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure choices list has expected structure without interpreting content."""
        if not v:
            raise ValueError("Choices list must not be empty")
        # Only structural validation - no content interpretation
        return v
    
    @property
    def raw_response_text(self) -> str:
        """
        Return raw response text from first choice if available.
        
        Note: This is a computed property, not stored state.
        The brain client should decide how to extract and interpret this.
        """
        if not self.choices:
            return ""
        
        first_choice = self.choices[0]
        message = first_choice.get('message', {})
        content = message.get('content', "")
        
        # Return content exactly as provided by API, no cleaning
        return str(content) if content is not None else ""


class DeepSeekBackend:
    """
    Stateless backend adapter for DeepSeek API.
    
    This class:
    1. Sends requests to DeepSeek API with provided configuration
    2. Returns raw responses exactly as received
    3. Captures timing and token usage
    4. Raises exceptions on HTTP or validation failures
    
    This class does NOT:
    1. Decide what to ask the model
    2. Interpret, filter, or modify responses
    3. Retry failed requests
    4. Add safety logic or content filtering
    5. Validate output correctness or format
    6. Maintain conversation state or context
    7. Extract or transform response content
    """
    
    def __init__(self, config: DeepSeekConfig) -> None:
        """
        Initialize DeepSeek backend with configuration.
        
        Args:
            config: DeepSeekConfig with API settings
        
        Raises:
            ValidationError: If config fails validation
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.DeepSeekBackend")
        self._api_request_log_level = API_REQUEST_LOG_LEVEL
        
        # Build API endpoint URL
        self._endpoint = f"{config.base_url}/{config.api_version}/chat/completions"
        
        # Setup HTTP headers
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatCompletionResponse:
        """
        Send request to DeepSeek API and return raw response.
        
        Args:
            system_prompt: System instruction (opaque, not interpreted)
            user_prompt: User/context payload (opaque, not interpreted)
            temperature: Optional override for temperature parameter
            max_tokens: Optional override for max_tokens parameter
        
        Returns:
            Raw ChatCompletionResponse exactly as received from API
        
        Raises:
            ValidationError: If request structure is invalid
            ValueError: If prompts are empty or API key missing
            RuntimeError: If HTTP request fails or response is malformed
        
        Note:
            - No retry on failure
            - No interpretation of prompts or response
            - No output validation or cleaning
            - No extraction of response content
            - Exactly one API call attempted
        """
        # Record request timing
        request_timestamp = datetime.utcnow()
        
        # Log request metadata (not content)
        self._log_request_metadata(system_prompt, user_prompt, request_timestamp)
        
        try:
            # Create messages list exactly as provided
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            
            # Build request with config values
            request_data = ChatCompletionRequest(
                model=self._config.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self._config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
                top_p=self._config.top_p,
                frequency_penalty=self._config.frequency_penalty,
                presence_penalty=self._config.presence_penalty,
                stream=False,
            )
            
            # Send HTTP request
            response_data = await self._send_http_request(request_data)
            
            # Record response timing
            response_timestamp = datetime.utcnow()
            
            # Create response object (API response as-is, no transformation)
            response = ChatCompletionResponse(
                **response_data,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
            )
            
            # Log response metadata (not content)
            self._log_response_metadata(response, request_timestamp)
            
            return response
            
        except Exception as e:
            # Log failure and re-raise without handling
            self._log_request_failure(e, request_timestamp)
            raise
    
    async def _send_http_request(
        self,
        request_data: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """
        Send HTTP request to DeepSeek API.
        
        Note: No retry logic, no backoff, exactly one attempt.
        """
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Prepare request payload
            payload = request_data.dict(exclude_none=True)
            
            try:
                # Send POST request
                async with session.post(
                    url=self._endpoint,
                    headers=self._headers,
                    json=payload,
                ) as response:
                    # Check HTTP status
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"HTTP {response.status}: {error_text[:200]}"
                        )
                    
                    # Parse response JSON exactly as received
                    response_json = await response.json()
                    
                    # Return raw response data without modification
                    return response_json
                    
            except aiohttp.ClientError as e:
                # HTTP client error (network, timeout, etc.)
                raise RuntimeError(f"HTTP client error: {str(e)}") from e
            except json.JSONDecodeError as e:
                # Malformed JSON response
                raise RuntimeError(f"Invalid JSON response: {str(e)}") from e
    
    def _log_request_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        timestamp: datetime
    ) -> None:
        """Log request metadata without content interpretation."""
        log_data = {
            "timestamp": timestamp.isoformat(),
            "operation": "deepseek_request",
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
            "total_prompt_length": len(system_prompt) + len(user_prompt),
            "model": self._config.model_name,
        }
        
        self._logger.log(
            self._api_request_log_level,
            "Sending request to DeepSeek API",
            extra=log_data
        )
    
    def _log_response_metadata(
        self,
        response: ChatCompletionResponse,
        request_timestamp: datetime
    ) -> None:
        """Log response metadata without content interpretation."""
        # Calculate request duration
        duration_ms = int(
            (response.response_timestamp - request_timestamp).total_seconds() * 1000
        )
        
        log_data = {
            "timestamp": response.response_timestamp.isoformat(),
            "operation": "deepseek_response",
            "request_id": response.request_id,
            "response_id": response.id,
            "model": response.model,
            "duration_ms": duration_ms,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "choices_count": len(response.choices),
        }
        
        self._logger.log(
            self._api_request_log_level,
            "Received response from DeepSeek API",
            extra=log_data
        )
    
    def _log_request_failure(
        self,
        error: Exception,
        request_timestamp: datetime
    ) -> None:
        """Log request failure without handling."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "deepseek_request_failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_timestamp": request_timestamp.isoformat(),
        }
        
        self._logger.log(
            self._api_request_log_level,
            "DeepSeek API request failed",
            extra=log_data
        )


# Factory function for creating backend from environment variables
def create_deepseek_backend_from_env() -> DeepSeekBackend:
    """
    Create DeepSeekBackend instance from environment variables.
    
    Environment variables:
    - DEEPSEEK_API_KEY: Required API key
    - DEEPSEEK_MODEL: Optional model name (default: deepseek-chat)
    - DEEPSEEK_BASE_URL: Optional base URL
    - DEEPSEEK_TEMPERATURE: Optional temperature (float)
    - DEEPSEEK_MAX_TOKENS: Optional max tokens (int)
    
    Returns:
        Configured DeepSeekBackend instance
    
    Raises:
        ValueError: If required environment variables are missing or invalid
    """
    # Get API key from environment
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    
    # Build configuration
    config = DeepSeekConfig(
        api_key=api_key,
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "0")) or None,
    )
    
    return DeepSeekBackend(config)


# Export public interface
__all__ = [
    "DeepSeekBackend",
    "DeepSeekConfig",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "TokenUsage",
    "create_deepseek_backend_from_env",
    "API_REQUEST_LOG_LEVEL",
]