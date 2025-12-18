"""
Brain client module - Stateless interface between cognitive system and reasoning model.

This module provides a pure mechanical interface for exchanging structured state information
with a reasoning model (LLM). It contains zero intelligence, zero behavior, and zero assumptions
about the system's goals, environment, or operational context.

Key Principles:
- Stateless between calls
- Deterministic behavior
- Model-agnostic backend support
- Explicit input/output validation
- No interpretation or modification of reasoning
- No injection of domain knowledge or heuristics

What this module DOES:
1. Validates structured input describing system state
2. Transfers validated input to configured model backend
3. Receives raw structured output from model
4. Validates output matches declared schema
5. Returns output unchanged
6. Logs all inputs/outputs verbatim

What this module DOES NOT do:
- Decide what the system should do next
- Modify, filter, rank, or sanitize reasoning
- Inject domain knowledge, goals, or tasks
- Encode safety, fallback, or exploration logic
- Maintain hidden state between calls
- Optimize, retry, or correct model outputs
- Encode success/failure semantics
- Assume anything about OS, UI, or user intent
- Help the model "decide better"
- Encode workflows, policies, or heuristics
- Apply defaults that imply behavior

All intelligence and decision-making resides exclusively in the reasoning model.
"""

import json
import logging
from typing import Any, Dict, Protocol, runtime_checkable
from datetime import datetime

from pydantic import BaseModel, ValidationError, ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

# Configure audit logging
_AUDIT_LOGGER = logging.getLogger("brain_client_audit")
_AUDIT_LOGGER.setLevel(logging.INFO)


# -------------------------------------------------------------------
# Input/Output Schema Definitions
# -------------------------------------------------------------------

class PerceivedState(BaseModel):
    """Structured representation of current system state perception.
    
    This is a passive container for already-interpreted state information.
    No validation implies behavior or correct interpretation.
    """
    model_config = ConfigDict(extra='forbid')
    
    sensory_inputs: Dict[str, Any] = Field(
        description="Raw sensory inputs as interpreted by upstream systems"
    )
    processed_observations: Dict[str, Any] = Field(
        description="Processed observations from perception systems"
    )
    telemetry: Dict[str, float] = Field(
        description="Current system telemetry readings"
    )


class RecentAction(BaseModel):
    """Record of a recently executed action.
    
    Contains only what was done, not whether it was correct or successful.
    """
    model_config = ConfigDict(extra='forbid')
    
    action_type: str = Field(description="Type identifier for the action")
    parameters: Dict[str, Any] = Field(description="Parameters used in action execution")
    timestamp: datetime = Field(description="When the action was initiated")


class MemorySnapshot(BaseModel):
    """Short-term memory contents at current moment.
    
    Passively contains whatever memory system provides.
    No implication about relevance or importance.
    """
    model_config = ConfigDict(extra='forbid')
    
    active_recall: Dict[str, Any] = Field(
        description="Currently active memory recalls"
    )
    recent_patterns: Dict[str, Any] = Field(
        description="Recently observed patterns"
    )
    contextual_tags: Dict[str, Any] = Field(
        description="Contextual tags applied by memory system"
    )


class BrainInput(BaseModel):
    """Complete input structure sent to reasoning model.
    
    This is the entire input contract with the model.
    All fields are required to enforce explicit communication.
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
        description="Unique identifier for this reasoning request"
    )


class ModelResponse(BaseModel):
    """Expected response structure from reasoning model.
    
    This defines the output contract only. No validation implies
    correctness, appropriateness, or safety of the response.
    """
    model_config = ConfigDict(extra='forbid')
    
    reasoning_artifacts: Dict[str, Any] = Field(
        description="Model's internal reasoning artifacts",
        min_items=1
    )
    selected_actions: list[Dict[str, Any]] = Field(
        description="Actions to execute, in proposed order",
        min_items=0
    )
    attention_focus: Dict[str, float] = Field(
        description="Attention weights for next perception cycle",
        min_items=1
    )
    memory_updates: Dict[str, Any] = Field(
        description="Proposed updates to memory systems",
        min_items=0
    )
    self_reported_confidence: float = Field(
        description="Model's self-reported confidence score",
        ge=0.0,
        le=1.0
    )


# -------------------------------------------------------------------
# Backend Protocol
# -------------------------------------------------------------------

@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model backend implementations.
    
    This allows swapping different reasoning models without
    changing the client interface.
    """
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response from reasoning model.
        
        Args:
            input_data: Validated BrainInput as dictionary
            
        Returns:
            Raw model response as dictionary
            
        Note:
            Backend must NOT modify or interpret the input.
            Backend must NOT apply any post-processing to output.
        """
        ...


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@pydantic_dataclass
class BrainClientConfig:
    """Configuration for brain client.
    
    Contains only mechanical settings. No behavioral defaults.
    """
    model_backend: ModelBackend = Field(
        description="Model backend implementation"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Whether to log all inputs/outputs verbatim"
    )


# -------------------------------------------------------------------
# Main Client
# -------------------------------------------------------------------

class BrainClient:
    """Stateless interface to reasoning model.
    
    This class is a pure mechanical pass-through. It contains:
    - Zero intelligence
    - Zero behavior
    - Zero assumptions
    - Zero hidden state
    
    All reasoning, decision-making, and intelligence resides
    exclusively in the configured model backend.
    """
    
    def __init__(self, config: BrainClientConfig):
        """Initialize brain client with mechanical configuration.
        
        Args:
            config: Configuration containing only mechanical settings
            
        Note:
            No behavioral configuration is accepted or stored.
        """
        self._config = config
        self._logger = logging.getLogger(__name__)
        
        # Validate audit logging configuration
        if config.enable_audit_logging and not _AUDIT_LOGGER.handlers:
            raise RuntimeError(
                "Audit logger has no handlers configured. "
                "Configure logging handlers for 'brain_client_audit' logger "
                "or disable audit logging."
            )
        
        # Log configuration for audit trail
        self._logger.info(
            "Brain client initialized with backend: %s",
            type(config.model_backend).__name__
        )
    
    def reason(self, input_data: BrainInput) -> ModelResponse:
        """Pass structured input to model and return structured output.
        
        This method:
        1. Validates input structure (already done by BrainInput type)
        2. Sends to configured backend
        3. Receives raw response
        4. Validates response matches schema
        5. Returns response unchanged
        6. Logs everything verbatim
        
        Args:
            input_data: Fully structured input containing:
                - Current perceived state
                - Recent actions
                - Memory snapshot
                
        Returns:
            ModelResponse: Raw, unmodified response from model
            
        Raises:
            ValidationError: If input or output fails schema validation
            RuntimeError: If backend fails mechanically
            ValueError: If response fails structural validation
            
        Note:
            - No retry logic is applied
            - No optimization is performed
            - No interpretation of content
            - No correction of reasoning
            - No ranking or filtering
            - No defaults are applied
        """
        # Input is already validated by BrainInput type
        input_dict = input_data.model_dump()
        
        # Audit log input (verbatim, no filtering)
        if self._config.enable_audit_logging:
            self._audit_log("input", input_dict)
        
        try:
            # Mechanical transfer to backend
            # No modification, no interpretation, no help
            raw_response = self._config.model_backend.generate(input_dict)
            
            # Validate response structure matches schema
            # This validates FORM only, not CONTENT correctness
            response = ModelResponse(**raw_response)
            
            # Audit log output (verbatim, no filtering)
            if self._config.enable_audit_logging:
                self._audit_log("output", raw_response)
            
            return response
            
        except ValidationError as e:
            # Schema validation failure - mechanical issue
            self._logger.error(
                "Model response validation failed: %s",
                str(e)
            )
            if self._config.enable_audit_logging:
                self._audit_log("validation_error", {
                    "input": input_dict,
                    "error": str(e),
                    "raw_response": raw_response if 'raw_response' in locals() else None
                })
            raise
            
        except Exception as e:
            # Mechanical failure in backend
            self._logger.error(
                "Backend mechanical failure: %s",
                str(e)
            )
            if self._config.enable_audit_logging:
                self._audit_log("backend_error", {
                    "input": input_dict,
                    "error": str(e)
                })
            raise RuntimeError(f"Backend mechanical failure: {str(e)}")
    
    def _audit_log(self, log_type: str, data: Dict[str, Any]) -> None:
        """Log data verbatim for audit trail.
        
        Args:
            log_type: Type of log entry
            data: Data to log (input, output, or error)
            
        Raises:
            RuntimeError: If audit logging is enabled but no handler is configured
            
        Note:
            No filtering, sanitization, or modification of data.
            Logs exactly what was received/sent.
        """
        if not self._config.enable_audit_logging:
            return
        
        # Double-check handler presence (may have been removed after initialization)
        if not _AUDIT_LOGGER.handlers:
            raise RuntimeError(
                "Audit logger has no handlers configured during logging attempt. "
                "Configure logging handlers for 'brain_client_audit' logger."
            )
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": log_type,
            "data": data
        }
        
        _AUDIT_LOGGER.info(
            json.dumps(audit_entry, default=str)
        )


# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------

__all__ = [
    "BrainClient",
    "BrainClientConfig",
    "BrainInput",
    "ModelResponse",
    "PerceivedState",
    "RecentAction",
    "MemorySnapshot",
    "ModelBackend"
]