"""
EME World Model - Authoritative Epistemic Core for OS Environment Mastery

Single source of truth about the OS environment. All environmental knowledge
must flow through this module. No other module may assert environmental facts.
"""

import json
import os
import threading
import tempfile
import hashlib
import time
import datetime
import sys
import platform
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum
import copy
import pathlib

# ============================================================================
# Types & Constants - EME Phase-1 Specific
# ============================================================================

class UpdateSource(Enum):
    """Source of a world model update - PHASE 1 RESTRICTED SET."""
    PERCEPTION = "perception"      # Direct sensor observation
    ACTION = "action"              # Result of executed action
    CORRECTION = "correction"      # Manual correction of error

class WorldModelError(Exception):
    """Base exception for world model failures."""
    pass

class SchemaViolationError(WorldModelError):
    """World model schema is violated."""
    pass

class AtomicWriteError(WorldModelError):
    """Atomic write operation failed."""
    pass

class DeltaValidationError(WorldModelError):
    """Delta fails validation."""
    pass

class SemanticDeltaError(WorldModelError):
    """Delta violates semantic constraints."""
    pass

@dataclass
class ChangeRecord:
    """Record of a single world model change."""
    timestamp: float
    source: UpdateSource
    source_id: str
    delta_hash: str
    world_model_hash_before: str
    world_model_hash_after: str
    delta_summary: str  # Human-readable summary of what changed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.datetime.fromtimestamp(self.timestamp).isoformat(),
            "source": self.source.value,
            "source_id": self.source_id,
            "delta_hash": self.delta_hash,
            "world_model_hash_before": self.world_model_hash_before,
            "world_model_hash_after": self.world_model_hash_after,
            "delta_summary": self.delta_summary
        }

@dataclass
class WorldModelSchema:
    """Schema definition for world model validation - EME PHASE 1 SPECIFIC."""
    version: int = 1
    required_root_keys: List[str] = field(default_factory=lambda: [
        "host",              # OS and hardware capabilities
        "applications",      # Installed and running applications
        "ui_elements",       # Currently visible UI elements
        "affordances",       # Possible actions in current context
        "motor",            # Input/output capabilities
        "metadata"          # System metadata
    ])
    allowed_delta_keys: Set[str] = field(default_factory=lambda: {
        "host", "applications", "ui_elements", "affordances", "motor"
    })
    
    # EME Phase-1 specific structure
    host_structure: Dict[str, Any] = field(default_factory=lambda: {
        "os": {
            "name": str,
            "version": str,
            "architecture": str
        },
        "python": {
            "version": str,
            "implementation": str
        },
        "capabilities": {
            "gui": {"detected": bool, "confidence": float, "last_checked": Optional[float]},
            "network": {"detected": bool, "confidence": float, "last_checked": Optional[float]},
            "clipboard": {"detected": bool, "confidence": float, "last_checked": Optional[float]}
        }
    })
    
    application_structure: Dict[str, Any] = field(default_factory=lambda: {
        "name": str,
        "version": Optional[str],
        "is_running": bool,
        "pid": Optional[int],
        "window_title": Optional[str],
        "detection_confidence": float,  # 0.0 to 1.0
        "first_seen": float,
        "last_seen": float
    })
    
    ui_element_structure: Dict[str, Any] = field(default_factory=lambda: {
        "type": str,           # "window", "button", "text_field", etc.
        "application": str,    # Which app this belongs to
        "bounds": {            # Position and size
            "x": int, "y": int, "width": int, "height": int
        },
        "text": Optional[str],
        "state": str,          # "enabled", "disabled", "selected", etc.
        "first_detected": float,
        "last_detected": float,
        "confidence": float    # 0.0 to 1.0
    })
    
    affordance_structure: Dict[str, Any] = field(default_factory=lambda: {
        "type": str,           # "click", "type", "scroll", "drag", etc.
        "target": str,         # ID of UI element
        "confidence": float,   # How sure we are this works
        "discovered_time": float,
        "last_success": Optional[float],  # Timestamp of last successful use
        "prerequisites": List[str]  # What must be true for this to work
    })
    
    motor_structure: Dict[str, Any] = field(default_factory=lambda: {
        "input_methods": {
            "keyboard": {"available": bool, "confidence": float, "last_tested": Optional[float]},
            "mouse": {"available": bool, "confidence": float, "last_tested": Optional[float]},
            "touch": {"available": bool, "confidence": float, "last_tested": Optional[float]},
            "voice": {"available": bool, "confidence": float, "last_tested": Optional[float]}
        },
        "constraints": {
            "min_click_delay": float,  # Seconds
            "typing_speed": float,     # Characters per second
            "screen_bounds": {         # Usable screen area
                "x": int, "y": int, "width": int, "height": int
            }
        },
        "reliability": {               # Success rates
            "click_accuracy": float,
            "typing_accuracy": float
        }
    })
    
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate world model against EME Phase-1 schema."""
        # 1. Check root structure
        for key in self.required_root_keys:
            if key not in data:
                raise SchemaViolationError(f"Missing required root key for EME Phase-1: {key}")
        
        # 2. Check JSON serializability
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            raise SchemaViolationError(f"Data not JSON serializable: {e}")
        
        # 3. Validate no unexpected root keys
        for key in data.keys():
            if key not in self.required_root_keys:
                raise SchemaViolationError(f"Unexpected root key for EME Phase-1: {key}")
        
        # 4. Type checking for critical fields
        if not isinstance(data.get("applications", {}), dict):
            raise SchemaViolationError("'applications' must be a dictionary")
        
        if not isinstance(data.get("ui_elements", {}), dict):
            raise SchemaViolationError("'ui_elements' must be a dictionary")
        
        if not isinstance(data.get("affordances", {}), dict):
            raise SchemaViolationError("'affordances' must be a dictionary")
        
        # 5. Check for unserializable objects
        def check_serializable(obj, path=""):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    check_serializable(item, f"{path}[{i}]")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str):
                        raise SchemaViolationError(f"Non-string key at {path}.{k}")
                    check_serializable(v, f"{path}.{k}")
            else:
                raise SchemaViolationError(f"Non-serializable object at {path}: {type(obj)}")
        
        check_serializable(data)

# ============================================================================
# Core World Model Class - EME Phase-1 Specialized
# ============================================================================

class WorldModel:
    """
    Authoritative representation of EME's OS environment - Phase 1 Specialized.
    
    EME PHASE 1 DOMAIN: OS Environment Mastery
    - What OS am I in? (GROUNDED - observable)
    - What applications exist? (GROUNDED - observable)
    - What UI elements are visible? (GROUNDED - observable)
    - What can I do right now? (DERIVED from grounded observations)
    - What are my motor constraints? (GROUNDED - testable)
    
    Guarantees:
    1. Authoritativeness - Only canonical truth about OS environment
    2. Durability - Survives restarts
    3. Atomicity - All-or-nothing updates
    4. Traceability - Complete audit trail
    5. Schema Stability - Deliberate evolution within Phase-1 domain
    
    PRIME DIRECTIVE: If an action occurs and the world model is not updated, 
    the action is epistemically invalid.
    """
    
    # Configuration constants
    UI_STALE_THRESHOLD = 30.0  # UI elements considered stale after 30 seconds
    LOW_CONFIDENCE = 0.2  # Confidence for unverified assumptions
    
    def __init__(self, filepath: str):
        """
        Initialize world model for EME Phase-1.
        
        Args:
            filepath: Path to persistent storage file.
        """
        self.filepath = pathlib.Path(filepath).resolve()
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.schema = WorldModelSchema()
        
        # EME Phase-1 specific runtime state
        self._world_state: Dict[str, Any] = {}
        self._change_history: List[ChangeRecord] = []
        self._boot_timestamp = time.time()
        self._session_id = hashlib.sha256(str(self._boot_timestamp).encode()).hexdigest()[:16]
        
        # EME Phase-1 allowed delta keys (tight validation)
        self._allowed_delta_keys = self.schema.allowed_delta_keys
        
        # Load or initialize with Phase-1 specific structure
        self._load_or_initialize()
        
        # Validate loaded state against Phase-1 schema
        self.assert_integrity()
    
    # ========================================================================
    # Public Interface - EME Phase-1 Specific
    # ========================================================================
    
    def get_state(self, include_stale: bool = True) -> Dict[str, Any]:
        """
        Get current world state (deep copy).
        
        Args:
            include_stale: Whether to include UI elements that might be stale
            
        Returns:
            Deep copy of current world state.
        """
        with self.lock:
            state = copy.deepcopy(self._world_state)
            
            if not include_stale:
                # Filter out possibly stale UI elements
                current_time = time.time()
                ui_elements = state.get("ui_elements", {})
                fresh_elements = {}
                
                for element_id, element in ui_elements.items():
                    last_seen = element.get("last_detected", 0)
                    if current_time - last_seen < self.UI_STALE_THRESHOLD:
                        fresh_elements[element_id] = element
                
                state["ui_elements"] = fresh_elements
            
            return state
    
    def apply_delta(self, delta: Dict[str, Any], source: UpdateSource, 
                   source_id: str, timestamp: Optional[float] = None) -> bool:
        """
        Apply a delta to the world model - EME PHASE 1 RESTRICTED.
        
        PHASE 1 RULES:
        1. Only PERCEPTION, ACTION, CORRECTION sources allowed
        2. Delta must map to known EME Phase-1 domains
        3. Delta must add new information, not just overwrite
        4. No "inference" or hallucinated facts
        5. Assumptions must be marked as low confidence
        
        Args:
            delta: Structured environmental change
            source: Source of the change
            source_id: Identifier for the source
            timestamp: When change occurred (defaults to now)
            
        Returns:
            True if delta was applied successfully
            
        Raises:
            DeltaValidationError: If delta is invalid
            SemanticDeltaError: If delta violates Phase-1 semantic rules
            AtomicWriteError: If persistence fails
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            # PHASE 1 RULE 1: Source validation
            if source not in [UpdateSource.PERCEPTION, UpdateSource.ACTION, UpdateSource.CORRECTION]:
                raise DeltaValidationError(
                    f"Phase 1 only allows PERCEPTION, ACTION, CORRECTION sources. Got: {source}"
                )
            
            # 1. Validate delta structure
            self._validate_delta_structure(delta)
            
            # 2. Validate delta semantics (PHASE 1 RULE 2, 3, 5)
            semantic_summary = self._validate_delta_semantics(delta, source)
            
            if not semantic_summary:
                raise SemanticDeltaError(
                    "Delta does not add new information or violates Phase-1 semantic rules"
                )
            
            # 3. Create change record
            world_hash_before = self._compute_state_hash()
            delta_hash = hashlib.sha256(json.dumps(delta, sort_keys=True).encode()).hexdigest()
            
            # 4. Apply delta with Phase-1 aware merge
            new_state = self._phase1_merge_delta(self._world_state, delta, timestamp)
            
            # 5. Validate new state against Phase-1 schema
            self.schema.validate(new_state)
            
            # 6. Update metadata
            new_state["metadata"]["last_update"] = timestamp
            new_state["metadata"]["update_count"] = new_state["metadata"].get("update_count", 0) + 1
            
            # 7. Atomic persistence
            world_hash_after = hashlib.sha256(json.dumps(new_state, sort_keys=True).encode()).hexdigest()
            
            try:
                self._atomic_write(new_state)
            except Exception as e:
                raise AtomicWriteError(f"Failed to persist world model: {e}")
            
            # 8. Update in-memory state
            self._world_state = new_state
            
            # 9. Record change
            change_record = ChangeRecord(
                timestamp=timestamp,
                source=source,
                source_id=source_id,
                delta_hash=delta_hash,
                world_model_hash_before=world_hash_before,
                world_model_hash_after=world_hash_after,
                delta_summary=semantic_summary
            )
            self._change_history.append(change_record)
            
            # 10. Trim history if needed
            if len(self._change_history) > 10000:  # Keep last 10k changes
                self._change_history = self._change_history[-10000:]
            
            return True
    
    def validate_schema(self) -> bool:
        """
        Validate current world model against EME Phase-1 schema.
        
        Returns:
            True if valid
            
        Raises:
            SchemaViolationError: If invalid for Phase-1
        """
        with self.lock:
            self.schema.validate(self._world_state)
            return True
    
    def assert_integrity(self) -> None:
        """
        Assert world model integrity - EME Phase-1 specific.
        
        Raises:
            SchemaViolationError: If integrity check fails
            WorldModelError: If other integrity issues
        """
        with self.lock:
            # 1. Phase-1 Schema validation
            self.validate_schema()
            
            # 2. Check metadata consistency
            metadata = self._world_state.get("metadata", {})
            if "creation_time" not in metadata:
                raise SchemaViolationError("Missing creation_time in metadata")
            
            # 3. Verify hash chain if we have history
            if len(self._change_history) > 1:
                for i in range(1, len(self._change_history)):
                    prev = self._change_history[i-1]
                    curr = self._change_history[i]
                    if prev.world_model_hash_after != curr.world_model_hash_before:
                        raise WorldModelError(f"Hash chain broken at change {i}")
            
            # 4. Phase-1 specific: Ensure no inference contamination
            for record in self._change_history:
                if record.source not in [UpdateSource.PERCEPTION, UpdateSource.ACTION, UpdateSource.CORRECTION]:
                    raise WorldModelError(
                        f"History contaminated with non-Phase-1 source: {record.source}"
                    )
            
            # 5. Check for unmarked assumptions
            self._check_for_unmarked_assumptions()
    
    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent change history.
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of change records as dictionaries
        """
        with self.lock:
            recent = self._change_history[-limit:] if self._change_history else []
            return [record.to_dict() for record in recent]
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get world model metadata.
        
        Returns:
            Metadata dictionary
        """
        with self.lock:
            return copy.deepcopy(self._world_state.get("metadata", {}))
    
    def get_phase1_summary(self) -> Dict[str, Any]:
        """
        Get EME Phase-1 specific summary of current state.
        
        Returns:
            Summary dictionary with Phase-1 domain counts
        """
        with self.lock:
            ui_elements = self._world_state.get("ui_elements", {})
            current_time = time.time()
            
            # Count fresh UI elements
            fresh_ui_count = 0
            for element in ui_elements.values():
                last_seen = element.get("last_detected", 0)
                if current_time - last_seen < self.UI_STALE_THRESHOLD:
                    fresh_ui_count += 1
            
            return {
                "host_os": self._world_state.get("host", {}).get("os", {}).get("name", "unknown"),
                "application_count": len(self._world_state.get("applications", {})),
                "ui_element_count": len(ui_elements),
                "fresh_ui_element_count": fresh_ui_count,
                "affordance_count": len(self._world_state.get("affordances", {})),
                "last_update": self._world_state.get("metadata", {}).get("last_update", 0),
                "update_count": self._world_state.get("metadata", {}).get("update_count", 0),
                "epistemic_status": self._compute_epistemic_status()
            }
    
    def prune_stale_ui_elements(self) -> int:
        """
        Remove UI elements that haven't been seen for a long time.
        
        Returns:
            Number of elements removed
        """
        with self.lock:
            current_time = time.time()
            ui_elements = self._world_state.get("ui_elements", {})
            stale_ids = []
            
            for element_id, element in ui_elements.items():
                last_seen = element.get("last_detected", 0)
                if current_time - last_seen >= self.UI_STALE_THRESHOLD * 10:  # 5 minutes
                    stale_ids.append(element_id)
            
            # Create a delta that explicitly removes stale elements
            if stale_ids:
                delta = {
                    "ui_elements": {element_id: None for element_id in stale_ids}
                }
                
                self.apply_delta(
                    delta=delta,
                    source=UpdateSource.CORRECTION,
                    source_id="stale_ui_cleanup",
                    timestamp=time.time()
                )
            
            return len(stale_ids)
    
    # ========================================================================
    # Internal Implementation - EME Phase-1 Specialized
    # ========================================================================
    
    def _load_or_initialize(self) -> None:
        """Load existing world model or initialize new one with Phase-1 structure."""
        with self.lock:
            if self.filepath.exists():
                try:
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        self._world_state = json.load(f)
                    
                    # Migrate to Phase-1 schema if needed
                    self._migrate_to_phase1_schema()
                    
                    # Load recent change history from state
                    metadata = self._world_state.get("metadata", {})
                    if "recent_changes" in metadata:
                        recent_changes = metadata.get("recent_changes", [])
                        self._change_history = [
                            ChangeRecord(
                                timestamp=ch["timestamp"],
                                source=UpdateSource(ch["source"]),
                                source_id=ch["source_id"],
                                delta_hash=ch["delta_hash"],
                                world_model_hash_before=ch["world_model_hash_before"],
                                world_model_hash_after=ch["world_model_hash_after"],
                                delta_summary=ch.get("delta_summary", "")
                            )
                            for ch in recent_changes[-1000:]
                        ]
                        
                except (json.JSONDecodeError, IOError) as e:
                    raise WorldModelError(f"Failed to load world model from {self.filepath}: {e}")
            else:
                # Initialize new world model with EME Phase-1 structure
                self._world_state = self._create_phase1_initial_state()
                
                # Create directory if needed
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Save initial state
                self._atomic_write(self._world_state)
    
    def _create_phase1_initial_state(self) -> Dict[str, Any]:
        """Create initial state for EME Phase-1 - NO ASSUMPTIONS AS FACTS."""
        current_time = time.time()
        
        return {
            "host": {
                "os": {
                    "name": platform.system(),
                    "version": platform.version(),
                    "architecture": platform.architecture()[0]
                },
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation()
                },
                "capabilities": {
                    "gui": {
                        "detected": False,
                        "confidence": self.LOW_CONFIDENCE,  # Low confidence assumption
                        "last_checked": None,
                        "note": "Assumption: GUI not yet verified"
                    },
                    "network": {
                        "detected": False,
                        "confidence": self.LOW_CONFIDENCE,  # Low confidence assumption
                        "last_checked": None,
                        "note": "Assumption: Network not yet verified"
                    },
                    "clipboard": {
                        "detected": False,
                        "confidence": self.LOW_CONFIDENCE,  # Low confidence assumption
                        "last_checked": None,
                        "note": "Assumption: Clipboard not yet verified"
                    }
                }
            },
            "applications": {
                # Start with empty applications dictionary
            },
            "ui_elements": {
                # Start with empty UI elements dictionary
            },
            "affordances": {
                # Start with empty affordances dictionary
            },
            "motor": {
                "input_methods": {
                    "keyboard": {
                        "available": True,
                        "confidence": 1.0,  # We know we have keyboard input
                        "last_tested": current_time,
                        "note": "Self-evident: running via keyboard input"
                    },
                    "mouse": {
                        "available": False,
                        "confidence": self.LOW_CONFIDENCE,  # Don't know yet
                        "last_tested": None,
                        "note": "Assumption: Mouse not yet verified"
                    },
                    "touch": {
                        "available": False,
                        "confidence": 0.0,  # High confidence it's not available
                        "last_tested": None,
                        "note": "Assumption: Touch not available in standard OS"
                    },
                    "voice": {
                        "available": False,
                        "confidence": 0.0,  # High confidence it's not available
                        "last_tested": None,
                        "note": "Assumption: Voice not available in standard OS"
                    }
                },
                "constraints": {
                    "min_click_delay": 0.0,  # Will be discovered through testing
                    "typing_speed": 0.0,      # Will be discovered through testing
                    "screen_bounds": {
                        "x": 0,
                        "y": 0,
                        "width": 0,  # Unknown until discovered
                        "height": 0
                    }
                },
                "reliability": {
                    "click_accuracy": 0.0,  # Unknown until tested
                    "typing_accuracy": 0.0
                }
            },
            "metadata": {
                "version": self.schema.version,
                "creation_time": current_time,
                "last_update": current_time,
                "update_count": 0,
                "session_id": self._session_id,
                "boot_timestamp": self._boot_timestamp,
                "phase": 1,
                "assumptions": [  # Track all initial assumptions
                    "host.capabilities.gui.detected=False (low confidence)",
                    "host.capabilities.network.detected=False (low confidence)",
                    "host.capabilities.clipboard.detected=False (low confidence)",
                    "motor.input_methods.mouse.available=False (low confidence)",
                    "motor.input_methods.touch.available=False (high confidence)",
                    "motor.input_methods.voice.available=False (high confidence)"
                ],
                "recent_changes": []
            }
        }
    
    def _migrate_to_phase1_schema(self) -> None:
        """Migrate world model to Phase-1 schema if needed."""
        metadata = self._world_state.get("metadata", {})
        current_version = metadata.get("version", 0)
        
        if current_version < self.schema.version:
            # Track assumptions that need to be migrated
            migrated_assumptions = []
            
            # Convert old boolean capabilities to confidence-based
            if "host" in self._world_state and "capabilities" in self._world_state["host"]:
                old_caps = self._world_state["host"]["capabilities"]
                if isinstance(old_caps.get("gui"), bool):
                    migrated_assumptions.append("host.capabilities.gui (migrated)")
                    self._world_state["host"]["capabilities"]["gui"] = {
                        "detected": old_caps["gui"],
                        "confidence": self.LOW_CONFIDENCE if old_caps["gui"] else 0.0,
                        "last_checked": None,
                        "note": "Migrated from boolean"
                    }
                
                # Similarly for other capabilities
                for cap in ["network", "clipboard"]:
                    if isinstance(old_caps.get(cap), bool):
                        migrated_assumptions.append(f"host.capabilities.{cap} (migrated)")
                        self._world_state["host"]["capabilities"][cap] = {
                            "detected": old_caps[cap],
                            "confidence": self.LOW_CONFIDENCE if old_caps[cap] else 0.0,
                            "last_checked": None,
                            "note": "Migrated from boolean"
                        }
            
            # Ensure all Phase-1 required keys exist
            for key in self.schema.required_root_keys:
                if key not in self._world_state:
                    if key == "affordances":
                        self._world_state[key] = {}
                    elif key == "motor":
                        self._world_state[key] = self._create_phase1_initial_state()["motor"]
            
            # Update metadata
            self._world_state["metadata"]["version"] = self.schema.version
            self._world_state["metadata"]["phase"] = 1
            self._world_state["metadata"].setdefault("assumptions", []).extend(migrated_assumptions)
            
            # Save migrated version
            self._atomic_write(self._world_state)
    
    def _check_for_unmarked_assumptions(self) -> None:
        """Check if there are facts that should be marked as assumptions."""
        # Check host capabilities
        caps = self._world_state.get("host", {}).get("capabilities", {})
        for cap_name, cap_info in caps.items():
            if isinstance(cap_info, dict) and cap_info.get("confidence", 1.0) > 0.8:
                # High confidence without recent check might be assumption
                last_checked = cap_info.get("last_checked")
                if last_checked is None or time.time() - last_checked > 3600:  # 1 hour
                    raise SchemaViolationError(
                        f"High confidence capability '{cap_name}' without recent verification"
                    )
        
        # Check motor input methods
        input_methods = self._world_state.get("motor", {}).get("input_methods", {})
        for method_name, method_info in input_methods.items():
            if isinstance(method_info, dict) and method_info.get("confidence", 1.0) > 0.8:
                last_tested = method_info.get("last_tested")
                if last_tested is None or time.time() - last_tested > 3600:
                    raise SchemaViolationError(
                        f"High confidence input method '{method_name}' without recent test"
                    )
    
    def _validate_delta_structure(self, delta: Dict[str, Any]) -> None:
        """Validate delta structure - EME Phase-1 specific."""
        if not delta:
            raise DeltaValidationError("Delta cannot be empty")
        
        if not isinstance(delta, dict):
            raise DeltaValidationError("Delta must be a dictionary")
        
        # Check all top-level keys are allowed Phase-1 domains
        for key in delta.keys():
            if key not in self._allowed_delta_keys:
                raise DeltaValidationError(
                    f"Delta key '{key}' not allowed in EME Phase-1. "
                    f"Allowed: {sorted(self._allowed_delta_keys)}"
                )
    
    def _validate_delta_semantics(self, delta: Dict[str, Any], source: UpdateSource) -> str:
        """
        Validate delta semantics - EME Phase-1 specific.
        
        Returns:
            Human-readable summary of what the delta adds/changes
            Empty string if delta is semantically invalid
        """
        changes = []
        
        for domain, domain_delta in delta.items():
            if domain == "applications":
                summary = self._validate_application_delta(domain_delta)
                if summary:
                    changes.append(f"Applications: {summary}")
            
            elif domain == "ui_elements":
                summary = self._validate_ui_delta(domain_delta)
                if summary:
                    changes.append(f"UI: {summary}")
            
            elif domain == "affordances":
                summary = self._validate_affordance_delta(domain_delta)
                if summary:
                    changes.append(f"Affordances: {summary}")
            
            elif domain == "host":
                summary = self._validate_host_delta(domain_delta, source)
                if summary:
                    changes.append(f"Host: {summary}")
            
            elif domain == "motor":
                summary = self._validate_motor_delta(domain_delta, source)
                if summary:
                    changes.append(f"Motor: {summary}")
        
        # No changes means delta is semantically empty
        if not changes:
            return ""
        
        return "; ".join(changes)
    
    def _validate_application_delta(self, delta: Dict[str, Any]) -> str:
        """Validate application delta adds new information."""
        current_apps = self._world_state.get("applications", {})
        
        new_apps = []
        updated_apps = []
        
        for app_id, app_info in delta.items():
            if app_info is None:  # Deletion
                if app_id in current_apps:
                    updated_apps.append(f"removed {app_id[:8]}...")
                continue
                
            if app_id not in current_apps:
                # New application
                if isinstance(app_info, dict) and app_info.get("name"):
                    new_apps.append(app_info.get("name", app_id))
            else:
                # Update to existing - check if it adds information
                current = current_apps[app_id]
                if self._dict_adds_information(app_info, current):
                    updated_apps.append(app_info.get("name", app_id))
        
        if new_apps:
            return f"discovered {len(new_apps)} apps: {', '.join(new_apps[:3])}{'...' if len(new_apps) > 3 else ''}"
        elif updated_apps:
            return f"updated {len(updated_apps)} apps"
        else:
            return ""
    
    def _validate_ui_delta(self, delta: Dict[str, Any]) -> str:
        """Validate UI element delta adds new information."""
        current_ui = self._world_state.get("ui_elements", {})
        
        new_elements = []
        updated_elements = []
        removed_elements = []
        
        for element_id, element_info in delta.items():
            if element_info is None:  # Explicit deletion
                if element_id in current_ui:
                    removed_elements.append(element_id[:8] + "...")
                continue
                
            if element_id not in current_ui:
                # New UI element
                if isinstance(element_info, dict) and element_info.get("type"):
                    new_elements.append(f"{element_info.get('type')}({element_id[:8]}...)")
            else:
                # Update - check if it adds information
                current = current_ui[element_id]
                if self._dict_adds_information(element_info, current):
                    updated_elements.append(element_id[:8] + "...")
        
        if new_elements:
            return f"found {len(new_elements)} new UI elements"
        elif updated_elements:
            return f"updated {len(updated_elements)} UI elements"
        elif removed_elements:
            return f"removed {len(removed_elements)} stale UI elements"
        else:
            return ""
    
    def _validate_affordance_delta(self, delta: Dict[str, Any]) -> str:
        """Validate affordance delta adds new information."""
        current_affordances = self._world_state.get("affordances", {})
        
        new_affordances = []
        removed_affordances = []
        
        for affordance_id, affordance_info in delta.items():
            if affordance_info is None:  # Deletion
                if affordance_id in current_affordances:
                    removed_affordances.append(affordance_id[:8] + "...")
                continue
                
            if affordance_id not in current_affordances:
                # New affordance
                if isinstance(affordance_info, dict) and affordance_info.get("type"):
                    target = affordance_info.get("target", "unknown")
                    new_affordances.append(f"{affordance_info['type']}->{target[:8]}...")
        
        if new_affordances:
            return f"discovered {len(new_affordances)} affordances"
        elif removed_affordances:
            return f"removed {len(removed_affordances)} affordances"
        else:
            return ""
    
    def _validate_host_delta(self, delta: Dict[str, Any], source: UpdateSource) -> str:
        """Validate host delta adds new information."""
        current_host = self._world_state.get("host", {})
        
        # Special check: if this is a PERCEPTION about capabilities, it should have confidence
        if "capabilities" in delta:
            caps_delta = delta["capabilities"]
            for cap_name, cap_info in caps_delta.items():
                if isinstance(cap_info, dict):
                    # PERCEPTION deltas should have timestamps
                    if source == UpdateSource.PERCEPTION:
                        if "last_checked" not in cap_info:
                            raise SemanticDeltaError(
                                f"Perception about capability '{cap_name}' missing timestamp"
                            )
        
        return "updated host info" if self._dict_adds_information(delta, current_host) else ""
    
    def _validate_motor_delta(self, delta: Dict[str, Any], source: UpdateSource) -> str:
        """Validate motor delta adds new information."""
        current_motor = self._world_state.get("motor", {})
        
        # Special check: ACTION deltas should update last_tested
        if source == UpdateSource.ACTION and "input_methods" in delta:
            for method_name, method_info in delta["input_methods"].items():
                if isinstance(method_info, dict) and "last_tested" not in method_info:
                    raise SemanticDeltaError(
                        f"Action using input method '{method_name}' missing test timestamp"
                    )
        
        return "updated motor constraints" if self._dict_adds_information(delta, current_motor) else ""
    
    def _dict_adds_information(self, new_dict: Dict[str, Any], old_dict: Dict[str, Any]) -> bool:
        """Check if new_dict adds information not in old_dict."""
        # Special case: None means deletion
        if new_dict is None:
            return True
            
        def get_leaves(d, path=""):
            leaves = []
            for k, v in d.items():
                current_path = f"{path}.{k}" if path else k
                if isinstance(v, dict):
                    leaves.extend(get_leaves(v, current_path))
                elif v is not None:  # Skip None values in comparison
                    leaves.append((current_path, v))
            return leaves
        
        new_leaves = set(get_leaves(new_dict))
        old_leaves = set(get_leaves(old_dict))
        
        # Check for new leaves or changed values
        diff = new_leaves - old_leaves
        
        # Also check if confidence increased significantly (more than 0.1)
        for path, new_value in new_leaves:
            if ".confidence" in path:
                # Find corresponding old value
                old_value = next((v for p, v in old_leaves if p == path), None)
                if old_value is not None and isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                    if abs(new_value - old_value) > 0.1:
                        return True
        
        return bool(diff)
    
    def _phase1_merge_delta(self, state: Dict[str, Any], delta: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """
        Merge delta into state with EME Phase-1 semantic awareness.
        
        Rules:
        1. Applications: Merge by ID, update timestamps
        2. UI Elements: Merge by ID, update detection_time; allow deletion
        3. Affordances: Add new, update existing if confidence higher; allow deletion
        4. Host: Deep merge with confidence tracking
        5. Motor: Deep merge, favor conservative constraints
        """
        new_state = copy.deepcopy(state)
        
        # Applications: Update existing, add new, allow deletion
        if "applications" in delta:
            for app_id, app_info in delta["applications"].items():
                if app_info is None:
                    # Explicit deletion
                    new_state["applications"].pop(app_id, None)
                elif app_id not in new_state["applications"]:
                    # New application
                    app_info["first_seen"] = timestamp
                    app_info["last_seen"] = timestamp
                    new_state["applications"][app_id] = app_info
                else:
                    # Update existing
                    existing = new_state["applications"][app_id]
                    existing.update(app_info)
                    existing["last_seen"] = timestamp
        
        # UI Elements: Update existing, add new, allow deletion
        if "ui_elements" in delta:
            for element_id, element_info in delta["ui_elements"].items():
                if element_info is None:
                    # Explicit deletion
                    new_state["ui_elements"].pop(element_id, None)
                elif element_id not in new_state["ui_elements"]:
                    # New UI element
                    element_info["first_detected"] = timestamp
                    element_info["last_detected"] = timestamp
                    new_state["ui_elements"][element_id] = element_info
                else:
                    # Update existing
                    existing = new_state["ui_elements"][element_id]
                    existing.update(element_info)
                    existing["last_detected"] = timestamp
        
        # Affordances: Add new, update if confidence is higher, allow deletion
        if "affordances" in delta:
            for affordance_id, affordance_info in delta["affordances"].items():
                if affordance_info is None:
                    # Explicit deletion
                    new_state["affordances"].pop(affordance_id, None)
                elif affordance_id not in new_state["affordances"]:
                    # New affordance
                    affordance_info["discovered_time"] = timestamp
                    new_state["affordances"][affordance_id] = affordance_info
                else:
                    # Update only if new confidence is higher
                    existing = new_state["affordances"][affordance_id]
                    new_confidence = affordance_info.get("confidence", 0)
                    if new_confidence > existing.get("confidence", 0):
                        existing.update(affordance_info)
                        existing["last_updated"] = timestamp
        
        # Host: Deep merge with confidence tracking
        if "host" in delta:
            self._deep_merge_with_confidence(new_state.setdefault("host", {}), delta["host"], timestamp)
        
        # Motor: Deep merge, favor conservative constraints
        if "motor" in delta:
            self._deep_merge_conservative(new_state.setdefault("motor", {}), delta["motor"], timestamp)
        
        return new_state
    
    def _deep_merge_with_confidence(self, target: Dict[str, Any], source: Dict[str, Any], timestamp: float) -> None:
        """Deep merge source into target with confidence tracking."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                if "confidence" in value:
                    # This is a confidence-based field
                    if value.get("confidence", 0) > target[key].get("confidence", 0):
                        # Higher confidence update
                        target[key].update(value)
                        if "last_checked" in value:
                            target[key]["last_checked"] = timestamp
                else:
                    # Recursive merge
                    self._deep_merge_with_confidence(target[key], value, timestamp)
            else:
                target[key] = copy.deepcopy(value)
    
    def _deep_merge_conservative(self, target: Dict[str, Any], source: Dict[str, Any], timestamp: float) -> None:
        """Deep merge for motor constraints, favoring conservative values."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                if key == "input_methods":
                    # For input methods, update with test timestamps
                    for method_name, method_info in value.items():
                        if method_name in target[key]:
                            # Update confidence and timestamp
                            if method_info.get("confidence", 0) > target[key][method_name].get("confidence", 0):
                                target[key][method_name].update(method_info)
                                if "last_tested" in method_info:
                                    target[key][method_name]["last_tested"] = timestamp
                        else:
                            target[key][method_name] = copy.deepcopy(method_info)
                elif key == "constraints":
                    # For constraints, take the larger (more conservative) value
                    for constraint_name, constraint_value in value.items():
                        if constraint_name in ["min_click_delay", "typing_speed"]:
                            # Larger values are more conservative
                            if constraint_name in target[key]:
                                target[key][constraint_name] = max(target[key][constraint_name], constraint_value)
                            else:
                                target[key][constraint_name] = constraint_value
                        else:
                            target[key][constraint_name] = copy.deepcopy(constraint_value)
                else:
                    self._deep_merge_conservative(target[key], value, timestamp)
            else:
                target[key] = copy.deepcopy(value)
    
    def _atomic_write(self, state: Dict[str, Any]) -> None:
        """Write world model atomically."""
        # Prepare state with limited recent changes
        state_to_write = copy.deepcopy(state)
        
        # Include recent changes in metadata (for persistence)
        recent_changes = [cr.to_dict() for cr in self._change_history[-100:]]  # Last 100 changes
        state_to_write["metadata"]["recent_changes"] = recent_changes
        
        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.filepath.parent,
            prefix=f".{self.filepath.name}.tmp.",
            suffix=".json"
        )
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(state_to_write, f, indent=2, sort_keys=True)
            
            # Atomic rename
            os.replace(temp_path, self.filepath)
            
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise AtomicWriteError(f"Atomic write failed: {e}")
    
    def _compute_state_hash(self) -> str:
        """Compute hash of current world state."""
        state_json = json.dumps(self._world_state, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def _compute_epistemic_status(self) -> str:
        """Compute overall epistemic status of the world model."""
        # Count verified vs assumed facts
        total_facts = 0
        verified_facts = 0
        high_confidence_threshold = 0.8
        
        # Check host capabilities
        caps = self._world_state.get("host", {}).get("capabilities", {})
        for cap_info in caps.values():
            if isinstance(cap_info, dict):
                total_facts += 1
                if cap_info.get("confidence", 0) > high_confidence_threshold:
                    verified_facts += 1
        
        # Check motor input methods
        input_methods = self._world_state.get("motor", {}).get("input_methods", {})
        for method_info in input_methods.values():
            if isinstance(method_info, dict):
                total_facts += 1
                if method_info.get("confidence", 0) > high_confidence_threshold:
                    verified_facts += 1
        
        if total_facts == 0:
            return "uninitialized"
        
        verification_ratio = verified_facts / total_facts
        
        if verification_ratio > 0.8:
            return "highly_grounded"
        elif verification_ratio > 0.5:
            return "partially_grounded"
        elif verification_ratio > 0.2:
            return "mostly_assumed"
        else:
            return "largely_assumed"
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup on context exit."""
        pass

# ============================================================================
# Factory Function
# ============================================================================

def create_world_model(filepath: str) -> WorldModel:
    """
    Create and initialize a world model for EME Phase-1.
    
    Args:
        filepath: Path to persistent storage
        
    Returns:
        Initialized WorldModel instance
    """
    return WorldModel(filepath)

# ============================================================================
# Export Public Interface
# ============================================================================

__all__ = [
    'WorldModel',
    'WorldModelError',
    'SchemaViolationError',
    'AtomicWriteError',
    'DeltaValidationError',
    'SemanticDeltaError',
    'UpdateSource',  # Note: Only PERCEPTION, ACTION, CORRECTION in Phase 1
    'ChangeRecord',
    'create_world_model'
]