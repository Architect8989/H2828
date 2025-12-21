"""
EME Causality Ledger v9.9
Pure causal recording - absolute temporal discipline, zero inference
Outputs: Action→effect relationships with verifiable timestamps
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
import uuid
from datetime import datetime
import os

# ============================================================================
# Core Data Structures - Absolute Temporal Precision
# ============================================================================

class ActionType(Enum):
    """Action categories - motor commands only"""
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_WHEEL = "mouse_wheel"
    KEYBOARD_PRESS = "keyboard_press"
    KEYBOARD_TYPE = "keyboard_type"
    SHELL_COMMAND = "shell_command"

@dataclass(frozen=True)
class ActionRecord:
    """Immutable action record - motor command only"""
    action_id: str                    # Unique identifier
    action_type: ActionType          # Motor action category
    parameters: Dict[str, Any]       # Raw action parameters
    timestamp_start: float           # Action initiation time (seconds, monotonic)
    timestamp_end: float             # Action completion time (seconds, monotonic)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate action record invariants"""
        if self.timestamp_start >= self.timestamp_end:
            raise ValueError(f"Invalid timestamps: {self.timestamp_start} >= {self.timestamp_end}")
        if not self.action_id:
            raise ValueError("action_id cannot be empty")

@dataclass(frozen=True)
class Delta:
    """
    Change detection measurement - absolute temporal precision required
    From change_detector.py with strict timestamp requirements
    """
    change_type: str
    coordinates: Tuple[int, int, int, int]
    before_hash: Optional[str]
    after_hash: Optional[str]
    timestamp_start: float           # Absolute measurement start (seconds)
    timestamp_end: float             # Absolute measurement end (seconds)
    measurement_reliability: float   # 0.0-1.0 (sensor measurement quality)
    provenance: Dict[str, Any]
    
    def __post_init__(self):
        """Validate Delta invariants"""
        if self.timestamp_start >= self.timestamp_end:
            raise ValueError(f"Invalid delta timestamps: {self.timestamp_start} >= {self.timestamp_end}")
        if not (0.0 <= self.measurement_reliability <= 1.0):
            raise ValueError(f"measurement_reliability {self.measurement_reliability} out of bounds")

@dataclass(frozen=True)
class CausalEntry:
    """Immutable causal record - action→effect binding with verified timing"""
    action_id: str
    action_fingerprint: str          # Deterministic hash of action
    time_window: Tuple[float, float]  # (start, end) - absolute action time
    deltas: List[Delta]              # Observed changes with absolute timestamps
    causal_reliability: float        # 0.0-1.0 (repeatability, not belief)
    stability_observations: int      # Number of times observed
    variance_metrics: Dict[str, float]  # Measurement variance
    ledger_version: str = "1.0"      # Schema version
    
    def __post_init__(self):
        """Validate causal record invariants"""
        if not (0.0 <= self.causal_reliability <= 1.0):
            raise ValueError(f"causal_reliability {self.causal_reliability} out of bounds")
        if self.stability_observations < 1:
            raise ValueError(f"stability_observations must be >=1, got {self.stability_observations}")
        start, end = self.time_window
        if start >= end:
            raise ValueError(f"Invalid time_window: {start} >= {end}")

# ============================================================================
# Action Fingerprinting - Deterministic Hashing
# ============================================================================

class ActionFingerprinter:
    """Deterministic action hashing - OS-agnostic, no semantics"""
    
    @staticmethod
    def create_fingerprint(action: ActionRecord) -> str:
        """Create deterministic hash of action parameters"""
        normalized = ActionFingerprinter._normalize_action(action)
        json_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()[:32]
    
    @staticmethod
    def _normalize_action(action: ActionRecord) -> Dict[str, Any]:
        """Normalize action for deterministic hashing - no timestamps"""
        return {
            "action_type": action.action_type.value,
            "parameters": ActionFingerprinter._normalize_parameters(action.parameters),
        }
    
    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize action parameters for deterministic hashing"""
        normalized = {}
        
        for key, value in sorted(parameters.items()):
            if isinstance(value, (int, float)):
                if key in ["x", "y", "dx", "dy"]:
                    normalized[key] = round(value, 1)  # 0.1 pixel precision
                elif key in ["keycode", "button"]:
                    normalized[key] = int(value)
                else:
                    normalized[key] = value
            elif isinstance(value, str):
                if key == "command":
                    cmd_hash = hashlib.sha256(value.encode()).hexdigest()[:16]
                    normalized[key] = f"command_hash:{cmd_hash}"
                else:
                    normalized[key] = value
            elif isinstance(value, list):
                normalized[key] = sorted(value)
            elif isinstance(value, dict):
                normalized[key] = ActionFingerprinter._normalize_parameters(value)
            else:
                normalized[key] = str(value)
        
        return normalized

# ============================================================================
# Delta Attribution Engine - Absolute Temporal Discipline
# ============================================================================

class DeltaAttributor:
    """Strict delta attribution - absolute timestamps required, no inference"""
    
    def __init__(self, max_window_overlap: float = 0.1):
        self.max_overlap = max_window_overlap
        self.action_windows: List[Tuple[str, float, float]] = []
    
    def record_action_window(self, action_id: str, start: float, end: float) -> None:
        """Record action time window, enforcing non-overlap"""
        self._validate_window(action_id, start, end)
        self.action_windows.append((action_id, start, end))
    
    def _validate_window(self, action_id: str, start: float, end: float) -> None:
        """Validate action window constraints"""
        if start >= end:
            raise ValueError(f"Non-monotonic window: {start} >= {end} for action {action_id}")
        
        for existing_id, existing_start, existing_end in self.action_windows:
            overlap_start = max(start, existing_start)
            overlap_end = min(end, existing_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                window_duration = min(end - start, existing_end - existing_start)
                
                if overlap_duration / window_duration > self.max_overlap:
                    raise ValueError(
                        f"Action window overlap for {action_id} with {existing_id}: "
                        f"{overlap_duration:.3f}s overlap"
                    )
    
    def attribute_deltas(
        self,
        deltas: List[Delta],
        action_start: float,
        action_end: float,
        attribution_timeout: float = 5.0
    ) -> List[Delta]:
        """
        Attribute deltas to action using absolute timestamps
        NO INFERENCE - timestamps must be provided by change detector
        """
        attributed_deltas = []
        attribution_end = action_end + attribution_timeout
        
        for delta in deltas:
            # Verify delta has absolute timestamps (not durations)
            delta_start, delta_end = delta.timestamp_start, delta.timestamp_end
            
            # Check if delta measurement occurred within attribution window
            delta_measured_during_action = (
                delta_start >= action_start and delta_end <= attribution_end
            )
            
            if not delta_measured_during_action:
                raise ValueError(
                    f"Delta outside attribution window: delta=({delta_start:.3f}-{delta_end:.3f}), "
                    f"action=({action_start:.3f}-{action_end:.3f}), "
                    f"timeout={attribution_timeout}s"
                )
            
            # Check if delta could belong to another action
            alternative_actions = self._find_alternative_attributions(
                delta_start, delta_end, action_id=None
            )
            
            if len(alternative_actions) > 1:
                raise ValueError(
                    f"Ambiguous delta attribution: {len(alternative_actions)} "
                    f"possible actions for delta at ({delta_start:.3f}-{delta_end:.3f})"
                )
            
            attributed_deltas.append(delta)
        
        return attributed_deltas
    
    def _find_alternative_attributions(
        self,
        delta_start: float,
        delta_end: float,
        action_id: Optional[str] = None
    ) -> List[str]:
        """Find all action windows that could claim this delta"""
        candidates = []
        
        for window_id, window_start, window_end in self.action_windows:
            if window_id == action_id:
                continue
            
            # Check temporal overlap with tolerance
            tolerance = 0.1  # 100ms tolerance
            window_expanded = (window_start - tolerance, window_end + tolerance)
            
            # Check if delta overlaps with expanded window
            if not (delta_end < window_expanded[0] or delta_start > window_expanded[1]):
                candidates.append(window_id)
        
        return candidates

# ============================================================================
# Repeatability Tracker - Statistical Measurement Only
# ============================================================================

class RepeatabilityTracker:
    """Track action-effect repeatability - no interpretation"""
    
    def __init__(self):
        self.history: Dict[str, List[Set[str]]] = {}  # fingerprint -> list of delta hash sets
        self.stats: Dict[str, Dict[str, Any]] = {}    # fingerprint -> statistics
    
    def record_observation(
        self,
        action_fingerprint: str,
        deltas: List[Delta]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Record action observation and compute repeatability metrics
        
        Returns:
            Tuple of (causal_reliability, stability_observations, variance_metrics)
            If <2 observations, reliability=0.0
        """
        delta_hashes = self._extract_delta_hashes(deltas)
        
        if action_fingerprint not in self.history:
            self.history[action_fingerprint] = []
            self.stats[action_fingerprint] = {
                "observations": 0,
                "consistent_observations": 0,
                "delta_count_variance": 0.0,
                "hash_similarity_history": []
            }
        
        self.history[action_fingerprint].append(delta_hashes)
        stats = self.stats[action_fingerprint]
        stats["observations"] += 1
        
        reliability = self._compute_reliability(action_fingerprint)
        variance_metrics = self._compute_variance_metrics(action_fingerprint)
        
        return reliability, stats["observations"], variance_metrics
    
    def _extract_delta_hashes(self, deltas: List[Delta]) -> Set[str]:
        """Extract unique hashes from deltas"""
        hashes = set()
        
        for delta in deltas:
            if delta.before_hash and delta.after_hash:
                pair_hash = hashlib.sha256(
                    f"{delta.before_hash}:{delta.after_hash}".encode()
                ).hexdigest()[:16]
                hashes.add(pair_hash)
        
        return hashes
    
    def _compute_reliability(self, action_fingerprint: str) -> float:
        """Compute causal reliability based on repeatability"""
        observations = self.history.get(action_fingerprint, [])
        
        if len(observations) < 2:
            # No reliability claim until sufficient observations
            return 0.0
        
        similarities = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                sim = self._compute_set_similarity(observations[i], observations[j])
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        reliability = sum(similarities) / len(similarities)
        return reliability
    
    def _compute_set_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _compute_variance_metrics(self, action_fingerprint: str) -> Dict[str, float]:
        """Compute variance metrics for this action"""
        observations = self.history.get(action_fingerprint, [])
        
        if len(observations) < 2:
            return {
                "delta_count_variance": 0.0,
                "hash_similarity_variance": 0.0,
                "observation_count": len(observations)
            }
        
        delta_counts = [len(obs) for obs in observations]
        count_variance = self._compute_variance(delta_counts)
        
        similarities = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                sim = self._compute_set_similarity(observations[i], observations[j])
                similarities.append(sim)
        
        similarity_variance = self._compute_variance(similarities) if similarities else 0.0
        
        return {
            "delta_count_variance": float(count_variance),
            "hash_similarity_variance": float(similarity_variance),
            "observation_count": len(observations),
            "average_delta_count": float(sum(delta_counts) / len(delta_counts))
        }
    
    @staticmethod
    def _compute_variance(values: List[float]) -> float:
        """Compute variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        variance = sum(squared_diffs) / (len(values) - 1)
        return variance

# ============================================================================
# Causality Ledger - Main Class
# ============================================================================

class CausalityLedger:
    """
    Pure causal recording - absolute temporal precision, no inference
    """
    
    def __init__(
        self,
        ledger_path: Optional[str] = None,
        max_window_overlap: float = 0.1,
        attribution_timeout: float = 5.0
    ):
        self.ledger_path = ledger_path
        self.max_overlap = max_window_overlap
        self.attribution_timeout = attribution_timeout
        
        self.fingerprinter = ActionFingerprinter()
        self.attributor = DeltaAttributor(max_window_overlap)
        self.tracker = RepeatabilityTracker()
        
        self.entries: List[CausalEntry] = []
        self.action_index: Dict[str, CausalEntry] = {}
        self.fingerprint_index: Dict[str, List[CausalEntry]] = {}
        
        if ledger_path and os.path.exists(ledger_path):
            self._load_ledger()
    
    def record(
        self,
        action: ActionRecord,
        deltas: List[Delta]
    ) -> CausalEntry:
        """
        Record action and its observed effects
        Requires absolute timestamps in deltas - no inference allowed
        """
        self._validate_inputs(action, deltas)
        
        action_fingerprint = self.fingerprinter.create_fingerprint(action)
        
        self.attributor.record_action_window(
            action.action_id,
            action.timestamp_start,
            action.timestamp_end
        )
        
        attributed_deltas = self.attributor.attribute_deltas(
            deltas,
            action.timestamp_start,
            action.timestamp_end,
            self.attribution_timeout
        )
        
        causal_reliability, stability_obs, variance_metrics = (
            self.tracker.record_observation(action_fingerprint, attributed_deltas)
        )
        
        entry = CausalEntry(
            action_id=action.action_id,
            action_fingerprint=action_fingerprint,
            time_window=(action.timestamp_start, action.timestamp_end),
            deltas=attributed_deltas,
            causal_reliability=causal_reliability,
            stability_observations=stability_obs,
            variance_metrics=variance_metrics
        )
        
        self.entries.append(entry)
        self.action_index[action.action_id] = entry
        
        if action_fingerprint not in self.fingerprint_index:
            self.fingerprint_index[action_fingerprint] = []
        self.fingerprint_index[action_fingerprint].append(entry)
        
        if self.ledger_path:
            self._persist_entry(entry)
        
        return entry
    
    def _validate_inputs(self, action: ActionRecord, deltas: List[Delta]) -> None:
        """Validate inputs before recording"""
        if action.action_id in self.action_index:
            raise ValueError(f"Duplicate action_id: {action.action_id}")
        
        for delta in deltas:
            if delta.timestamp_start >= delta.timestamp_end:
                raise ValueError(
                    f"Invalid delta timestamps: {delta.timestamp_start} >= {delta.timestamp_end}"
                )
    
    def _persist_entry(self, entry: CausalEntry) -> None:
        """Persist causal entry to storage"""
        if not self.ledger_path:
            return
        
        entry_dict = asdict(entry)
        entry_dict["deltas"] = [asdict(delta) for delta in entry.deltas]
        
        with open(self.ledger_path, 'a') as f:
            json_str = json.dumps(entry_dict, separators=(',', ':'))
            f.write(json_str + '\n')
    
    def _load_ledger(self) -> None:
        """Load ledger from storage"""
        if not self.ledger_path or not os.path.exists(self.ledger_path):
            return
        
        try:
            with open(self.ledger_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry_dict = json.loads(line)
                        
                        delta_dicts = entry_dict.pop("deltas", [])
                        deltas = []
                        for delta_dict in delta_dicts:
                            if "coordinates" in delta_dict:
                                delta_dict["coordinates"] = tuple(delta_dict["coordinates"])
                            deltas.append(Delta(**delta_dict))
                        
                        if "time_window" in entry_dict:
                            entry_dict["time_window"] = tuple(entry_dict["time_window"])
                        
                        entry_dict["deltas"] = deltas
                        entry = CausalEntry(**entry_dict)
                        
                        self.entries.append(entry)
                        self.action_index[entry.action_id] = entry
                        
                        if entry.action_fingerprint not in self.fingerprint_index:
                            self.fingerprint_index[entry.action_fingerprint] = []
                        self.fingerprint_index[entry.action_fingerprint].append(entry)
                        
                        start, end = entry.time_window
                        self.attributor.action_windows.append(
                            (entry.action_id, start, end)
                        )
                        
                        self.tracker.record_observation(
                            entry.action_fingerprint,
                            entry.deltas
                        )
        
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load ledger: {e}")
    
    # ========================================================================
    # Query Methods - Fact Retrieval Only
    # ========================================================================
    
    def get_by_action_id(self, action_id: str) -> Optional[CausalEntry]:
        """Get causal entry by action_id"""
        return self.action_index.get(action_id)
    
    def get_by_fingerprint(self, action_fingerprint: str) -> List[CausalEntry]:
        """Get all causal entries with given action fingerprint"""
        return self.fingerprint_index.get(action_fingerprint, [])
    
    def get_recent_entries(self, count: int = 100) -> List[CausalEntry]:
        """Get most recent causal entries"""
        return self.entries[-count:] if self.entries else []
    
    def get_entries_with_reliability(
        self,
        min_reliability: float = 0.7,
        max_reliability: float = 1.0
    ) -> List[CausalEntry]:
        """Get entries within reliability range"""
        result = []
        for entry in self.entries:
            if min_reliability <= entry.causal_reliability <= max_reliability:
                result.append(entry)
        return result
    
    def get_action_fingerprints(self) -> List[str]:
        """Get all unique action fingerprints in ledger"""
        return list(self.fingerprint_index.keys())
    
    # ========================================================================
    # Statistical Methods - Measurement Only
    # ========================================================================
    
    def compute_ledger_stats(self) -> Dict[str, Any]:
        """Compute ledger statistics - no interpretation"""
        if not self.entries:
            return {
                "total_entries": 0,
                "unique_fingerprints": 0,
                "average_reliability": 0.0,
                "total_deltas": 0
            }
        
        total_entries = len(self.entries)
        unique_fingerprints = len(self.fingerprint_index)
        
        total_reliability = sum(e.causal_reliability for e in self.entries)
        average_reliability = total_reliability / total_entries
        
        total_deltas = sum(len(e.deltas) for e in self.entries)
        
        # Most frequently observed action fingerprint (statistical frequency only)
        most_frequent_fingerprint = None
        highest_frequency = 0
        for fingerprint, entries in self.fingerprint_index.items():
            frequency = len(entries)
            if frequency > highest_frequency:
                highest_frequency = frequency
                most_frequent_fingerprint = fingerprint
        
        return {
            "total_entries": total_entries,
            "unique_fingerprints": unique_fingerprints,
            "average_reliability": float(average_reliability),
            "total_deltas": total_deltas,
            "most_frequent_fingerprint": most_frequent_fingerprint,  # Statistical frequency only
            "most_frequent_count": highest_frequency
        }
    
    def export_ledger(self, path: str) -> None:
        """Export complete ledger to JSON file"""
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "ledger_version": "1.0",
                "total_entries": len(self.entries)
            },
            "entries": []
        }
        
        for entry in self.entries:
            entry_dict = asdict(entry)
            entry_dict["deltas"] = [asdict(delta) for delta in entry.deltas]
            export_data["entries"].append(entry_dict)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

# ============================================================================
# Utility Functions
# ============================================================================

def create_action_id() -> str:
    """Generate unique action identifier"""
    return f"act_{uuid.uuid4().hex[:16]}"

def create_timedelta_id() -> str:
    """Generate unique identifier for delta measurements"""
    return f"delta_{uuid.uuid4().hex[:16]}"

def create_standard_ledger(storage_path: str) -> CausalityLedger:
    """Create standard causality ledger"""
    return CausalityLedger(
        ledger_path=storage_path,
        max_window_overlap=0.1,
        attribution_timeout=5.0
    )

def create_strict_ledger(storage_path: str) -> CausalityLedger:
    """Create strict causality ledger (less tolerance)"""
    return CausalityLedger(
        ledger_path=storage_path,
        max_window_overlap=0.05,
        attribution_timeout=3.0
    )