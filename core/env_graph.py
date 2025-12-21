"""
EME Environment Navigation Graph v9.9 - FINAL
Pure topological memory - explicit ambiguity encoding, no corruption
"""

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from datetime import datetime
import time

# ============================================================================
# Core Data Structures - Frozen Topology
# ============================================================================

@dataclass(frozen=True)
class StateNode:
    """Immutable state record - hash-based identity only"""
    state_hash: str           # Deterministic screen/state hash
    first_seen: float        # Absolute timestamp (seconds)
    last_seen: float         # Absolute timestamp (seconds)
    visit_count: int         # Number of times this state observed
    
    def __post_init__(self):
        """Validate state node invariants"""
        if not self.state_hash:
            raise ValueError("state_hash cannot be empty")
        if self.first_seen > self.last_seen:
            raise ValueError(f"first_seen {self.first_seen} > last_seen {self.last_seen}")
        if self.visit_count < 1:
            raise ValueError(f"visit_count must be >=1, got {self.visit_count}")

@dataclass(frozen=True)
class StateTransition:
    """Immutable transition record - mechanical observation only"""
    from_state: str          # Source state hash
    to_state: str            # Destination state hash
    action_fingerprint: str  # Deterministic action hash
    reliability: float       # 0.0-1.0 (repeatability of this transition)
    observation_count: int   # Number of times observed
    ambiguous: bool = False  # True if multiple destinations for same action from same state
    reversible: Optional[bool] = None  # None until proven
    
    def __post_init__(self):
        """Validate transition invariants"""
        if not self.from_state or not self.to_state:
            raise ValueError("from_state and to_state cannot be empty")
        if not self.action_fingerprint:
            raise ValueError("action_fingerprint cannot be empty")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability {self.reliability} out of bounds")
        if self.observation_count < 1:
            raise ValueError(f"observation_count must be >=1, got {self.observation_count}")

# ============================================================================
# Graph Events - Append-Only Record
# ============================================================================

class GraphEventType(Enum):
    """Types of events that modify graph state"""
    STATE_OBSERVED = "state_observed"
    TRANSITION_OBSERVED = "transition_observed"
    TRANSITION_REVERSIBILITY_ESTABLISHED = "transition_reversibility_established"
    TRANSITION_AMBIGUITY_ESTABLISHED = "transition_ambiguity_established"

@dataclass(frozen=True)
class GraphEvent:
    """Immutable event record - all graph modifications via events"""
    event_type: GraphEventType
    timestamp: float
    data: Dict[str, Any]
    event_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate deterministic event hash"""
        hash_data = {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": json.dumps(self.data, sort_keys=True, separators=(',', ':'))
        }
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        object.__setattr__(self, 'event_hash', hash_obj.hexdigest()[:32])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "event_hash": self.event_hash
        }

# ============================================================================
# Graph Integrity Engine
# ============================================================================

class GraphIntegrity:
    """Enforces graph invariants - explicit ambiguity encoding"""
    
    @staticmethod
    def validate_state_hash(state_hash: str) -> None:
        """Validate state hash format"""
        if not isinstance(state_hash, str):
            raise TypeError(f"state_hash must be string, got {type(state_hash)}")
        if len(state_hash) < 8:
            raise ValueError(f"state_hash too short: {len(state_hash)} chars")
    
    @staticmethod
    def detect_ambiguity(
        from_state: str,
        to_state: str,
        action_fingerprint: str,
        existing_transitions: Dict[Tuple[str, str, str], StateTransition]
    ) -> Tuple[bool, List[StateTransition]]:
        """
        Detect if action from state has multiple destinations
        Returns (is_ambiguous, conflicting_transitions)
        """
        conflicting = [
            t for t in existing_transitions.values()
            if t.from_state == from_state and 
               t.action_fingerprint == action_fingerprint and
               t.to_state != to_state
        ]
        
        return len(conflicting) > 0, conflicting
    
    @staticmethod
    def validate_no_orphan_transitions(
        transition: StateTransition,
        state_nodes: Dict[str, StateNode]
    ) -> None:
        """Ensure transition references existing states"""
        if transition.from_state not in state_nodes:
            raise ValueError(f"Orphan transition: from_state {transition.from_state} not in graph")
        if transition.to_state not in state_nodes:
            raise ValueError(f"Orphan transition: to_state {transition.to_state} not in graph")

# ============================================================================
# Environment Navigation Graph - FIXED VERSION
# ============================================================================

class EnvironmentNavigationGraph:
    """
    Pure navigation memory - records state transitions only
    No pathfinding, no planning, no interpretation
    """
    
    def __init__(self, graph_path: Optional[str] = None):
        """
        Initialize navigation graph
        
        Args:
            graph_path: Path to append-only event log (None for in-memory only)
        """
        self.graph_path = graph_path
        
        # Core graph structures
        self.state_nodes: Dict[str, StateNode] = {}
        self.transitions: Dict[Tuple[str, str, str], StateTransition] = {}
        
        # Reverse indexes for query performance (maintained without duplication)
        self.transitions_from: Dict[str, List[StateTransition]] = {}
        self.transitions_to: Dict[str, List[StateTransition]] = {}
        self.action_transitions: Dict[str, List[StateTransition]] = {}
        
        # State observation tracking to prevent duplication
        self.state_observations: Set[Tuple[str, float]] = set()  # (state_hash, timestamp)
        
        # Event history (for rebuild)
        self.event_history: List[GraphEvent] = []
        
        # Load existing graph if path provided
        if graph_path and os.path.exists(graph_path):
            self._load_graph()
    
    # ========================================================================
    # Core Graph Operations - FIXED
    # ========================================================================
    
    def record_transition(
        self,
        state_hash_before: str,
        state_hash_after: str,
        action_fingerprint: str,
        causal_reliability: float,  # From action_ledger: observed repeatability
        timestamp_start: float,
        timestamp_end: float
    ) -> Tuple[StateNode, StateNode, StateTransition]:
        """
        Record a state transition with absolute temporal precision
        
        Returns:
            Tuple of (from_state_node, to_state_node, transition)
            
        Raises:
            ValueError: On any graph integrity violation
        """
        # --------------------------------------------------------------------
        # 1. Validate inputs
        # --------------------------------------------------------------------
        GraphIntegrity.validate_state_hash(state_hash_before)
        GraphIntegrity.validate_state_hash(state_hash_after)
        
        if not (0.0 <= causal_reliability <= 1.0):
            raise ValueError(f"causal_reliability {causal_reliability} out of bounds")
        
        # --------------------------------------------------------------------
        # 2. Record states (creates or updates state nodes)
        # --------------------------------------------------------------------
        from_state_node = self._record_state_observation(state_hash_before, timestamp_start)
        to_state_node = self._record_state_observation(state_hash_after, timestamp_end)
        
        # --------------------------------------------------------------------
        # 3. Detect and handle ambiguity
        # --------------------------------------------------------------------
        is_ambiguous, conflicting_transitions = GraphIntegrity.detect_ambiguity(
            state_hash_before,
            state_hash_after,
            action_fingerprint,
            self.transitions
        )
        
        # --------------------------------------------------------------------
        # 4. Create or update transition
        # --------------------------------------------------------------------
        transition_key = (state_hash_before, state_hash_after, action_fingerprint)
        
        if transition_key in self.transitions:
            # Update existing transition
            existing = self.transitions[transition_key]
            transition = self._update_transition(existing, causal_reliability, is_ambiguous)
        else:
            # Create new transition
            transition = StateTransition(
                from_state=state_hash_before,
                to_state=state_hash_after,
                action_fingerprint=action_fingerprint,
                reliability=causal_reliability,  # Observed reliability mean
                observation_count=1,
                ambiguous=is_ambiguous
            )
        
        # If ambiguous, mark all conflicting transitions as ambiguous
        if is_ambiguous:
            self._mark_transitions_as_ambiguous(state_hash_before, action_fingerprint)
        
        # --------------------------------------------------------------------
        # 5. Validate no orphan transitions
        # --------------------------------------------------------------------
        GraphIntegrity.validate_no_orphan_transitions(transition, self.state_nodes)
        
        # --------------------------------------------------------------------
        # 6. Update graph structures (with index deduplication)
        # --------------------------------------------------------------------
        self._update_transition_with_deduplication(transition_key, transition)
        
        # --------------------------------------------------------------------
        # 7. Check for reversibility
        # --------------------------------------------------------------------
        self._establish_reversibility_if_possible(transition)
        
        # --------------------------------------------------------------------
        # 8. Record events and persist
        # --------------------------------------------------------------------
        self._record_transition_observed_event(
            state_hash_before, state_hash_after, action_fingerprint,
            causal_reliability, timestamp_end, is_ambiguous
        )
        
        if self.graph_path:
            self._persist_events()
        
        return from_state_node, to_state_node, transition
    
    def _record_state_observation(self, state_hash: str, timestamp: float) -> StateNode:
        """
        Record state observation - creates or updates node
        Deduplicates by (state_hash, timestamp) to prevent artificial inflation
        """
        # Check if we've already recorded this exact observation
        observation_key = (state_hash, timestamp)
        if observation_key in self.state_observations:
            # Return existing node (no duplicate recording)
            return self.state_nodes[state_hash]
        
        # Mark this observation as recorded
        self.state_observations.add(observation_key)
        
        if state_hash in self.state_nodes:
            # Update existing state node
            existing = self.state_nodes[state_hash]
            node = StateNode(
                state_hash=state_hash,
                first_seen=existing.first_seen,
                last_seen=timestamp,
                visit_count=existing.visit_count + 1
            )
        else:
            # Create new state node
            node = StateNode(
                state_hash=state_hash,
                first_seen=timestamp,
                last_seen=timestamp,
                visit_count=1
            )
        
        self.state_nodes[state_hash] = node
        
        # Record state observation event (only once per observation)
        self._record_state_observed_event(state_hash, timestamp)
        
        return node
    
    def _update_transition(
        self,
        existing: StateTransition,
        new_reliability: float,
        is_ambiguous: bool
    ) -> StateTransition:
        """
        Update transition with new observation
        Computes weighted mean of observed reliabilities (no interpretation)
        """
        # Compute weighted average of observed reliabilities
        total_weight = existing.observation_count + 1
        # observed_reliability_mean = (existing_reliability * existing_observations + new_reliability) / total_observations
        weighted_reliability = (
            (existing.reliability * existing.observation_count) + new_reliability
        ) / total_weight
        
        return StateTransition(
            from_state=existing.from_state,
            to_state=existing.to_state,
            action_fingerprint=existing.action_fingerprint,
            reliability=weighted_reliability,
            observation_count=total_weight,
            ambiguous=is_ambiguous or existing.ambiguous,  # Ambiguity is permanent
            reversible=existing.reversible
        )
    
    def _update_transition_with_deduplication(
        self,
        transition_key: Tuple[str, str, str],
        transition: StateTransition
    ) -> None:
        """
        Update transition and indexes with deduplication
        Removes old transition from indexes before adding new one
        """
        from_state, to_state, action_fp = transition_key
        
        # Remove old transition from indexes (if exists)
        old_transition = self.transitions.get(transition_key)
        if old_transition:
            self._remove_transition_from_indexes(old_transition)
        
        # Update main transition dictionary
        self.transitions[transition_key] = transition
        
        # Add to indexes
        if from_state not in self.transitions_from:
            self.transitions_from[from_state] = []
        self.transitions_from[from_state].append(transition)
        
        if to_state not in self.transitions_to:
            self.transitions_to[to_state] = []
        self.transitions_to[to_state].append(transition)
        
        if action_fp not in self.action_transitions:
            self.action_transitions[action_fp] = []
        self.action_transitions[action_fp].append(transition)
    
    def _remove_transition_from_indexes(self, transition: StateTransition) -> None:
        """Remove transition from all indexes"""
        # Remove from from_state index
        if transition.from_state in self.transitions_from:
            self.transitions_from[transition.from_state] = [
                t for t in self.transitions_from[transition.from_state]
                if not (t.to_state == transition.to_state and 
                        t.action_fingerprint == transition.action_fingerprint)
            ]
        
        # Remove from to_state index
        if transition.to_state in self.transitions_to:
            self.transitions_to[transition.to_state] = [
                t for t in self.transitions_to[transition.to_state]
                if not (t.from_state == transition.from_state and 
                        t.action_fingerprint == transition.action_fingerprint)
            ]
        
        # Remove from action_fingerprint index
        if transition.action_fingerprint in self.action_transitions:
            self.action_transitions[transition.action_fingerprint] = [
                t for t in self.action_transitions[transition.action_fingerprint]
                if not (t.from_state == transition.from_state and 
                        t.to_state == transition.to_state)
            ]
    
    def _mark_transitions_as_ambiguous(self, from_state: str, action_fingerprint: str) -> None:
        """Mark all transitions with given (from_state, action_fingerprint) as ambiguous"""
        transitions_to_update = []
        
        # Find all transitions to update
        for key, transition in self.transitions.items():
            if (transition.from_state == from_state and 
                transition.action_fingerprint == action_fingerprint and
                not transition.ambiguous):
                transitions_to_update.append((key, transition))
        
        # Update each transition
        for key, old_transition in transitions_to_update:
            new_transition = StateTransition(
                from_state=old_transition.from_state,
                to_state=old_transition.to_state,
                action_fingerprint=old_transition.action_fingerprint,
                reliability=old_transition.reliability,
                observation_count=old_transition.observation_count,
                ambiguous=True,  # Mark as ambiguous
                reversible=old_transition.reversible
            )
            
            # Update with deduplication
            self._update_transition_with_deduplication(key, new_transition)
            
            # Record ambiguity event
            self._record_ambiguity_established_event(
                old_transition.from_state,
                old_transition.to_state,
                old_transition.action_fingerprint
            )
    
    def _establish_reversibility_if_possible(self, transition: StateTransition) -> None:
        """Check and establish reversibility if reverse transition exists"""
        reverse_key = (
            transition.to_state,
            transition.from_state,
            transition.action_fingerprint
        )
        
        if reverse_key in self.transitions:
            # Both directions observed - establish reversibility
            forward = self.transitions[(transition.from_state, transition.to_state, transition.action_fingerprint)]
            reverse = self.transitions[reverse_key]
            
            # Update forward transition
            updated_forward = StateTransition(
                from_state=forward.from_state,
                to_state=forward.to_state,
                action_fingerprint=forward.action_fingerprint,
                reliability=forward.reliability,
                observation_count=forward.observation_count,
                ambiguous=forward.ambiguous,
                reversible=True
            )
            self._update_transition_with_deduplication(
                (forward.from_state, forward.to_state, forward.action_fingerprint),
                updated_forward
            )
            
            # Update reverse transition
            updated_reverse = StateTransition(
                from_state=reverse.from_state,
                to_state=reverse.to_state,
                action_fingerprint=reverse.action_fingerprint,
                reliability=reverse.reliability,
                observation_count=reverse.observation_count,
                ambiguous=reverse.ambiguous,
                reversible=True
            )
            self._update_transition_with_deduplication(reverse_key, updated_reverse)
            
            # Record reversibility event
            self._record_reversibility_established_event(
                forward.from_state, forward.to_state, forward.action_fingerprint
            )
    
    # ========================================================================
    # Query Surface - Read-Only Navigation Memory
    # ========================================================================
    
    def get_neighbors(self, state_hash: str) -> List[StateTransition]:
        """Get all transitions originating from this state"""
        return self.transitions_from.get(state_hash, [])
    
    def get_transitions(self, from_state: str) -> List[StateTransition]:
        """Alias for get_neighbors (maintains interface consistency)"""
        return self.get_neighbors(from_state)
    
    def get_transitions_to(self, to_state: str) -> List[StateTransition]:
        """Get all transitions leading to this state"""
        return self.transitions_to.get(to_state, [])
    
    def has_seen_state(self, state_hash: str) -> bool:
        """Check if state has been observed"""
        return state_hash in self.state_nodes
    
    def transition_stats(
        self,
        from_state: str,
        action_fingerprint: str
    ) -> Dict[str, Any]:
        """
        Get statistics for transitions from state with given action
        
        Note: Returns raw statistics only, no ranking or interpretation
        Returns observed frequency measurements only, not usefulness
        """
        transitions = [
            t for t in self.transitions_from.get(from_state, [])
            if t.action_fingerprint == action_fingerprint
        ]
        
        if not transitions:
            return {
                "transition_count": 0,
                "destinations": [],
                "observed_reliability_mean": 0.0,
                "total_observations": 0,
                "ambiguous": False
            }
        
        destinations = [t.to_state for t in transitions]
        total_observations = sum(t.observation_count for t in transitions)
        avg_reliability = sum(t.reliability * t.observation_count for t in transitions) / total_observations
        is_ambiguous = any(t.ambiguous for t in transitions)
        
        return {
            "transition_count": len(transitions),
            "destinations": destinations,
            "observed_reliability_mean": float(avg_reliability),  # Weighted mean of observed reliabilities
            "total_observations": total_observations,
            "ambiguous": is_ambiguous
        }
    
    def get_state_node(self, state_hash: str) -> Optional[StateNode]:
        """Get state node by hash"""
        return self.state_nodes.get(state_hash)
    
    def get_all_states(self) -> List[StateNode]:
        """Get all observed states (no ordering implied)"""
        return list(self.state_nodes.values())
    
    def get_all_transitions(self) -> List[StateTransition]:
        """Get all observed transitions (no ordering implied)"""
        return list(self.transitions.values())
    
    # ========================================================================
    # Graph Statistics - Measurement Only
    # ========================================================================
    
    def compute_graph_stats(self) -> Dict[str, Any]:
        """
        Compute graph statistics - no interpretation
        
        Note: All statistics are frequency measurements only,
        not rankings of usefulness or importance
        """
        total_states = len(self.state_nodes)
        total_transitions = len(self.transitions)
        
        if total_transitions == 0:
            return {
                "total_states": total_states,
                "total_transitions": total_transitions,
                "average_out_degree": 0.0,
                "reversible_transitions": 0,
                "ambiguous_transitions": 0,
                "most_frequent_state": None,  # Statistical frequency only
                "most_frequent_action": None  # Statistical frequency only
            }
        
        # Average out-degree (statistical measurement)
        total_out_edges = sum(len(transitions) for transitions in self.transitions_from.values())
        avg_out_degree = total_out_edges / total_states if total_states > 0 else 0.0
        
        # Reversible transitions (count)
        reversible_count = sum(1 for t in self.transitions.values() if t.reversible is True)
        
        # Ambiguous transitions (count)
        ambiguous_count = sum(1 for t in self.transitions.values() if t.ambiguous)
        
        # Most frequent state (by visit count)
        most_frequent_state = max(self.state_nodes.values(), key=lambda n: n.visit_count, default=None)
        
        # Most frequent action (by total observation count)
        action_counts: Dict[str, int] = {}
        for transition in self.transitions.values():
            action_counts[transition.action_fingerprint] = \
                action_counts.get(transition.action_fingerprint, 0) + transition.observation_count
        
        most_frequent_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None
        
        return {
            "total_states": total_states,
            "total_transitions": total_transitions,
            "average_out_degree": float(avg_out_degree),
            "reversible_transitions": reversible_count,
            "ambiguous_transitions": ambiguous_count,
            "most_frequent_state": most_frequent_state.state_hash if most_frequent_state else None,
            "most_frequent_state_count": most_frequent_state.visit_count if most_frequent_state else 0,
            "most_frequent_action": most_frequent_action,  # Statistical frequency only
            "most_frequent_action_count": action_counts.get(most_frequent_action, 0) if most_frequent_action else 0
        }
    
    # ========================================================================
    # Event Recording and Persistence - FIXED
    # ========================================================================
    
    def _record_state_observed_event(self, state_hash: str, timestamp: float) -> None:
        """Record state observation event (deduplicated)"""
        event = GraphEvent(
            event_type=GraphEventType.STATE_OBSERVED,
            timestamp=timestamp,
            data={"state_hash": state_hash}
        )
        self.event_history.append(event)
    
    def _record_transition_observed_event(
        self,
        from_state: str,
        to_state: str,
        action_fingerprint: str,
        reliability: float,
        timestamp: float,
        ambiguous: bool
    ) -> None:
        """Record transition observation event"""
        event = GraphEvent(
            event_type=GraphEventType.TRANSITION_OBSERVED,
            timestamp=timestamp,
            data={
                "from_state": from_state,
                "to_state": to_state,
                "action_fingerprint": action_fingerprint,
                "reliability": reliability,
                "ambiguous": ambiguous
            }
        )
        self.event_history.append(event)
    
    def _record_reversibility_established_event(
        self,
        from_state: str,
        to_state: str,
        action_fingerprint: str
    ) -> None:
        """Record reversibility established event"""
        event = GraphEvent(
            event_type=GraphEventType.TRANSITION_REVERSIBILITY_ESTABLISHED,
            timestamp=time.time(),
            data={
                "from_state": from_state,
                "to_state": to_state,
                "action_fingerprint": action_fingerprint
            }
        )
        self.event_history.append(event)
    
    def _record_ambiguity_established_event(
        self,
        from_state: str,
        to_state: str,
        action_fingerprint: str
    ) -> None:
        """Record ambiguity established event"""
        event = GraphEvent(
            event_type=GraphEventType.TRANSITION_AMBIGUITY_ESTABLISHED,
            timestamp=time.time(),
            data={
                "from_state": from_state,
                "to_state": to_state,
                "action_fingerprint": action_fingerprint
            }
        )
        self.event_history.append(event)
    
    def _persist_events(self) -> None:
        """Append new events to event log"""
        if not self.graph_path:
            return
        
        # Get events not yet persisted
        if not hasattr(self, '_last_persisted_index'):
            self._last_persisted_index = 0
        
        new_events = self.event_history[self._last_persisted_index:]
        
        if not new_events:
            return
        
        with open(self.graph_path, 'a') as f:
            for event in new_events:
                f.write(json.dumps(event.to_dict()) + '\n')
        
        self._last_persisted_index = len(self.event_history)
    
    def _load_graph(self) -> None:
        """Rebuild graph from event log"""
        if not self.graph_path or not os.path.exists(self.graph_path):
            return
        
        try:
            with open(self.graph_path, 'r') as f:
                for line in f:
                    if line.strip():
                        event_dict = json.loads(line)
                        
                        # Recreate event
                        event = GraphEvent(
                            event_type=GraphEventType(event_dict["event_type"]),
                            timestamp=event_dict["timestamp"],
                            data=event_dict["data"]
                        )
                        
                        # Apply event to rebuild graph
                        self._apply_event(event)
                        
                        # Record in history
                        self.event_history.append(event)
            
            # Set persisted index
            self._last_persisted_index = len(self.event_history)
        
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load graph: {e}")
    
    def _apply_event(self, event: GraphEvent) -> None:
        """Apply event to rebuild graph state"""
        if event.event_type == GraphEventType.STATE_OBSERVED:
            state_hash = event.data["state_hash"]
            timestamp = event.timestamp
            
            # Check for duplicate observation
            observation_key = (state_hash, timestamp)
            if observation_key in self.state_observations:
                return
                
            self.state_observations.add(observation_key)
            
            if state_hash in self.state_nodes:
                existing = self.state_nodes[state_hash]
                node = StateNode(
                    state_hash=state_hash,
                    first_seen=existing.first_seen,
                    last_seen=timestamp,
                    visit_count=existing.visit_count + 1
                )
            else:
                node = StateNode(
                    state_hash=state_hash,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    visit_count=1
                )
            
            self.state_nodes[state_hash] = node
            
        elif event.event_type == GraphEventType.TRANSITION_OBSERVED:
            from_state = event.data["from_state"]
            to_state = event.data["to_state"]
            action_fp = event.data["action_fingerprint"]
            reliability = event.data["reliability"]
            ambiguous = event.data.get("ambiguous", False)
            
            # Ensure states exist (create placeholder nodes if needed)
            if from_state not in self.state_nodes:
                self.state_nodes[from_state] = StateNode(
                    state_hash=from_state,
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    visit_count=1
                )
                self.state_observations.add((from_state, event.timestamp))
            
            if to_state not in self.state_nodes:
                self.state_nodes[to_state] = StateNode(
                    state_hash=to_state,
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    visit_count=1
                )
                self.state_observations.add((to_state, event.timestamp))
            
            # Create or update transition
            key = (from_state, to_state, action_fp)
            if key in self.transitions:
                existing = self.transitions[key]
                transition = self._update_transition(existing, reliability, ambiguous)
            else:
                transition = StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    action_fingerprint=action_fp,
                    reliability=reliability,
                    observation_count=1,
                    ambiguous=ambiguous,
                    reversible=None
                )
            
            self._update_transition_with_deduplication(key, transition)
            
        elif event.event_type == GraphEventType.TRANSITION_REVERSIBILITY_ESTABLISHED:
            from_state = event.data["from_state"]
            to_state = event.data["to_state"]
            action_fp = event.data["action_fingerprint"]
            
            # Update both directions to reversible
            forward_key = (from_state, to_state, action_fp)
            reverse_key = (to_state, from_state, action_fp)
            
            if forward_key in self.transitions and reverse_key in self.transitions:
                forward = self.transitions[forward_key]
                reverse = self.transitions[reverse_key]
                
                updated_forward = StateTransition(
                    from_state=forward.from_state,
                    to_state=forward.to_state,
                    action_fingerprint=forward.action_fingerprint,
                    reliability=forward.reliability,
                    observation_count=forward.observation_count,
                    ambiguous=forward.ambiguous,
                    reversible=True
                )
                
                updated_reverse = StateTransition(
                    from_state=reverse.from_state,
                    to_state=reverse.to_state,
                    action_fingerprint=reverse.action_fingerprint,
                    reliability=reverse.reliability,
                    observation_count=reverse.observation_count,
                    ambiguous=reverse.ambiguous,
                    reversible=True
                )
                
                self._update_transition_with_deduplication(forward_key, updated_forward)
                self._update_transition_with_deduplication(reverse_key, updated_reverse)
                
        elif event.event_type == GraphEventType.TRANSITION_AMBIGUITY_ESTABLISHED:
            from_state = event.data["from_state"]
            to_state = event.data["to_state"]
            action_fp = event.data["action_fingerprint"]
            
            # Mark the specific transition as ambiguous
            key = (from_state, to_state, action_fp)
            if key in self.transitions:
                old_transition = self.transitions[key]
                if not old_transition.ambiguous:
                    new_transition = StateTransition(
                        from_state=old_transition.from_state,
                        to_state=old_transition.to_state,
                        action_fingerprint=old_transition.action_fingerprint,
                        reliability=old_transition.reliability,
                        observation_count=old_transition.observation_count,
                        ambiguous=True,
                        reversible=old_transition.reversible
                    )
                    self._update_transition_with_deduplication(key, new_transition)
    
    # ========================================================================
    # Export and Serialization
    # ========================================================================
    
    def export_graph(self, path: str) -> None:
        """Export complete graph to JSON file"""
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "graph_version": "1.0",
                "total_states": len(self.state_nodes),
                "total_transitions": len(self.transitions),
                "note": "Statistics are frequency measurements only, not rankings"
            },
            "states": [asdict(node) for node in self.state_nodes.values()],
            "transitions": [asdict(trans) for trans in self.transitions.values()]
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_event_log(self, path: str) -> None:
        """Export event log to JSON file"""
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_events": len(self.event_history)
            },
            "events": [event.to_dict() for event in self.event_history]
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

# ============================================================================
# Utility Functions
# ============================================================================

def create_navigation_graph(storage_path: str) -> EnvironmentNavigationGraph:
    """Create navigation graph with persistent storage"""
    return EnvironmentNavigationGraph(graph_path=storage_path)

def create_memory_only_graph() -> EnvironmentNavigationGraph:
    """Create in-memory navigation graph (for testing)"""
    return EnvironmentNavigationGraph(graph_path=None)