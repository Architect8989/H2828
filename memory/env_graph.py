"""
Environment Graph - Monotonic Memory of Environmental Facts
Accumulates validated changes without interpretation or inference.
"""

import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from core.logger import log_event


class EnvGraph:
    """
    Append-only, monotonic memory of environmental changes.
    Records facts without deletion or contradiction.
    """
    
    def __init__(self, storage_path: str = "memory/long_term/env_facts.jsonl"):
        """
        Initialize environment memory.
        
        Args:
            storage_path: Path to append-only storage file
        """
        self.storage_path = storage_path
        self._loaded_fingerprints: Set[str] = set()
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Create storage file and directory if they don't exist."""
        directory = os.path.dirname(self.storage_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Load existing fingerprints for deduplication
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                fact = json.loads(line.strip())
                                fingerprint = fact.get('fingerprint')
                                if fingerprint:
                                    self._loaded_fingerprints.add(fingerprint)
                            except json.JSONDecodeError:
                                log_event(f"EnvGraph: Line {line_num} invalid JSON, skipping")
            except Exception as e:
                # If we can't read existing data, log and start fresh
                log_event(f"EnvGraph: Failed to load existing facts: {e}")
                self._loaded_fingerprints.clear()
        else:
            log_event("EnvGraph: Creating new env_facts storage")
    
    def _compute_fingerprint(self, fact: Dict[str, Any]) -> str:
        """
        Compute deterministic fingerprint for a fact.
        
        Args:
            fact: Fact dictionary
            
        Returns:
            SHA256 fingerprint as hex string
        """
        # Create stable string representation
        fact_str = json.dumps(fact, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(fact_str.encode()).hexdigest()
    
    def update(self, deltas: List[Any]) -> None:
        """
        Add validated environmental changes to memory.
        
        Args:
            deltas: List of validated Delta objects
        """
        try:
            new_facts = []
            
            for delta in deltas:
                try:
                    # Extract basic facts from delta (no interpretation)
                    fact = {
                        "change_type": getattr(delta, 'change_type', 'unknown'),
                        "timestamp": getattr(delta, 'timestamp', datetime.now(timezone.utc)).isoformat(),
                        "confidence": getattr(delta, 'confidence', 1.0),
                        "recorded_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Add before/after states if available
                    if hasattr(delta, 'before_state'):
                        fact["before_state"] = getattr(delta, 'before_state', {})
                    if hasattr(delta, 'after_state'):
                        fact["after_state"] = getattr(delta, 'after_state', {})
                    
                    # Compute fingerprint for deduplication
                    fingerprint = self._compute_fingerprint(fact)
                    fact["fingerprint"] = fingerprint
                    
                    # Skip if already recorded
                    if fingerprint in self._loaded_fingerprints:
                        continue
                    
                    new_facts.append(fact)
                    self._loaded_fingerprints.add(fingerprint)
                    
                except Exception as delta_error:
                    # Individual delta failure doesn't stop the process
                    log_event(f"EnvGraph: Failed to process delta: {delta_error}")
                    continue
            
            # Append new facts to storage
            if new_facts:
                self._store_facts(new_facts)
                
        except Exception as update_error:
            # Isolated failure: log but don't crash
            log_event(f"EnvGraph: Update failed: {update_error}")
    
    def _store_facts(self, facts: List[Dict[str, Any]]) -> None:
        """
        Append facts to storage with immediate durability.
        
        Args:
            facts: List of fact dictionaries to store
        """
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                for fact in facts:
                    line = json.dumps(fact, separators=(',', ':')) + '\n'
                    f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as storage_error:
            log_event(f"EnvGraph: Storage failed: {storage_error}")
            # Don't re-raise - memory failure shouldn't crash the experiment
