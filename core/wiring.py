"""
EME-Core Composition Layer
Pure wiring of concrete components - no behavior, no intelligence.
"""

from execution.opencu_backend import OpenCUBackend
from execution.action_executor import ActionExecutor
from perception.screen_capture import ScreenCapture
from perception.change_detector import ChangeDetector
from memory.action_ledger import ActionLedger
from memory.env_graph import EnvGraph
from core.life_loop import LifeLoop


def build_life_loop() -> LifeLoop:
    """
    Factory function that wires concrete components into a LifeLoop instance.
    
    Returns:
        LifeLoop instance with all dependencies injected
    """
    # Instantiate OS backend
    backend = OpenCUBackend()
    
    # Wire action execution system
    action_executor = ActionExecutor(backend)
    
    # Wire perception system
    perception_capturer = ScreenCapture()
    change_detector = ChangeDetector()
    
    # Wire delta validator (separate from change detector)
    delta_validator = DeltaValidator()
    
    # Wire memory system
    action_ledger = ActionLedger()
    env_graph = EnvGraph()
    
    # Create LifeLoop with all dependencies injected
    life_loop = LifeLoop(
        action_executor=action_executor,
        perception_capturer=perception_capturer,
        change_detector=change_detector,
        delta_validator=delta_validator,
        action_ledger=action_ledger,
        env_graph_updater=env_graph
    )
    
    return life_loop


class DeltaValidator:
    """
    Separate validation component for attribution.
    Preserves epistemic separation: measurement vs validation.
    """
    def __init__(self):
        pass
    
    def validate(self, delta, time_window):
        """
        Validate whether a delta is attributable to the action.
        
        Args:
            delta: Delta object to validate
            time_window: (start, end) tuple of action timing
            
        Returns:
            bool: True if delta is attributable, False otherwise
        """
        # Implementation will be provided by the perception module
        # For now, this is a placeholder that preserves the interface
        return True
