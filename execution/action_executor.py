"""Action Dispatcher

One action → one run() call. No transformation. No retries. No timing.
Pure pass-through executor with zero intelligence.
"""


class ActionExecutor:
    """
    Pure dispatcher between LifeLoop and action.run().
    
    This is a spinal cord: transmits intent without understanding.
    """
    
    def execute(self, action) -> None:
        """
        Execute exactly one action exactly once.
        
        Args:
            action: Any object with a callable 'run' method
            
        Raises:
            AttributeError: If action.run doesn't exist
            Any exception raised by action.run()
        """
        # Contract: verify action has run() method
        if not hasattr(action, 'run'):
            raise AttributeError("Action missing required 'run' method")
        
        if not callable(action.run):
            raise AttributeError("Action 'run' attribute is not callable")
        
        # Execute exactly once
        return action.run()
