#!/usr/bin/env python3
"""
EME-Core Activation Entrypoint
Power switch for the EME nervous system - no intelligence, just activation.
"""

from execution.opencu_backend import OpenCUBackend
from execution.action_executor import ActionExecutor
from core.life_loop import LifeLoop


def main() -> None:
    """Bring EME-Core to life with minimal activation sequence."""
    # Create OpenCU backend for low-level OS control
    backend = OpenCUBackend()
    
    # Create action executor with the backend
    executor = ActionExecutor(backend)
    
    # Create life loop with the executor - using exact constructor signature
    life_loop = LifeLoop(action_executor=executor)
    
    # Create a simple, deterministic mouse move action
    # Using the exact Action constructor pattern from execution/action_executor.py
    # The Action class requires id, executable_call, and automatically computes fingerprint
    experiment_action = executor.create_action(
        action_id="first_mouse_move",
        executable_call=lambda: backend.mouse_move_abs(100, 100)
    )
    
    # Execute the experiment - let any exceptions propagate
    life_loop.run_experiment(experiment_action)


if __name__ == "__main__":
    main()
