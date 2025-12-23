#!/usr/bin/env python3
"""
EME-Core Activation Entrypoint
Power switch for the EME nervous system - no intelligence, just activation.
"""

from core.wiring import build_life_loop
from execution.action_executor import Action, ActionType


def main() -> None:
    """Bring EME-Core to life with minimal activation sequence."""
    # Create LifeLoop with all dependencies via wiring layer
    life_loop = build_life_loop()
    
    # Create a simple, deterministic mouse move action
    # Using the Action dataclass directly as defined in execution.action_executor
    experiment_action = Action(
        type=ActionType.MOVE,
        parameters=(100, 100)
    )
    
    # Execute the experiment - let any exceptions propagate
    life_loop.run_experiment(experiment_action)


if __name__ == "__main__":
    main()
