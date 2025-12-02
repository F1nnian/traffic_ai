"""
TrainedAgent: Inference-only agent that uses a saved Q-table.

- Loads a NumPy `.npy` file (default: models/q_table.npy).
- Exposes `act(state)` which chooses the greedy action: argmax_a Q[state, a].
- Compatible with environment observations shaped like (phase, bucketNS, bucketEW).

Usage:
    from src.trainedagent import TrainedAgent
    agent = TrainedAgent()  # or TrainedAgent(path_to_q_table)
    action = agent.act(state)
"""

import os
import numpy as np
from src.config import MODELS_DIR


class TrainedAgent:
    def __init__(self, q_table_path: str | None = None):
        """Load a trained Q-table.

        Args:
            q_table_path: Optional path to a `.npy` file. If None, uses
                `os.path.join(MODELS_DIR, "q_table.npy")`.
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the loaded array does not have at least one action dim.
        """
        default_path = os.path.join(MODELS_DIR, "q_table.npy")
        self.path = q_table_path or default_path

        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                f"Q-table not found at '{self.path}'. Train first or provide a valid path."
            )

        self.q_table = np.load(self.path)
        if self.q_table.ndim < 1:
            raise ValueError("Loaded Q-table has invalid shape.")
        if self.q_table.shape[-1] < 1:
            raise ValueError("Loaded Q-table has no action dimension.")

        self.n_actions = int(self.q_table.shape[-1])

    def act(self, state) -> int:
        """Choose the greedy action for the given discrete state.

        Args:
            state: Iterable of indices (phase, bucketNS, bucketEW). Can be tuple or ndarray.
        Returns:
            int: action index (0..n_actions-1)
        """
        idx = tuple(int(x) for x in state)
        # Validate index length vs Q-table dims
        expected_state_rank = self.q_table.ndim - 1
        if len(idx) != expected_state_rank:
            raise ValueError(
                f"State rank {len(idx)} does not match Q-table state rank {expected_state_rank}."
            )
        return int(np.argmax(self.q_table[idx]))

    def load(self, path: str):
        """Reload a different Q-table from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Q-table not found at '{path}'")
        self.q_table = np.load(path)
        self.path = path
        self.n_actions = int(self.q_table.shape[-1])
