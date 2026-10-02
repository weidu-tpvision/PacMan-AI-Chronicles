"""
Reinforcement Learning subsystem:
- Deep Q-Networks (PyTorch CNN)
- Approximate Q-Learning (TD-error updates)
- Direct Policy Search (Cross-Entropy Optimization)
- Model checkpoints, weights, and training telemetry in rl/weights/
"""

import os

from rl.dqn_model import ACTION_TO_IDX, ACTIONS, IDX_TO_ACTION, PacmanDQN, encode_state
from rl.optimize_policy import optimize_policy
from rl.train_dqn import train_dqn
from rl.train_q_learning import train_q_learning

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

__all__ = [
    "PacmanDQN",
    "encode_state",
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "train_dqn",
    "train_q_learning",
    "optimize_policy",
    "WEIGHTS_DIR",
]
