"""
Reinforcement Learning subsystem:
- Deep Q-Networks (PyTorch CNN)
- Approximate Q-Learning (TD-error updates)
- Direct Policy Search (Cross-Entropy Optimization)
- Model checkpoints, weights, and training telemetry in rl/weights/

Only the lightweight model module is imported eagerly. Trainers are resolved lazily so
that `import agents` never pulls in training code, pygame (plotting) or torch-only
type annotations.
"""

import importlib
import os

from rl.dqn_model import ACTION_TO_IDX, ACTIONS, IDX_TO_ACTION, PacmanDQN, encode_state

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

_LAZY = {
    "train_dqn": "rl.train_dqn",
    "train_q_learning": "rl.train_q_learning",
    "optimize_policy": "rl.optimize_policy",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'rl' has no attribute {name!r}")


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
