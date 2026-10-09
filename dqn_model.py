"""
Backward-compatibility facade for DQN model architecture.
All neural models have been modularized under `rl.dqn_model`.
"""

from rl.dqn_model import (
    ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NUM_CHANNELS,
    NUM_SCALARS,
    PacmanDQN,
    encode_state,
)

__all__ = [
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "NUM_CHANNELS",
    "NUM_SCALARS",
    "PacmanDQN",
    "encode_state",
]
