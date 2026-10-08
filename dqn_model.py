"""
Backward-compatibility facade for DQN model architecture.
All neural models have been modularized under `rl.dqn_model`.
"""

from rl.dqn_model import (
    ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    PacmanDQN,
    encode_frame,
    encode_state,
    NUM_CHANNELS,
)

__all__ = [
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "PacmanDQN",
    "encode_frame",
    "encode_state",
    "NUM_CHANNELS",
]
