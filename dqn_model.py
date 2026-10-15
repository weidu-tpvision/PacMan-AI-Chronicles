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
    CONV_LAYERS,
    CONV_DILATIONS,
    PacmanDQN,
    encode_state,
    CompactObservation,
    pack_pellets,
    decode_grids_batch,
    PELLET_COORDS,
    N_PELLETS,
    N_PELLET_BYTES,
)

__all__ = [
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "NUM_CHANNELS",
    "NUM_SCALARS",
    "CONV_LAYERS",
    "CONV_DILATIONS",
    "PacmanDQN",
    "encode_state",
    "CompactObservation",
    "pack_pellets",
    "decode_grids_batch",
    "PELLET_COORDS",
    "N_PELLETS",
    "N_PELLET_BYTES",
]

