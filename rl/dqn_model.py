"""
Deep Q-Network (DQN) PyTorch Architecture for Pac-Man.

An observation is a pair (grid, scalars):
- grid: binary spatial planes (maze, pellets, actor positions and headings) plus the
  Scatter/Chase planes, which change what ghosts do locally;
- scalars: stall progress and remaining horizon, which only change how much the future
  is worth and therefore feed the dense layer directly.
The encoding is a function of the current game state only (no history); see dqn.md.
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object

from core.environment import DEFAULT_MAX_STEPS, MODE_CYCLE, SCATTER_STEPS
from core.maze_data import GRID_HEIGHT, GRID_WIDTH, MAZE_LAYOUT, START_POSITIONS, WALL_CELLS

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

GHOST_SLOTS = 3

# Grid plane layout
CH_WALLS = 0
CH_PELLETS = 1
CH_PACMAN = 2
CH_GHOSTS = 3                                   # one plane per ghost identity
CH_PACMAN_HEADING = CH_GHOSTS + GHOST_SLOTS     # 4 one-hot planes, set at Pac-Man's tile
CH_GHOST_HEADINGS = CH_PACMAN_HEADING + 4       # 4 one-hot planes per ghost, at its tile
CH_SCATTER = CH_GHOST_HEADINGS + 4 * GHOST_SLOTS
CH_CYCLE_PHASE = CH_SCATTER + 1
CH_COORD_X = CH_CYCLE_PHASE + 1
CH_COORD_Y = CH_COORD_X + 1
NUM_CHANNELS = CH_COORD_Y + 1

# Scalar inputs to the dense layer
SCALAR_STALL = 0
SCALAR_HORIZON = 1
NUM_SCALARS = 2

# Steps without a pellet after which the trainer applies its stall penalty; the stall
# scalar saturates here, which keeps that penalty a function of the observation.
STALL_STEPS = 45

# Full-resolution network: ConvNet trunk with standard 3x3 convs and a dilated conv layer (d=2)
# to expand the receptive field across maze corridors without pooling, then a 1x1 reduction
# to REDUCED_CHANNELS before the dense layer.
CONV_WIDTH = 32
CONV_DILATIONS = (1, 1, 1, 1, 2, 1, 1)
CONV_LAYERS = len(CONV_DILATIONS)
REDUCED_CHANNELS = 16
HIDDEN_UNITS = 128

WALL_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        WALL_MAP[y, x] = 1.0

# Precomputed static CoordConv spatial coordinate planes normalized to [0, 1]
COORD_X_MAP = np.tile(np.linspace(0.0, 1.0, GRID_WIDTH, dtype=np.float32), (GRID_HEIGHT, 1))
COORD_Y_MAP = np.tile(np.linspace(0.0, 1.0, GRID_HEIGHT, dtype=np.float32)[:, None], (1, GRID_WIDTH))


def _on_grid(x: int, y: int) -> bool:
    return 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH


def encode_state(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
    last_move: Optional[str] = None,
    ghost_dirs: Optional[List[str]] = None,
    mode_step: int = 0,
    steps_without_pellet: int = 0,
    steps_remaining: int = DEFAULT_MAX_STEPS,
    horizon: int = DEFAULT_MAX_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the current game state as (grid [NUM_CHANNELS, H, W], scalars [NUM_SCALARS]).

    All values are binary or normalized to [0, 1]. `mode_step` is the environment clock
    *before* the step being decided (the ghosts' next move uses mode_step + 1).
    """
    grid = np.zeros((NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    grid[CH_WALLS] = WALL_MAP

    for fx, fy in pellets:
        if _on_grid(fx, fy):
            grid[CH_PELLETS, fy, fx] = 1.0

    px, py = pacman_pos
    if _on_grid(px, py):
        grid[CH_PACMAN, py, px] = 1.0
        if last_move in ACTION_TO_IDX:
            grid[CH_PACMAN_HEADING + ACTION_TO_IDX[last_move], py, px] = 1.0

    ghost_dirs = ghost_dirs or ["up"] * GHOST_SLOTS
    for index, (gx, gy) in enumerate(ghost_positions[:GHOST_SLOTS]):
        if not _on_grid(gx, gy):
            continue
        grid[CH_GHOSTS + index, gy, gx] = 1.0
        if index < len(ghost_dirs) and ghost_dirs[index] in ACTION_TO_IDX:
            grid[CH_GHOST_HEADINGS + 4 * index + ACTION_TO_IDX[ghost_dirs[index]], gy, gx] = 1.0

    mode_in_cycle = mode_step % MODE_CYCLE  # phase of the ghosts' next move (clock mode_step + 1)
    grid[CH_SCATTER] = float(mode_in_cycle < SCATTER_STEPS)
    grid[CH_CYCLE_PHASE] = mode_in_cycle / max(1, MODE_CYCLE - 1)
    grid[CH_COORD_X] = COORD_X_MAP
    grid[CH_COORD_Y] = COORD_Y_MAP

    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    scalars[SCALAR_STALL] = min(max(steps_without_pellet, 0), STALL_STEPS) / STALL_STEPS
    scalars[SCALAR_HORIZON] = min(max(steps_remaining, 0), max(1, horizon)) / max(1, horizon)
    return grid, scalars


# Canonical pellet ordering for bitmask packing (150 pellets, excluding Pac-Man's start tile)
PELLET_COORDS = np.array(
    sorted([
        (x, y)
        for y, row in enumerate(MAZE_LAYOUT)
        for x, char in enumerate(row)
        if char == "." and (x, y) != tuple(START_POSITIONS["pacman"])
    ]),
    dtype=np.int32,
)
N_PELLETS = len(PELLET_COORDS)
N_PELLET_BYTES = (N_PELLETS + 7) // 8
PELLET_TO_IDX = {tuple(p): i for i, p in enumerate(PELLET_COORDS)}
P_XS = PELLET_COORDS[:, 0]
P_YS = PELLET_COORDS[:, 1]


def pack_pellets(pellets_set: set) -> np.ndarray:
    """Pack active pellet positions into a compact bitmask (19 bytes for 150 pellets)."""
    bits = np.zeros(N_PELLET_BYTES * 8, dtype=np.uint8)
    for p in pellets_set:
        idx = PELLET_TO_IDX.get(p)
        if idx is not None:
            bits[idx] = 1
    return np.packbits(bits)


def decode_grids_batch(
    px: np.ndarray,
    py: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    p_head: np.ndarray,
    g_head: np.ndarray,
    mode_step: np.ndarray,
    pellet_bits: np.ndarray,
):
    """Vectorized batch decoding of compact state arrays into float32 (B, NUM_CHANNELS, H, W) PyTorch tensors.

    Reconstructs exact full spatial grid representations in ~1 ms for 128 items,
    guaranteeing 100% numerical parity with encode_state().
    """
    B = len(px)
    grids = np.zeros((B, NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    grids[:, CH_WALLS] = WALL_MAP

    # Pellets plane
    unpacked = np.unpackbits(pellet_bits, axis=1)[:, :N_PELLETS]
    b_idx, p_idx = np.where(unpacked)
    grids[b_idx, CH_PELLETS, P_YS[p_idx], P_XS[p_idx]] = 1.0

    # Pac-Man position and heading
    b_range = np.arange(B)
    grids[b_range, CH_PACMAN, py, px] = 1.0
    valid_ph = p_head >= 0
    if np.any(valid_ph):
        grids[b_range[valid_ph], CH_PACMAN_HEADING + p_head[valid_ph], py[valid_ph], px[valid_ph]] = 1.0

    # Ghost positions and headings
    for k in range(GHOST_SLOTS):
        grids[b_range, CH_GHOSTS + k, gy[:, k], gx[:, k]] = 1.0
        valid_gh = g_head[:, k] >= 0
        if np.any(valid_gh):
            grids[b_range[valid_gh], CH_GHOST_HEADINGS + 4 * k + g_head[valid_gh, k], gy[valid_gh, k], gx[valid_gh, k]] = 1.0

    # Scatter and cycle phase planes
    m_cycle = mode_step % MODE_CYCLE
    grids[:, CH_SCATTER] = (m_cycle < SCATTER_STEPS).astype(np.float32)[:, None, None]
    grids[:, CH_CYCLE_PHASE] = (m_cycle / max(1, MODE_CYCLE - 1)).astype(np.float32)[:, None, None]

    # CoordConv coordinate planes
    grids[:, CH_COORD_X] = COORD_X_MAP
    grids[:, CH_COORD_Y] = COORD_Y_MAP

    if TORCH_AVAILABLE:
        return torch.from_numpy(grids)
    return grids


class CompactObservation:
    """Wrapper holding full observation (grid, scalars) and compact representation.

    Implements tuple unpacking (grid, scalars = obs) and indexing (obs[0], obs[1])
    for seamless 100% backward compatibility with all agents and evaluation routines.
    """
    __slots__ = ("grid", "scalars", "compact")

    def __init__(self, grid: np.ndarray, scalars: np.ndarray, compact: tuple):
        self.grid = grid
        self.scalars = scalars
        self.compact = compact

    def __iter__(self):
        yield self.grid
        yield self.scalars

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.grid
        if idx == 1:
            return self.scalars
        raise IndexError(f"Index out of range for CompactObservation: {idx}")

    def __len__(self):
        return 2


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Dueling Deep Q-Network: a stack of 3x3 convolutions at full grid resolution (no
        pooling, so exact tile offsets between actors survive), a 1x1 channel reduction,
        then one dense layer that also receives the scalar inputs, and V / A heads
        combined as Q = V + A - mean(A).
        """

        def __init__(
            self,
            in_channels: int = NUM_CHANNELS,
            num_scalars: int = NUM_SCALARS,
            num_actions: int = 4,
            dilations: Tuple[int, ...] = CONV_DILATIONS,
        ):
            super().__init__()
            layers, channels = [], in_channels
            for d in dilations:
                layers += [
                    nn.Conv2d(channels, CONV_WIDTH, kernel_size=3, padding=d, dilation=d),
                    nn.ReLU(),
                ]
                channels = CONV_WIDTH
            layers += [nn.Conv2d(channels, REDUCED_CHANNELS, kernel_size=1), nn.ReLU(), nn.Flatten()]
            self.trunk = nn.Sequential(*layers)
            self.feature_head = nn.Sequential(
                nn.Linear(REDUCED_CHANNELS * GRID_HEIGHT * GRID_WIDTH + num_scalars, HIDDEN_UNITS),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(HIDDEN_UNITS, 1)
            self.advantage_head = nn.Linear(HIDDEN_UNITS, num_actions)

        def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
            features = self.feature_head(torch.cat([self.trunk(grid), scalars], dim=1))
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class PacmanDQN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PacmanDQN. Install torch to enable.")
