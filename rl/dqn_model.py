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
from core.maze_data import GRID_HEIGHT, GRID_WIDTH, WALL_CELLS

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
NUM_CHANNELS = CH_CYCLE_PHASE + 1

# Scalar inputs to the dense layer
SCALAR_STALL = 0
SCALAR_HORIZON = 1
NUM_SCALARS = 2

# Steps without a pellet after which the trainer applies its stall penalty; the stall
# scalar saturates here, which keeps that penalty a function of the observation.
STALL_STEPS = 45

# Full-resolution network: CONV_LAYERS x (3x3 conv, CONV_WIDTH channels) -> receptive
# field (2 * CONV_LAYERS + 1) tiles, then a 1x1 reduction to REDUCED_CHANNELS before
# the dense layer.
CONV_LAYERS = 6
CONV_WIDTH = 32
REDUCED_CHANNELS = 16
HIDDEN_UNITS = 128

WALL_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        WALL_MAP[y, x] = 1.0


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

    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    scalars[SCALAR_STALL] = min(max(steps_without_pellet, 0), STALL_STEPS) / STALL_STEPS
    scalars[SCALAR_HORIZON] = min(max(steps_remaining, 0), max(1, horizon)) / max(1, horizon)
    return grid, scalars


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Dueling Deep Q-Network: a stack of 3x3 convolutions at full grid resolution (no
        pooling, so exact tile offsets between actors survive), a 1x1 channel reduction,
        then one dense layer that also receives the scalar inputs, and V / A heads
        combined as Q = V + A - mean(A).
        """

        def __init__(self, in_channels: int = NUM_CHANNELS, num_scalars: int = NUM_SCALARS, num_actions: int = 4):
            super().__init__()
            layers, channels = [], in_channels
            for _ in range(CONV_LAYERS):
                layers += [nn.Conv2d(channels, CONV_WIDTH, kernel_size=3, padding=1), nn.ReLU()]
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
