"""
Deep Q-Network (DQN) PyTorch Architecture for Pac-Man.
Encodes the 19x21 maze into a 4-channel spatial tensor and outputs Q-values for actions.
"""

from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object

from core.maze_data import GRID_HEIGHT, GRID_WIDTH, WALL_CELLS

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

# Visual cell values
VAL_EMPTY = 0.0
VAL_WALL = -0.5
VAL_PELLET = 0.5
VAL_PACMAN = 1.0
VAL_GHOST = -1.0
FRAME_STACK_SIZE = 3

# Precompute static walls mask
BASE_FRAME = np.full((GRID_HEIGHT, GRID_WIDTH), VAL_EMPTY, dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        BASE_FRAME[y, x] = VAL_WALL


def encode_frame(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
) -> np.ndarray:
    """
    Render the current game state into a single 2D grid frame (21, 19)
    using natural visual Z-ordering:
      Empty: 0.0
      Wall: -0.5
      Pellet: +0.5
      Pac-Man: +1.0
      Ghost: -1.0 (overrides pellet when occupying the same tile)
    """
    frame = BASE_FRAME.copy()

    # 1. Pellets (positive targets)
    for fx, fy in pellets:
        if 0 <= fy < GRID_HEIGHT and 0 <= fx < GRID_WIDTH:
            frame[fy, fx] = VAL_PELLET

    # 2. Pac-Man (self position)
    px, py = pacman_pos
    if 0 <= py < GRID_HEIGHT and 0 <= px < GRID_WIDTH:
        frame[py, px] = VAL_PACMAN

    # 3. Ghosts (highest priority hazard; overrides pellet if on same tile)
    for gx, gy in ghost_positions:
        if 0 <= gy < GRID_HEIGHT and 0 <= gx < GRID_WIDTH:
            frame[gy, gx] = VAL_GHOST

    return frame


def encode_state(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
    history: List[np.ndarray] = None,
    k: int = FRAME_STACK_SIZE,
) -> np.ndarray:
    """
    Encode state with k stacked temporal frames (k, 21, 19).
    If no history is provided, repeats the current frame k times.
    """
    curr = encode_frame(pacman_pos, ghost_positions, pellets)
    if history and len(history) > 0:
        frames = list(history)[-(k - 1):] + [curr]
        while len(frames) < k:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0)
    return np.repeat(curr[np.newaxis, :, :], k, axis=0)


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Deep Q-Network with temporal frame stacking (k=3).
        Preserves exact spatial maze coordinates without destructive MaxPool downsampling.
        """

        def __init__(self, in_channels: int = FRAME_STACK_SIZE, num_actions: int = 4):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                # 1x1 conv to compress channels to 16 without destroying 21x19 coordinate resolution
                nn.Conv2d(64, 16, kernel_size=1),
                nn.ReLU(),
            )

            # 16 channels * 21 height * 19 width = 6,384 features
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * GRID_HEIGHT * GRID_WIDTH, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.conv(x)
            return self.fc(features)
else:
    class PacmanDQN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PacmanDQN. Install torch to enable.")
