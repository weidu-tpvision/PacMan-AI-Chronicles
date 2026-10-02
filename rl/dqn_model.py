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

# Precompute static walls mask (Channel 0)
WALL_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        WALL_MAP[y, x] = 1.0


def encode_state(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
) -> np.ndarray:
    """
    Encode game state into a (4, 21, 19) float32 tensor:
    Channel 0: Walls
    Channel 1: Pac-Man location
    Channel 2: Ghost locations
    Channel 3: Pellets
    """
    state = np.zeros((4, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    # Channel 0: Walls
    state[0] = WALL_MAP

    # Channel 1: Pac-Man
    px, py = pacman_pos
    if 0 <= py < GRID_HEIGHT and 0 <= px < GRID_WIDTH:
        state[1, py, px] = 1.0

    # Channel 2: Ghosts
    for gx, gy in ghost_positions:
        if 0 <= gy < GRID_HEIGHT and 0 <= gx < GRID_WIDTH:
            state[2, gy, gx] = 1.0

    # Channel 3: Pellets
    for fx, fy in pellets:
        if 0 <= fy < GRID_HEIGHT and 0 <= fx < GRID_WIDTH:
            state[3, fy, fx] = 1.0

    return state


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Convolutional Deep Q-Network adapted for the 19x21 Pac-Man maze.
        Extracts spatial corridor and entity features via 2D Convolutions and Max Pooling.
        """

        def __init__(self, in_channels: int = 4, num_actions: int = 4):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 21x19 -> 10x9
            )

            # 64 channels * 10 height * 9 width = 5760 features
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 10 * 9, 128),
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
