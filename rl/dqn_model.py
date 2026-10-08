"""
Deep Q-Network (DQN) PyTorch Architecture for Pac-Man.
Encodes the 21x19 maze into an unentangled 6-channel spatial tensor with temporal velocity tracking:
  Channel 0: Static Walls (1.0 = Wall, 0.0 = Corridor)
  Channel 1: Pellets remaining (1.0 = Pellet, 0.0 = Empty)
  Channel 2: Pac-Man current position at t (1.0 = Pac-Man)
  Channel 3: Ghosts current positions at t (1.0 = Ghost)
  Channel 4: Pac-Man previous position at t-1 (1.0 = Pac-Man)
  Channel 5: Ghosts previous positions at t-1 (1.0 = Ghost)
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

from core.maze_data import GRID_HEIGHT, GRID_WIDTH, WALL_CELLS

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

NUM_CHANNELS = 6
FRAME_STACK_SIZE = NUM_CHANNELS  # Backwards-compatibility alias

# Precompute static walls mask (Channel 0)
WALL_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        WALL_MAP[y, x] = 1.0


def encode_state(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
    prev_pacman_pos: Optional[Tuple[int, int]] = None,
    prev_ghost_positions: Optional[List[Tuple[int, int]]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Encode game state into an unentangled (6, 21, 19) float32 binary tensor:
      Channel 0: Static Walls (1.0 = Wall, 0.0 = Corridor)
      Channel 1: Pellets remaining (1.0 = Pellet, 0.0 = Empty)
      Channel 2: Pac-Man current position at t (1.0 = Pac-Man)
      Channel 3: Ghosts current positions at t (1.0 = Ghost)
      Channel 4: Pac-Man previous position at t-1 (1.0 = Pac-Man)
      Channel 5: Ghosts previous positions at t-1 (1.0 = Ghost)

    All channels contain strictly discrete binary values {0.0, 1.0} ensuring
    optimal gradient flow under ReLU without sign or polarity interference.
    """
    state = np.zeros((NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    state[0] = WALL_MAP

    for fx, fy in pellets:
        if 0 <= fy < GRID_HEIGHT and 0 <= fx < GRID_WIDTH:
            state[1, fy, fx] = 1.0

    px, py = pacman_pos
    if 0 <= py < GRID_HEIGHT and 0 <= px < GRID_WIDTH:
        state[2, py, px] = 1.0

    for gx, gy in ghost_positions:
        if 0 <= gy < GRID_HEIGHT and 0 <= gx < GRID_WIDTH:
            state[3, gy, gx] = 1.0

    # Temporal positions for velocity / direction inference
    ppx, ppy = prev_pacman_pos if prev_pacman_pos is not None else pacman_pos
    if 0 <= ppy < GRID_HEIGHT and 0 <= ppx < GRID_WIDTH:
        state[4, ppy, ppx] = 1.0

    prev_ghosts = prev_ghost_positions if prev_ghost_positions is not None else ghost_positions
    for pgx, pgy in prev_ghosts:
        if 0 <= pgy < GRID_HEIGHT and 0 <= pgx < GRID_WIDTH:
            state[5, pgy, pgx] = 1.0

    return state


def encode_frame(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
) -> np.ndarray:
    """Backwards-compatibility alias for single-step encoding without history."""
    return encode_state(pacman_pos, ghost_positions, pellets)


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Deep Q-Network with temporal velocity tracking across 6 unentangled channels.
        Uses MaxPool2d(2) to provide a 10x10 receptive field for global maze vision.
        """

        def __init__(self, in_channels: int = NUM_CHANNELS, num_actions: int = 4, dueling: bool = True):
            super().__init__()
            self.dueling = dueling

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 21x19 -> 10x9
            )

            # 64 channels * 10 height * 9 width = 5,760 features
            if dueling:
                self.feature_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 10 * 9, 128),
                    nn.ReLU(),
                )
                self.value_head = nn.Linear(128, 1)
                self.advantage_head = nn.Linear(128, num_actions)
            else:
                # Retain the original module names and shapes so historical checkpoints load.
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 10 * 9, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.conv(x)
            if not self.dueling:
                return self.fc(features)
            features = self.feature_head(features)
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class PacmanDQN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PacmanDQN. Install torch to enable.")
