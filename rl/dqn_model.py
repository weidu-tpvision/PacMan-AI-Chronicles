"""
Deep Q-Network (DQN) PyTorch Architecture for Pac-Man.
Encodes the maze using object-identity, motion, and environment-phase channels. Global
features are repeated spatial planes so they can be consumed by the same convolutional
network as the object maps.
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

from core.environment import MODE_CYCLE, SCATTER_STEPS
from core.maze_data import GRID_HEIGHT, GRID_WIDTH, WALL_CELLS

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

GHOST_SLOTS = 3
BASE_CHANNELS = 4  # walls, pellets, Pac-Man now, Pac-Man previously
GHOST_POSITION_CHANNELS = 2 * GHOST_SLOTS
PACMAN_HEADING_CHANNELS = 4
GHOST_HEADING_CHANNELS = 4 * GHOST_SLOTS
SCALAR_CHANNELS = 4  # scatter flag, cycle phase, stall progress, horizon remaining
NUM_CHANNELS = BASE_CHANNELS + GHOST_POSITION_CHANNELS + PACMAN_HEADING_CHANNELS + GHOST_HEADING_CHANNELS + SCALAR_CHANNELS

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
    last_move: Optional[str] = None,
    ghost_dirs: Optional[List[str]] = None,
    mode_step: int = 0,
    steps_without_pellet: int = 0,
    steps_remaining: int = 300,
    horizon: int = 300,
) -> np.ndarray:
    """
    Encode a Markov-oriented observation. Channel groups are ordered as base maps,
    per-ghost current/previous maps, Pac-Man heading, per-ghost headings, then global
    scatter/phase/stall/horizon planes. Heading and scalar channels are binary or
    normalized to [0, 1].
    """
    state = np.zeros((NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    state[0] = WALL_MAP

    for fx, fy in pellets:
        if 0 <= fy < GRID_HEIGHT and 0 <= fx < GRID_WIDTH:
            state[1, fy, fx] = 1.0

    px, py = pacman_pos
    if 0 <= py < GRID_HEIGHT and 0 <= px < GRID_WIDTH:
        state[2, py, px] = 1.0

    ghost_current_start = BASE_CHANNELS
    ghost_previous_start = ghost_current_start + GHOST_SLOTS
    for index, (gx, gy) in enumerate(ghost_positions[:GHOST_SLOTS]):
        if 0 <= gy < GRID_HEIGHT and 0 <= gx < GRID_WIDTH:
            state[ghost_current_start + index, gy, gx] = 1.0

    # Temporal positions for velocity / direction inference
    ppx, ppy = prev_pacman_pos if prev_pacman_pos is not None else pacman_pos
    if 0 <= ppy < GRID_HEIGHT and 0 <= ppx < GRID_WIDTH:
        state[3, ppy, ppx] = 1.0

    prev_ghosts = prev_ghost_positions if prev_ghost_positions is not None else ghost_positions
    for index, (pgx, pgy) in enumerate(prev_ghosts[:GHOST_SLOTS]):
        if 0 <= pgy < GRID_HEIGHT and 0 <= pgx < GRID_WIDTH:
            state[ghost_previous_start + index, pgy, pgx] = 1.0

    cursor = BASE_CHANNELS + GHOST_POSITION_CHANNELS
    directions = ["up", "down", "left", "right"]
    if last_move in directions:
        state[cursor + directions.index(last_move), :, :] = 1.0
    cursor += PACMAN_HEADING_CHANNELS

    ghost_dirs = ghost_dirs or ["up"] * GHOST_SLOTS
    for index, (gx, gy) in enumerate(ghost_positions[:GHOST_SLOTS]):
        if index < len(ghost_dirs) and ghost_dirs[index] in directions and 0 <= gy < GRID_HEIGHT and 0 <= gx < GRID_WIDTH:
            direction_channel = cursor + index * 4 + directions.index(ghost_dirs[index])
            state[direction_channel, gy, gx] = 1.0
    cursor += GHOST_HEADING_CHANNELS

    # The current decision's ghost move occurs after mode_step increments.
    next_mode_step = mode_step + 1
    mode_in_cycle = (next_mode_step - 1) % MODE_CYCLE
    state[cursor, :, :] = float(mode_in_cycle < SCATTER_STEPS)
    state[cursor + 1, :, :] = mode_in_cycle / max(1, MODE_CYCLE - 1)
    state[cursor + 2, :, :] = min(max(steps_without_pellet, 0), 45) / 45.0
    state[cursor + 3, :, :] = min(max(steps_remaining, 0), max(1, horizon)) / max(1, horizon)

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
        Deep Q-Network with identity-preserving spatial and state-context channels.
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
