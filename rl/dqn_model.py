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

# Steps without a pellet after which the trainer applies its stall penalty; the stall
# plane saturates here, which keeps that penalty a function of the observation.
STALL_STEPS = 45

# Full-resolution network ("deep"): DEEP_CONV_LAYERS x (3x3 conv, DEEP_CONV_WIDTH ch)
# -> receptive field (2 * DEEP_CONV_LAYERS + 1) tiles, then a 1x1 reduction to
# DEEP_REDUCED_CHANNELS before the dense layer.
DEEP_CONV_LAYERS = 6
DEEP_CONV_WIDTH = 32
DEEP_REDUCED_CHANNELS = 16
HIDDEN_UNITS = 128
ARCHITECTURES = ("deep", "pool")

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
    state[cursor + 2, :, :] = min(max(steps_without_pellet, 0), STALL_STEPS) / STALL_STEPS
    state[cursor + 3, :, :] = min(max(steps_remaining, 0), max(1, horizon)) / max(1, horizon)

    return state


def encode_frame(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
) -> np.ndarray:
    """Backwards-compatibility alias for single-step encoding without history."""
    return encode_state(pacman_pos, ghost_positions, pellets)


def detect_architecture(state_dict) -> Tuple[str, bool]:
    """(arch, dueling) of a saved PacmanDQN state dict, from its parameter names."""
    arch = "deep" if "trunk.0.weight" in state_dict else "pool"
    return arch, "value_head.weight" in state_dict


if TORCH_AVAILABLE:
    class PacmanDQN(nn.Module):
        """
        Deep Q-Network with identity-preserving spatial and state-context channels.

        arch="deep" (default): a stack of 3x3 convolutions at full grid resolution (no
        pooling, so exact tile offsets between actors survive), a 1x1 channel reduction,
        one dense layer and dueling heads.
        arch="pool": the earlier two-conv + MaxPool2d(2) network, kept so existing
        checkpoints load (dueling=False reproduces the oldest single-head checkpoints).
        """

        def __init__(self, in_channels: int = NUM_CHANNELS, num_actions: int = 4, dueling: bool = True, arch: str = "deep"):
            super().__init__()
            if arch not in ARCHITECTURES:
                raise ValueError(f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}")
            self.arch = arch
            self.dueling = dueling

            if arch == "deep":
                layers, channels = [], in_channels
                for _ in range(DEEP_CONV_LAYERS):
                    layers += [nn.Conv2d(channels, DEEP_CONV_WIDTH, kernel_size=3, padding=1), nn.ReLU()]
                    channels = DEEP_CONV_WIDTH
                layers += [nn.Conv2d(channels, DEEP_REDUCED_CHANNELS, kernel_size=1), nn.ReLU()]
                self.trunk = nn.Sequential(*layers)
                flat_features = DEEP_REDUCED_CHANNELS * GRID_HEIGHT * GRID_WIDTH
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 21x19 -> 10x9
                )
                flat_features = 64 * (GRID_HEIGHT // 2) * (GRID_WIDTH // 2)

            if dueling:
                self.feature_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_features, HIDDEN_UNITS),
                    nn.ReLU(),
                )
                self.value_head = nn.Linear(HIDDEN_UNITS, 1)
                self.advantage_head = nn.Linear(HIDDEN_UNITS, num_actions)
            else:
                # Retain the original module names and shapes so historical checkpoints load.
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_features, HIDDEN_UNITS),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_UNITS, num_actions),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.trunk(x) if self.arch == "deep" else self.conv(x)
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
