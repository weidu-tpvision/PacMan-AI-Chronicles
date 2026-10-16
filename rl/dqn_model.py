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

# Full-resolution network:
# - Option 2 (default, egocentric=False): 7-layer ConvNet trunk with dilated convolutions
#   CONV_DILATIONS = (1, 1, 2, 2, 4, 1, 1) giving a 25x25 full-board receptive field (> 19x21),
#   circular horizontal (X) padding across the warp tunnel, and 1x1 reduction to
#   REDUCED_CHANNELS = 8 before the dense layer.
# - Option 1 (optional, egocentric=True): rolls all spatial planes so Pac-Man is always centered
#   at (CENTER_Y=10, CENTER_X=9) for full translational invariance in both Conv and Linear heads.
CONV_WIDTH = 32
CONV_DILATIONS = (1, 1, 2, 2, 4, 1, 1)
CONV_LAYERS = len(CONV_DILATIONS)
REDUCED_CHANNELS = 8
HIDDEN_UNITS = 128

CENTER_X = GRID_WIDTH // 2   # 9
CENTER_Y = GRID_HEIGHT // 2  # 10

WALL_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for x, y in WALL_CELLS:
    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
        WALL_MAP[y, x] = 1.0

# Precomputed static CoordConv spatial coordinate planes normalized to [0, 1]
COORD_X_MAP = np.tile(np.linspace(0.0, 1.0, GRID_WIDTH, dtype=np.float32), (GRID_HEIGHT, 1))
COORD_Y_MAP = np.tile(np.linspace(0.0, 1.0, GRID_HEIGHT, dtype=np.float32)[:, None], (1, GRID_WIDTH))

# Precomputed rolled lookup tables for egocentric=True (Option 1) so batch decoding stays O(1)
ROLLED_WALL_MAPS = np.empty((GRID_HEIGHT, GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
ROLLED_COORD_X = np.empty((GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
ROLLED_COORD_Y = np.empty((GRID_HEIGHT, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
for _py in range(GRID_HEIGHT):
    _sy = CENTER_Y - _py
    ROLLED_COORD_Y[_py] = np.roll(COORD_Y_MAP, _sy, axis=0)
    for _px in range(GRID_WIDTH):
        _sx = CENTER_X - _px
        ROLLED_WALL_MAPS[_py, _px] = np.roll(WALL_MAP, (_sy, _sx), axis=(0, 1))
for _px in range(GRID_WIDTH):
    ROLLED_COORD_X[_px] = np.roll(COORD_X_MAP, CENTER_X - _px, axis=1)


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
    egocentric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the current game state as (grid [NUM_CHANNELS, H, W], scalars [NUM_SCALARS]).

    All values are binary or normalized to [0, 1]. `mode_step` is the environment clock
    *before* the step being decided (the ghosts' next move uses mode_step + 1).
    When `egocentric=False` (Option 2, default), uses fixed top-down board coordinates.
    When `egocentric=True` (Option 1), rolls all spatial planes so Pac-Man is centered at
    `(CENTER_Y, CENTER_X) = (10, 9)`.
    """
    grid = np.zeros((NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    px, py = int(pacman_pos[0]), int(pacman_pos[1])

    if egocentric and _on_grid(px, py):
        sy = CENTER_Y - py
        sx = CENTER_X - px
        grid[CH_WALLS] = ROLLED_WALL_MAPS[py, px]
        for fx, fy in pellets:
            if _on_grid(fx, fy):
                grid[CH_PELLETS, (fy + sy) % GRID_HEIGHT, (fx + sx) % GRID_WIDTH] = 1.0
        grid[CH_PACMAN, CENTER_Y, CENTER_X] = 1.0
        if last_move in ACTION_TO_IDX:
            grid[CH_PACMAN_HEADING + ACTION_TO_IDX[last_move], CENTER_Y, CENTER_X] = 1.0

        ghost_dirs = ghost_dirs or ["up"] * GHOST_SLOTS
        for index, (gx, gy) in enumerate(ghost_positions[:GHOST_SLOTS]):
            if not _on_grid(gx, gy):
                continue
            rgy = (gy + sy) % GRID_HEIGHT
            rgx = (gx + sx) % GRID_WIDTH
            grid[CH_GHOSTS + index, rgy, rgx] = 1.0
            if index < len(ghost_dirs) and ghost_dirs[index] in ACTION_TO_IDX:
                grid[CH_GHOST_HEADINGS + 4 * index + ACTION_TO_IDX[ghost_dirs[index]], rgy, rgx] = 1.0

        grid[CH_COORD_X] = ROLLED_COORD_X[px]
        grid[CH_COORD_Y] = ROLLED_COORD_Y[py]
    else:
        grid[CH_WALLS] = WALL_MAP
        for fx, fy in pellets:
            if _on_grid(fx, fy):
                grid[CH_PELLETS, fy, fx] = 1.0
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

        grid[CH_COORD_X] = COORD_X_MAP
        grid[CH_COORD_Y] = COORD_Y_MAP

    mode_in_cycle = mode_step % MODE_CYCLE  # phase of the ghosts' next move (clock mode_step + 1)
    grid[CH_SCATTER] = float(mode_in_cycle < SCATTER_STEPS)
    grid[CH_CYCLE_PHASE] = mode_in_cycle / max(1, MODE_CYCLE - 1)

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
    egocentric: bool = False,
):
    """Vectorized batch decoding of compact state arrays into float32 (B, NUM_CHANNELS, H, W) PyTorch tensors.

    Reconstructs exact full spatial grid representations in ~1 ms for 128 items,
    guaranteeing 100% numerical parity with encode_state(..., egocentric=egocentric).
    """
    B = len(px)
    grids = np.zeros((B, NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    b_range = np.arange(B)

    unpacked = np.unpackbits(pellet_bits, axis=1)[:, :N_PELLETS]
    b_idx, p_idx = np.where(unpacked)

    if egocentric:
        px_i = px.astype(np.int32)
        py_i = py.astype(np.int32)
        sy = CENTER_Y - py_i
        sx = CENTER_X - px_i

        grids[:, CH_WALLS] = ROLLED_WALL_MAPS[py_i, px_i]
        grids[b_idx, CH_PELLETS, (P_YS[p_idx] + sy[b_idx]) % GRID_HEIGHT, (P_XS[p_idx] + sx[b_idx]) % GRID_WIDTH] = 1.0

        grids[b_range, CH_PACMAN, CENTER_Y, CENTER_X] = 1.0
        valid_ph = p_head >= 0
        if np.any(valid_ph):
            grids[b_range[valid_ph], CH_PACMAN_HEADING + p_head[valid_ph], CENTER_Y, CENTER_X] = 1.0

        for k in range(GHOST_SLOTS):
            rgy = (gy[:, k].astype(np.int32) + sy) % GRID_HEIGHT
            rgx = (gx[:, k].astype(np.int32) + sx) % GRID_WIDTH
            grids[b_range, CH_GHOSTS + k, rgy, rgx] = 1.0
            valid_gh = g_head[:, k] >= 0
            if np.any(valid_gh):
                grids[b_range[valid_gh], CH_GHOST_HEADINGS + 4 * k + g_head[valid_gh, k], rgy[valid_gh], rgx[valid_gh]] = 1.0

        grids[:, CH_COORD_X] = ROLLED_COORD_X[px_i]
        grids[:, CH_COORD_Y] = ROLLED_COORD_Y[py_i]
    else:
        grids[:, CH_WALLS] = WALL_MAP
        grids[b_idx, CH_PELLETS, P_YS[p_idx], P_XS[p_idx]] = 1.0

        grids[b_range, CH_PACMAN, py, px] = 1.0
        valid_ph = p_head >= 0
        if np.any(valid_ph):
            grids[b_range[valid_ph], CH_PACMAN_HEADING + p_head[valid_ph], py[valid_ph], px[valid_ph]] = 1.0

        for k in range(GHOST_SLOTS):
            grids[b_range, CH_GHOSTS + k, gy[:, k], gx[:, k]] = 1.0
            valid_gh = g_head[:, k] >= 0
            if np.any(valid_gh):
                grids[b_range[valid_gh], CH_GHOST_HEADINGS + 4 * k + g_head[valid_gh, k], gy[valid_gh, k], gx[valid_gh, k]] = 1.0

        grids[:, CH_COORD_X] = COORD_X_MAP
        grids[:, CH_COORD_Y] = COORD_Y_MAP

    # Scatter and cycle phase planes
    m_cycle = mode_step % MODE_CYCLE
    grids[:, CH_SCATTER] = (m_cycle < SCATTER_STEPS).astype(np.float32)[:, None, None]
    grids[:, CH_CYCLE_PHASE] = (m_cycle / max(1, MODE_CYCLE - 1)).astype(np.float32)[:, None, None]

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
    class CircularXConv2d(nn.Conv2d):
        """3x3 Conv2d with circular horizontal (X) padding across the warp tunnel
        and standard zero vertical (Y) padding."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
            pad = dilation * (kernel_size // 2)
            super().__init__(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(pad, 0),
                dilation=dilation,
            )
            self._pad_x = pad

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._pad_x > 0:
                x = nn.functional.pad(x, (self._pad_x, self._pad_x, 0, 0), mode="circular")
            return super().forward(x)


    class PacmanDQN(nn.Module):
        """
        Dueling Deep Q-Network: a stack of 3x3 convolutions at full grid resolution (no
        pooling, so exact tile offsets between actors survive) with circular horizontal
        padding across the warp tunnel and dilated receptive field (25x25 > 19x21), a 1x1
        channel reduction to REDUCED_CHANNELS, then one dense layer that also receives the
        scalar inputs, and V / A heads combined as Q = V + A - mean(A).

        Supports both architectural modes:
          - egocentric=False (Option 2, default): fixed top-down allocentric board view.
          - egocentric=True  (Option 1): Pac-Man-centered camera view at (10, 9).
        """

        def __init__(
            self,
            in_channels: int = NUM_CHANNELS,
            num_scalars: int = NUM_SCALARS,
            num_actions: int = 4,
            dilations: Tuple[int, ...] = CONV_DILATIONS,
            reduced_channels: int = REDUCED_CHANNELS,
            egocentric: bool = False,
        ):
            super().__init__()
            self.register_buffer("_egocentric", torch.tensor(bool(egocentric), dtype=torch.bool))
            layers, channels = [], in_channels
            for d in dilations:
                layers += [
                    CircularXConv2d(channels, CONV_WIDTH, kernel_size=3, dilation=d),
                    nn.ReLU(),
                ]
                channels = CONV_WIDTH
            layers += [nn.Conv2d(channels, reduced_channels, kernel_size=1), nn.ReLU(), nn.Flatten()]
            self.trunk = nn.Sequential(*layers)
            self.feature_head = nn.Sequential(
                nn.Linear(reduced_channels * GRID_HEIGHT * GRID_WIDTH + num_scalars, HIDDEN_UNITS),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(HIDDEN_UNITS, 1)
            self.advantage_head = nn.Linear(HIDDEN_UNITS, num_actions)

        @property
        def egocentric(self) -> bool:
            return bool(self._egocentric.item())

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Allow loading state_dicts that omitted _egocentric (defaults to False)
            ego_key = prefix + "_egocentric"
            if ego_key not in state_dict:
                state_dict = dict(state_dict)
                state_dict[ego_key] = torch.tensor(False, dtype=torch.bool)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
            features = self.feature_head(torch.cat([self.trunk(grid), scalars], dim=1))
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class PacmanDQN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PacmanDQN. Install torch to enable.")

