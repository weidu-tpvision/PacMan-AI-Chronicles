"""
Standardized headless environment simulator for tournament benchmarking and RL training.
Includes:
- Toroidal wrap-around logic
- Pellet collection tracking
- Deterministic/stochastic ghost AI with multi-target scatter/chase behaviors
- Non-overlapping ghost dispersion logic
- Collision detection: same tile, or Pac-Man entering a tile a ghost occupied this step
  (covers head-on swaps and stepping onto a ghost that then moves away)
- Move validation (illegal moves are counted and treated as a no-op)
- Life-loss respawn handling shared by every frontend
"""

import random
from typing import List, Optional, Set, Tuple

from core.maze_data import (
    DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAZE_LAYOUT,
    OPPOSITE_DIRECTIONS,
    START_POSITIONS,
    WALL_CELLS,
)

# Arcade Scatter / Chase cycle lengths (in environment steps)
CHASE_STEPS = 28
SCATTER_STEPS = 7
MODE_CYCLE = CHASE_STEPS + SCATTER_STEPS

# Canonical scoring rules shared by tournament, trainers' reporting and frontends
SCORE_PELLET = 10
SCORE_DEATH = -200
SCORE_WIN = 500

# Canonical episode horizon shared by training, validation, CEM search and the tournament
DEFAULT_MAX_STEPS = 300


def pacman_collides(
    old_pac: List[int], new_pac: List[int], old_ghosts: List[List[int]], new_ghosts: List[List[int]]
) -> bool:
    """Collision rule for one step.

    Hit if Pac-Man ends on a ghost's tile, or Pac-Man moved onto a tile a ghost occupied
    at the start of the step (head-on swaps and "walking through" a ghost that moved
    elsewhere). A ghost entering the tile Pac-Man just left is a miss.
    """
    moved = new_pac != old_pac
    return any(
        new_pac == new_g or (moved and new_pac == old_g)
        for old_g, new_g in zip(old_ghosts, new_ghosts)
    )


class Environment:
    """Standardized environment simulator for fair tournament comparisons and RL training."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.step_count = 0
        self.mode_step = 0  # Scatter/Chase clock, restarted on respawn (arcade behaviour)
        self.steps_without_pellet = 0
        self.illegal_moves = 0

        self.walls: Set[Tuple[int, int]] = set(WALL_CELLS)
        self.pellets: Set[Tuple[int, int]] = set()

        for y, row in enumerate(MAZE_LAYOUT):
            for x, char in enumerate(row):
                if char == ".":
                    self.pellets.add((x, y))

        self._place_actors()
        if tuple(self.pacman_pos) in self.pellets:
            self.pellets.remove(tuple(self.pacman_pos))

    def _place_actors(self) -> None:
        self.pacman_pos: List[int] = list(START_POSITIONS["pacman"])
        self.ghost_positions: List[List[int]] = [list(g) for g in START_POSITIONS["ghosts"]]
        self.ghost_dirs: List[str] = ["up"] * len(self.ghost_positions)
        self.last_move: str = "left"

    def respawn(self) -> None:
        """Reset Pac-Man and ghosts to their start tiles after a life is lost (pellets persist)."""
        self._place_actors()
        self.mode_step = 0
        self.steps_without_pellet = 0

    @property
    def in_scatter(self) -> bool:
        """True while ghosts are in Scatter mode.

        Arcade-authentic ordering (Namco 1980, level 1 starts in Scatter): each 35-step
        cycle opens with exactly SCATTER_STEPS (7) of Scatter followed by CHASE_STEPS (28)
        of Chase. The clock restarts on respawn.
        """
        return self.mode_step > 0 and (self.mode_step - 1) % MODE_CYCLE < SCATTER_STEPS

    def get_legal_moves(self, x: int, y: int) -> List[str]:
        legal = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx, ny = (x + dx) % GRID_WIDTH, y + dy
            if 0 <= ny < GRID_HEIGHT and (nx, ny) not in self.walls:
                legal.append(d)
        return legal

    def step(self, move: str) -> Tuple[bool, bool, bool]:
        """
        Execute Pac-Man move and advance ghost positions.

        Illegal moves (into a wall) are counted in ``self.illegal_moves`` and Pac-Man stays
        in place while ghosts still advance. Unknown direction strings raise ``ValueError``.

        If Pac-Man eats the last pellet the board is cleared immediately (ghosts do not move),
        so ``collided`` and ``won`` are never both True.

        Returns:
            (collided: bool, ate_pellet: bool, won: bool)
        """
        if move not in DIRECTIONS:
            raise ValueError(f"Unknown move {move!r}; expected one of {list(DIRECTIONS)}")

        self.step_count += 1
        self.mode_step += 1
        old_pac = list(self.pacman_pos)

        if move in self.get_legal_moves(old_pac[0], old_pac[1]):
            dx, dy = DIRECTIONS[move]
            nx, ny = (old_pac[0] + dx) % GRID_WIDTH, old_pac[1] + dy
            self.pacman_pos = [nx, ny]
            self.last_move = move
        else:
            self.illegal_moves += 1
            nx, ny = old_pac

        ate_pellet = False
        if (nx, ny) in self.pellets:
            self.pellets.remove((nx, ny))
            ate_pellet = True
        self.steps_without_pellet = 0 if ate_pellet else self.steps_without_pellet + 1

        if not self.pellets:
            return False, ate_pellet, True

        # Advance Ghosts
        old_ghosts = [list(g) for g in self.ghost_positions]
        pdx, pdy = DIRECTIONS.get(self.last_move, (0, 0))
        new_ghosts: List[List[int]] = []
        in_scatter = self.in_scatter

        for i, gpos in enumerate(self.ghost_positions):
            glegal = self.get_legal_moves(gpos[0], gpos[1])
            if not glegal:
                new_ghosts.append(gpos)
                continue

            opp = OPPOSITE_DIRECTIONS.get(self.ghost_dirs[i])
            filtered = [m for m in glegal if m != opp] or glegal

            if in_scatter:
                # Scatter home corners to break stalemates
                if i == 0:
                    tx, ty = GRID_WIDTH - 2, 1  # Top-right
                elif i == 1:
                    tx, ty = 1, 1               # Top-left
                else:
                    tx, ty = 1, GRID_HEIGHT - 2 # Bottom-left
            else:
                # Ghost behaviors (Chase):
                # Ghost 0 (Red / Blinky): Direct intercept to Pac-Man position
                # Ghost 1 (Pink / Pinky): Predictive ambush (3 cells ahead of Pac-Man)
                # Ghost 2 (Orange / Clyde): Flank from behind when far, retreat to corner when close
                if i == 0:
                    tx, ty = self.pacman_pos[0], self.pacman_pos[1]
                elif i == 1:
                    tx, ty = self.pacman_pos[0] + pdx * 3, self.pacman_pos[1] + pdy * 3
                else:
                    dist = abs(gpos[0] - self.pacman_pos[0]) + abs(gpos[1] - self.pacman_pos[1])
                    if dist > 5:
                        tx, ty = self.pacman_pos[0] - pdx * 2, self.pacman_pos[1] - pdy * 2
                    else:
                        tx, ty = 1, GRID_HEIGHT - 2

            # Horizontal tunnel coordinates are cyclic, including predictive targets
            # that extend beyond either edge of the board.
            tx %= GRID_WIDTH

            # Dispersion: prevent ghosts from overlapping on identical tiles
            claimed = {tuple(p) for p in new_ghosts}

            def score_ghost(m: str) -> float:
                _dx, _dy = DIRECTIONS[m]
                _nx = (gpos[0] + _dx) % GRID_WIDTH
                _ny = gpos[1] + _dy
                dx = abs(_nx - tx)
                dx = min(dx, GRID_WIDTH - dx)
                score = dx + abs(_ny - ty)
                if (_nx, _ny) in claimed:
                    score += 500.0  # heavy penalty to diverge paths
                return score

            # Controlled junction jitter (10% chance to diversify paths across seeds)
            if len(filtered) > 1 and self.rng.random() < 0.10:
                gm = self.rng.choice(filtered)
            else:
                gm = min(filtered, key=score_ghost)

            self.ghost_dirs[i] = gm
            gx = (gpos[0] + DIRECTIONS[gm][0]) % GRID_WIDTH
            gy = gpos[1] + DIRECTIONS[gm][1]
            new_ghosts.append([gx, gy])

        self.ghost_positions = new_ghosts

        collided = pacman_collides(old_pac, self.pacman_pos, old_ghosts, self.ghost_positions)
        return collided, ate_pellet, False
