"""
Standardized headless environment simulator for tournament benchmarking and RL training.
Includes:
- Toroidal wrap-around logic
- Pellet collection tracking
- Deterministic/stochastic ghost AI with multi-target scatter/chase behaviors
- Non-overlapping ghost dispersion logic
- Strict head-on pass-through collision detection
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


class Environment:
    """Standardized environment simulator for fair tournament comparisons and RL training."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.walls: Set[Tuple[int, int]] = set(WALL_CELLS)
        self.pellets: Set[Tuple[int, int]] = set()

        for y, row in enumerate(MAZE_LAYOUT):
            for x, char in enumerate(row):
                if char == ".":
                    self.pellets.add((x, y))

        self.pacman_pos: List[int] = list(START_POSITIONS["pacman"])
        if tuple(self.pacman_pos) in self.pellets:
            self.pellets.remove(tuple(self.pacman_pos))

        self.ghost_positions: List[List[int]] = [list(g) for g in START_POSITIONS["ghosts"]]
        self.ghost_dirs: List[str] = ["up", "up", "up"]
        self.last_move: str = "left"

    def get_legal_moves(self, x: int, y: int) -> List[str]:
        legal = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if nx < 0:
                nx = GRID_WIDTH - 1
            elif nx >= GRID_WIDTH:
                nx = 0
            if (nx, ny) not in self.walls:
                legal.append(d)
        return legal

    def step(self, move: str) -> Tuple[bool, bool, bool]:
        """
        Execute Pac-Man move and advance ghost positions.
        Returns:
            (collided: bool, ate_pellet: bool, won: bool)
        """
        old_pac = list(self.pacman_pos)
        dx, dy = DIRECTIONS[move]
        nx, ny = old_pac[0] + dx, old_pac[1] + dy

        # Toroidal wrap-around
        if nx < 0:
            nx = GRID_WIDTH - 1
        elif nx >= GRID_WIDTH:
            nx = 0

        self.pacman_pos = [nx, ny]
        self.last_move = move

        ate_pellet = False
        if (nx, ny) in self.pellets:
            self.pellets.remove((nx, ny))
            ate_pellet = True

        won = len(self.pellets) == 0

        # Advance Ghosts
        old_ghosts = [list(g) for g in self.ghost_positions]
        pdx, pdy = DIRECTIONS.get(self.last_move, (0, 0))
        new_ghosts = []

        for i, gpos in enumerate(self.ghost_positions):
            glegal = self.get_legal_moves(gpos[0], gpos[1])
            if not glegal:
                new_ghosts.append(gpos)
                continue

            opp = OPPOSITE_DIRECTIONS.get(self.ghost_dirs[i])
            filtered = [m for m in glegal if m != opp] or glegal

            # Ghost behaviors:
            # Ghost 0 (Red / Blinky): Direct intercept to Pac-Man position
            # Ghost 1 (Pink / Pinky): Predictive ambush (3 cells ahead of Pac-Man)
            # Ghost 2 (Orange / Clyde): Scatter/flank behavior
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

            # Dispersion: prevent ghosts from overlapping on identical tiles
            claimed = {tuple(p) for p in new_ghosts}

            def score_ghost(m: str) -> float:
                _dx, _dy = DIRECTIONS[m]
                _nx = (gpos[0] + _dx) % GRID_WIDTH
                _ny = gpos[1] + _dy
                score = abs(_nx - tx) + abs(_ny - ty)
                if (_nx, _ny) in claimed:
                    score += 500.0  # heavy penalty to diverge paths
                return score

            gm = min(filtered, key=score_ghost)
            self.ghost_dirs[i] = gm
            gx = (gpos[0] + DIRECTIONS[gm][0]) % GRID_WIDTH
            gy = gpos[1] + DIRECTIONS[gm][1]
            new_ghosts.append([gx, gy])

        self.ghost_positions = new_ghosts

        # Collision detection (same cell or head-on pass-through swap)
        collided = False
        for i in range(len(self.ghost_positions)):
            if self.pacman_pos == self.ghost_positions[i] or (
                old_pac == self.ghost_positions[i] and self.pacman_pos == old_ghosts[i]
            ):
                collided = True
                break

        return collided, ate_pellet, won
