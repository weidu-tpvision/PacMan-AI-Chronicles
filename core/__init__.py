"""
Core game engine components: maze layout, graph analytics, and environment simulator.
"""

from core.environment import Environment
from core.maze_data import (
    DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAZE_ADJ,
    MAZE_DEAD_ENDS,
    MAZE_DIST_MATRIX,
    MAZE_JUNCTIONS,
    MAZE_LAYOUT,
    OPPOSITE_DIRECTIONS,
    START_POSITIONS,
    WALKABLE_CELLS,
    WALL_CELLS,
)

__all__ = [
    "Environment",
    "MAZE_LAYOUT",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "START_POSITIONS",
    "DIRECTIONS",
    "OPPOSITE_DIRECTIONS",
    "WALKABLE_CELLS",
    "WALL_CELLS",
    "MAZE_ADJ",
    "MAZE_DIST_MATRIX",
    "MAZE_DEAD_ENDS",
    "MAZE_JUNCTIONS",
]
