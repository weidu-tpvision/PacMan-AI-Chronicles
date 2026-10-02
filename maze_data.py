"""
Backward-compatibility facade for maze data and graph topology.
All core definitions have been modularized under `core.maze_data`.
"""

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
