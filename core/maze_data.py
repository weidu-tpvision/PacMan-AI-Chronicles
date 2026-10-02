"""
PacMan-AI-Chronicles: Core Maze Data & Graph Topology.
Defines the layout, dimensions, directions, and graph analytics
(BFS distance matrix, dead-end depths, safe junctions) used across all agents and engines.
"""


import collections
from typing import Dict, List, Set, Tuple

MAZE_LAYOUT = [
    "###################",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.### # ###.####",
    "   #.#       #.#   ",
    "####.# ##### #.####",
    "    .  #   #  .    ",
    "####.# ##### #.####",
    "   #.#       #.#   ",
    "####.# ##### #.####",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#..#...........#..#",
    "##.#.#.#####.#.#.##",
    "#....#...#...#....#",
    "#.######.#.######.#",
    "#.................#",
    "###################",
]

GRID_WIDTH = len(MAZE_LAYOUT[0])
GRID_HEIGHT = len(MAZE_LAYOUT)

START_POSITIONS = {
    "pacman": [9, 15],
    "ghosts": [
        [9, 7],   # Red (Blinky style)
        [7, 7],   # Pink (Pinky style)
        [11, 7],  # Orange (Clyde style)
    ],
}

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITE_DIRECTIONS: Dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# --- Graph BFS & Topological Analytics ---
WALKABLE_CELLS: Set[Tuple[int, int]] = set()
WALL_CELLS: Set[Tuple[int, int]] = set()
for _y, _row in enumerate(MAZE_LAYOUT):
    for _x, _char in enumerate(_row):
        if _char != "#":
            WALKABLE_CELLS.add((_x, _y))
        else:
            WALL_CELLS.add((_x, _y))

MAZE_ADJ: Dict[Tuple[int, int], List[Tuple[int, int]]] = collections.defaultdict(list)
for _x, _y in WALKABLE_CELLS:
    for _d, (_dx, _dy) in DIRECTIONS.items():
        _nx = (_x + _dx) % GRID_WIDTH
        _ny = _y + _dy
        if 0 <= _ny < GRID_HEIGHT and (_nx, _ny) in WALKABLE_CELLS:
            MAZE_ADJ[(_x, _y)].append((_nx, _ny))

MAZE_DIST_MATRIX: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
for _start in WALKABLE_CELLS:
    _q = collections.deque([(_start, 0)])
    _vis = {_start: 0}
    while _q:
        _curr, _d = _q.popleft()
        MAZE_DIST_MATRIX[(_start, _curr)] = _d
        for _nxt in MAZE_ADJ[_curr]:
            if _nxt not in _vis:
                _vis[_nxt] = _d + 1
                _q.append((_nxt, _d + 1))

_degrees = {node: len(neighbors) for node, neighbors in MAZE_ADJ.items()}
MAZE_DEAD_ENDS: Dict[Tuple[int, int], int] = {}
_depth = 1
_active = [node for node, deg in _degrees.items() if deg <= 1]
while _active:
    _next_active = []
    for node in _active:
        MAZE_DEAD_ENDS[node] = _depth
        for _nxt in MAZE_ADJ[node]:
            _degrees[_nxt] -= 1
            if _degrees[_nxt] == 1 and _nxt not in MAZE_DEAD_ENDS:
                _next_active.append(_nxt)
    _depth += 1
    _active = _next_active

MAZE_JUNCTIONS: Set[Tuple[int, int]] = {
    node for node, neighbors in MAZE_ADJ.items() if len(neighbors) >= 3
}
