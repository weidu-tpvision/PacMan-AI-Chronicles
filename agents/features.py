"""
Shared linear-feature extractors for the Q-learning family of agents and their trainers.

Single source of truth used by:
  - agents.q_learning_agent.PretrainedQLearningAgent  (TEXTBOOK features)
  - rl.train_q_learning                               (TEXTBOOK features)
  - agents.q_learning_agent.TrainedQLearningAgent     (ENHANCED features)
  - rl.optimize_policy                                (ENHANCED features)

Keeping agents and trainers on the same function guarantees train/inference parity.
"""

import collections
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.maze_data import (
    DIRECTIONS,
    GRID_WIDTH,
    MAZE_ADJ,
    MAZE_DEAD_ENDS,
    MAZE_DIST_MATRIX,
    MAZE_JUNCTIONS,
    OPPOSITE_DIRECTIONS,
)

Pos = Tuple[int, int]
FAR = 99  # Sentinel distance for unreachable / absent targets

TEXTBOOK_FEATURES: List[str] = [
    "ghost_1_step",
    "ghost_2_step",
    "ghost_3_step",
    "ghost_safe_dist",
    "eats_pellet",
    "nearest_pellet_dist",
    "reverse_penalty",
]

ENHANCED_FEATURES: List[str] = [
    "ghost_1_step",
    "ghost_2_step",
    "ghost_3_step",
    "ghost_safe_dist",
    "dead_end_trap",
    "safe_junction",
    "eats_pellet",
    "poisoned_pellet",
    "nearest_pellet_dist",
    "reverse_penalty",
]

# Distance features in the textbook set are scaled by 1/10 (CS188-style normalisation)
# to keep TD updates numerically stable.
TEXTBOOK_DIST_SCALE = 10.0


def next_cell(pos: Pos, action: str) -> Pos:
    dx, dy = DIRECTIONS[action]
    return ((pos[0] + dx) % GRID_WIDTH, pos[1] + dy)


def toroidal_manhattan(a: Pos, b: Pos) -> int:
    dx = abs(a[0] - b[0])
    return min(dx, GRID_WIDTH - dx) + abs(a[1] - b[1])


def nearest_pellet_bfs(start: Pos, pellets: Set[Pos], max_depth: int = FAR) -> int:
    """Exact maze (BFS) distance from `start` to the closest pellet, with early exit."""
    if not pellets:
        return 0
    if start in pellets:
        return 0
    seen = {start}
    q = collections.deque([(start, 0)])
    while q:
        cur, d = q.popleft()
        if d >= max_depth:
            break
        for nxt in MAZE_ADJ.get(cur, ()):
            if nxt in seen:
                continue
            if nxt in pellets:
                return d + 1
            seen.add(nxt)
            q.append((nxt, d + 1))
    return FAR


def min_ghost_bfs(cell: Pos, ghosts: Iterable[Pos]) -> int:
    return min((MAZE_DIST_MATRIX.get((cell, tuple(g)), FAR) for g in ghosts), default=FAR)


def min_ghost_manhattan(cell: Pos, ghosts: Iterable[Pos]) -> int:
    return min((toroidal_manhattan(cell, tuple(g)) for g in ghosts), default=FAR)


def is_reversal(action: str, last_move: Optional[str]) -> bool:
    return bool(last_move) and action == OPPOSITE_DIRECTIONS.get(last_move)


def textbook_features(
    pacman_pos: Pos,
    ghost_positions: List[Pos],
    pellets: Set[Pos],
    action: str,
    last_move: Optional[str] = None,
) -> Dict[str, float]:
    """Classic linear features: Manhattan ghost proximity + BFS closest food (CS188 SimpleExtractor)."""
    n = next_cell(pacman_pos, action)
    g = min_ghost_manhattan(n, ghost_positions)
    return {
        "ghost_1_step": 1.0 if g <= 1 else 0.0,
        "ghost_2_step": 1.0 if g == 2 else 0.0,
        "ghost_3_step": 1.0 if g == 3 else 0.0,
        "ghost_safe_dist": (min(g, 30) / TEXTBOOK_DIST_SCALE) if g > 3 else 0.0,
        "eats_pellet": 1.0 if n in pellets else 0.0,
        "nearest_pellet_dist": nearest_pellet_bfs(n, pellets) / TEXTBOOK_DIST_SCALE,
        "reverse_penalty": 1.0 if (is_reversal(action, last_move) and g > 3) else 0.0,
    }


def enhanced_features(
    pacman_pos: Pos,
    ghost_positions: List[Pos],
    pellets: Set[Pos],
    action: str,
    last_move: Optional[str] = None,
) -> Dict[str, float]:
    """Topological features: BFS ghost distance, dead-end traps, safe junctions, poisoned pellets."""
    n = next_cell(pacman_pos, action)
    g = min_ghost_bfs(n, ghost_positions)

    depth = MAZE_DEAD_ENDS.get(n, 0)
    dead_end_trap = 1.0 if (n in MAZE_DEAD_ENDS and g <= depth + 3) else 0.0
    safe_junction = 1.0 if (n in MAZE_JUNCTIONS and g > 2) else 0.0
    eats = 1.0 if n in pellets else 0.0
    poisoned = 1.0 if (eats and (g <= 2 or dead_end_trap)) else 0.0

    return {
        "ghost_1_step": 1.0 if g <= 1 else 0.0,
        "ghost_2_step": 1.0 if g == 2 else 0.0,
        "ghost_3_step": 1.0 if g == 3 else 0.0,
        "ghost_safe_dist": float(g) if g > 3 else 0.0,
        "dead_end_trap": dead_end_trap,
        "safe_junction": safe_junction,
        "eats_pellet": eats,
        "poisoned_pellet": poisoned,
        "nearest_pellet_dist": float(nearest_pellet_bfs(n, pellets)),
        "reverse_penalty": 1.0 if (is_reversal(action, last_move) and g > 3) else 0.0,
    }


def linear_q(weights: Dict[str, float], feats: Dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in feats.items())
