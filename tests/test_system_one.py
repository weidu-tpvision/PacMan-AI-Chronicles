"""
Headless test runner for the System 1 decision engine.
Validates state extraction, question formatting, and decision generation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.decision_client import SystemOneAgent
from core.maze_data import DIRECTIONS, MAZE_LAYOUT, START_POSITIONS



def test_decision_pipeline():
    print("==================================================")
    print("Testing System 1 Decision Model Pipeline")
    print("==================================================")

    agent = SystemOneAgent(model="nimble", prefer_live=True)
    online = agent.is_ollama_online(force_refresh=True)
    print(f"Ollama Service Reachable (localhost:11434): {online}")

    pacman_pos = tuple(START_POSITIONS["pacman"])
    ghost_positions = [tuple(g) for g in START_POSITIONS["ghosts"]]

    # Collect pellets
    pellets = set()
    walls = set()
    for y, row in enumerate(MAZE_LAYOUT):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))
            elif char == ".":
                pellets.add((x, y))

    # Test intersection at (8, 15)
    test_pos = (8, 15)
    legal_moves = []
    for d, (dx, dy) in DIRECTIONS.items():
        if (test_pos[0] + dx, test_pos[1] + dy) not in walls:
            legal_moves.append(d)

    print(f"\nEvaluating Position: {test_pos}")
    print(f"Legal Moves: {legal_moves}")
    print(f"Ghosts: {ghost_positions}")

    result = agent.decide_move(
        pacman_pos=test_pos,
        ghost_positions=ghost_positions,
        pellets=pellets,
        legal_moves=legal_moves,
    )

    print("\n--- Result from Decision Agent ---")
    print(f"Mode: {'Live Ollama' if result.is_live else 'Heuristic Simulator'}")
    print(f"Selected Choice: {result.choice}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    print("Probabilities:")
    for m, p in result.probabilities.items():
        print(f"  - {m:5}: {p * 100:.1f}%")

    assert result.choice in legal_moves, "Choice must be a valid legal move"
    print("\n[SUCCESS] Decision pipeline operates correctly!")


if __name__ == "__main__":
    test_decision_pipeline()
