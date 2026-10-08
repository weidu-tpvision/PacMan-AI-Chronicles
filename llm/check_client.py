"""
Manual smoke runner for the System 1 decision engine (live Ollama or offline simulator).
Validates state extraction, question formatting, and decision generation.

Not a unit test (it may contact a live server): run it explicitly with
    python -m llm.check_client
"""

from core.environment import Environment
from core.maze_data import START_POSITIONS
from llm.decision_client import DEFAULT_MODEL, SystemOneAgent


def check_decision_pipeline():
    print("==================================================")
    print("Testing System 1 Decision Model Pipeline")
    print("==================================================")

    agent = SystemOneAgent(model=DEFAULT_MODEL, prefer_live=True)
    online = agent.is_ollama_online(force_refresh=True)
    print(f"Ollama Service Reachable ({agent.host}): {online}")

    env = Environment()
    ghost_positions = [tuple(g) for g in START_POSITIONS["ghosts"]]

    # Junction next to the spawn tile; legal moves come from the real environment
    test_pos = (8, 15)
    legal_moves = env.get_legal_moves(*test_pos)

    print(f"\nEvaluating Position: {test_pos}")
    print(f"Legal Moves: {legal_moves}")
    print(f"Ghosts: {ghost_positions}")

    result = agent.decide_move(
        pacman_pos=test_pos,
        ghost_positions=ghost_positions,
        pellets=env.pellets,
        legal_moves=legal_moves,
    )

    print("\n--- Result from Decision Agent ---")
    print(f"Mode: {'Live Ollama' if result.is_live else 'Heuristic Simulator'}")
    print(f"Selected Choice: {result.choice}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    if result.error_msg:
        print(f"Note: {result.error_msg}")
    print("Probabilities:")
    for m, p in result.probabilities.items():
        print(f"  - {m:5}: {p * 100:.1f}%")

    assert result.choice in legal_moves, "Choice must be a valid legal move"
    print("\n[SUCCESS] Decision pipeline operates correctly!")


if __name__ == "__main__":
    check_decision_pipeline()
