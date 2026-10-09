"""
Shared controller registry for the interactive frontends (pacman_game.py, web_arena.py).

Score badges are read from the latest tournament results file
(results/tournament_results.json, written by compare_baselines.py) so the UI can never
drift from the measured numbers.
"""

import json
import os
from typing import Dict, List, Optional

from agents.dqn_agent import DQNAgent
from agents.greedy_agent import GreedyHeuristicAgent
from agents.q_learning_agent import PretrainedQLearningAgent, TrainedQLearningAgent
from agents.random_agent import RandomAgent
from agents.system_one_agent import SystemOneBaselineAgent

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tournament_results.json"
)


def load_tournament_scores(path: str = RESULTS_PATH) -> Dict[str, float]:
    """Map tournament agent name -> mean score. Empty dict if no results file exists."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {row["name"]: row["score"] for row in data.get("summary", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _uses_trained_weights(agent) -> bool:
    """False when an agent fell back to untrained / built-in default parameters."""
    if hasattr(agent, "model_loaded"):
        return agent.model_loaded
    return getattr(agent, "weights_loaded", True)


def _badge(label: str, result_name: Optional[str], scores: Dict[str, float]) -> str:
    if result_name and result_name in scores:
        return f"{label} ({scores[result_name]:.0f} pts)"
    return label


def build_controllers(sys1_backend, model: str) -> List[dict]:
    """Return the 6 arena controllers in hotkey order [1]-[6]."""
    scores = load_tournament_scores()
    specs = [
        dict(id="rl_opt", name="RL (Policy Optimized)", label="Graph-Aware RL", result="RL (Policy Optimized, CEM)",
             sub="Cross-Entropy policy optimization on topological features",
             color=(168, 85, 247), type_label="POLICY Q-VALUES", agent=TrainedQLearningAgent()),
        dict(id="dqn", name="Deep Q-Network (DQN)", label="PyTorch CNN (full-res dueling)", result="DQN (+inference heuristics)",
             sub="Dueling Double-DQN on raw spatial grid tensors (+ inference anti-orbit heuristics)",
             color=(236, 72, 153), type_label="DEEP Q-VALUES", agent=DQNAgent(heuristics=True)),
        dict(id="sys1", name=f"System 1 ({model})", label="Neural Zero-Shot", result=None,
             sub="Spatial prompt reasoning via structured JSON logits",
             color=(56, 189, 248), type_label="LOGIT PROBABILITIES", agent=SystemOneBaselineAgent(sys1_backend)),
        dict(id="greedy", name="Greedy Heuristic", label="Handcrafted Rules", result="Greedy Heuristic",
             sub="Symbolic rules balancing BFS food search & ghost evasion",
             color=(34, 197, 94), type_label="HEURISTIC WEIGHTS", agent=GreedyHeuristicAgent()),
        dict(id="rl_textbook", name="Q-Learning (Textbook)", label="Classic TD-Learning", result="Q-Learning (Textbook TD)",
             sub="Linear approximation trained with classic Bellman (TD) updates",
             color=(245, 158, 11), type_label="BELLMAN Q-VALUES", agent=PretrainedQLearningAgent()),
        dict(id="random", name="Random Agent", label="Noise Floor", result="Random Agent (Baseline)",
             sub="Uniform random distribution across legal corridors",
             color=(239, 68, 68), type_label="UNIFORM SELECTION", agent=RandomAgent()),
    ]
    controllers = []
    for s in specs:
        label, result = s.pop("label"), s.pop("result")
        if _uses_trained_weights(s["agent"]):
            s["badge"] = _badge(label, result, scores)
        else:
            # The tournament score belongs to the trained weights, not this fallback policy.
            s["name"] = f"{s['name']} [UNTRAINED]"
            s["badge"] = f"{label} (weights missing)"
        controllers.append(s)
    return controllers
