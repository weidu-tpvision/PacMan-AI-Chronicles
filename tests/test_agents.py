"""
Quick verification script for the modular architecture.
Tests importing core, rl, agents, and llm, and instantiating each agent.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("1. Python executable:", sys.executable, flush=True)


import core
print("2. Core package loaded successfully.", flush=True)

import core.environment
print("3. Environment class loaded successfully.", flush=True)

import rl
print("4. RL package loaded successfully.", flush=True)

from rl.dqn_model import PacmanDQN, encode_state
print("5. PacmanDQN and encode_state loaded successfully.", flush=True)

import agents
print("6. Agents package loaded successfully.", flush=True)

from agents import (
    RandomAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    TrainedQLearningAgent,
    DQNAgent,
)
print("7. All agent classes imported successfully.", flush=True)

r = RandomAgent()
g = GreedyHeuristicAgent()
q = PretrainedQLearningAgent()
opt = TrainedQLearningAgent()
dqn = DQNAgent()
print(f"8. Agents initialized:")
print(f"   - {r.name}")
print(f"   - {g.name}")
print(f"   - {q.name}")
print(f"   - {opt.name} (weights keys: {len(opt.weights)})")
print(f"   - {dqn.name} (PyTorch model loaded: {dqn.model_loaded})", flush=True)

# Test one decision step for each
legal = ["up", "down", "left", "right"]
pos = (9, 15)
ghosts = [(9, 7), (7, 7), (11, 7)]
pellets = {(1, 1), (2, 1)}

for ag in [r, g, q, opt, dqn]:
    res = ag.decide(pos, ghosts, pellets, legal)
    print(f"   -> {ag.name} chose '{res.choice}' in {res.latency_ms:.2f}ms", flush=True)

print("\n>>> ALL SYSTEM CHECKS PASSED SUCCESSFULLY! <<<", flush=True)
