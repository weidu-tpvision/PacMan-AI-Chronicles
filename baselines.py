"""
Backward-compatibility facade for baseline and RL agents.
All agents are now modularized inside the `agents/` package:
- RandomAgent: agents.random_agent
- GreedyHeuristicAgent: agents.greedy_agent
- PretrainedQLearningAgent, TrainedQLearningAgent: agents.q_learning_agent
- DQNAgent: agents.dqn_agent
- SystemOneBaselineAgent: agents.system_one_agent
"""

from agents.base import AgentProtocol, DecisionResult, softmax
from agents.dqn_agent import DQNAgent
from agents.greedy_agent import GreedyHeuristicAgent
from agents.q_learning_agent import (
    PretrainedQLearningAgent,
    TrainedQLearningAgent,
)
from agents.random_agent import RandomAgent
from agents.system_one_agent import SystemOneBaselineAgent

__all__ = [
    "AgentProtocol",
    "DecisionResult",
    "softmax",
    "RandomAgent",
    "GreedyHeuristicAgent",
    "PretrainedQLearningAgent",
    "TrainedQLearningAgent",
    "DQNAgent",
    "SystemOneBaselineAgent",
]
