"""
Root runner / compatibility facade for Approximate Q-Learning.
The full implementation is located in `rl.train_q_learning`.
"""

import argparse
from rl.train_q_learning import train_q_learning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Approximate Q-Learning on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes")
    args = parser.parse_args()

    train_q_learning(episodes=args.episodes)
