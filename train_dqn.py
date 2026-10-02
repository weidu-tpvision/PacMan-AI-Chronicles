"""
Root runner / compatibility facade for DQN training.
The full implementation is located in `rl.train_dqn`.
"""

import argparse
from rl.train_dqn import train_dqn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train_dqn(episodes=args.episodes, lr=args.lr, batch_size=args.batch_size)
