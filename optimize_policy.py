"""
Root runner / compatibility facade for Direct Policy Search.
The full implementation is located in `rl.optimize_policy`.
"""

import argparse
from rl.optimize_policy import optimize_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Pac-Man Policy via Cross-Entropy Method")
    parser.add_argument("--generations", type=int, default=35, help="Number of generations")
    args = parser.parse_args()

    optimize_policy(generations=args.generations)
