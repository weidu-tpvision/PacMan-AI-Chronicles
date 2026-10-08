"""
Root runner / compatibility facade for DQN training.
The full implementation (and CLI) lives in `rl.train_dqn`; CLI args pass through.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("rl.train_dqn", run_name="__main__")
