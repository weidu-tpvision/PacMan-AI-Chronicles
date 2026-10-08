"""
Root runner / compatibility facade for Approximate Q-Learning.
The full implementation (and CLI) lives in `rl.train_q_learning`; CLI args pass through.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("rl.train_q_learning", run_name="__main__")
