"""
Root runner / compatibility facade for Direct Policy Search.
The full implementation (and CLI) lives in `rl.optimize_policy`; CLI args pass through.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("rl.optimize_policy", run_name="__main__")
