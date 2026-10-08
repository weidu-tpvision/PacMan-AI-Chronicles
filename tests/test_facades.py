"""Guard tests for the root-level backward-compatibility facade modules.

Facades must re-export the current package symbols and must be importable without
side effects (in particular, importing a runner facade must not start training).
"""

import importlib.util
import unittest

import agents.dqn_agent
import core.maze_data
import llm.decision_client

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional (dqn_model / trainer facades need it)")
class TestRootFacades(unittest.TestCase):
    def test_module_facades_reexport_current_symbols(self):
        import rl.dqn_model
        import baselines
        import decision_client
        import dqn_model
        import maze_data

        self.assertIs(dqn_model.PacmanDQN, rl.dqn_model.PacmanDQN)
        self.assertIs(dqn_model.encode_state, rl.dqn_model.encode_state)
        self.assertIs(maze_data.MAZE_LAYOUT, core.maze_data.MAZE_LAYOUT)
        self.assertIs(decision_client.SystemOneAgent, llm.decision_client.SystemOneAgent)
        self.assertIs(baselines.DQNAgent, agents.dqn_agent.DQNAgent)
        self.assertEqual(baselines.RandomAgent.__module__, "agents.random_agent")

    def test_training_runner_facades_import_without_running(self):
        # runpy-based launchers: importing must not start a training run
        import optimize_policy  # noqa: F401
        import train_dqn  # noqa: F401
        import train_q_learning  # noqa: F401


if __name__ == "__main__":
    unittest.main()
