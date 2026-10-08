# Code Review: PacMan-AI-Chronicles

**Review scope:** Current working-tree changes across the shared environment, agent refactor, training pipelines, tournament runner, and DQN updates. The repository already contained extensive uncommitted edits; this review did not reset or rewrite them.

## Findings

### [Fixed, originally P1] DQN targets maximize over actions that are illegal in the next state

In the Double-DQN update, `policy_net(b_ns).argmax(...)` selects from all four directions. The agent and behavior policy only choose legal moves, but the replay transition does not retain a next-state legal-action mask. A blocked direction can therefore have the largest Q-value and be used as the bootstrap target, teaching the network to value an action it can never execute. This affects both the new dueling network and the previous architecture.

**Resolution:** Replay transitions now carry the next state's legal-action mask. Double-DQN action selection masks illegal Q-values before `argmax`; terminal transitions remain suppressed by the terminal flag.

**Location:** [rl/train_dqn.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/rl/train_dqn.py:320)

### [Fixed, originally P2] Stall cutoff omits the documented stall penalty

`compute_reward` detects the 45-step no-pellet cutoff and returns `stalled=True`, but leaves the transition reward unchanged. The project context describes this cutoff as a `-50` penalty. As written, training ends the episode without the intended signal against getting trapped in a loop.

**Resolution:** `compute_reward` now applies the documented `-50` penalty at the stall cutoff. Stalls remain truncations, so bootstrapping from the cutoff state is preserved.

**Location:** [rl/train_dqn.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/rl/train_dqn.py:82)

### [Fixed, originally P2] Tournament runner trusts an agent's move even when it is illegal

The tournament counts `error_msg`, but passes `res.choice` directly to `Environment.step`. An agent returning an invalid direction raises `ValueError` and stops the tournament; an in-range direction that hits a wall is counted by the environment as an illegal move and consumes a turn. That makes the benchmark fragile precisely when it should be reporting agent errors and illegal actions.

**Resolution:** The tournament now records invalid choices and errors. Known blocked directions retain the environment's counted no-op behavior; unknown choices use a deterministic legal fallback so an invalid response cannot abort the tournament.

**Location:** [compare_baselines.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/compare_baselines.py:68)

### [Fixed] Pygame frontend did not parse and the collision respawn path was inconsistent

The shared-controller refactor left an orphaned duplicate controller block in `pacman_game.py`, causing an `IndentationError` at import time. The collision respawn code also referenced an unimported `START_POSITIONS` and bypassed `Environment.respawn()`, leaving the shared Scatter/Chase clock unchanged.

**Resolution:** Removed the orphaned block and routed life-loss respawns through `Environment.respawn()`, then synchronized the visual actor positions from the environment.

### [Fixed] Pygame could apply a stale async decision

Resetting the game or switching agents while an asynchronous decision was in flight could leave the old result available to the next state. The declared decision epoch was not used.

**Resolution:** Requests capture the current epoch, and their results are accepted only if the epoch still matches. Reset, switch, and respawn invalidate pending results.

## Review checks

- `python -m unittest discover -s tests -v` — **13 tests passed**, including regression coverage for replay masks, stall penalty, invalid tournament actions, Pygame collision respawn, environment clock reset, and stale async decisions.
- `python -m py_compile pacman_game.py rl/train_dqn.py compare_baselines.py` — **passed**.
- A one-episode DQN smoke run with short horizon exercised prioritized replay, legal-action masks, dueling-network optimization, validation, and checkpoint writing — **completed**.
- Loaded that temporary dueling checkpoint through `DQNAgent(require_weights=True)` — **loaded successfully**. Temporary smoke artifacts were removed.
- Full 5,000-episode DQN run with cosine annealing — **completed**; best validation mean **+1,018.0** at episode **4,900**. Checkpoint and SVG/PNG diagnostics were saved.

## Overall assessment

The three findings above are fixed and covered by regressions. The shared architecture is a meaningful improvement: training and inference share feature extractors, frontends share an agent registry, and simulation uses a common environment. The test suite is still small and does not prove all environment and tournament behavior correct; the DQN smoke run confirms execution, not learning quality over a full training schedule.
