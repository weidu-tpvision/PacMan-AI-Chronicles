# Code Review Report

**Date:** 2026-10-08  
**Scope:** Full repository source review, plus the current working-tree changes.  
**Review basis:** Static inspection. Fixes for the findings were applied; no test suite was run.

## Findings

### [Fixed, originally P1] DQN input omitted variables that change transitions and rewards

**Location:** [rl/dqn_model.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/rl/dqn_model.py:48), [rl/train_dqn.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/rl/train_dqn.py:57), [core/environment.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/core/environment.py:75)

The former six-channel encoder omitted ghost identity, Scatter/Chase phase, heading, stall progress, and remaining horizon, although these affect transitions or rewards. The new 30-channel encoder represents these features and preserves each ghost's current/history position and heading. Training, validation, tournament, and UI calls now supply environment context. Stall and horizon cutoffs are terminal in replay to match the finite episodes used for scoring. Legacy checkpoints load through a first-layer channel expansion, but their policies should be retrained for the new representation.

### [Fixed, originally P2] Ghost chase distance did not account for the horizontal tunnel

**Location:** [core/environment.py](/C:/Users/wei.du/WorkAtTPVision/test/system_one/core/environment.py:170)

`score_ghost` ranked candidate moves using `abs(_nx - tx)`. The maze wraps horizontally, so this could prefer a route nearly one full row long over a short route through the tunnel. Predictive chase targets could also lie outside the grid, making the raw coordinate difference even less representative.

**Resolution:** `tx` is now normalized modulo the grid width, and candidate scoring uses the shortest wrapped horizontal distance. No tests were run for this fix.

## Working-tree notes

- **TODO:** Add DQN training resume support. The trainer currently starts from fresh weights and saves only the best model state; resuming optimizer, replay buffer, scheduler, exploration, counters, and RNG state requires a full training checkpoint format.
- `results/tournament_results.json` is modified only in reported latency values; the reviewed source does not show corresponding behavioral changes.
- `dqn.md` documents the new DQN observation channels and checkpoint retraining requirement.
- `core/environment.py` contains the P2 fix and tracks steps without a pellet for DQN observations.
- The DQN encoder, training pipeline, inference agent, and callers were updated for the P1 fix.
- The updated 30-channel model was trained for 2,000 episodes (seed 0); the training log reports a best validation score of 651.0 at episode 1,600.

## Review limitations

No test suite was run after the fixes. The training run above exercises the updated DQN training and validation paths. The large existing `CODE_REVIEW.md` records prior checks, but those historical tests were not rerun for this review.
