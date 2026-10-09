# DQN design

Design notes for the Deep Q-Network agent: what it observes, how the network is shaped, what it is trained to optimize, and why. Values are referenced by their code constants (`rl/dqn_model.py`, `rl/train_dqn.py`); experiment results and run parameters are deliberately not recorded here.

> **Status:** the shipped checkpoint (`rl/weights/dqn_pacman.pt`) comes from a short run that only verifies the pipeline end to end on the design below; it is not a tuned model. Checkpoints from earlier observation encodings or networks are rejected (the agent reports `[UNTRAINED]`, the tournament skips the DQN rows).

## 1. Observation

`encode_state()` returns a pair `(grid, scalars)` computed from the **current game state only** (no history):

| Grid planes | Content |
| --- | --- |
| `CH_WALLS` | Walls |
| `CH_PELLETS` | Remaining pellets |
| `CH_PACMAN` | Pac-Man position |
| `CH_GHOSTS` (3) | One position map per ghost identity |
| `CH_PACMAN_HEADING` (4) | Pac-Man heading, one-hot, set at Pac-Man's tile |
| `CH_GHOST_HEADINGS` (12) | Each ghost's heading, one-hot, set at that ghost's tile |
| `CH_SCATTER`, `CH_CYCLE_PHASE` | Scatter flag and Scatter/Chase cycle phase for the ghosts' next move (full planes) |

| Scalars (dense-layer input) | Content |
| --- | --- |
| `SCALAR_STALL` | Steps without a pellet, saturating at `STALL_STEPS` |
| `SCALAR_HORIZON` | Remaining fraction of the episode horizon |

**Placement rule:** an input that changes *what happens locally* is spatial (positions, headings, and the Scatter/Chase phase, which changes where each ghost heads next); an input that only changes *how much the future is worth* (stall progress drives the stall penalty, remaining horizon bounds future reward) goes straight to the dense layer, next to the value and advantage heads.

### Separate binary channels, not one categorical map
For a small symbolic grid, one 0/1 map per object type is preferred over a single integer-coded map (`0=empty, 1=wall, 2=pellet, …`):

- A CNN fed integer codes treats them as ordered magnitudes (category 6 "closer" to 5 than to 2), which is meaningless.
- One label per cell cannot represent overlapping objects (e.g. at a collision) without a priority rule that hides information.
- Per-ghost maps keep ghost identity, which matters because each ghost follows a different chase rule.
- Sparse position maps are expected for single-tile entities and are not, by themselves, a reason to merge channels.

### Markov completeness
The encoder is built around the question "does this variable change future transitions or rewards?":

- **Encoded:** ghost positions and headings (ghosts may not reverse), Pac-Man position and heading (Pinky/Clyde targets use it), Scatter/Chase phase for the *next* ghost move, remaining pellets, stall progress (it drives the stall penalty), remaining horizon (the episode is finite).
- **Not encoded, by design:**
  - *Ghost RNG state* — junction jitter is exogenous noise; the agent should not model the generator.
  - *Cumulative score* — affects neither dynamics nor reward.
  - *Illegal-move count* — action selection is masked to legal moves.
  - *Lives* — training episodes are single-life (a collision is terminal). If a multi-life objective is ever wanted, training and observation must change together.

### Encoding decisions
- **No previous-position maps.** For every actor, previous position = current position − heading (verified over random-policy rollouts; ghosts move every step, and Pac-Man only fails to move after a blocked move, which the masked policy never makes). The dynamics depend on headings, not on previous positions, and the old maps contradicted the headings at episode start (prev = current while heading says "left"/"up"). Dropping them also removes the agent's position history, the source of reset/respawn bugs.
- **Headings are local** (at the actor's tile) for Pac-Man and ghosts alike, so convolution filters see "Pac-Man here, facing left" directly.
- **Ghost position maps are kept** although each equals the sum of that ghost's heading planes: the redundancy is trivial for the first layer and keeps "ghost here" explicit.
- **The wall plane is kept** although the maze is fixed: it makes corridor features easy to learn and keeps an egocentric view possible.

## 2. Network

A stack of `CONV_LAYERS` `3 × 3` convolutions (`CONV_WIDTH` channels) with padding and **no pooling**, a `1 × 1` convolution to `REDUCED_CHANNELS`, then one dense layer (`HIDDEN_UNITS`) that receives the flattened features **and the scalars**, then dueling heads (`V(s)` and `A(s, a)`, combined as `Q = V + A − mean(A)`).

Why no pooling: on a tile maze one tile is the difference between "ghost adjacent" and "ghost two steps away". The earlier design pooled 2× right after a `5 × 5` receptive field, discarding exact relative positions before the network could use them, and left almost all parameters in one dense layer that had to do all maze-wide reasoning. Stacking convolutions at full resolution grows the receptive field by two tiles per layer without losing tile precision; the `1 × 1` reduction keeps the dense layer small. The cost is compute (every layer sees all 399 tiles), so the stack is kept narrow.

### Not chosen (yet): Pac-Man-centered view
Re-centering the grid on Pac-Man (wrapping horizontally through the tunnel) aligns convolution features with the action outputs and is the common choice for grid games. It is a larger change (coordinate transform in encoder, agent and replay) and remains a candidate experiment once the full-resolution network has a baseline.

## 3. Objective

### Reward follows the evaluation score
The training reward is the **tournament score delta scaled by `REWARD_SCALE`** (pellet, death and win use `SCORE_PELLET`, `SCORE_DEATH`, `SCORE_WIN` from `core.environment`) plus two small shaping terms:

- `R_REVERSAL` — discourages reversing in a corridor when other moves exist (anti-oscillation).
- `R_STALL_PER_STEP` — applied on every step once `STALL_STEPS` steps pass without a pellet. The stall scalar in the observation makes this penalty Markov.

Previously the reward used its own pellet/death/win values whose ratios differed from the score (a death cost half as much, relative to pellets, as it does in the tournament), plus a per-step cost. A policy optimized for that reward is not optimizing the score it is judged on.

### Discount
The discount's effective horizon, `1 / (1 − γ)`, must cover the span over which risk and reward matter. With a short horizon a death a few dozen steps ahead is nearly free compared to a pellet now, and the win bonus is invisible; that produces a greedy "eat until caught" policy. The default `gamma` is therefore long relative to typical episode lengths.

### Episode ends
- **Collision** and **board cleared**: terminal.
- **Time limit** (`max_steps`): terminal for bootstrapping. This is correct *because* remaining horizon is part of the observation; without it, time-limit cutoffs would have to bootstrap.
- **Stall: not an episode end.** The former stall cutoff existed only in training (evaluation and arenas never cut episodes), so the agent never learned what happens after a long pellet-less stretch. It is replaced by the per-step stall penalty above.

## 4. Learning algorithm
- **Double DQN** targets with **legal-action masks**: the next action is chosen by the online network among moves that are legal in the next state, and evaluated by the target network.
- **Dueling** value/advantage heads.
- **Prioritized replay** (proportional, exponent `alpha`) with importance-sampling weights; the IS exponent β is annealed from `per_beta_start` to 1 over the *episode* schedule (annealing by environment steps assumed every episode ran to the horizon, so β never reached 1).
- **Huber loss**, gradient-norm clipping, periodic hard target sync, Adam with cosine learning-rate annealing.
- **Exploration:** epsilon-greedy, decaying exponentially from `epsilon_start` to `epsilon_min` over `epsilon_decay_fraction` of the planned episodes. A fixed per-episode decay factor would tie the exploration profile to one run length (short runs would end mostly random, long runs would sit at the floor), and the schedule is a pure function of the episode count, so resuming stays exact.
- **Replay storage:** preallocated ring arrays (fp16 grids, fp32 scalars); full training-state checkpoints for `--resume` (see README).

## 5. Comparison with the Atari DQN
[*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) learned from preprocessed grayscale frames stacked four deep so a feed-forward network could infer motion.

| Aspect | This project | Atari DQN |
| --- | --- | --- |
| Observation | Symbolic 0/1 maps of maze, pellets, actors | Emulator pixels |
| Motion | Explicit per-actor headings | Four stacked frames |
| Object identity | Per-ghost channels | Inferred from pixels |
| Network | Narrow full-resolution `3 × 3` stack, scalars into the dense layer, dueling heads | Strided conv stack for large images |
| Learning | Double DQN, dueling, prioritized replay, legal-action masks, score-aligned reward | Replay memory and target network |

The simulator exposes exact state, so imitating the pixel pipeline would only add work; the paper is a reference for end-to-end value learning, not a template for the input.

## 6. Experiment protocol
- Change **one thing per run** and compare on identical VAL seeds and training budgets.
- Train long runs in sessions with `--stop-after` / `--resume` (resuming from a periodic or `--stop-after` checkpoint is exact).
- Picking the best of many periodic validations on a small VAL set is optimistic. The trainer therefore re-evaluates the best checkpoint *and* the final weights on `final_val_episodes` VAL seeds that follow the selection seeds (never used for selection) and writes `rl/weights/dqn_final_eval.json`; compare variants on those numbers, not on the selection score.

## 7. Open design items
1. Pac-Man-centered view as an alternative to the absolute-grid network.
2. Inference heuristics (`REVERSAL_PENALTY`, `ORBIT_PENALTY_PER_VISIT` in `agents/dqn_agent.py`): reassess once the score-aligned objective is properly trained, and drop them if they no longer help.

## 8. Verification checklist
- Identical encoded observations imply the same legal actions and reward semantics (up to ghost jitter).
- Encodings differ when ghost identity, phase or headings differ and those change future transitions.
- The observation is a function of the current state only (`tests/test_dqn_encoding.py`, `tests/test_agents.py`).
- Replay stores exactly the observation used at inference time.
- Stop/resume reproduces the uninterrupted run (`tests/test_train_resume.py`).

## 9. Implementation files
- `core/environment.py` — transition rules, ghost modes/headings, collision rule, scoring constants.
- `rl/dqn_model.py` — observation encoder (plane / scalar layout), `STALL_STEPS`, network.
- `rl/train_dqn.py` — reward, episode ends, replay, validation, checkpoints.
- `agents/dqn_agent.py` — inference, checkpoint compatibility check, fallbacks for callers without environment context, optional heuristics.
