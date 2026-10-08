# DQN design

Design notes for the Deep Q-Network agent: what it observes, how the network is shaped, what it is trained to optimize, and why. Values are referenced by their code constants (`rl/dqn_model.py`, `rl/train_dqn.py`); experiment results and run parameters are deliberately not recorded here.

> **Status:** the shipped checkpoint (`rl/weights/dqn_pacman.pt`) was trained before the collision-rule fix, the objective change and the full-resolution network described below. It still loads (legacy architecture), but it must be retrained before its scores mean anything.

## 1. Observation

`encode_state()` produces a stack of 0/1 spatial maps plus a few normalized global planes on the `21 × 19` grid:

| Channels | Content |
| --- | --- |
| 0 | Walls |
| 1 | Remaining pellets |
| 2–3 | Pac-Man now / previous position |
| 4–9 | Each ghost now (4–6) / previous (7–9) — one map per ghost identity |
| 10–13 | Pac-Man heading (one-hot) |
| 14–25 | Ghost headings (one-hot per ghost, set at the ghost's tile) |
| 26–29 | Scatter flag, Scatter/Chase cycle phase, stall progress (saturates at `STALL_STEPS`), remaining horizon |

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

### Known redundancies (candidates for a later cleanup)
- **Previous-position maps are mostly redundant with headings:** ghosts move every step, so previous = current − heading; Pac-Man's differ only after a blocked move, which the masked training policy never makes.
- **Pac-Man heading is broadcast over the whole grid**, while ghost headings are placed at the ghost's tile; placing it at Pac-Man's tile would be consistent.
- **The four global values are broadcast as full planes** and pass through every convolution; appending them to the dense-layer input would be cheaper and clearer.

These are harmless but change the input format, so they are deferred to one combined encoding change (old checkpoints would no longer load).

## 2. Network

### Full-resolution convolution stack (default, `arch="deep"`)
A stack of `3 × 3` convolutions with padding and **no pooling**, a `1 × 1` convolution that reduces channels before flattening, one dense layer, then dueling heads (`V(s)` and `A(s, a)`, combined as `Q = V + A − mean(A)`).

Why no pooling: on a tile maze one tile is the difference between "ghost adjacent" and "ghost two steps away". The previous design pooled 2× right after a `5 × 5` receptive field, discarding exact relative positions before the network could use them, and left ~96% of the parameters in one dense layer that had to do all maze-wide reasoning. Stacking convolutions at full resolution grows the receptive field without losing tile precision; the `1 × 1` reduction keeps the dense layer small. The cost is compute (every layer sees all 399 tiles), so the stack is kept narrow.

### Legacy pooled network (`arch="pool"`)
Two convolutions, `MaxPool2d(2)`, dense layer, dueling (or a single linear head for the oldest checkpoints). Kept only so existing checkpoints load; `DQNAgent` detects the architecture from the checkpoint's parameter names.

### Not chosen (yet): Pac-Man-centered view
Re-centering the grid on Pac-Man (wrapping horizontally through the tunnel) aligns convolution features with the action outputs and is the common choice for grid games. It is a larger change (coordinate transform in encoder, agent and replay) and remains a candidate experiment once the full-resolution network has a baseline.

## 3. Objective

### Reward follows the evaluation score
The training reward is the **tournament score delta scaled by `REWARD_SCALE`** (pellet, death and win use `SCORE_PELLET`, `SCORE_DEATH`, `SCORE_WIN` from `core.environment`) plus two small shaping terms:

- `R_REVERSAL` — discourages reversing in a corridor when other moves exist (anti-oscillation).
- `R_STALL_PER_STEP` — applied on every step once `STALL_STEPS` steps pass without a pellet. The stall plane in the observation makes this penalty Markov.

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
- **Replay storage:** preallocated fp16 ring arrays; full training-state checkpoints for `--resume` (see README).

## 5. Comparison with the Atari DQN
[*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) learned from preprocessed grayscale frames stacked four deep so a feed-forward network could infer motion.

| Aspect | This project | Atari DQN |
| --- | --- | --- |
| Observation | Symbolic 0/1 maps of maze, pellets, actors | Emulator pixels |
| Motion | Previous positions + explicit headings | Four stacked frames |
| Object identity | Per-ghost channels | Inferred from pixels |
| Network | Narrow full-resolution `3 × 3` stack, dueling heads | Strided conv stack for large images |
| Learning | Double DQN, dueling, prioritized replay, legal-action masks, score-aligned reward | Replay memory and target network |

The simulator exposes exact state, so imitating the pixel pipeline would only add work; the paper is a reference for end-to-end value learning, not a template for the input.

## 6. Experiment protocol
- Change **one thing per run** and compare on identical VAL seeds and training budgets.
- Train long runs in sessions with `--stop-after` / `--resume` (resuming from a periodic or `--stop-after` checkpoint is exact).
- Picking the best of many periodic validations on a small VAL set is optimistic; re-evaluate the selected checkpoint on a larger VAL set before comparing agents.

## 7. Open design items
1. Encoding cleanup (Section 1 redundancies) as one input-format change.
2. Pac-Man-centered view as an alternative to the absolute-grid network.
3. Inference heuristics (`REVERSAL_PENALTY`, `ORBIT_PENALTY_PER_VISIT` in `agents/dqn_agent.py`): the orbit penalty is large relative to Q-values and mostly masks training failures; reassess once the score-aligned objective is trained, and drop if no longer useful.
4. Remove the legacy 6-channel checkpoint migration in `agents/dqn_agent.py` once no such checkpoints remain.

## 8. Verification checklist
- Identical encoded observations imply the same legal actions and reward semantics (up to ghost jitter).
- Encodings differ when ghost identity, phase or headings differ and those change future transitions.
- Reset/respawn observations are correct, including initial headings.
- Replay stores exactly the observation used at inference time.
- Stop/resume reproduces the uninterrupted run (`tests/test_train_resume.py`).

## 9. Implementation files
- `core/environment.py` — transition rules, ghost modes/headings, collision rule, scoring constants.
- `rl/dqn_model.py` — observation encoder, `STALL_STEPS`, network architectures.
- `rl/train_dqn.py` — reward, episode ends, replay, validation, checkpoints.
- `agents/dqn_agent.py` — inference-time history, architecture detection, optional heuristics.
