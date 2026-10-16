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
| `CH_COORD_X`, `CH_COORD_Y` | Static normalized $[0, 1]$ X and Y coordinate planes (CoordConv) |

| Scalars (dense-layer input) | Content |
| --- | --- |
| `SCALAR_STALL` | Steps without a pellet, saturating at `STALL_STEPS` |
| `SCALAR_HORIZON` | Remaining fraction of the episode horizon |

**Placement rule:** an input that changes *what happens locally* is spatial (positions, headings, coordinate planes, and the Scatter/Chase phase, which changes where each ghost heads next); an input that only changes *how much the future is worth* (stall progress drives the stall penalty, remaining horizon bounds future reward) goes straight to the dense layer, next to the value and advantage heads.

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

`PacmanDQN` uses a stack of `CONV_LAYERS` (7) `3 × 3` convolutions (`CONV_WIDTH` channels, dilations `CONV_DILATIONS = (1, 1, 2, 2, 4, 1, 1)`) using `CircularXConv2d` (circular horizontal `X` padding across the row-9 warp tunnel and zero vertical `Y` padding) with **no pooling**, a `1 × 1` convolution to `REDUCED_CHANNELS` (`8`), then one dense layer (`HIDDEN_UNITS = 128`) that receives the flattened features **and the scalars**, then dueling heads (`V(s)` and `A(s, a)`, combined as `Q = V + A − mean(A)`).

Why no pooling + full-board dilation + circular X padding:
- On a tile maze one tile is the difference between "ghost adjacent" and "ghost two steps away". Pooling discards exact relative positions before the network can use them.
- Stacking 7 convolutions at full resolution with dilations `(1, 1, 2, 2, 4, 1, 1)` expands the effective receptive field to $1 + 2 \times \sum d_i = 25 \times 25$ tiles, which exceeds the $19 \times 21$ maze so every tile's convolutional feature vector sees the **entire board** (including across the horizontal warp tunnel via `CircularXConv2d`) before reaching the dense layer.
- Reducing channels to `REDUCED_CHANNELS = 8` via the `1 × 1` bottleneck halves the coordinate-locked `Linear` head parameter count (`~409k` instead of `~818k`), shifting capacity into the translation-invariant Conv trunk.

### Two supported spatial framing modes (`egocentric=False` default vs `egocentric=True`)
Both spatial framing options are implemented and stored in the checkpoint's `_egocentric` buffer so `DQNAgent` and evaluators auto-detect the framing of any loaded model:
1. **Option 2 — Allocentric Fixed Board (`egocentric=False`, default):**
   - Pure Atari-style fixed top-down board view where `encode_state` and `decode_grids_batch` place walls, pellets, and actors at their absolute `(y, x)` board coordinates, and the $25 \times 25$ dilated `CircularXConv2d` trunk routes long-range value across the board.
2. **Option 1 — Egocentric Pac-Man-Centered View (`egocentric=True` / `--egocentric`):**
   - Rolls all spatial planes by `(CENTER_Y - py, CENTER_X - px)` so Pac-Man is always centered at `(10, 9)`. Precomputed rolled wall and coordinate lookup tables (`ROLLED_WALL_MAPS`, `ROLLED_COORD_X`, `ROLLED_COORD_Y`) keep vectorized batch decoding $O(1)$ (`~1 ms` per batch) with 100% mathematical parity with `np.roll`.

## 3. Objective

### Reward & shaping (`compute_reward`)
The training reward is strictly proportional to the true game score delta (`SCORE_PELLET = +10`, `SCORE_DEATH = -500`, `SCORE_WIN = +500` from `core.environment` scaled by `REWARD_SCALE = 0.1`), plus two small shaping terms:

- `R_REVERSAL` — discourages reversing in a corridor when other moves exist (anti-oscillation).
- `R_STALL_PER_STEP` — applied on every step once `STALL_STEPS` steps pass without a pellet. The stall scalar in the observation makes this penalty Markov.

Keeping `SCORE_PELLET = +10` constant across the entire episode preserves the `50:1` death-to-pellet risk ratio in the endgame (whereas progressive pellet tiers caused the agent to take fatal 10:1 gambles near ghosts when fewer than 50 pellets remained).

### Discount
The discount's effective horizon, `1 / (1 − γ)`, must cover the span over which risk and reward matter. With a short horizon a death a few dozen steps ahead is nearly free compared to a pellet now, and the win bonus is invisible; that produces a greedy "eat until caught" policy. The default `gamma` is therefore long relative to typical episode lengths.

### Episode ends
- **Collision** and **board cleared**: terminal.
- **Time limit** (`max_steps`): terminal for bootstrapping. This is correct *because* remaining horizon is part of the observation; without it, time-limit cutoffs would have to bootstrap.
- **Stall: not an episode end.** The former stall cutoff existed only in training (evaluation and arenas never cut episodes), so the agent never learned what happens after a long pellet-less stretch. It is replaced by the per-step stall penalty above.

## 4. Learning algorithm
- **Double DQN** targets with **legal-action masks**: the next action is chosen by the online network among moves that are legal in the next state, and evaluated by the target network.
- **Target network synchronization:** supports Polyak soft updates (`target_tau`, e.g. 0.005 via `target <- (1 - tau) * target + tau * policy`) for smooth value tracking, or periodic hard sync every `target_update_steps` environment steps (`target_tau = 0.0`).
- **Multi-step ($n$-step) returns:** supported via `--n-step N` (default 1). Rolling episode deque accumulates discounted returns $R = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$ and bootstraps from $s_{t+n}$ with discount $\gamma^n$. In empirical tests, $n=1$ proved superior during high $\epsilon$-greedy exploration regimes because naive multi-step returns lock in random exploratory actions for intermediate steps, whereas 1-step targets evaluate the greedy argmax at $s_{t+1}$.
- **Dueling** value/advantage heads.
- **Prioritized replay** (proportional, exponent `alpha`) with importance-sampling weights; the IS exponent β is annealed from `per_beta_start` to 1 over the step budget as a pure function of environment steps.
- **Live max-priority tracking:** new transitions receive `max(np.max(priorities), priority_epsilon)`, allowing initial priorities to adjust downward when earlier TD error spikes are sampled and resolved (eliminating permanent oversampling drift).
- **Huber loss**, gradient-norm clipping, Adam with cosine learning-rate annealing over the step budget.
- **Compact replay storage:** Transitions are stored in compact native state representation (41 bytes per state: uint8 actor coordinates, int8 headings, 19-byte pellet bitmask, fp32 scalars), reducing buffer RAM footprint by **~360×** (200,000 transitions require only ~21 MB of RAM, down from 7.6 GB; full training checkpoints shrink from ~3 GB to ~20 MB). On-the-fly vectorized decoding via `decode_grids_batch` reconstructs full PyTorch spatial tensors in ~1 ms per batch with 100% exact numerical parity. Default capacity scaled to **200,000 transitions** (checkpoint v8).
- **Update cadence & replay ratio:** one gradient update every `train_every` environment steps. The replay ratio is `batch_size / train_every` samples replayed per environment step. Decoupling cadence with `train_every = 4` and `batch_size = 128` (replay ratio 32) nearly doubles CPU simulation throughput (from ~26 to ~54 steps/s) while providing higher-quality gradient estimates and stabilizing soft target tracking.
- **SumTree PER sampling:** $O(\log N)$ priority updates and inverse-CDF sampling implemented via a complete power-of-2 binary `SumTree` and `MaxTree`, replacing the linear CDF array. Provides exact mathematical equivalence to `random.choices` and $O(1)$ live max priority tracking without full array reductions.
- **CPU threads:** training pins torch intra-op threads (`--threads`, default 3/4 of logical CPUs); all logical CPUs oversubscribe this narrow network. Thread count is a runtime setting, not part of the checkpointed config.
- **Rejected speedups:** running the online network on `s` and `s'` as one concatenated batch is *slower* (backward then runs over the doubled batch); `channels_last` memory format gave no gain on CPU.

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
