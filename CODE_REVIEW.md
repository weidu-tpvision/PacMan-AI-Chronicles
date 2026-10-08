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

---

# Code Review (2026-10-08, follow-up round)

**Scope:** Full repository review after the fixes above. Verified with `python -m unittest discover -s tests` (13 passed) in the `systemone` venv.

## New findings (fixed in this round)

### [Fixed] README/GEMINI hardcoded experiment results that no longer matched the code
The README tournament table claimed 1,000 seeds / 150-move horizon with scores (926.2 CEM, 570.0 DQN, …) that match no reproducible configuration; the committed `results/tournament_results.json` was a 100-seed / 400-move run with entirely different numbers. The README roadmap and GEMINI claimed a 5,000-episode DQN run with best validation 1,018.0, while the committed metrics are a 2,000-episode run (best val 651.0). Stale architecture labels ("6-Channel CNN", "k=3 frame stack", "No MaxPool", "Peak val: 440.0") persisted in `agents/registry.py`, `web/index.html`, `rl/plot_metrics.py`, and GEMINI.

**Resolution:** All hardcoded experimental results were removed from README/GEMINI in favor of reproduction instructions; the generated artifacts (`results/tournament_results.json`, `rl/weights/`) are now the single source of truth (arena badges already read from them). Architecture labels corrected to the 30-channel dueling encoder; diagnostic figures regenerated.

### [Fixed] Tournament artifact used a 400-move horizon; training/validation use 300
`core.environment.DEFAULT_MAX_STEPS = 300` is the canonical horizon for DQN training, validation, CEM search, and Q-learning, but the committed tournament results were generated with `--max-moves 400`. The DQN observation contains a `steps_remaining / horizon` channel, so the policy was evaluated out-of-distribution for the final 100 steps of each episode.

**Resolution:** Re-ran the tournament at the canonical horizon (`python compare_baselines.py --episodes 100 --offline`, 300 moves) so the recorded artifact matches training/validation semantics.

## Open findings (not addressed in this round)

1. **Trainers are single-life; arenas are 3-lives.** `Environment.respawn()` gives the frontends multi-life episodes, but both RL trainers treat any collision as terminal. Agents never learn post-death play. Decide on one semantics or train multi-life episodes.
2. **`DQNAgent` respawn heuristic** (wipe temporal state when `dx+dy > 2`) can fail for deaths near the spawn point; frontends already call `agent.reset()` explicitly, so consider deleting the heuristic.
3. **Simulated-latency jitter uses `hash(tuple(pos))`** (`llm/decision_client.py`), which varies per process (PYTHONHASHSEED) — simulator latencies are not reproducible across runs.
4. **No `.gitignore`**; `__pycache__/` noise in the working tree and large binary checkpoints committed to git.
5. **Root facade modules** (`baselines.py`, `dqn_model.py`, `maze_data.py`, `decision_client.py`, `train_*.py`, `optimize_policy.py`) are unused internally and already drifting (root `train_dqn.py` lost `--seed`/`--save-path`).
6. **Test gaps on core dynamics:** scatter/chase phase boundaries, ghost tunnel wrap (a prior fix shipped untested), head-on pass-through collision, and `in_scatter` at `mode_step=0` have no regression coverage.
7. **Performance:** `ReplayBuffer.sample` re-stacks numpy arrays per batch; a tensor ring buffer would speed DQN training substantially.

---

# Code Review (2026-10-08, round 3: training-code focus)

## Fixed

1. **Replay buffer RAM halved** ([rl/train_dqn.py](rl/train_dqn.py)): transitions now store states as `float16` (binary channels exact, scalar planes keep ~3 decimals); a full 40k buffer drops from ~3.6 GiB to ~1.8 GiB. `sample()` upcasts to float32 tensors.
2. **`compute_reward` docstring contradiction**: the docstring claimed stalls "are not treated as terminal for bootstrapping" while the loop pushes them as `done=True`. Docstring now records the actual (deliberate) semantics: stall and time-limit cutoffs are terminal in replay, matching finite-episode scoring, with the horizon channel keeping value semantics consistent.
3. **`DQNAgent` respawn heuristic removed** ([agents/dqn_agent.py](agents/dqn_agent.py)): the `dx+dy > 2` teleport detector could leak stale pre-death temporal state for deaths near spawn. All callers (both arenas, tournament, trainers) already call `reset()` explicitly; the lifecycle contract is now documented on the class and pinned in `tests/test_agents.py`.
4. **Reproducible simulated latency** ([llm/decision_client.py](llm/decision_client.py)): `hash()` is process-salted; jitter is now a deterministic function of position.
5. **`.gitignore` extended** with `.pytest_cache/` (the file already existed and covers `__pycache__`/venvs; the untracked `.pyc` noise visible in the working tree is ignored but not auto-removed).
6. **Root training facades de-drifted**: `train_dqn.py`, `train_q_learning.py`, `optimize_policy.py` are now `runpy` launchers that reuse the real module CLIs (the old hand-copied argparse had already lost `--seed`/`--save-path`). New `tests/test_facades.py` guards re-export identity and side-effect-free imports.

## Verification

- `python -m unittest discover -s tests` — **15 tests pass** (13 existing + 2 facade guards; the toroidal-tracking test now also pins the explicit `reset()` contract).
- fp16 buffer push/sample round-trip, deterministic simulator latency (92.0 ms at spawn), a 60-move tournament episode via `run_episode(DQNAgent())`, and a 1-episode DQN smoke training run (warmup + PER sample + masked Double-DQN update + validation + checkpoint) — all passed.

## Still open (deliberately deferred)

- **P3 life semantics — DECIDED (keep current code)**: trainers/tournament stay single-life; the policy optimizes score per life and respawn handling is an arena concern. No change intended.
- Structural replay improvements (uint8 split channels, index-linked frame storage) and capacity tuning.
- Environment-dynamics test gaps (scatter/chase boundaries, ghost tunnel wrap, head-on pass-through).
- `web_arena.py` single-threaded server blocks during live LLM calls.
- **Arena DQN horizon saturation (minor)**: neither arena passes `steps_remaining`, so after 300 decisions in an open-ended arena session the horizon channel saturates at 0 ("no time left"). Display-only impact; tournament/training pass it correctly.

---

# Code Review (2026-10-09, round 4: frontends & tournament reporting)

## Fixed

1. **Half-finished threaded web server** ([web_arena.py](web_arena.py)): `ArenaHTTPServer` and `session_lock` were defined but unused (`main()` still ran a single-threaded `TCPServer`). The threaded server is now used, and every session access (step / reset / switch / state) holds `session_lock`.
2. **Web client request pile-up** ([web/index.html](web/index.html)): `setInterval(fetchState, 70)` fired regardless of in-flight requests. Replaced with a self-scheduling loop that waits for each response; `[S]` (advertised in the hint) now cycles speed.
3. **Web HUD ignored lives / level / events**: lives were hardcoded to three hearts and `game_over` / `life_lost` / `level_cleared` were never shown. Now rendered (status badge shows events and level). Static walls are fetched once via `GET /api/state` instead of on every step.
4. **Tournament double-counted blocked moves** ([compare_baselines.py](compare_baselines.py)): a known blocked direction was counted by both the environment and `invalid_actions`. `illegal_moves` now adds only substituted unknown choices.
5. **"Live" System 1 label trusted `/api/version`**: a stock Ollama answers it but has no `/v1/systemone`, so every move silently fell back while labelled Live. The tournament now probes the real endpoint with one decision, falls back to the simulator with a warning, and records a per-agent `live_rate`.
6. **Arena badges for untrained agents** ([agents/registry.py](agents/registry.py)): an agent running on fallback/default weights now shows `[UNTRAINED]` and no tournament score.
7. **Arena DQN horizon saturation** (closes the round-3 open item): both arenas pass `steps_remaining = horizon`, keeping the horizon plane in-distribution for open-ended play.

## Verification

- `python -m unittest discover -s tests` — **21 tests pass** (6 new: blocked-move count, live probe, untrained badge, wall-less step payload, DQN horizon kwargs, concurrent HTTP steps).
- Web arena smoke run (GET `/`, `/api/state`, POST `/api/step`, `/api/switch_agent`) and `node --check` on the client script; 2-seed tournament against an unreachable host.

## Still open

- Pygame worker thread can race `agent.reset()` (stale DQN history after `[R]`/switch mid-decision).
- Collision misses Pac-Man stepping onto a ghost's tile when that ghost moves elsewhere.
- Test scenarios in `tests/test_agents.py` use legal-move lists that don't match the maze; tests are not guarded for missing numpy/torch/pygame.
- Docs: personal absolute paths in README/GEMINI; stale GEMINI channel description.
