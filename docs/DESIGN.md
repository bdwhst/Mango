# Mango: AlphaGo Zero Reproduction — Design Document

Status: **v5 — v4 contracts plus the single-GPU acceleration profile (§13, off by default) and MGO2 optional fields; M1–M3a done, M3a′ next**
Date: 2026-09-17 (v1–v4), 2026-09-21 (v5)
Target paper: Silver et al., *Mastering the game of Go without human knowledge*, Nature 550, 354–359 (2017) — "AlphaGo Zero" (AGZ).

Change logs: §15 (v1 → v2), §16 (v2 → v3), §17 (v3 → v4), §18 (v4 → v5).

---

## 1. Goals and non-goals

### Goals

1. A faithful, readable reproduction of the AlphaGo Zero algorithm: one residual network with policy and value heads, PUCT tree search with no rollouts, and a closed loop of self-play → training → gated evaluation.
2. Runs on two platforms from one code base: **Windows + NVIDIA CUDA** (dev machine: RTX 5080, CUDA 13.2, MSVC 2026) and **macOS Apple Silicon + Metal (MPS)**.
3. Board size is a **runtime** parameter of the engine (5 ≤ N ≤ 19). A trained model is nevertheless **size-specific** (both heads contain fully connected layers whose width depends on N²); the engine validates the model's declared size against the requested board size. The first milestone is a complete 9×9 loop; 19×19 is the same code with a different config and its own model.
4. Every component is testable in isolation: rules, search, network export, data format.
5. **Search correctness and evaluation validity are the primary risks** and get the most design and test attention (§§5.4, 5.5, 5.8, 9).
6. **Single machine, single GPU, for both self-play inference and training.** The whole loop must run on one consumer GPU (RTX 5080 class, or an Apple Silicon GPU) with the phases sharing that device; there is no distributed or multi-GPU code path. Efficiency on that one device is a design goal, not an afterthought: batched inference across games, fp16 inference, an evaluation cache and mixed-precision training are part of the baseline, not deviations.
7. **Measure, on that one GPU, how much of AGZ's compute the later literature removes.** A second profile (§13, `accel`) adds the self-play accelerations of KataGo (playout cap randomization, forced playouts with policy-target pruning, auxiliary ownership/score targets, global pooling) and two loop-level changes (no gating, growing window), **each behind a switch that is off by default**, so that the AGZ reproduction (goal 1) stays untouched and every switch is judged against it on the same frozen ladder at equal GPU-hours (M4b). The deliverable is the ablation report, not a stronger engine.

### Non-goals (for now)

- Hand-written CUDA / Metal inference kernels. LibTorch is the inference backend; a custom backend (CUDA Graphs, TensorRT) can be added later behind the same C++ interface if §10 shows launch overhead dominating.
- KataGo as a product: multiple rule sets, variable komi and handicap, analysis features, the score-distribution head, score utility in search. Only the self-play accelerations listed in goal 7 are in scope, and only as switches.
- Gumbel AlphaZero search and MuZero-style reanalyze. Both are promising for one GPU and are deliberately deferred until the M4b baseline exists (§13).
- Distributed self-play across machines; multi-GPU training; running self-play and training **concurrently** on the one GPU (phases are sequential, §6.5 — reconsidered only for 19×19, §13).
- Human game data (SL bootstrap) — that is AlphaGo 2016, not AGZ. Human games may serve as an evaluation set (move-prediction accuracy) but never as training data.
- **Reproducing 19×19 strength.** The 19×19 configuration exists to prove the code path runs end to end; see §8.1 for the compute arithmetic.

---

## 2. AlphaGo Zero in one page (what we are reproducing)

| Item | Paper (19×19) | Notes |
|---|---|---|
| Input | 17 binary planes: 8 history planes of the player-to-move's stones, 8 of the opponent's, 1 constant plane (1 if black to move) | From the perspective of the side to move |
| Network | Initial convolutional block (conv 3×3×256 + BN + ReLU) **plus** 19 or 39 residual blocks (2× conv 3×3×256 + BN, skip, ReLU) | Our config counts residual blocks only; the initial block is always present (§6.1) |
| Policy head | Conv 1×1×2 + BN + ReLU → FC → N²+1 logits (last = pass) | |
| Value head | Conv 1×1×1 + BN + ReLU → FC 256 → ReLU → FC 1 → tanh | |
| Loss | (z − v)² − πᵀ log p + c‖θ‖², c = 10⁻⁴, over **all** parameters | See deviation D9 |
| Optimizer | SGD, momentum 0.9, LR 0.01 → 0.001 (400k steps) → 0.0001 (600k steps), batch 2048, on **64 GPU workers** (inference during self-play ran on TPUs) | |
| Search | PUCT: a = argmax Q(s,a) + U(s,a), U = c_puct · P(s,a) · √Σ_b N(s,b) / (1+N(s,a)); 1600 simulations/move; leaves evaluated in mini-batches of 8 with virtual loss; tree reuse | c_puct value not stated in AGZ |
| Edge init | Each new edge initialised to N = 0, W = 0, **Q = 0**, P = p_a | Explicit in Methods |
| Search-time symmetry | Each leaf position is transformed by a uniformly random dihedral symmetry d_i before evaluation; the policy is mapped back | |
| Root noise | P(s,a) = (1−ε) p_a + ε η_a, η ~ Dir(0.03), ε = 0.25 | |
| Move selection / stored π | π(a\|s₀) ∝ N(s₀,a)^(1/τ); τ = 1 for the first 30 moves, τ → 0 afterwards. The stored training target is this temperature-adjusted π | See deviation D4 |
| Resignation | Resign if root value and best-child value < v_resign; v_resign **selected automatically** to keep false-positive resignations below 5%; 10% of games played without resignation | |
| Game length cap | Games are terminated at **722 moves (2·19²)** | |
| Training data | Positions sampled uniformly from the last 500,000 games; random 8-fold symmetry augmentation | |
| Gating | Checkpoint every 1000 steps; 400 games vs current best; promote if the candidate wins **> 55%** (point estimate) | |
| Rules | Tromp–Taylor **scoring** (area), komi 7.5, positional superko, **suicide prohibited** (original Tromp–Taylor permits suicide) | |
| Value target | z = ±1 game result from the perspective of the player to move at that position | |

---

## 3. Architecture overview

```
┌────────────────────────────── C++ engine (cpp/) ───────────────────────────────┐
│                                                                                │
│   Board (+8 position snapshots) ──► Feature planes ──► NNEvaluator (abstract)  │
│        ▲                                                   │                   │
│        │                                                   ├─ TorchEvaluator   │
│   MCTS (PUCT) ◄────────────────────────────────────────────┤  (TorchScript,    │
│        ▲                                                   │   CUDA|MPS|CPU)   │
│        │                                                   └─ FakeEvaluator    │
│   Self-play driver ──► game chunks (*.mgo, snapshots + moves + visits) + SGF   │
│   Match driver (eval) ──► win-rate report (paired, unique-trajectory count)    │
│   GTP front-end ──► GoGui / Sabaki / gogui-twogtp / gui.py (§6.7)              │
└────────────────────────────────────────────────────────────────────────────────┘
                 ▲ immutable model version (model.pt + model.json)      │ *.mgo chunks (atomic publish)
                 │                                                      ▼
┌────────────────────────────── Python (python/mango/) ──────────────────────────┐
│   model.py (ResNet)  ─►  train.py (loss, SGD)  ─►  export.py (TorchScript)     │
│   data.py (chunk reader: assembles planes from snapshots, symmetries, holdout) │
│   pipeline.py (sequential: selfplay → train → export → gate → promote)         │
│   strength.py (frozen-checkpoint ladder + external anchors → Elo)              │
│   gui.py (tkinter: play a model / analyse a game, over GTP)                    │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Language split** (same as minigo `cc/`, Leela Zero, KataGo, OpenSpiel `alpha_zero_torch`): everything on the hot path of self-play is C++; everything about learning is Python. The only two contracts between them are (a) the model version directory and (b) the game-chunk file format. Both are versioned (§5.6, §6.4).

---

## 4. Repository layout

```
Mango/
  CMakeLists.txt                 top-level; finds LibTorch from the venv's torch package
  CMakePresets.json              windows-cuda / macos-mps / cpu presets
  requirements.txt
  configs/
    5x5-smoke.json               end-to-end smoke / M3a' learning check
    9x9.json                     default (fast validation)
    19x19.json                   full-budget 19×19 configuration (§8.1; throughput measurement only)
    19x19-smoke.json             full-size network, tiny budget — the M6 engineering acceptance config
  cpp/
    core/      board.h/.cpp  history.h/.cpp  zobrist.h  features.h/.cpp  symmetry.h/.cpp  sgf.h/.cpp  random.h  config.h/.cpp
    nn/        evaluator.h   torch_evaluator.h/.cpp  fake_evaluator.h  model_meta.h/.cpp
    search/    node.h/.cpp   mcts.h/.cpp   search_params.h
    selfplay/  game_runner.h/.cpp  chunk_writer.h/.cpp  selfplay_main.cpp
    match/     match_main.cpp
    gtp/       gtp.h/.cpp    gtp_options.h/.cpp  mcts_player.h/.cpp  gtp_main.cpp
    tests/     ref_board.h/.cpp  test_board.cpp  test_rules.cpp  test_history.cpp  test_features.cpp  test_symmetry.cpp
               test_mcts.cpp  test_mcts_batched.cpp  test_chunk.cpp  test_torch_eval.cpp
    third_party/doctest/  doctest.h  LICENSE.txt        (vendored, MIT, version pinned in README)
    third_party/nlohmann/ json.hpp   LICENSE.MIT        (vendored, MIT, version pinned in README)
  python/
    mango/     __init__.py  model.py  data.py  train.py  export.py  pipeline.py  strength.py  chunk.py  config.py  state.py
               resign.py  gtp.py  gui.py
    tests/     test_model.py  test_chunk.py  test_export_roundtrip.py  test_feature_parity.py  test_symmetry_parity.py
  scripts/
    setup_env.ps1  setup_env.sh  build.ps1  build.sh
  docs/
    DESIGN.md (this file)  RULES.md  DATA_FORMAT.md  MODEL_FORMAT.md
  runs/<run-name>/            (gitignored) see §6.5 for the layout
```

Build produces three executables: `mango_selfplay`, `mango_match`, `mango_gtp`, plus `mango_tests`.

---

## 5. C++ engine

### 5.1 Board, history and rules (`cpp/core`)

#### 5.1.1 Two objects: `Board` (position) and `GameHistory`

The position state and the game history are separated so that search can copy the former cheaply and share the latter.

**`Board`** — the current position plus what the network and the rules need locally. Trivially copyable, no heap, fixed capacity for N ≤ 19:

| Field | Size (19×19) | Purpose |
|---|---|---|
| `cells[441]` (bordered 21×21, `EMPTY/BLACK/WHITE/WALL`) | 441 B | stone lookup |
| union-find `parent[441]` (u16) | 882 B | chains |
| liberty bitsets per chain root: `uint64[6]` over the 361 unbordered points | 441 × 48 B ≈ 21 KB | exact liberties: merge = OR, count = popcount, "would capture" = neighbouring enemy chain with popcount 1 |
| **position snapshots** ring buffer: 8 × (black `uint64[6]`, white `uint64[6]`) | 768 B | the 8 history time steps for the features (§5.2) |
| `hash` (Zobrist over stones only), `toMove`, `consecutivePasses`, `moveCount`, `koPointHint` | few bytes | |

Total ≈ 24 KB at 19×19 (≈ 3 KB at 9×9 if bitset width is chosen by N at compile time of the template; first version uses the fixed 19×19 layout everywhere). One `memcpy` of 24 KB per simulation is ~1 µs and is accepted (§5.4.4).

`play(move)` updates cells/chains/liberties/hash, then **pushes a snapshot** of the new stone configuration into the ring buffer (a pass pushes a duplicate of the current configuration, because history is indexed by time step, not by stone changes). `applySymmetry(sym)` transforms cells **and all eight snapshots** (and rebuilds chains). The 8-step history can therefore be encoded from the `Board` alone. (The move list plus the initial position and the rules would also determine every past configuration by replay; the snapshots exist to avoid replaying the game at every leaf, not because replay is impossible.)

**`GameHistory`** — owned by the game, shared read-only by the search:

```cpp
struct GameHistory {
  std::vector<Move>     moves;           // for SGF / debugging
  std::vector<uint64_t> hashes;          // stone-configuration hash after every move (incl. initial)
  std::unordered_set<uint64_t> hashSet;  // same content, O(1) membership
};
```

**Legality** is `Board::isLegal(move, color, const HashHistory& hist)` where `HashHistory` is a small view `{ const GameHistory* game; const uint64_t* pathHashes; int pathLen; }`. Search passes the root's game history plus the hashes along the current descent path (≤ tree depth, scanned linearly). No hash set is ever copied during search.

#### 5.1.2 Rules ("Tromp–Taylor scoring with suicide prohibited")

- Suicide is illegal.
- **Positional superko**: a *stone placement* may not recreate any previous whole-board stone configuration (game history ∪ path). **Passes are exempt**: a pass never changes the configuration and is always legal.
- Game ends after two consecutive passes, or when `moveCount` reaches `move_cap = 2·N²` (722 on 19×19, 162 on 9×9, 50 on 5×5). A capped game is scored as it stands; the termination reason is recorded.
- **Area scoring** (stones + empty points reached only by one colour), komi 7.5 (configurable). Dead stones are not removed.

#### 5.1.3 Superko check cost

- For a **non-capturing** placement the resulting hash is `hash ^ zobrist[point][color]`: one XOR plus one set lookup plus a path scan. Whether a placement captures is O(1) from the liberty bitsets of the ≤ 4 neighbouring enemy chains, so `legalMoves()` does not simulate every empty point.
- Only **capturing** candidates need the removal simulated to obtain the post-capture hash (also computed incrementally: XOR out each captured stone).
- **Every placement is checked**, capturing or not. The widespread shortcut "only captures can repeat a position" is wrong under pass-allowed rules (opponent passes, then a non-capturing placement can recreate an earlier configuration). This must not be "optimised" away later.

#### 5.1.4 Interface sketch

```cpp
class Board {
public:
  explicit Board(int size, float komi);
  bool  isLegal(Move m, Color c, const HashHistory& hist) const;   // suicide + superko; pass always legal
  void  play(Move m);                                              // asserts legality in debug; pushes snapshot
  Color toMove() const;   int size() const;   int moveCount() const;
  bool  gameOver() const;                                          // two passes or move cap
  float score() const;                                             // black − white − komi (area)
  uint64_t hash() const;                                           // stones only
  void  legalMoves(const HashHistory& hist, std::vector<Move>& out) const;
  void  encodeFeatures(uint8_t* planes17) const;                   // §5.2, from the snapshots
  const Snapshot& snapshot(int stepsBack) const;                   // 0 = current, 1..7 = history (zeros before start)
  Board applySymmetry(int sym) const;                              // 0..7, incl. snapshots
};
```

**Reference implementation for tests.** A slow, obviously-correct `RefBoard` (flood-fill liberties, full-board copy per move, list of full stone arrays for both superko and history) lives in the test tree and is fuzzed against `Board` (§9).

### 5.2 Feature planes (`cpp/core/features`)

Exactly the AGZ encoding, `17 × N × N`, `uint8` 0/1, from the perspective of the side to move:

| Plane | Content |
|---|---|
| 0–7 | Stones of the player to move at t, t−1, …, t−7 (snapshots 0..7) |
| 8–15 | Stones of the opponent at t, t−1, …, t−7 |
| 16 | All ones if black to move, else all zeros |

History before the start of the game is all zeros. The C++ encoder reads the eight snapshots (§5.1.1). The Python loader (§6.3) assembles the same planes from the per-position snapshots stored in the chunk (§5.6); it therefore needs **no rules engine**, only indexing. Parity is verified by a fixture test.

**Feature schema version** `features_v1` is written into every chunk header and model metadata; the engine refuses a model whose schema differs from its own.

### 5.3 Neural-network interface (`cpp/nn`)

```cpp
struct NNInput  { const uint8_t* planes; };        // 17*N*N; owned by the caller, must stay valid
                                                   // until evaluate() returns (the call is synchronous)
struct NNOutput { std::vector<float> policy;       // N*N+1 probabilities: fp32 softmax over ALL actions,
                                                   // computed by the evaluator from the model's logits,
                                                   // BEFORE legal-move masking
                  float value; };                  // tanh in [-1,1], perspective: player to move

class NNEvaluator {
public:
  virtual ~NNEvaluator() = default;
  virtual int boardSize() const = 0;
  virtual const std::string& modelId() const = 0;
  virtual void evaluate(const std::vector<NNInput>& in, std::vector<NNOutput>& out) = 0;  // one batch
};
```

**Contract (all implementations):**

- `evaluate` is synchronous: outputs are complete on return, and the caller's `planes` memory is not referenced afterwards.
- The exported model returns **logits and value** as the first two entries of its output tuple (optional auxiliary heads follow and are ignored by the engine, §6.1); the evaluator computes the **softmax in fp32** (logits are cast to fp32 first) so that fp16 execution cannot underflow small priors to exactly 0. Legal-move masking and renormalisation are done by the search, never by the evaluator, because legality depends on history the 17 planes do not contain.
- The evaluator is not required to be thread-safe; one evaluator is used from one thread.

**`TorchEvaluator`**

- Loads a **model version directory** (§6.4): `model.pt` (TorchScript) + `model.json` (`model_id`, board size, plane count, feature schema, residual-block and filter counts, export dtype, git hash, export time). Selects the device at runtime: `--device auto` → CUDA if `torch::cuda::is_available()`, else MPS if `torch::mps::is_available()`, else CPU.
- Runs under `c10::InferenceMode`; the module is in eval mode as exported (BN uses running statistics). On CUDA the module is converted to fp16 by default and the input tensor is cast to the module's dtype; fp32 on MPS and CPU; `--fp32` forces fp32 everywhere for parity checks.
- Input planes are copied into a host tensor `[B,17,N,N]` (uint8), moved to the device, converted to the model dtype; one forward call per batch returns `(logits [B,N²+1], value [B], …)`; logits are brought back as fp32 and soft-maxed on the CPU (a few µs per row).
- **TorchScript is deprecated upstream** but still shipped and loadable in the pinned PyTorch 2.14. We pin to it deliberately for the prototype. **Known risk (observed in M2):** PyTorch 2.14 emits `FutureWarning: torch.jit.trace is not supported in Python 3.14+ and may break` on the dev machine's Python 3.14 venv; tracing and loading currently work and the round-trip tests pass. If tracing breaks on a future PyTorch, the fallback is a Python 3.12 venv for the export step (the C++ loader does not depend on the Python version). `torch.export` / AOTInductor are *not* drop-in replacements for `torch::jit::load` and would require a different C++ loader; that is a separate future task. The exact deployment path (trace in eval mode → save → `torch::jit::load` → forward under InferenceMode on CUDA and on MPS) is verified by `test_torch_eval` on both platforms in **M2**.

**CUDA fast path (M4a step 2, implemented 2026-09-28).** Reason: the Nsight Systems profile of 2026-09-25 (`runs/9x9-r0/profile/2026-09-25`, self-play T = 8, G = 1024 and the 200-pair gate) shows the evaluation thread launch-bound, not GPU-bound — 78 kernels per forward, ≈ 6 µs of CPU per launch (≈ 470 µs per forward) against ≈ 226 µs of GPU execution, the GPU busy 17–20 % of the time, and 29–30 % of the GPU time spent in the NCHW↔NHWC transposes cuDNN wraps around every fp16 convolution. This is the condition §2 and §13 name for a CUDA-Graphs backend. Two switches, both CUDA-only, both inside `TorchEvaluator`, the `NNEvaluator` contract unchanged:

- **`inference.channels_last`** (true; **fp16 only**). At load the module's 4-D parameters are converted to the channels-last memory format and the input batch is made channels-last before the forward, so cuDNN runs its NHWC kernels without transposes. It is not applied with `--fp32`: cuDNN's NHWC fp32 kernels compute at TF32-class precision even with TF32 switched off (measured on the 9×9 model `0026`: 1.4·10⁻³ against the NCHW fp32 path, the same as `allowTf32`; the NCHW path's own batch-shape noise is 2·10⁻⁶), and fp32 has to stay IEEE fp32 for the parity checks. Done in the evaluator, not at export: `model.pt` (format `torchscript-v1`) is unchanged, every existing model version (ladder entries, the incumbent of a gate) gets it, nothing is re-exported.
- **`inference.cuda_graphs`** (true). Batch sizes are rounded up to a bucket — 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096 — and each bucket in use owns a static GPU input tensor (uint8 `[B,17,N,N]`), a captured graph of *cast to the model dtype → forward → logits and value cast to fp32 and concatenated to `[B, N²+2]`*, and the static output tensor. `evaluate` with b positions copies them into the bucket's host buffer (rows ≥ b keep their previous content: they are evaluated and ignored — in eval mode no operation mixes samples), does one host-to-device copy, `replay()`, one device-to-host copy, and the fp32 softmax on the CPU for the first b rows. A bucket is captured on its first use after 3 plain forwards of that shape (JIT profiling/fusion and cuDNN's algorithm choice settle before the capture). If a capture throws, the evaluator reports it once on stderr and uses the plain path for the rest of its life; batches above the largest bucket use the plain path. Memory: each graph keeps its own activations (≈ 0.4 GB at B = 1024 for 6×64); the buckets in use sum to at most ≈ 3× the largest.

Outputs of the fast path are not bit-identical to the plain path with fp16 (padding changes the batch shape, the memory format changes the kernels; the same class of difference as batch composition, §5.5.1); the parity tolerances of §9 "NN backend" apply to every combination of the two switches. Measured on the 9×9 model `0026`, 1,024 positions from random games: fp16 fast path against the plain path 1.9·10⁻³ (policy) / 4.9·10⁻⁴ (value), the plain path against itself at another batch shape 2.1·10⁻³ / 9.8·10⁻⁴; fp32 1.7·10⁻⁶ against 1.9·10⁻⁶. A replay of the same input is bit-identical to the previous replay. The summary of `mango_selfplay` reports `channels_last`, `cuda_graphs`, `graphs_captured`, `graph_replays`. `--no-channels-last` / `--no-cuda-graphs` on `mango_selfplay`, `mango_match` and `mango_gtp` switch them off for A/B measurements. Not part of the config fingerprint (no change to the data semantics). Out of scope: folding BN into the convolutions, TensorRT, pinned host memory (the copies are < 1.5 % in the profile).

**Captures are exclusive (2026-09-28).** PyTorch captures in the global mode, where a CUDA call of any other thread during the capture invalidates it. Observed with two evaluators used from two threads at once (an experiment with several evaluation threads, reverted): a capture failed and that evaluator silently fell back to the plain path. A process-wide reader-writer lock therefore guards the GPU work of every `TorchEvaluator`: held shared by the constructor, `evaluate` and the destructor, exclusively by a capture, which runs after the call that completes the bucket's warm-up. The drivers use one evaluation thread (the match driver's owns both evaluators), so today the lock is never contended; captures happen only while buckets warm up, so it costs nothing in steady state.

**`FakeEvaluator`**: deterministic policy (uniform, or a table keyed by position hash) and value; used by MCTS tests so search logic can be tested without a GPU.

**NN cache** (disabled by default in the first version; `--nn-cache-size 0`):

- Key = `(model_id, hash of the complete 17×N×N encoded input)`. The board hash is **not** a valid key: different move orders can reach the same stones with different history planes.
- Value = the raw `NNOutput` **before** legal-move masking.
- One cache per evaluator, so candidate and incumbent in a match never share entries.

### 5.4 Search (`cpp/search`)

#### 5.4.1 Perspective invariants (normative)

1. `NN.value(s)` is the expected outcome **for the player to move at s**, in [−1, 1].
2. For an edge `(s, a)`, `Q(s,a) = W(s,a)/N(s,a)` is from the perspective of **the player who chooses a at s**, i.e. the player to move at s.
3. Playing `a` at `s` leads to `s'` where the *other* player moves. A value `v(s')` obtained at `s'` (from the network or from a terminal result) is backed up into edge `(s,a)` **negated**: `W(s,a) += −v(s')`. Along a path root→leaf the sign alternates at every ply.
4. A terminal state's value is computed from the score **for the player to move at that terminal state**, so rule 3 applies unchanged.
5. The **root value** used for resignation is `max_a Q(s₀,a)` over visited children and the root's own network value `v(s₀)`; both are already in the root player's perspective. The mover resigns iff both are `< v_resign`.
6. Root Dirichlet noise and temperature apply only at the root, only in self-play.

#### 5.4.2 Data structures and memory

```cpp
enum class NodeState : uint8_t { UNEXPANDED, PENDING, EXPANDED, TERMINAL };

struct Edge {                       // 32 bytes with alignment
  Move     move;                    // u16
  uint16_t virtualLoss;
  float    P;
  uint32_t N;
  float    W;
  std::unique_ptr<Node> child;      // 8 B; nullptr until first traversal
};

struct Node {
  NodeState state;
  Color toMove;
  float nnValue;                    // v(s) from the network (or terminal value), perspective: toMove
  std::vector<Edge> edges;          // legal moves only, filled on expansion
  uint32_t visitCount;              // = 1 + Σ edges[i].N once EXPANDED
  float terminalValue;              // valid iff TERMINAL
};
```

Children are legal moves only; priors are the raw policy renormalised over the legal set. Nodes own their children through `unique_ptr`; tree reuse moves the chosen child's pointer to become the new root and frees the rest.

**Memory estimate.** One new node per simulation; a node costs ≈ 64 B + 32 B × (legal moves).

| | 9×9 (≤ 82 edges) | 19×19 (≤ 362 edges) |
|---|---|---|
| per node | ≈ 2.7 KB | ≈ 11.7 KB |
| per move (S sims) | 200 × 2.7 KB ≈ 0.5 MB | 800 × 11.7 KB ≈ 9.4 MB |
| G games in flight | 128 × 0.5 MB ≈ 70 MB | 64 × 9.4 MB ≈ 600 MB |
| with tree reuse (retained subtree, worst case ×2) | ≈ 140 MB | ≈ 1.2 GB |

The "×2 for tree reuse" row is an **estimate, not a bound**: a retained subtree can accumulate over many moves. The driver records the peak live node count per game and per process (`--stats`), and M3b reports the measured peak instead of this estimate. K > 1 adds at most K−1 pending nodes per game. Edges of a node are freed when the node is freed, so tree reuse never leaks siblings.

#### 5.4.3 Selection

Score `S(a) = Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))`, with **Q = 0 for unvisited edges** (paper). When `Σ_b N(s,b) = 0` every `S(a)` is 0, so the rule is completed by an explicit **tie-break: higher P first, then lower move index**. Consequently the first selection at a fresh node is the highest-prior legal move, deterministically.

**Known interaction (to be swept in M4, §8.2).** With Q = 0 for unvisited edges, a position where all visited children have Q < 0 makes unvisited moves look comparatively attractive (search spreads out), and one where all Q > 0 makes them look bad (search narrows); a larger `c_puct` amplifies this asymmetry. `c_puct = 1.5` and `Q = 0` are the **starting point**, not constants: both are config values and are the first parameters swept once the 9×9 loop runs. "Parent-value FPU" (`Q_init = v(s)`, the parent's own network value, which is already in the chooser's perspective) is the experiment flag.

#### 5.4.4 One simulation (sequential reference, K = 1)

```
simulate(rootNode, rootBoard, gameHistory):
  board = rootBoard                      // one 24 KB copy per simulation
  pathHashes = []; path = []; node = rootNode
  loop:
    if node.state == TERMINAL:  v = node.terminalValue; break
    if node.state == UNEXPANDED:
        expandAndEvaluate(node, board, {gameHistory, pathHashes})   // legal moves need the history view
        v = node.nnValue (or terminalValue); break
    edge = select(node); path.push(edge)
    board.play(edge.move); pathHashes.push(board.hash())
    node = edge.child (created lazily as UNEXPANDED)
  backup(path, v)                         // from leaf to root: edge.N += 1; edge.W += ±v (sign alternating,
                                          // starting with −v for the edge into the leaf); node.visitCount += 1
```

The root is expanded (and evaluated) once before the first simulation so that `visitCount(root) = 1` and root noise can be applied to its priors.

**Two counters, kept separate (normative).**

- `simulations_this_move` — the number of **new, completed** simulations run for the current move (collisions and abandoned descents do not count). The search budget `S` applies to this counter: a move's search ends when `simulations_this_move == S`.
- `root_total_visits = Σ_a N(s₀,a)` — includes visits **inherited through tree reuse**. π and the stored visit counts are computed from this total. With tree reuse the two differ: a new root that already carried 80 child visits and receives 200 new simulations stores visits summing to 280.

Inherited visits are therefore *not* charged against the budget (the paper's "1600 simulations" are new simulations with the subtree retained). A config flag `budget_includes_inherited = false` documents this; setting it to `true` ends the search when `root_total_visits ≥ S` instead, which reduces the actual search per move.

Visit counts are `uint32` everywhere — in the tree, in the chunk (§5.6) and in the loader — so accumulated statistics with a configurable budget cannot overflow a 16-bit field.

#### 5.4.5 Batched simulations: pending-evaluation protocol (K > 1, and multi-game batching)

Virtual loss only *discourages* re-selecting a path; it cannot guarantee distinct leaves. The protocol below makes this explicit:

- **Reservation.** During a descent, each traversed edge gets `virtualLoss += 1`, and `Q` and `√ΣN` are computed *with* virtual losses counted as extra visits with value −1 (from the chooser's perspective). Nothing else changes: `N` and `W` are committed only at backup.
- **Reaching an UNEXPANDED node**: it becomes `PENDING`, its input planes are encoded and appended to the batch, and the path (edges and the board copy's hashes) is stored with the request.
- **Reaching a PENDING node** ("collision"): the descent is **abandoned**: all reservations on its path are removed, no visit is committed, and a `collisions` counter is incremented. The game contributes fewer than K leaves this step.
- **Reaching a TERMINAL node**: no evaluation is needed; it is backed up immediately (visit committed, reservations removed).
- **Commit.** After the batch returns, for each request: the node becomes `EXPANDED` with its edges/priors/`nnValue`; then the normal backup runs, which commits `N`/`W` and removes the reservations along its path.
- **Failure (first version: no transactional rollback).** If `evaluate` throws, the driver first **retries the same requests once** (the PENDING nodes, their planes and symmetries are kept, so no RNG state is consumed and every game stays reproducible from its seed — re-collecting would draw new symmetries). If the retry throws too, every PENDING node in the batch reverts to `UNEXPANDED` and all reservations are removed. Simulations that completed *within the same step without the network* (terminal backups) are **kept** — their visits are already committed and are valid. Lazily created `UNEXPANDED` children with `N = 0` are also kept; they are harmless. `simulations_this_move` reflects only completed simulations, so the search simply continues from the actual count. The exception then propagates and the process aborts loudly. Restoring the exact pre-batch tree would require journaling every modification and is not attempted.
- **Invariant checks (debug builds).** After every step: no node is PENDING, all `virtualLoss == 0`, and for every EXPANDED node `visitCount == 1 + Σ N(edges)`.

**What batching does and does not guarantee (normative for the tests).**

- **K = 1, batching across independent games**: each game's search is *exactly* the sequential search — every selection within a game sees fully committed statistics. Test: with a deterministic `FakeEvaluator` and one RNG stream per game, every game in a batched run reproduces, move by move and count by count, a standalone sequential run of that game.
- **K > 1 within one game**: the second and later descents of a step select on statistics that do not yet include the first descent's result, so the result legitimately **differs from sequential search even when no collision occurs**. Tests for K > 1 therefore check only: the pending set has no duplicate nodes; collisions are counted and leave no reservation behind; backup signs and counts are correct; after commit all invariants hold; after a simulated evaluator failure no node is PENDING, no reservation remains, and completed simulations are retained.

#### 5.4.6 Root and move choice

- Dirichlet noise `Dir(α)` mixed with `ε = 0.25` into the root priors after expansion, self-play only. `α = 0.03 · 361 / N²` (deviation D3).
- **Search-time symmetry (paper).** By default each leaf is evaluated under a uniformly random dihedral symmetry and the policy is mapped back. Flag `--search-symmetry off` for deterministic tests.
- **Move choice**: `π(a) ∝ N(s₀,a)^(1/τ)`; `τ = 1` for `move_number < temperature_moves` (30 on 19×19, 8 on 9×9); afterwards `τ → 0`, i.e. argmax with ties broken by prior.
- **Stored search result**: the chunk stores the **raw root visit counts** `N(s₀,·)` including inherited visits (sparse, `uint32`, §5.6) together with the move actually played and the root's `v(s₀)` and `max_a Q(s₀,a)`. The training target is derived in the loader: `store_pi = "visits"` (normalised visits at every move; our default, deviation D4) or `"temperature"` (paper: normalised `N^(1/τ)` before the cutoff; **after the cutoff the one-hot target is the stored played move**, not an argmax recomputed by the loader — the search broke visit ties by prior, which the file does not contain). Because raw counts are stored, switching the target never requires regenerating data.

#### 5.4.7 Resignation

Disabled in the first version (`resign_threshold = -1`). When enabled: per invariant 5, the mover resigns if `v(s₀) < v_resign` and `max_a Q(s₀,a) < v_resign`, i.e. iff `r_t = max(v(s₀), max_a Q(s₀,a)) < v_resign`. 10% of self-play games are tagged `no_resign` and played out.

**Data for the automatic threshold.** Every move of every game stores `root_value = v(s₀)` and `root_max_q` in the chunk (§5.6), so `r_t` is available offline without re-running any search. With §5.4.10 on, `root_value` is `value(s₀)` = `v(s₀)` raised to the exact value of a game-ending move at the root when that is better (a player who can end the game with a win must not resign).

**Selection rule (as in the paper, made precise).** After each iteration, over the **no-resign games of the current window**: for a candidate threshold `t`, a game is a *false positive* if its eventual winner had `r_t < t` at any of their moves. `false_positive_rate(t) = (#false-positive games) / (#no-resign games)`. `v_resign` is the largest `t` in the allowed range `[−0.99, −0.50]` (step 0.01) with `false_positive_rate(t) < 0.05`. If fewer than **100 no-resign games** are in the window, or no `t` in the range qualifies, resignation stays **disabled** for the next iteration. The chosen `t`, the sample size and the rate are logged per iteration. **Measured on the first 9×9 run (M4 review):** a threshold selected on the window and applied to the *next* model showed 5–12 % false positives on that model's no-resign games (the value head sharpens faster than the window turns over). That rate is the control quantity, not a label-error rate (the game ends at the first dip by either player, usually the loser's); the pipeline also logs the **counterfactual label-error estimate** (`resign.counterfactual_false_resign_rate`: over the no-resign games, the first player below the threshold resigns, mislabelled iff that player went on to win) — 8.8 % of the games resignation would have ended, pooled over R0's iterations 5–15. Two pipeline-only knobs exist to act on this without leaving the paper's rule: `search.resign_select_scope = "latest"` selects on the previous iteration's no-resign games only (≈ 200 games ≥ the 100-game floor; on the R0 data it lowered the realised rate in 7 of 10 iterations, e.g. 11.3 % → 4.8 % at iteration 15) and `search.resign_fpr_target` (default 0.05; 0.025 kept the realised rate at 0.5–5.5 % on the same data). Defaults stay at the paper's values; whether a control run without resignation changes strength or value learning is an open experiment (§11 M4).

#### 5.4.8 Tree reuse

After a move is played the chosen child becomes the root, keeping its subtree and statistics. Root noise is re-applied to the new root's priors. The tree is **never shared between two different players/models** (§5.8). **GTP `undo`, `clear_board`, `boardsize` and `komi` discard the whole tree** (tree reuse only ever follows a forward move).

#### 5.4.9 Acceleration hooks in the search (all off by default; §13)

The three search-side techniques of the acceleration profile are specified here so that the search contract stays in one place. With every `accel` switch off, the search is exactly §5.4.1–§5.4.8.

- **Playout cap randomization (PCR, KataGo).** Before each self-play move, with probability `full_prob` the move is a **full search**: budget `S` new simulations, root Dirichlet noise on, forced playouts on (if enabled), and the position becomes a **policy-target position** (`search_kind = 1`). Otherwise it is a **reduced search**: budget `reduced_simulations`, root noise **off**, forced playouts **off**, and — **as in the paper (KataGo 2019 §3.1)** — the position is **not a training sample** (`search_kind = 0`; the loader skips it). The gain comes from more games per GPU-hour, hence more independent game outcomes for the value head, not from training on the cheap positions. `accel.playout_cap.reduced_positions = "value_only"` is an **additional experimental variant** that keeps reduced positions as value/auxiliary samples with the policy term masked; it changes the objective's policy/value balance and is specified in §6.2 (normalisation) and §6.3 (sampling). The move is still chosen from the visit counts with the usual temperature rule (§5.4.6). Tree reuse applies to both kinds; `simulations_this_move` counts new simulations in both. The decision is drawn from the game's RNG stream so a game is reproducible from its seed. Match/GTP/ladder searches are always full searches.
- **Forced playouts (KataGo).** At the root of a full search, a child `a` **that has already been visited** with prior `P(a)` is selected regardless of its PUCT score while `0 < N(a) < n_forced(a) = sqrt(k_forced · P(a) · Σ_b N(b))`, `k_forced = 2`. An unvisited child is never forced (its first visit must be earned through PUCT; without the `N(a) > 0` condition every action with `P(a) > 0` would qualify as soon as the root has one visit, and tiny-prior moves would consume the budget — KataGo 2019 §3.2 applies the rule to visited children only). Ties among forced children follow the normal PUCT order. Never applied below the root, never in reduced searches, never in match play.
- **Policy-target pruning (KataGo).** After a full search, with `b` the most-visited child, all of `Q(·)` **frozen** at their end-of-search values (`W/N` before any removal; utilities are held constant as in KataGo 2019 §3.2), the exploration term's total **frozen** at `N_tot = Σ_a N(a)` (pre-pruning), and the per-child removal budget `m(c) = floor(sqrt(k_forced · P(c) · N_tot))` — the paper's `n_forced` threshold evaluated at the final total and rounded down, **not** the number of selections that were actually forced (a child that reached its visits through ordinary PUCT and turned out bad is prunable too, as in the paper): for every child `c ≠ b`, let `n = N(c)`; while `n > N(c) − m(c)` and `Q(c) + c_puct · P(c) · sqrt(N_tot) / (1 + (n − 1)) ≤ Q(b) + c_puct · P(b) · sqrt(N_tot) / (1 + N(b))`, decrement `n` — i.e. remove up to `m(c)` visits one at a time as long as the child, with one fewer visit, would still not have out-scored `b`. The target count is `n`; a child ending with `n ≤ 1` is dropped from the target (count 0), except the played move, which keeps `max(n, 1)`. `b` is never pruned. **Only the stored target is pruned**: the tree, `root_total_visits`, `root_value` and `root_max_q` are untouched, and move selection has already happened. In the chunk this shows as `Σ count ≤ root_total_visits` (§5.6, `record_extras` bit 2).

#### 5.4.10 Game-ending moves (M4 review; `search.resolve_terminal_moves`, default on; deviation D18)

**Why.** On the first 9×9 run every loss of a strong model to the iteration-1 model was a game ended by two consecutive passes before move 40 with a 0.5–2.5-point margin: the strong model passed first while behind by komi (or accepted a pass while behind). With the plain search of §5.4.4 the pass edge at `s₀` is valued by the network at `s₁` (the position after the pass) and by the replies searched from `s₁`; the opponent's answering pass — which ends the game at an exact, unfavourable score — has a tiny prior at `s₁` and is never visited within the budget, so the exact result is invisible and the value head's near-komi misjudgement decides (`docs/RUN_9x9_R0.md`, reading 1).

**Rule.** A move `a` at position `s` *ends the game* if `s` has one consecutive pass and `a` is a pass, or if `s` is one move before the move cap (then every move ends it). Its outcome is exact and costs no evaluation. When a node is expanded:

1. every game-ending edge gets `terminalValue(a) = −v_T(s·a)` — the finished game's value for the player to move there (rule 4), negated into the chooser's perspective (rule 3);
2. the node's `terminalMoveValue = max_a terminalValue(a)` over those edges, and the value backed up for the expansion is `value(s) = max(v(s), terminalMoveValue)`: the mover can guarantee the best game-ending outcome, so the true minimax value is at least that — a lower bound that only ever raises the network's estimate (a winning pass makes the node exactly `+1`; a losing pass changes nothing);
3. in selection, an **unvisited** game-ending edge uses `terminalValue(a)` in place of the FPU value `Q = 0`, so a winning pass is chosen from the first selection on and a losing one is deprioritised. Visited edges use `W/N` as always.

`rootValue()` (resignation, §5.4.7, and the stored `root_value`, §5.6) is `value(s₀)` — the raw `v(s₀)` stays available as `rootNetValue()`. Nothing else changes: no extra visits, `visitCount == 1 + Σ N` holds, the collect/commit protocol is untouched, so the K = 1 batched-equals-sequential guarantee (§5.4.5) carries over. Cost: one board copy and score per game-ending edge at expansion (one for the pass case; all legal moves only at the cap). `resolve_terminal_moves = false` restores the paper's plain search (tests of §5.4.4's Q arithmetic use it). AlphaGo Zero does not describe such a rule; it is a correctness patch of the search, kept behind a switch and listed as D18.

### 5.5 Self-play driver (`cpp/selfplay`)

First version, deliberately simple (**one evaluator, one thread, G games, K = 1**):

```
own: one TorchEvaluator, G concurrent games, each with its own Board, GameHistory and tree
loop:
    for each game g: run selection until it produces one pending leaf (or a terminal backup)
    evaluate the ≤ G pending leaves in one batch
    commit + backup
    for games with simulations_this_move == S: choose a move, record root visit counts (total), v(s₀), max Q, play it
    for games that ended: write the game record to the chunk + SGF, start a new game
```

With K = 1 per game there can be no within-game collision, so the batch is exactly the set of games with a pending leaf. Defaults for 9×9: `S = 200`; `G = 128` with the first-version driver (R0 iterations 1–23), `G = 1024` with T = 8 search threads from M4a on (§5.5.1, numbers in §11). After M3b measures throughput (§10) we add, in this order and only if needed: K > 1 per game (§5.4.5), multiple threads each with its own evaluator, a shared cross-thread batching queue.

#### 5.5.1 Multi-threaded driver (M4a, planned; reviewed 2026-09-25)

R0 showed the single search thread to be the limit (§10, §11 M4a): the evaluator *call* (transfer + forward + softmax + copy-back, §5.3) takes 49 % of the wall time and the single-threaded search the rest. **G is the global concurrency**: G games in flight in total, at most one pending leaf per game (K = 1), so a batch never exceeds G whatever the thread count. Threads therefore do not enlarge batches; they shorten the search time between two evaluator calls. Larger batches come only from a larger G, which the driver allows (G is bounded by memory, not by threads) and which the benchmark covers separately.

**T = 1 is the first-version driver, unchanged** (the same code path, bit-identical output on the same seed). For T > 1:

```
T search threads; thread t owns games t, t+T, ... (each with its own Board, GameHistory, tree, RNG stream, game_seed)
    loop: for every own game: selection until one pending leaf (or a terminal backup);
          submit all own pending leaves (≤ ceil(G/T) requests, each tagged (thread, game)) to the queue as one round;
          wait until every request of the round has its result; commit + backup; moves / new games as in the first version
1 evaluation thread: wait until the queue is non-empty, take up to B = selfplay.max_batch (default G) requests from its
          head in FIFO order — a thread's round may be split over several forwards when it exceeds the room left —
          one forward, write each result to its request, and signal a thread once the last request of its round is done
chunk writer: one mutex; game records appended in completion order (the chunk format is order-free, §5.6)
```

The queue holds requests, not rounds, so B may be smaller than one thread's share (B < ceil(G/T) is legal; a round then takes several forwards and its thread waits for all of them); B ≥ G means every queued request fits one forward. A retry (below) repeats the failed forward with exactly its requests. Which requests share a forward depends on timing, so with the real fp16 network a T > 1 run is **not** bit-reproducible and is not claimed to be (the batch shape changes fp16 rounding). The contract for T > 1 is the per-game one: under the deterministic `FakeEvaluator`, the record of every game — identified by its `game_seed` — equals the record of the same seed in a standalone sequential run (moves, root visit counts, `root_value`, `root_max_q`, `z`, termination); chunk bytes and game order are not compared. Everything per game is untouched: K = 1, the collect/commit protocol, seeds and RNG streams, resignation and move choice, the recorded fields. The summary merges the threads' counters (evaluations, batches, average batch, retries, terminations; `seconds` is wall time).

**Failure protocol (both drivers).** A failed forward is retried once by the evaluation thread with the same requests while the submitting threads wait (§5.4.5). On the second failure the evaluation thread sets a global abort flag and broadcasts on the queue's condition variable; every search thread wakes, calls `SearchTree::abort` on each of its pending leaves (node back to Unexpanded, reservations removed), drops its queued sub-batch and returns; the driver joins all threads, publishes nothing from the affected batch (games in progress are lost, as in the first version) and exits non-zero. Tests: after a simulated single failure no node is Pending and no reservation remains in any thread and the retried batch is identical; after two failures every thread has exited and the chunk writer has published only complete chunks.

**Match driver.** `playMatch` gets the same structure with **one evaluation thread that owns both evaluators**: requests are bucketed by side (the two sides are different models with their own caches, §5.8) and each bucket is a separate forward; a bucket is never mixed with the other side's requests. T search threads over the G game slots as above; T = 1 is the first-version match driver unchanged.

**Step 0: processes instead of threads (pipeline-only, `selfplay.processes = N`).** **N = 1 is the existing pipeline path, unchanged** (one process, seed = the iteration seed, `next_chunk_id` advanced by the chunks actually published; this is what a resumed run keeps using). For N ≥ 2, no C++ change: N `mango_selfplay` processes, each the first-version driver with **its own G** games in flight (so the total concurrency is N·G, which the benchmark must state), its own evaluator and its own seed. The self-play plan persists, before any process starts, one record per worker: `{k, games, seed = derive_seed(iteration seed, k), chunk_id_start, chunk_id_end}` with `chunk_id_end − chunk_id_start = games` (one id per game is enough for uniqueness, ids need not be contiguous across workers; `next_chunk_id` advances to the last block's end). A restart re-reads the plan and never re-splits: each worker's remaining quota is its `games` minus the games in the published chunks of *its* id block, and it restarts at `max(published id in its block) + 1`. If one worker exits non-zero the pipeline terminates the others (they publish only complete chunks; `.tmp` files are ignored, §5.6), logs the failure and raises; the next start resumes every worker from the plan. Per-worker logs `selfplay_k.log`; the summaries are merged (games, positions, evaluations, terminations summed; `seconds` = wall time of the phase; `positions_per_s` from those). Whatever interrupts the launch or the wait (a failed `Popen`, Ctrl-C) terminates and waits for every started worker before the error propagates, so a restart never runs beside a leftover. On a restart the recorded summary's counts (games, positions, terminations, black wins, chunks) are the whole iteration's, rebuilt from the published chunks; the relaunch's own counts and timing are kept under `launch`, and the fields the interrupted launches alone could have supplied (timing, evaluator counters) are listed under `incomplete`. **Implemented 2026-09-25** (`split_selfplay_workers`, `merge_selfplay_summaries`, `Pipeline._run_workers`, `_selfplay_workers`; tests in `tests/test_pipeline_workers.py` with `tests/fake_selfplay.py`). **Measured 2026-09-25 (step 0, 9×9, S = 200, fp16, RTX 5080, model `0022`, 900 games per configuration, wall time incl. process start-up):** 1×G128 299 positions/s (avg batch 112; R0's 350 was measured on a 2,000-game iteration, whose tail — fewer games in flight at the end — weighs less); (a) same total concurrency 3×G43 = 129: **232** (avg batch 38, the evaluator call 86 % of each process's wall time — splitting a fixed G over processes only shrinks the batches); (b) 2×G128 430, 3×G128 485, **4×G128 501** (avg batch 85, 100k evals/s — at the forward-only ceiling of ≈ 109k at batch 128, §10). So step 0 gives 1.7× at 4× the concurrency and the GPU is now evaluator-bound at batch ≈ 90: **≥ 700 is not met at step 0** (as expected — it is the baseline for step 1), and step 1 only reaches it if the shared queue produces batches ≥ 256 (ceiling 237k evals/s), i.e. G ≥ 512 in one process.

**Step 1 implemented 2026-09-25** (`EvalQueue` in `selfplay/eval_queue.h`: rounds, FIFO `max_batch` batches, abort/stop; `BatchedSelfplay(ev, G, model, threads, max_batch)` with `runSingleThread` = the M3b code and `runThreaded`; `playMatch(..., threads, max_batch)` with one evaluation thread owning both evaluators and one forward per side; `--threads` / `--max-batch` on both executables; `SelfplayGameOptions::checkInvariants` for the tests). The summary line gained `threads` and `max_batch`; otherwise T = 1 writes the same chunks and summary as before. **Measured 2026-09-25 (step 1, 9×9, S = 200, fp16, RTX 5080, model `0022`, one process, 900 games per configuration, wall time incl. start-up; `max_batch` = G):** (a) same total concurrency G = 128: T = 1 **293**, T = 2 **324**, T = 4 311, T = 8 291 positions/s — the threads shorten the search share but the rounds arrive out of phase, so the average batch halves (110 → 55) and the evaluation thread is busy 96–98 % of the wall time: at G = 128 the driver is evaluator-bound and threads gain nothing; (b) larger G: T4 G256 543 (batch 99), T4 G512 595 (125), T8 G512 690 (158), **T8 G1024 820 positions/s** (batch 206, 164k evals/s, evaluation thread 97 % busy) — **≥ 700 met** (2.8× the T = 1 number of the same day, 2.3× R0's 350). **Gate (200 pairs, `0022` vs `0021`):** T1 G128 189 s (R0's ≈ 180), T4 G128 251 s (slower: batches per side halve), T4 G400 **117 s**, T8 G400 128 s — **≤ 90 s not met**: with 400 fixed games the per-side batch is ≤ 200 and shrinks as games finish, and the evaluation thread is already saturated, so the queue cannot take the gate below ≈ 115 s; a faster forward per batch (CUDA graphs / TensorRT, §10) or a different gate size would be needed, neither is part of M4a. With the real fp16 network T > 1 is not bit-reproducible (batch shapes change rounding): the four gate runs scored 0.655, 0.630, 0.662, 0.637 for the same seed, all inside the interval.

**Driver profile (measurement for the next self-play steps, implemented 2026-09-28).** Counters only — nothing the drivers do depends on them, T = 1 still writes the same chunks. `BatchStats::profile` (`DriverProfile`) records: for the search threads (seconds summed over the threads) `search_collect_s` (selection up to the pending leaves, moves, game ends), `search_commit_s` (commit + backup), `search_callback_s` (game start and `onGameDone` — the chunk writer — including the wait for its mutex), `search_wait_s` (T > 1: blocked in `submitRound`), `rounds`; for the evaluation thread (T > 1) `eval_wait_s` (idle, no request queued), `eval_call_s` (= `eval_seconds`, inside `evaluate()`), `eval_handback_s` (request list, results, wake-up), `rounds_per_batch` (rounds a forward carried requests of); `eval_busy` = (call + handback) / wall, `search_busy` = (collect + commit + callbacks) / (T · wall); a batch-size histogram (power-of-two bins, forwards and requests per bin); and a timeline of 1-second bins (forwards, requests, average games in flight at each forward), which shows the tail of a phase. `TorchEvaluator::timings()` splits `evaluate()` into `pack_s` (planes into the host buffer), `device_s` (host → device, forward or replay, device → host, which the copy back waits for) and `unpack_s` (fp32 softmax, outputs), with `rows` and `padded_rows` (rows the forwards ran on after padding to a bucket). `mango_selfplay` puts everything but the timeline into the summary's `profile` object and prints a profile line to stderr; `--profile-out FILE` writes the profile with the timeline; the pipeline passes `logs/selfplay_profile/<iteration>.json` (`<iteration>_k<k>.json` per worker with `selfplay.processes` ≥ 2) and logs the busy fractions. The match driver is not instrumented yet. **Measured 2026-09-28 (model `0061`, 2,000 games, T8 G1024, fp16 channels-last graphs, GPU otherwise idle):** 1,441 positions/s; the evaluation thread is busy 99 % (670 µs per call: pack 8, **device 584**, unpack 79, handback 16); the search threads are busy 38 % and wait the rest; 3.0 rounds per forward, average batch 205, 7.9 % padded rows; the tail — from the first second with fewer than G/2 games in flight — lasts 11.9 s of 53.9 s (22 %) and evaluates 14 % of the positions. nsys (900 games, graph-level trace) puts the GPU at 84 % busy with ≈ 90 µs idle between forwards; a forward's GPU span grows about linearly from batch 128 on (≈ 2.4 µs per row at 192, 1.9 at 512) and is ≈ 150–170 µs for any batch up to 16 (medians; bucket 48: 230 µs, 64: 227, 96: 308). The mean per bucket is misleading: the first plain (warm-up) forwards of a new shape in the first ≈ 2 s of a process occasionally take 0.5–0.9 s (TorchScript re-specialisation, cuDNN set-up for the shape) and land on whichever buckets appear first — one run put 1.3 s on bucket 64 (mean 1,137 µs, median 233), another 0.9 s on bucket 768 — ≈ 1–1.5 s per launch, 2–3 % of a 54 s phase. A node-level trace of bucket 64 (138 µs of kernels): 12 residual-block convolutions 60 µs, 13 fused BN+ReLU / BN+add+ReLU kernels 46 µs (33 %), the rest 32 µs; the span from the copy in to the copy back exceeds the kernel time by 70–230 µs (copies, graph launch). So the device time is GPU work, not transfers.

**Result buffers owned by the driver (fix, 2026-09-28).** A search thread's result buffer (`outs`) lived in the thread; after an abort (a second evaluator failure or an error in any thread) a search thread returns while the evaluation thread may still be finishing a forward that carries its requests and then writes into the freed buffer — a latent use-after-free of both M4a drivers (self-play and match). The buffers are now owned by the driver and outlive every thread. No test observes the bug deterministically (it needs an abort while a forward is in flight, and a sanitizer to see the write); the existing abort tests of both drivers run the path. **Tried and reverted the same day:** pinned host buffers with asynchronous copies of only the batch's rows (1,485 positions/s against 1,417–1,441 — the device time per call did not move), and several evaluation threads with one evaluator and stream each (E = 2: 1,505, +4–6 %, average batch 119; E = 3: 1,443). With the GPU computing 84 % of the time neither pays; what remains is GPU compute per position and the tail of the phase.

**Resignation.** `mango_selfplay --resign-threshold T` overrides `search.resign_threshold`; the pipeline passes the `v_resign` it selected for the iteration (§5.4.7, §6.5), `-1` disables. The summary line reports the threshold used and the termination counts.

**Seeds.** The run has a `run_seed`. Each game's seed is `game_seed = hash(run_seed, chunk_id, game_index)`; it drives root noise, temperature sampling and search symmetries for that game, is stored in the game record (§5.6) and in the SGF `GC` comment, and makes any individual self-play game reproducible.

### 5.6 Game-chunk format (`*.mgo`, version 2, `docs/DATA_FORMAT.md`)

Stored **per game, not per position**: position snapshots + moves + sparse root visit counts + result. Planes and targets are assembled by the loader (§6.3). This replaces the per-record 17-plane layout of v2, which does not scale (19×19: 7.6 KB per position, ≈ 190 GB for a 100k-game window).

**Encoding rules (normative).** All integers little-endian. Fields are serialised **one by one at the offsets below** on both sides (C++ writer/reader and Python reader); no struct is ever `memcpy`'d or `fromfile`'d as a whole, so compiler padding cannot leak into the format. The header is exactly **128 bytes**; `header_size` is stored and checked. Strings are ASCII, zero-padded; readers truncate at the first NUL.

**File header (128 bytes)**

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | char[4] | magic `"MGO2"` |
| 4 | 1 | u8 | version = 2 |
| 5 | 1 | u8 | board_size N |
| 6 | 1 | u8 | planes = 17 |
| 7 | 1 | u8 | feature_schema = 1 |
| 8 | 1 | u8 | rules_id = 1 (TT scoring, no suicide, positional superko) |
| 9 | 3 | u8[3] | reserved (zero) |
| 12 | 4 | f32 | komi |
| 16 | 4 | u32 | header_size = 128 |
| 20 | 4 | u32 | num_games |
| 24 | 4 | u32 | num_positions = Σ T over games |
| 28 | 4 | u32 | simulations_per_move (config S) |
| 32 | 32 | char[32] | model_id |
| 64 | 16 | char[16] | config_fingerprint (SHA-256 prefix of board+search+selfplay config) |
| 80 | 8 | u64 | chunk_id |
| 88 | 8 | u64 | run_seed |
| 96 | 2 | u16 | move_cap |
| 98 | 2 | u16 | temperature_moves (needed by the loader for `store_pi = "temperature"`) |
| 100 | 1 | u8 | record_extras bitmask: bit 0 = every game record ends with `final_ownership`; bit 1 = every game record carries per-move `search_kind`; bit 2 = stored visit counts are policy-target-pruned (§5.4.9). Zero under the AGZ profile |
| 101 | 1 | u8 | reserved (zero) |
| 102 | 2 | u16 | reduced_simulations (PCR reduced budget; 0 = PCR off) |
| 104 | 4 | f32 | full_search_prob (PCR; 1.0 when PCR off) |
| 108 | 20 | u8[20] | reserved (zero) |

**Game record** (repeated `num_games` times, immediately after the header, no alignment):

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 2 | u16 | game_index (local to this chunk) |
| 2 | 2 | u16 | T = moves played |
| 4 | 1 | i8 | result: +1 black wins, −1 white wins, 0 draw (only possible with integer komi) |
| 5 | 1 | u8 | termination: 0 two passes, 1 resign, 2 move cap |
| 6 | 1 | u8 | no_resign_game |
| 7 | 1 | u8 | reserved (zero) |
| 8 | 4 | f32 | score = black − white − komi (area) |
| 12 | 8 | u64 | game_seed |
| 20 | (T+1)·P | u8 | snapshots, P = ceil(N²/4) bytes each, 2 bits/point (0 empty, 1 black, 2 white), point index = row·N + col, little-endian within a byte (point 4k in bits 0–1). Snapshot t is the configuration **before** move t; snapshot T is final |
| — | T·2 | u16 | moves: point index 0..N²−1, N² = pass |
| — | T·1 | u8 | **only if record_extras bit 1**: `search_kind` per move: 1 = full search (policy-target position), 0 = reduced search (value/aux targets only) (§5.4.9) |
| — | T·8 | f32, f32 | per move: `root_value` v(s₀), `root_max_q` (both in the mover's perspective; §5.4.7) |
| — | variable | | per move: `u32 root_total_visits; u16 nnz; nnz × (u16 action, u32 count)` — root visit counts including inherited visits; Σ count == root_total_visits, except with record_extras bit 2 where the counts are pruned and Σ count ≤ root_total_visits (nnz ≥ 1: the played move is always present) |
| — | N²·1 | i8 | **only if record_extras bit 0**: `final_ownership` of the final snapshot per point, +1 black area, −1 white area, 0 neutral — the engine's `areaOwnership` (§5.1), stored so that Python never needs the rules engine |

Derived by the loader, never stored: `ownership_t = final_ownership · (+1 if t even else −1)` and `score_t = score · (+1 if t even else −1)` (mover's perspective, like `z_t`), with an **auxiliary mask `aux_t = 0` for resigned games** (`termination = 1`: the final snapshot is unfinished and its area score need not favour the winner; two-pass and move-cap games are complete positions and keep `aux_t = 1`); `final_ownership` is still written for resigned games so the file stays uniform and the resignation diagnostics can use it. Positions with `search_kind = 0` are not training samples unless `reduced_positions = "value_only"` (§5.4.9). Planes for position `t` come from snapshots `t, t−1, …, t−7` (zeros before 0) split into mover/opponent by parity (**black moves at even t**: no handicap, and passes alternate turns); `z_t = result · (+1 if t even else −1)`; `π_t` from the visit counts with the chosen `store_pi` rule (§5.4.6).

**Cross-language fixture.** `tests/fixtures/mgo2_fixture.bin` is a hand-constructed chunk with two short 5×5 games and known byte content (record_extras = 0), and `mgo2_fixture_extras.bin` the same games with all three extras bits set; both the C++ reader and `chunk.py` must parse it to the same values, and both writers must reproduce it byte for byte from the same in-memory games.

**Size** per position: 9×9 ≈ 21 B snapshot + 2 B move + 8 B root values + 6 B + 6·nnz B (nnz typically 20–40) ≈ 200 B; 19×19 ≈ 91 B + 2 B + 8 B + 6 B + 6·nnz (nnz ≤ ~100) ≈ 700 B.

| | per iteration | window |
|---|---|---|
| 9×9: 2,000 games × ~70 positions | ≈ 28 MB | 20,000 games ≈ 280 MB |
| 19×19: 5,000 games × ~250 positions | ≈ 875 MB | 100,000 games ≈ 17.5 GB |

- **Atomic publish.** The writer writes `chunk_<id>.mgo.tmp` and renames it to `chunk_<id>.mgo` only after the last record and a flush. Readers ignore `.tmp` files. On a normal shutdown (SIGINT / `--games` reached) the partial chunk is completed with the games finished so far; games in progress are discarded, not written.
- **Provenance.** The header ties every game to the model, rules, komi, feature schema and config that produced it; the trainer refuses chunks whose feature schema, board size or rules differ from the run config.
- **Holdout tag.** Games with `game_seed % 20 == 0` (5%) are **holdout games**: never sampled for training, used for the held-out loss monitor (§6.6). The tag is a pure function of the stored seed, so it needs no field and survives re-reads.
- A reader in Python (`chunk.py`) and the C++ writer/reader share fixture tests.

### 5.7 GTP front-end (`cpp/gtp`)

Minimal GTP v2: `protocol_version, name, version, boardsize, clear_board, komi, play, genmove, undo, final_score, showboard, list_commands, known_command, quit`, plus `mango-analyze` (top moves with N/Q/P). `undo` is implemented by replaying the game history from the start (cheap) and **discards the search tree** (§5.4.8). Command-line precedence (`gtp_options.cpp`, unit-tested): `--size`/`--komi`/`--sims` > model metadata (size, komi) > config file > defaults; `--komi` may be negative; with `--allow-komi-mismatch` both the start-up check and the GTP `komi` command are unrestricted, otherwise `komi` must equal the model's komi (whatever its sign). Enough for GoGui, Sabaki and `gogui-twogtp` matches against GNU Go or KataGo.

### 5.8 Match driver (`cpp/match`) and evaluation validity

`mango_match --a <model_dir> --b <model_dir> --pairs 200 --sims 200 --seed <s>`:

- **Independence of players.** Each side has its **own evaluator (own model, own cache), own search parameters and own tree**. Trees are never shared or "switched"; after each move the side to move searches from its own tree (reused from its previous turn), the opponent's move is applied to both trees.
- **Diversity.** Two mechanisms, both on by default:
  1. Search-time random symmetry (§5.4.6, the paper's method) — already makes search stochastic.
  2. An **opening set**: `pairs` distinct openings, each played twice with colours swapped (a *pair*). Openings are the first `k` moves (9×9: k = 2–4) sampled from the incumbent's policy at temperature 1 with the match seed, deduplicated, and written to the match report so the match can be reproduced.
- **Reporting.** Number of **unique trajectories** (hash of the move list) out of games played, as a **diagnostic** printed in the report (identical models can legitimately repeat games; no threshold is enforced). The score is estimated **on pairs**: each pair's score ∈ {0, ½, 1} (candidate's mean over its two games); the point estimate is the mean pair score. The interval is a **95% bootstrap percentile interval over pairs** (10,000 resamples of the pair-score vector, seeded), not a Wilson interval — pair scores are three-valued, not Bernoulli, and pairs are the independent units.
- **Promotion rule (explicit).** The candidate is promoted iff the **point estimate** of its mean pair score is **> 0.55** (paper). The interval is reported and logged but is not part of the rule. The standard error depends on the score distribution; near the worst case (all pairs decisive, p ≈ 0.5) it is ≈ 3.5% with 200 pairs and ≈ 7% with 50 pairs, which is why 9×9 uses 200 pairs (§8) — games are cheap.
- No root noise; `τ → 0` from move 1 (paper); identical simulation budget for both sides.
- **Random anchor and fixed openings (M4, for the ladder §6.6).** `--a random` / `--b random` is the uniform-random legal-move player (no evaluator, no tree; uniform over the legal board moves, pass only when no board move is legal, seeded per game like a tree). `--openings-file F` plays a stored opening set (a JSON array of move arrays, validated by replay) instead of generating one from the seed; `--write-openings F` only generates and writes the set. JSON move lists (opening files, the report's `openings` and `games_detail[].moves`) encode pass as `N²`, like the chunk format, so a report's openings or games can be fed back as an opening file.

### 5.9 Threading model summary (first version)

| Component | Threads |
|---|---|
| Self-play | `selfplay.threads` = 1: 1 thread, 1 evaluator, G games, K = 1 (§5.5). T > 1 (M4a, §5.5.1): T search threads + 1 evaluation thread, one evaluator, G the global concurrency; 9×9 runs T = 8, G = 1024. `selfplay.processes` > 1: N such processes. |
| Match | `eval.threads` = 1: 1 thread, 2 evaluators, G game pairs. T > 1 (M4a, §5.5.1): T search threads + 1 evaluation thread owning both evaluators, one forward per side; 9×9 runs T = 4. |
| GTP | Single game, 1 evaluator, K = 1 (K > 1 once §5.4.5 is tested). |
| Training | Python; DataLoader workers for chunk reading and augmentation. |

Self-play, training and evaluation run **sequentially** in the first version (§6.5).

---

## 6. Python training (`python/mango`)

### 6.1 Model (`model.py`)

`AGZNet(board_size, res_blocks, filters, planes=17)` exactly as in §2: initial convolutional block **plus** `res_blocks` residual blocks (config counts residual blocks only), policy head, value head.

Optional parts, present only when the corresponding `accel` switch is on (§13); the exported `model.json` lists them under `heads` and `trunk` so the C++ evaluator can check what it loads:
- **Global pooling** (`accel.global_pooling`): in the residual blocks listed in `global_pooling_blocks`, half of the channels after the first conv go through a pooling branch — per channel the mean, the mean scaled by `(N − 14)/10` (KataGo's board-size feature; constant on one board size but kept so the module is size-agnostic) and the max — followed by a linear layer whose output is added as a per-channel bias to the other half before the second conv. The 1×1 policy and value convs are unchanged.
- **Ownership head** (`accel.aux.ownership`): 1×1 conv (1 filter) → tanh, one value per point, target `ownership_t` (§5.6).
- **Score head** (`accel.aux.score`): from the value head's 256-unit layer a second linear output, `ŝ`, predicting `score_t / N²` (a scalar; KataGo's score-distribution head is deliberately not reproduced).
**Output contract (training, export and C++ alike).** `forward` returns a **positional tuple**: `(logits [B,N²+1], value [B])`, followed — only when the corresponding head exists — by `ownership [B,N²]` and then `score [B]`, in that fixed order. There is no dict output and no lookup by name. `model.json` lists `outputs` (e.g. `["policy_logits", "value"]` or `["policy_logits", "value", "ownership", "score"]`) and `trunk.global_pooling_blocks`; `export.py` traces the full tuple; the C++ evaluator checks that the tuple length equals `len(outputs)` and that entries 0 and 1 are named `policy_logits` and `value`, uses those two, and ignores the rest. `train.py` unpacks the same positions. Softmax is the evaluator's job (§5.3).

### 6.2 Loss, optimiser and how much to train per iteration (`train.py`)

- `loss = mse(v, z) + CE_policy + w_own · L_own + w_score · L_score + c · Σ‖θ‖²`. Per batch: `CE_policy` is the cross-entropy **averaged over the positions with `search_kind = 1`** (under the AGZ profile and under PCR with `reduced_positions = "drop"` that is every position, so the term is the paper's; with `"value_only"` it is the mean over the policy-target positions present, or 0 if none — the policy gradient per policy-target position is then unchanged, only the value head sees more samples); `L_own = mean over positions with aux_t = 1 of mean_points(ô − ownership_t)²` and `L_score = mean over positions with aux_t = 1 of (ŝ − score_t/N²)²`, both 0 when no such position is in the batch (`aux_t` masks resigned games, §5.6). The auxiliary terms exist only with their heads (`w_own = 1.0`, `w_score = 0.5` initial values, swept in M4b); `c = 1e-4` as an explicit term over conv and linear **weights** (BN affine parameters and biases excluded — deviation D9). Implementation note: the gradient of `c‖θ‖²` is `2cθ`; if this is ever replaced by `SGD(weight_decay=…)` the equivalent value is `2e-4`, not `1e-4`.
- SGD, momentum 0.9, LR schedule by global step from the config (9×9 default: 0.01 for 0–30k, 0.001 for 30k–60k, 0.0001 after). Batch 256.
- **Steps per iteration are bounded by the data.** `steps = min(train_steps, ceil(samples_per_position × window_train_positions / batch))` with `samples_per_position = 0.25` per iteration by default, where **`window_train_positions` is the size of the dataset index actually built for this phase (§6.3)**: holdout games excluded and, under PCR with `reduced_positions = "drop"`, only `search_kind = 1` positions — not the manifest's `num_positions` total (with `full_prob = 0.25` the total would make the cap four times too loose and change the per-position sampling rate). The count is computed by the dataset at the start of the train phase, written into the phase plan next to `target_global_step` (§6.5), and logged. **If it is 0** (possible only in smoke configs), the train phase is skipped: `target_global_step = start_global_step`, the candidate is the incumbent's weights re-exported with the new iteration number, and the skip is recorded in `state.json`. On iteration 1 (≈ 2,000 games ≈ 140k positions) that is ≈ 140 steps rather than 1,000, so early networks never see the tiny first window for multiple epochs; at steady state (window 20k games ≈ 1.4M positions) the cap is inactive and each position is sampled ≈ 0.25 × 10 iterations ≈ 2.5 times over its lifetime. The paper's value head over-fitting on consecutive positions of the same game (noted in AlphaGo 2016) is the reason for this cap and for the held-out monitor (§6.6).
- Mixed precision (`torch.autocast` + `GradScaler`) on CUDA; fp32 on MPS.
- **Learner checkpoint** (`learner/ckpt_<step>.pt`): model weights, optimizer state, GradScaler state, global step, RNG states (torch, CUDA, NumPy, Python), the config hash. Distinct from the best self-play model: a failed gate never touches the learner state, and training always resumes from the latest learner checkpoint.

### 6.3 Data pipeline (`data.py`, `chunk.py`)

- The **replay manifest** (`replay/manifest.json`) lists chunk files in creation order with `chunk_id`, `model_id`, game count, position count and policy-target position count (Σ `search_kind`; equal to the position count without PCR). These are bookkeeping numbers; the step cap uses the built index (§6.2). The window is the most recent `window_games` games by this manifest (9×9 default 20,000).
- `ChunkDataset` **memory-maps** each chunk in the window (`np.memmap`, read-only) and builds a small index of (game offset, T, holdout flag) once; the OS page cache is then shared by all readers, so a 19×19 window of ~17 GB is never duplicated per process. Sampling is uniform over **training positions** (holdout games excluded): pick a game with probability ∝ T, then t uniform in [0, T). For the sampled (game, t) it unpacks snapshots t…t−7, builds the 17 planes and `z_t`, builds `π_t` from the visit counts, and — when the heads exist — `ownership_t`, `score_t` and `aux_t`; then applies **one and the same** of the 8 symmetries to the planes, to the board part of π (pass untouched) **and to the ownership map** (the score, `z` and the masks are invariant), and yields `(planes float32, π, z[, ownership, score, aux_mask])`. Under PCR the index excludes positions with `search_kind = 0` (`reduced_positions = "drop"`, the paper), or includes them with a `policy_mask` (`"value_only"`); a game's sampling weight is its number of **indexed** positions, so sampling stays uniform over training positions either way. The first version runs with `num_workers = 0` (assembly is cheap NumPy indexing); workers are added only if the GPU is measurably starved, and then they share the maps rather than copy them.
- `HoldoutDataset` iterates every position of the holdout games once (no symmetry), for the monitor in §6.6.
- **Windows file locking.** Chunks are only ever deleted by the pipeline process, between phases, when no trainer process is alive (§6.5). The trainer never deletes; DataLoader workers are torn down at the end of the training phase. Eviction of chunks that fall out of the window happens at the start of the next self-play phase.
- Symmetry index tables are precomputed per board size and shared with the C++ `applySymmetry` via a fixture test.

### 6.4 Export and model versions (`export.py`, `docs/MODEL_FORMAT.md`)

- `model.eval()` is required before tracing; export runs `torch.jit.trace` on CPU, fp32, input `[B,17,N,N]` float32, **traced at batch 8** and checked at batch 1 and 64 (tracing at batch 1 risks specialising the batch dimension into view/reshape ops). The exported graph returns the positional tuple of §6.1 (`(logits, value)` under the AGZ profile).
- **Metadata carries the game contract.** `model.json` contains, besides the fields in §5.3: `komi`, `rules_id`, `move_cap`, `feature_schema`, `board_size`. The model has no komi input, so its value head is only meaningful for the komi and rules it was trained under. The engine refuses to run a model whose `board_size`, `feature_schema` or `rules_id` differ from the run config, and refuses (or warns with `--allow-komi-mismatch`) when `komi` differs; the trainer refuses chunks whose komi/rules differ from the learner's. GTP `komi` accepts only the model's komi and answers `? unsupported komi <x>; this model was trained with <k>` otherwise. If a run is configured with an integer komi, draws are possible: `result = 0` and `z = 0` for every position of a drawn game.
- A **model version** is an immutable directory `models/<model_id>/` with `model.pt`, `model.json` and `weights.pt` (eager state dict for the ladder / re-export). `model_id = <iteration:04d>-<8 hex of sha256 over the parameter bytes>` (the digest is over the state dict, not over `model.pt`, whose zip serialisation is not guaranteed byte-stable; ≤ 31 chars, fits the zero-padded `char[32]` header field). Directories are written to `models/.tmp-<id>-<pid>` and renamed into place; re-exporting identical weights is a no-op.
- `best.json` holds the current best `model_id`; it is updated last, after the version directory exists.
- Round-trip tests: Python `torch.jit.load` vs eager, and C++ `test_torch_eval` on the same fixture (CUDA fp16/fp32, MPS fp32, CPU fp32) with stored reference numbers (§9 thresholds).

### 6.5 Orchestration (`pipeline.py`, sequential)

```
runs/<name>/
  config.json  state.json  models/<model_id>/  best.json  learner/ckpt_*.pt
  replay/manifest.json  replay/chunk_*.mgo  sgf/<iteration>/  matches/<iteration>.json
  strength/{openings_v1.json, ladder.json, matches/, ratings.json, ratings.csv}  monitor/holdout.csv  logs/
```

**Restart protocol.** `state.json` is the single authority and is rewritten atomically **before a phase starts** (with the phase's *plan*) and again when it ends. Other files (checkpoints, model directories, match reports, `best.json`) are outputs that the restart logic **reconciles against the plan**, never trusts on their own — atomic writes of individual files do not give cross-file transactions.

```
state.json = {
  run_seed, iteration, best_model_id, v_resign,
  phase: "selfplay" | "settle" | "train" | "monitor" | "export" | "gate" | "promote" | "strength" | "launch_eval",   # settle, launch_eval: §6.5.1
  promotions: [ { gate_iteration, candidate, effective_selfplay_iteration } ],   # §6.5.1
  background: { gate: null | {...}, ladder: [ {...} ] },                        # §6.5.1
  plan: {                                   # written before the phase starts; fixed until it ends
    selfplay: { task_id: "<iteration>", model_id, target_games, chunk_prefix: "chunk_<iteration>_",
                v_resign, resign_selection },   # v_resign chosen from the window (§5.4.7) or the config constant
    train:    { input_ckpt, start_global_step, target_global_step },
    export:   { input_ckpt, candidate_model_id },
    gate:     { candidate_model_id, incumbent_model_id, report: "matches/<iteration>.json", match_seed },
    promote:  { candidate_model_id, decision: null | true | false },
    strength: { add: bool, model_id, promoted }  # add = promoted or iteration % ladder_every == 0
  }
}
```

Resumption rules per phase:

- **selfplay**: before the plan is written, chunks that fell out of the window are evicted (files deleted, manifest entries kept and marked `evicted`; `selfplay.evict_old_chunks`) and `v_resign` is selected (§5.4.7; with `search.resign_auto = false` the config constant is used) and stored in the plan, so a restart reuses the same threshold. Then: count games in *published* chunks whose name starts with `chunk_prefix`; run self-play for `target_games − existing` more games (new chunk ids continue the sequence) with `--resign-threshold v_resign`. `.tmp` chunks are deleted. After the phase the measured false-positive rate of `v_resign` on the iteration's no-resign games is logged (diagnostic).
- **train**: load the newest learner checkpoint whose `global_step ≤ target_global_step` (this may be newer than `input_ckpt` if a checkpoint was saved before the crash); train **until `global_step == target_global_step`**, never "for N more steps". If a checkpoint already reaches the target, the phase is complete.
- **export**: if `models/<candidate_model_id>/` exists and validates, done; otherwise re-export from `input_ckpt` (deterministic, same id).
- **gate**: if the report exists and names the same candidate/incumbent/seed, done; otherwise re-run the match (seeded → same openings).
- **promote**: `decision` is computed from the report and written to the plan first; then `best.json` is written; then the phase is marked done. On restart: if `best.json` already equals the candidate and `decision == true`, done; if `decision` is recorded but `best.json` differs, redo the pointer write; if `decision` is null, recompute it from the report.
- **strength**: if `add`, the candidate is added to the ladder (the initial model is added first, at iteration 0, when the ladder has no model entry) and plays its missing matches (§6.6); `ladder.json` is rewritten after every match and a match whose report already names the same two players is not replayed, so a restart finishes the remaining matches only. The fit is recomputed at the end of the phase. The iteration counter advances after this phase.

Training steps per iteration are therefore expressed as a **fixed target step** (`target_global_step = start + bounded_steps`, §6.2, with `window_train_positions` stored in the plan so the bound is reproducible), so a crash after a checkpoint but before the state update can neither repeat nor skip work.

```
iteration i:
  1. selfplay:  mango_selfplay --model models/<best> --seed <run_seed, i>  until games_per_iteration new games are published
  2. train:     bounded steps (§6.2) from the window, resuming from learner/ckpt_*.pt; save new learner ckpt
  3. monitor:   held-out value MSE and policy CE (§6.6) → monitor/holdout.csv
  4. export:    models/<candidate_id>/ (immutable)
  5. gate:      mango_match candidate vs best (§5.8) → matches/<i>.json
  6. promote:   if pair win-rate > 0.55 → best.json = candidate  (logged as a gating event, not as Elo)
  7. strength:  every promoted model and every `ladder_every`-th candidate joins the frozen ladder (§6.6)
```

`accel.gating = false` / `eval.gating = false` (AlphaZero variant) skips the match and always promotes.

### 6.5.1 Overlapped evaluation: `pipeline.async_ladder`, `pipeline.async_gate` (design 2026-09-28, approved after two review rounds; implemented 2026-09-28)

**Why.** A 9×9 iteration without the ladder takes ≈ 123 s sequentially — self-play 51 s, train 9 s, monitor 9 s, export 1 s, gate 44 s, bookkeeping ≈ 9 s — and ≈ 230 s when the ladder (≈ 105 s) runs, which is every promotion or `ladder_every`-th iteration (about every other iteration in R0). Evaluation is 35–60 % of the wall time, and nothing in the training loop reads the ladder. During self-play the GPU runs kernels only ≈ 40–50 % of a forward's span (the driver's loop latency and the tail of the phase, §5.5.1), so a second process has room. **Measured 2026-09-28** (model `0061`, GPU otherwise idle; `scratchpad/concurrency.py`): self-play alone 51.1 s, the 200-pair gate alone 43.9 s, **both started together 75.5 s** (self-play 69.5 s, gate 75.3 s) against 95.0 s in sequence (−21 %); two self-play processes of 1,000 games each: +5 % over one — overlapping *different* work is the lever. A fully asynchronous pipeline (the paper's long-running workers) is not proposed: it needs model hot reload in the C++ drivers, replaces the plan-based restart protocol, makes the data each training step sees depend on timing, and its extra gain over this design is small while the GPU is the shared bottleneck.

**Two switches, pipeline-only** (a new config section `pipeline`; the C++ side ignores it), independent of each other; **defaults since 2026-09-29: `async_ladder` true, `async_gate` false** (decision below; both were false until then):
- **`pipeline.async_ladder`** — the strength phase (§6.6) runs in a background *ladder lane*. No semantic change: the ladder is measurement only.
- **`pipeline.async_gate`** — the gate of iteration i runs in the background while iteration i+1 plays its games. **Semantic change:** a promoted candidate starts to play self-play one iteration later than in the sequential pipeline. Recorded as deviation D19 (§12).

A switch is read at an iteration boundary; background work pending when a switch changes is completed under the rules it was started with (its delay is recorded with it, below), so a run can be switched between iterations.

**Which model plays self-play: decision and effect are separate.** A gate decision says *whether* a candidate becomes the incumbent; *when* it starts to produce self-play data is recorded with it and never derived from wall-clock order. `state.json` keeps
```
promotions: [ { gate_iteration: i, candidate, effective_selfplay_iteration: e } ]
```
with **e = i + 1 for a candidate of iteration i gated synchronously and e = i + 2 when its gate ran with `async_gate`** (the value is fixed when the gate is launched and stored in its record). An entry is keyed by `(gate_iteration, candidate)` and added only if absent. The self-play plan of iteration j takes, among the entries with `effective_selfplay_iteration ≤ j`, the one with the largest `(effective_selfplay_iteration, gate_iteration)` — never the list order; with no such entry, the run's initial model. Two entries can share an effective iteration when the switch changes (candidate i gated asynchronously, e = i + 2; candidate i+1 gated synchronously after the switch, e = i + 2): the later gate wins, which is right, since candidate i+1 was gated against the incumbent that already included the decision on candidate i. `best_model_id` / `best.json` keep their meaning — the incumbent after the latest decision: the gate's opponent and what external tools (`mango_gtp`, `vs_gnugo.py`) load. So a drain may complete a decision early (it must, so that a finished run leaves no pending gate) without changing any iteration's data producer: running 10 iterations in one go and running them as ten restarts give the same self-play model sequence. In the sequential pipeline e = i + 1 is exactly today's behaviour; the self-play plan reads the promotion list instead of `best_model_id`, and existing runs are migrated once (the list is rebuilt from the recorded gate events, each with `gate_iteration` = its iteration and e = its iteration + 1; the migration is keyed, so repeating it after a crash adds nothing).

**Background jobs.** A background job is an OS process launched by a small supervisor, `python -m mango.bgjob` (below), and polled by the pipeline's main thread. Only the main thread writes `state.json` and `best.json`; a job writes only its own outputs.
- **Gate job**: the supervisor runs today's gate `mango_match` invocation, unchanged (arguments, report `matches/<iteration>.json`, seed).
- **Ladder job**: the supervisor runs the ladder step in-process — construct `Ladder`, add the initial model if there is no model entry, `add_model(M, i, promoted)`, fit — with a runner that launches each `mango_match` / GTP engine as a polled child (never a blocking `subprocess.run`). The **lane runs at most one ladder job at a time**, in iteration order, so the ladder files keep a single writer and every entry's opponents are chosen from the ladder as it stood after the previous entry, as in the sequential pipeline. While ladder jobs are enabled the main process never writes ladder files.
- **Polling**: at every phase boundary and while the main thread waits for a subprocess of its own (self-play, a synchronous gate or ladder match) — `_run_subprocess` / `_run_workers` gain a poll hook — not during in-process training or monitoring (≈ 20 s). A job that exited 0 is collected (result applied, state saved, the next queued ladder job launched); a job that exited non-zero stops the run like a failing phase (every process of the run terminated as below, error logged and raised); the recorded state lets a restart resume.
- **Logs**: `logs/match_bg.log` (gate job), `logs/ladder_job.log` (ladder job and its matches); `pipeline.log` gets `background: launched / finished (<s> s) / failed` lines, and every phase that overlapped background work says so in its summary line (`overlapped: gate 0071, ladder 0070`).

**Process lifecycle: no process outlives its pipeline, and no job runs twice.** An exception handler does not run when a process is killed, so both rules rest on the operating system, and on the processes that actually write (a supervisor's `mango_match` writes the report, not the supervisor).
1. **Containers that kill whole trees.** *Windows:* the pipeline creates a run-level Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and, per background job, a nested Job Object inside it; every process it launches (self-play, match, each supervisor) is assigned to one of them, and the children a process spawns inherit the membership. If the pipeline dies in any way the OS closes the run-level handle and terminates everything; to stop one job (it failed, its supervisor died, Ctrl-C) the pipeline calls `TerminateJobObject` on that job's object and waits until its active-process count is 0. **`go` handshake:** a supervisor starts no work until it reads the line `go` on stdin, which the pipeline writes only after the assignment succeeded; on EOF or any other input the supervisor exits at once, and if the assignment fails the pipeline kills the (idle) supervisor and raises. *macOS/Linux:* each supervisor is the leader of a new process group containing its children; it checks once a second that its parent pipeline is alive (`os.getppid()` against the pid recorded at launch) and on the parent's death or SIGTERM kills its group, **waits for every child to exit**, and exits last. The pipeline records every job's process-group id in the job's record; to stop a job it signals the group and waits for it.
2. **Locks held by every writer.** The pipeline holds an OS file lock on `runs/<name>/.pipeline.lock` for its life (a second pipeline on the same run refuses to start). Each job has `runs/<name>/jobs/<job id>.lock` (`gate_<iteration>`, `ladder_<iteration>`), taken by the pipeline before launch and **inherited by the supervisor and by every child it starts**: on POSIX an `flock` lock on a descriptor passed down with `pass_fds` is held until the last process holding that open file description has exited — so the lock stays held while a `mango_match` of the job is alive, even after its supervisor died; on Windows the job's nested Job Object plays that role (a job counts as running while its active-process count is non-zero), and the lock file serialises pipelines. **Relaunch rule:** a job is (re)launched only after its lock could be acquired (POSIX: all holders gone) or its Job Object is empty (Windows). If it is still held at a restart — the previous pipeline and supervisor both died by force and a child survived on POSIX — the pipeline signals the recorded process group and waits up to 60 s for the lock, then fails with a message naming the pids; it never starts a second writer.
3. **The pipeline's own subprocesses (review 2026-09-30).** Self-play, a synchronous gate and the matches of an inline strength step are written by the pipeline's foreground subprocesses, which need the same guarantees. *Windows:* they are in the run-level Job Object (item 1). *POSIX:* each runs under `python -m mango.bgjob --exec -- <command>` — its own process group, the parent watch of item 1, the command's stdout and stderr passed through, its exit code returned — and inherits the descriptor of `runs/<name>/jobs/foreground.lock`, which the pipeline holds for its life; the supervisors' process groups are appended to `jobs/foreground.pgids`. A restarted pipeline acquires that lock before launching anything; if an orphan still holds it, the recorded groups are signalled and the lock awaited for up to 60 s, then the pipeline fails with the group ids — so a phase is never dispatched while an earlier dispatch of it is alive.

**Settle: an idempotent commit with stable keys.** With `async_gate`, iteration i's phases are
```
selfplay(i) -> settle(i) -> train(i) -> monitor(i) -> export(i) -> launch_eval(i)       (iteration i+1 follows)
```
`launch_eval(i)` records `background.gate = {iteration: i, plan: {candidate i, incumbent: best_model_id now, report: matches/<i>.json, match_seed: derive_seed(run_seed, 1000 + i)}, effective_selfplay_iteration: i + 2, decision: null}`, saves, launches the gate job and advances the iteration counter (with `eval.gating = false`: decision `true`, nothing launched). `settle(i)` handles the gate recorded for iteration i−1 (none in iteration 1) in these steps, each idempotent by a stable key, and is **re-run from its first step after any crash**:
1. wait for the job (polling), unless a valid report exists (today's validity check); compute the decision from the report and write it into `background.gate.decision` if still null; save.
2. if promoted: add `{gate_iteration: i−1, candidate, effective_selfplay_iteration}` to `promotions` unless the key `(i−1, candidate)` exists; save; write `best.json` (idempotent).
3. add the gating event unless `events` already holds `{iteration: i−1, event: "gate", candidate}` — events are keyed by `(iteration, event, candidate)` everywhere (today's promote phase and `train_skipped` included, which also fixes their duplication on restart).
4. the strength step for candidate i−1: with `async_ladder`, enqueue `{iteration: i−1, model_id, promoted}` unless the queue or `state.strength` already has iteration i−1; without it, run the ladder inline (idempotent as today, ratings as below).
5. clear `background.gate`; save.
A crash between any two steps — for example after the ladder job was enqueued and saved but before the gate was cleared — re-runs settle, which finds each effect already present and adds nothing twice.

**Ratings history keyed by iteration.** Today `Ladder.fit` appends to `ratings.csv`, so a fit repeated after a crash duplicates rows (or leaves a partial one). Instead every fit is published as **`strength/fits/<fit_iteration>.json`** (atomic write; a repeated fit of the same iteration replaces it), `ratings.json` is the latest one, and **`ratings.csv` is regenerated** (written to a temporary file, then atomically replaced) from two sources with a fixed precedence: **for every fit iteration that has a fit file, only the fit file's rows; for every other fit iteration, the rows of `strength/fits_legacy.csv`**; rows sorted by fit iteration, then entry order. So a legacy iteration that is fitted again is replaced as a whole, and one that is not keeps its old rows exactly once. **Migration, repeatable after an interruption:** before the first fit file of a run is written, if `fits_legacy.csv` does not exist and `ratings.csv` does, `ratings.csv` is copied to `fits_legacy.csv.tmp` and renamed to `fits_legacy.csv`; a leftover `.tmp` is discarded and the copy redone; `ratings.csv` itself is only ever replaced by a regeneration. A run that never had a CSV has no legacy file. The ladder job's result is its fit file; collecting the job copies the candidate's rating into `state.strength[<iteration>]`, keyed by iteration. Because the fit file is written before `ratings.json` and `ratings.csv`, it marks completion only together with them: **collecting a job always completes the publication first** (`finish_fit_publication`: `ratings.json` rewritten from the fit file, `ratings.csv` regenerated; idempotent), so a job interrupted between the fit file and the derived files leaves nothing unpublished (review 2026-09-30). The sequential strength phase uses the same publication, so both paths produce the same files.

**Drain** at the end of `--iterations` / `--hours`: the loop stops starting iterations, settles a pending gate (the decision and `best.json` are completed; the promotion keeps its recorded `effective_selfplay_iteration`, so the next iteration's data producer is unchanged), then waits for the ladder lane to empty.

**Restart.** At start-up: acquire the run lock; then, for every recorded job — `background.gate` with `decision == null` and no valid report, the head of `background.ladder` without a fit file for its iteration — acquire the job lock and relaunch it; a gate with a recorded decision is completed by `settle`'s idempotent steps. Then the main phase resumes by the per-phase rules of §6.5. The ladder job is idempotent end to end: `Ladder` keeps the existing entry and does not replay reported matches, and the fit is published by iteration. A crash can happen anywhere — during self-play with the gate running, between the report and the recorded decision, between the decision and `best.json`, after a ladder job was enqueued but before the gate was cleared, inside a ladder job with a `mango_match` running, after a fit file was written but before the job was collected — and the resumed run reaches the same state, reports, ratings and model sequence as an uninterrupted one.

**What changes, and what does not.** Every candidate is gated against the incumbent at its launch — after the decision on the previous candidate — so gate pairings, seeds, the 55 % rule and the report format are the sequential ones. The self-play model of every iteration is a function of the gate reports and the recorded effective iterations only, so the sequence of self-play models, seeds and gate pairings is reproducible from the plans; only wall-clock timings and the fp16 batch-shape noise already accepted in §5.5.1 differ. AlphaGo Zero's own pipeline is asynchronous in this sense.

**Resources.** Self-play (T = 8 + 1), a gate (T = 4 + 1) and ladder matches share the GPU and 8 cores; training may overlap a ladder job. No pinning or priorities; the contention is part of what the acceptance measures.

**Config** (§8, pipeline-only keys): `pipeline.async_ladder` (true since 2026-09-29), `pipeline.async_gate` (false).

**Tests (§9 row "Pipeline, overlapped evaluation").** Integration tests with the real executables on the 5×5 test config (as `test_pipeline.py`), the fake self-play driver where timing matters, and a scripted gate driver through a test seam `_match_command()` (writes a valid report with a given mean pair score, can sleep, fail, or record its pid, start and end):
1. `async_ladder` only, 3 iterations, `ladder_every = 1`: `ladder.json`, `strength/fits/*`, `ratings.json`, `ratings.csv` and `state.strength` equal a sequential run's; a sleeping ladder job delays nothing; the drain empties the lane.
2. `async_gate`, 4 iterations, outcomes reject, promote, reject, promote: every iteration's self-play model is the one the recorded effective iterations predict (a promotion of iteration i plays from i+2); every gate's incumbent is the best after the previous decision; **segmented runs** — the same 4 iterations as one run, as 4 one-iteration runs with drains in between, and as 2 × 2 — give the same self-play model sequence, promotions and `best.json`.
3. Restart after a crash at each point listed under *Restart*, including a crash after the ladder job was enqueued and saved but before `background.gate` was cleared: the resumed run ends in the same `state.json` (timings excluded), `best.json`, ladder, fit files, `ratings.csv` and reports as an uninterrupted run; `events` and the ladder queue have no duplicates.
4. **Forced termination** while a ladder job runs a slow scripted `mango_match`: (a) of the pipeline — within 10 s no process of the run is alive (checked by the recorded pids); (b) of the supervisor only, the pipeline alive — the pipeline stops the job's tree, waits until the lock / Job Object is free and then stops the run (a job that exited non-zero, test 5). In both cases the restart runs the job again and the recorded start/end times of every `mango_match` writing one report never overlap. A supervisor that gets EOF instead of `go` exits without starting a child. A second pipeline started on the same run while one is alive refuses to start. (c) Of the pipeline while its self-play is running (a driver that hangs after its first chunk): within 10 s the self-play process is gone, and the restart dispatches the remaining games after that moment only (item 3; review 2026-09-30). The `--exec` wrapper passes output and exit code through.
5. A background job exiting non-zero stops the run, every other process of the run is terminated, the state keeps the job, the restart resumes it.
6. Switching `async_gate` off with a gate pending settles it first (its recorded e = i + 2 stays); with both candidate i (asynchronous, e = i + 2) and candidate i+1 (synchronous after the switch, e = i + 2) promoted, iteration i+2 plays candidate i+1, whatever the order of the list; with no promotion recorded, the initial model; both switches false: the existing pipeline tests pass, and the self-play models, promotions and state transitions are today's (the promotion list migration included).
7. `ratings.csv` regeneration: a fit repeated for the same iteration leaves one set of rows; a legacy CSV is preserved as `fits_legacy.csv` and its rows appear once; an iteration present in the legacy file and then fitted again appears only with the new fit's rows; a migration interrupted after the `.tmp` copy is redone and yields the same files; a ladder job interrupted after its fit file but before `ratings.json` / `ratings.csv` (the run's last job, so no later fit repairs it) is collected with the publication completed — the run ends with the files of an uninterrupted one.

**Implementation (2026-09-28).** `mango/proc.py`: `JobObject` (ctypes; `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, `TerminateJobObject`, active-process count), `RunContainer` (the run-level object; `adopt(proc, job)` assigns to it and then to the job's object, which thereby nests), `FileLock` (`msvcrt.locking` / `flock`; on POSIX `release` only closes the descriptor, never `LOCK_UN`, which would release the lock for every inheriting process), the run lock (shared by `Pipeline` objects of one process, refused to another process), `run_polled` (a subprocess with a stdout reader thread and a poll hook; any exception terminates and waits for it), `pid_alive`. `mango/bgjob.py`: the supervisor (exit codes 0 done, 1 failed — a failing `mango_match` included, 2 no `go`, 3 aborted). `mango/strength.py`: `ladder_step` (the strength step shared by the inline phase and the ladder job), `publish_fit` / `regenerate_ratings_csv` / `migrate_legacy_ratings_csv`; `Ladder(match_command=…)` is the test seam; `remeasure_ladder` archives `fits/` and `fits_legacy.csv` too and publishes its fit as that of the last entry's iteration. `mango/pipeline.py`: phases `settle` and `launch_eval`; `state.plan.mode` holds the switches of the iteration (written with the self-play plan); `state.promotions`, `state.background = {gate, ladder: [queue]}` (each record gets the supervisor's `pid` / `pgid` when launched); an inline strength step (either path, `async_ladder` off) first waits for the ladder lane to empty, so the ladder keeps one writer and iteration order when the switch is turned off; the drain runs at the end of `run_iterations` / `run_for_seconds`, and any exception stops every background tree before it propagates. `--async-ladder on|off`, `--async-gate on|off` set the switches in the run's `config.json` (effective at the next iteration boundary). Every phase appends `{iteration, phase, seconds, overlapped}` to `logs/phase_times.jsonl` (the acceptance data). Migration of an existing run: the promotion list rebuilt from R0's 71 gate events reproduces the recorded self-play model of every one of its 72 iterations; if the events do not end at `best_model_id` the pipeline refuses to start instead of changing the model. **Verified on Windows only**: the POSIX branch (process groups, parent watch, inherited `flock`, the `--exec` wrapper and `jobs/foreground.lock` of item 3) is written but has not run; the forced-kill tests exercise it only on a POSIX machine.

**Acceptance (a copy of `runs/9x9-r0` at iteration 72, 10 iterations per configuration, the same day, otherwise idle machine):** mean wall time per iteration, drain included — `async_ladder`: ≤ 0.85× the sequential run; `async_ladder` + `async_gate`: ≤ 0.75× (the measured overlap predicts ≈ 0.70–0.75×). Per-phase times of overlapped and non-overlapped phases reported separately. The tests above pass. A configuration that misses its bound is reported as not met, with the per-phase breakdown.

**Acceptance result (2026-09-29): not met for either configuration.** Copies of `runs/9x9-r0`, iteration 72 finished sequentially (untimed), then iterations 73–82 timed with the drain included, one configuration after the other on an otherwise idle machine (`scratchpad/accept_async.py`; the per-phase sums below; the measurement runs were deleted afterwards). Mean wall time per iteration: **sequential 137.1 s; `async_ladder` 125.1 s = 0.91× (bound ≤ 0.85×: not met); `async_ladder` + `async_gate` 115.5 s = 0.84× (bound ≤ 0.75×: not met).** Sums over the 10 iterations (s):

| | self-play | train | monitor | gate | strength (inline) | settle | drain | total |
|---|---|---|---|---|---|---|---|---|
| sequential | 549 | 126 | 86 | 405 | 200 | — | 2 | 1,371 |
| `async_ladder` | 578 | 130 | 94 | 444 | 0 | — | 2 | 1,251 |
| both | 722 | 140 | 102 | — | 0 | 29 | 159 | 1,155 |

Reading: (1) the ladder is a smaller share than the design assumed — the runs' gates promoted rarely (sequential: no promotion in 73–82, ladder entries only at 75 and 80, 200 s in all), so removing it can save at most ≈ 15 %, and the overlap costs back part of it (gate +39 s, self-play +29 s under contention). (2) With `async_gate`, self-play grew by 173 s (+32 %) while the 405 s of gates left the critical path, but the last gate and ladder job are paid in the drain (159 s), which is ≈ 1/10 of a gate's and ladder's cost that a 10-iteration window cannot amortise. (3) The runs are not the same workload: the three copies diverged (fp16 batch-shape noise, §5.5.1) — promotions: sequential none, `async_ladder` candidate 77, both candidate 82 — so the ladder did 2, 3 and 3 entries respectively; the `async_ladder` run did more ladder work than the sequential one. The bounds are not re-stated; what would change the verdict is a longer window (drain amortised) or a measurement with equal ladder workloads, to be decided in review.

**Second measurement with a 10×128 network (2026-09-29).** `runs/big-base`: R0's config with 10 blocks × 128 filters, R0's replay window at iteration 72 (22,000 games, 879k positions) copied in, the network pretrained on it for 8,000 steps at lr 0.01 (146 s; policy CE ≈ 2.1) and exported as the initial model at iteration 73 (`scratchpad/big_base.py`); three copies timed over iterations 73–82, drain included (`scratchpad/accept_big.py`). Mean wall time per iteration: **sequential 396.0 s; `async_ladder` 314.2 s = 0.79× (bound ≤ 0.85×: met as measured); both switches 310.1 s = 0.78× (bound ≤ 0.75×: not met).** Sums over the 10 iterations (s):

| | self-play | train | monitor | gate | strength (inline) | settle | drain | total | ladder entries |
|---|---|---|---|---|---|---|---|---|---|
| sequential | 1,697 | 191 | 104 | 947 | 1,016 | — | 2 | 3,960 | 7 |
| `async_ladder` | 1,817 | 212 | 113 | 994 | 0 | — | 2 | 3,142 | 5 |
| both | 2,570 | 239 | 117 | — | 0 | 8 | 161 | 3,101 | 5 |

**The workloads again differ, and not in favour of the verdict:** the sequential run promoted 5 candidates and added 7 ladder entries (1,016 s inline), the asynchronous runs 3–4 promotions and 5 entries. Re-costing the sequential run with 5 entries — its own inline times of the first five entries, 76 + 102 + 136 + 173 + 181 s = 668 s instead of 1,016 s — gives ≈ 361 s per iteration, against which `async_ladder` is ≈ 0.87× and both ≈ 0.86×: **with comparable ladder work neither bound is met.** Reading: with the larger network the GPU is busier, so overlapping costs more — ladder jobs in the background took 198–512 s (inline 76–181 s), self-play grew 7 % under the ladder lane and 51 % (+873 s) when the gate also overlapped it, which ate nearly all of the 947 s of gates taken off the critical path; `async_gate` adds ≈ 1 % over `async_ladder` alone at this size (≈ 0.84× vs 0.91× with 6×64). The ladder lane is where the saving is; the gate overlap is not worth its semantic cost (D19) at either size on this machine.

**Decision (user, 2026-09-29): `async_ladder` on by default, `async_gate` off by default.** Taken with the acceptance results above as they stand (neither bound met with comparable ladder work; `async_ladder` ≈ 0.87–0.91× of sequential, `async_gate` adds ≈ 1–7 % at the cost of D19): the ladder lane has no semantic cost and is where the saving is; the gate overlap stays implemented and switchable but off. The acceptance bounds are not re-stated — the records above remain "not met". Consequence for existing runs: a run whose `config.json` has no `pipeline` section (R0 included) uses the ladder lane from its next iteration on; `--async-ladder off` restores the sequential strength phase. The sequential test (`test_pipeline.py`) sets `async_ladder` false explicitly.

**Open points for the review.** (a) The one-iteration delay of promotion is the price of `async_gate`; `async_ladder` alone has no semantic cost. (b) No polling during in-process training (≈ 20 s latency for a queued ladder job) — accepted for simplicity. (c) Contention between a ladder job and training is measured by the acceptance run. (d) The Job Object / process-group layer and the run lock also protect the sequential pipeline (self-play and match processes die with it); they are introduced with `async_ladder` but apply whatever the switches.

### 6.6 Monitoring and strength measurement (`strength.py`, `monitor/`)

**Held-out monitor (every iteration, cheap, earliest warning).** On the holdout games of the current window (5% of games, never trained on) the trainer reports value MSE and policy cross-entropy, next to the same quantities on a training sample. A held-out value MSE that rises while the training MSE keeps falling is an **alarm**, not a verdict: self-play data and targets drift from iteration to iteration, so a gap between the two curves can also come from distribution shift. The alarm triggers a look at the ladder and at the resign/termination statistics; it does not by itself decide anything. Both curves are recorded per iteration in `monitor/holdout.csv`. The training-sample curve is a seeded uniform random subset of the window's training positions (4,096), not the head of the index. Because the window and the self-play targets move every iteration, a falling held-out loss alone does not show generalisation on a fixed task; with `training.fixed_holdout_iteration = k` (9×9: 5) the holdout games of iteration `k` are frozen once into `monitor/fixed_holdout.mgo` (holdout games are never trained on, so the set stays clean for the whole run) and every later iteration also reports `fixed_value_mse` / `fixed_policy_ce` on that fixed set.

**Strength (frozen ladder).** The promotion log is **not** a strength curve. Strength is measured separately:

- A **frozen ladder**: every promoted model plus every `ladder_every`-th candidate regardless of gating.
- **Anchors** with fixed identity: (a) a uniform-random legal-move player, (b) the first promoted model, (c) an external engine over GTP — GNU Go level 10 on 9×9 (later KataGo with a small net as a stronger anchor). **(c) is played on demand, not by the pipeline** (decision 2026-09-24: a GTP match is one game per process at batch-1 inference, 12 min per 100 games even with 4 parallel trios, and dominated the iteration): `scripts/vs_gnugo.py --run R` plays the best (or a given) model against GNU Go on the ladder's openings with the ladder's seed and budget, stores the report under `strength/gnugo/`, and refits the ladder with GNU Go as an extra player (`fit_with_external`; the ladder file is never modified). `eval.gtp_anchor` stays `null`.
- **Fixed opening set.** The ladder uses a versioned opening file `strength/openings_v1.json`, generated **once** per run (from uniformly random legal moves, k = 2–4, deduplicated, seeded) and never regenerated, so that strength measurements taken at different times see the same opening distribution. Gating matches (§5.8) may sample openings from the incumbent; the ladder must not.
- Each new ladder entry plays fixed-budget matches (same simulations, no noise, fixed opening set + search symmetry) against the anchors and against 2–3 nearest ladder neighbours. All results go into one table.
- **Rating fit.** Bradley–Terry ratings are fitted by **MAP with a Gaussian prior** on every rating (prior σ = 350 Elo, equivalently a ridge penalty in natural units), with anchor (a) pinned at 0. An unregularised MLE has no finite solution when a player wins or loses every game or when the comparison graph is disconnected, which is exactly the situation early ladders are in. Before fitting, `strength.py` checks that the comparison graph is **connected** (otherwise the disconnected component is reported unrated) and reports **complete separation** (any player with all wins or all losses) so those ratings are read as prior-dominated. Intervals come from the Laplace approximation at the MAP or from a bootstrap over games.
- This fitted curve is what is reported as "strength". "Beats random 100%" is a sanity check only. The M4 acceptance criterion is Elo growth on this ladder over ≥ 10 iterations with non-overlapping intervals between the first and last entries.
- **What the fit may be read as (M4 review).** The Laplace interval is conditional on the single-scale Bradley–Terry model and on the prior; neither is checked by the interval. `strength.py` therefore reports, and `scripts/run_report.py` prints, three diagnostics next to every fit: (1) **predicted vs observed** per match with the standardised residual `z = (obs − pred) / sqrt(pred (1 − pred) / n)` — a well-specified model has mean |z| ≈ 1; (2) **prior sensitivity** — the same matches under σ ∈ {200, 350, 700, 1400} Elo; only conclusions that hold for every σ (order, non-overlap of two entries) are prior-robust; an entry *all* of whose results are 100–0 is unidentified, while one that also has non-separated results is identified by the data but shrunk by the prior (R0 re-measured: iteration 15 = 1,925 at σ = 350, 2,413 unregularised), so a level is quoted with its σ; (3) an **opening-adjusted fit** (`fit_bradley_terry_openings`: logit P(black wins) = r_black − r_white + o_k with a per-opening colour term, Gaussian prior on the `o_k`) plus per-opening black win rates, to tell opening/colour effects from strength. On the first 9×9 run the plain fit had mean |z| = 3.5 (max 49: iteration 15 vs iteration 1 predicted 0.999, observed 0.88) and the opening terms were all < 130 Elo, so the misfit was not the opening set; it came from opponent-specific early two-pass endings (`docs/RUN_9x9_R0.md`). The rule for reports: the **order** and the **head-to-head results** are the findings; an Elo number is quoted with its σ and its residual summary, never alone.

**Implementation (M4, `strength.py`).** Files under `runs/<name>/strength/`: `openings_v1.json` (written once by `mango_match --write-openings` with `eval.ladder_pairs` openings of `eval.opening_moves` moves, seed derived from `run_seed`), `ladder.json` (entries and every match), `matches/<a>__<b>.json` (reports), `ratings.json` (latest fit) and `ratings.csv` (one row per player per fit, the history). Entries: the anchor `random` (§5.8 random player, pinned at 0), the optional GTP anchor, the initial model (iteration 0) and the models added by the pipeline; the **first promoted model** is the entry with the lowest iteration whose `promoted` flag is true. "Nearest neighbours" are the `eval.ladder_neighbours` most recently added model entries (entries arrive in iteration order and strength is expected to rise with it; a rating-based choice would need a rating the new entry does not have yet). Matches: `eval.ladder_pairs` colour-swapped pairs on the fixed opening set, the eval simulation budget, no noise, per-pairing seed derived from `run_seed` and the pair of names (so a replayed match is identical). **External GTP anchor** (`eval.gtp_anchor = {name, command}`): games are driven by `gtp.py` between the engine and `mango_gtp --model`, refereed by a third `mango_gtp` process (random player) that receives every move and scores the final position — so games against external engines are scored under Mango's rules (Tromp–Taylor area scoring, §5.1); an opponent move illegal under those rules (superko, suicide) loses the game; `resign` is honoured; the report has the shape of `mango_match`'s. The GTP anchor plays only model entries. Fit: `fit_bradley_terry` (MAP, Gaussian prior σ = 350 Elo on every non-anchor rating, damped Newton on the concave log-posterior, Laplace SE from the inverse Hessian, ±1.96 SE intervals; draws count ½; connectivity by BFS from the anchor; separation flag per player). Every fit re-positions every entry (new matches change old ratings), so the per-iteration log lines are not a curve; `ratings.csv` keeps each fit and `scripts/run_report.py` refits the whole ladder — read strength from the latest fit. `python -m mango.strength crossrun` plays the final models of several runs round-robin plus the random anchor on one opening set and fits them together — the tool for the §8.2 sweep.

### 6.7 Human interface (`gui.py`)

A tkinter window (standard library, Windows and macOS) for two uses: **(1) a human plays against a model**, **(2) a human plays both colours locally and asks the search where to play next** (analysis). It contains **no rules and no search**: it drives one `mango_gtp --model DIR --sims N` process over GTP (§5.7) and treats the engine as the single authority — `play` decides legality (occupied points, suicide, positional superko), `genmove` is the model's move, `showboard` is re-read after every change and is what gets drawn (the position is never simulated in Python), `final_score` is the Tromp–Taylor score with the model's komi, and `mango-analyze` runs one search budget from the current position and returns the top moves with N, Q (mean value for the side to move) and P; the GUI shows them on the board as win probabilities `(Q + 1) / 2` with visit counts, best move highlighted, and the root value as the side-to-move's win probability. Consequences of the search contract (§5.4.8): a move played after an analysis reuses the analysed subtree, so "analyse, then play the suggestion" costs one budget, not two; `undo`, `clear_board` and an engine restart discard the tree. The simulation budget is a launch option of `mango_gtp`, so changing it in the window restarts the engine at the next new game. Game end: two consecutive passes or the move cap (§5.1, both scored by the engine), or a resignation by the side to move (the human; `MctsPlayer` never resigns, §5.4.7 is self-play only — a `resign` reply is nevertheless handled). Undo retracts one move in analysis mode and the model's reply plus the human's move in play mode. Mode and colour apply to the game in progress (switching to analysis turns automatic analysis on and the human plays both colours from there; switching to play, or changing colour, hands the side to move to the model at once); "New game" only clears the board. Engine calls run on a worker thread so the window stays responsive during a search; the controls are disabled meanwhile. Launch: double-click `Mango.cmd` (Windows, `pythonw` from the venv) or `Mango.command` (macOS) in the repository root — without arguments the GUI opens a chooser listing every run under `runs/` with a best model (newest first, the run's `eval_simulations` preset) and a browse button for a model directory, and start-up errors are shown in a message box; or `python scripts/gui.py --run runs/9x9-r0` (the run's best model) / `--model DIR --sims N`; `--mode analysis`, `--colour W`, `--top-k`, `--device`, `--bin`.

---

## 7. Cross-platform strategy

| Concern | Decision |
|---|---|
| Build system | CMake ≥ 3.24 with `CMakePresets.json` (`windows-cuda`, `macos-mps`, `cpu`). Visual Studio 18 2026 generator on Windows (Ninja also works with `vcvars`); Ninja on macOS. |
| Language standard | **C++20** (LibTorch 2.14 headers require it; verified on MSVC 2026). AppleClang ≥ 15 on macOS. |
| LibTorch source | **The pip-installed torch package** in `.venv` on both platforms (`torch.utils.cmake_prefix_path`). One version for training and inference; MPS guaranteed on macOS. CMake locates it by running the venv Python if `CMAKE_PREFIX_PATH` is not given. |
| Runtime libraries | Windows: post-build step copies `torch/lib/*.dll` next to the executables. macOS: `RPATH` set to `torch/lib`. |
| Device selection | Runtime flag `--device auto|cuda|mps|cpu`; platform `#ifdef`s only in `torch_evaluator.cpp`. |
| Precision | fp16 on CUDA (logits only; softmax in fp32 on the CPU), fp32 on MPS/CPU; parity test thresholds in §9. |
| Dependencies | LibTorch; vendored `doctest.h` and `nlohmann/json.hpp` with licences and pinned versions. |
| Verified so far | **Windows only**: MSVC 2026 + CMake 4.3.2 + PyTorch 2.14.0+cu130 build and run a LibTorch CUDA smoke test (conv on an RTX 5080) — done in this session. **macOS: not verified.** The MPS load-and-forward test is part of **M2 acceptance** and needs to be run by the user on the Mac before M3 starts. |

---

## 8. Configuration

Single JSON per run, sections `board`, `search`, `selfplay`, `training`, `eval`. Defaults:

| Parameter | 5×5 smoke | 9×9 default | 19×19 ("runs") | Paper |
|---|---|---|---|---|
| residual blocks / filters (initial conv block not counted) | 2 / 32 | 6 / 64 | 20 / 128 | 19 or 39 / 256 |
| simulations per move (self-play) | 50 | 200 | 800 | 1600 |
| simulations per move (eval/GTP/ladder) | 50 | 200 | 800 | 1600 |
| c_puct (swept in M4) | 1.5 | 1.5 | 1.5 | not stated |
| FPU for unvisited edges (swept in M4) | Q = 0 | Q = 0 | Q = 0 | Q = 0 |
| Dirichlet α / ε | 0.43 / 0.25 | 0.134 / 0.25 | 0.03 / 0.25 | 0.03 / 0.25 |
| temperature moves | 4 | 8 | 30 | 30 |
| search-time symmetry | off (M3a') / on | on | on | on |
| move cap (2·N²) | 50 | 162 | 722 | 722 |
| resign threshold / no-resign fraction | disabled | auto (`resign_auto`) / 10% | auto / 10% | auto / 10% |
| komi | 7.5 | 7.5 | 7.5 | 7.5 |
| games per iteration | 100 | 2,000 | 5,000 | 25,000 |
| window (games) | 1,000 | 20,000 | 100,000 | 500,000 |
| batch / max train steps per iteration | 64 / 100 | 256 / 1,000 | 256 / 2,000 | 2048 / 1,000 |
| samples per position per iteration (cap) | 0.25 | 0.25 | 0.25 | — |
| holdout games | 5% | 5% | 5% | — |
| eval pairs (games) / gate | 20 (40) / > 55% | 200 (400) / > 55% | 100 (200) / > 55% | 400 games / > 55% |
| G games in flight × K leaves | 16 × 1 | 128 × 1 | 64 × 1 | — |
| NN cache | off | off | off | (not described) |

**`accel` section (§13), every switch off by default so that the defaults above are the AGZ profile:**

| Key | Default | Meaning |
|---|---|---|
| `accel.playout_cap.enabled` | false | PCR (§5.4.9) |
| `accel.playout_cap.full_prob` | 0.25 | probability of a full search per move |
| `accel.playout_cap.reduced_simulations` | 9×9: 50, 19×19: 100 | reduced budget; the full budget is `search.simulations` |
| `accel.playout_cap.reduced_positions` | "drop" | "drop" = reduced-search positions are not training samples (paper); "value_only" = experimental variant (§5.4.9, §6.2, §6.3) |
| `accel.forced_playouts` | false | root forced playouts, `k_forced = 2` |
| `accel.policy_target_pruning` | false | prune forced visits from the stored target (requires `forced_playouts`) |
| `accel.aux.ownership` / `accel.aux.score` | false / false | auxiliary heads and targets (§6.1, §6.2); `w_own = 1.0`, `w_score = 0.5` |
| `accel.global_pooling` / `global_pooling_blocks` | false / 9×9: [2, 4], 19×19: [5, 10, 15] | pooling branch in the listed residual blocks (1-based) |
| `accel.gating` | true | false = AlphaZero-style: the candidate is always promoted, the evaluation match is skipped (the ladder still measures strength, §6.6) |
| `accel.window` | "fixed" | "growing": window in games `= w₀ · (1 + β · ((n/w₀)^α − 1)/α)`, KataGo's schedule with `w₀ = 2,000` (9×9), `α = 0.75`, `β = 0.4`, `n` = games generated so far, capped at `training.window_games` |

The chunk header records the PCR parameters and which extras are present (§5.6); the trainer refuses to mix chunks whose `record_extras` differ from the run's profile.

**Pipeline-only keys (M4; the C++ side ignores unknown keys):** `pipeline.async_ladder` (true since 2026-09-29), `pipeline.async_gate` (false) — overlapped evaluation (§6.5.1); `search.resign_auto` (false; 9×9: true — select `v_resign` per iteration, §5.4.7), `search.resign_select_scope` ("window" | "latest"), `search.resign_fpr_target` (0.05), `selfplay.evict_old_chunks` (true), `training.fixed_holdout_iteration` (0 = off; 9×9: 5, §6.6), `eval.ladder_pairs` (50), `eval.ladder_neighbours` (3), `eval.gtp_anchor` (null or `{name, command}`, §6.6), `eval.gtp_workers` (4: engine trios playing a GTP match in parallel — `scripts/vs_gnugo.py` — each a contiguous block of openings; the report records `gtp_workers` and is reproducible for the same count only). **M4a:** `selfplay.processes` (1; 9×9: 1 since step 1 — pipeline-only, implemented: N self-play processes per iteration, each with its own `games_in_flight`, per-worker plan records `{k, games, seed, chunk_id_start, chunk_id_end}` persisted before launch, §5.5.1; a self-play phase whose plan was written with N = 1 finishes single-process, N applies from the next plan). **M4a (read by the C++ side):** `selfplay.threads` (1; 9×9: 8) / `eval.threads` (1; 9×9: 4) — 1 = the first-version drivers unchanged, > 1 = search threads of the M4a drivers, `games_in_flight` stays the global concurrency (9×9: 1024 with the threads); `selfplay.max_batch` (0 = G; the evaluation thread's batch cap). `--threads` / `--max-batch` override them on `mango_selfplay` and `mango_match`. **M4a step 2 (read by the C++ side):** `inference.channels_last` (true; takes effect with fp16 only), `inference.cuda_graphs` (true) — the CUDA fast path of `TorchEvaluator` (§5.3), ignored on MPS and CPU; `--no-channels-last` / `--no-cuda-graphs` on the three executables.

### 8.1 Compute arithmetic for 19×19 (why it is a "runs end to end" config)

Per iteration: 5,000 games × ~250 positions × 800 simulations = **10⁹ network evaluations**. A 20×128 network on one RTX 5080 will plausibly reach 5–10k evaluations/s end to end (to be measured), i.e. **30–55 hours per iteration**, and a meaningful 19×19 run needs hundreds of iterations. The 19×19 config therefore proves that the code path (formats, memory, pipeline) works at full size; it is **not** expected to produce a strong player on this hardware, and M6's acceptance is phrased accordingly (§11). Reaching real 19×19 strength would need a much faster backend, more GPUs, and/or a reduced budget (fewer sims, smaller net), which is out of scope.

### 8.2 First parameter sweep (M4)

Once the 9×9 loop runs, sweep jointly: `c_puct ∈ {0.8, 1.1, 1.5, 2.5}` × FPU ∈ {Q = 0, parent value}, judged on the frozen ladder (§6.6) at equal wall-clock budget. Tooling (M4): `scripts/sweep_9x9.py --hours H` runs the 8 configurations sequentially under `runs/sweep/` with `Pipeline.run_for_seconds` (whole iterations until the budget is used, restart-safe) and `--compare` fits their final best models plus the random anchor in one Bradley–Terry table (`mango.strength.cross_run`). R0 took ≈ 13 min per iteration, so one GPU-hour per configuration is ≈ 4–5 iterations; the budget is chosen when the sweep is launched.

---

## 9. Testing strategy

| Layer | Tests |
|---|---|
| Rules (C++) | Captures, multi-chain captures with shared liberties (merge = OR), suicide, simple ko, positional superko incl. triple ko and the **pass-then-non-capturing-repeat** case, pass exempt from superko, pass/pass end, move cap end, area scoring with seki and dame, Zobrist consistency, **fuzz vs `RefBoard`** for 10k random games (legality, liberties, score, hash). |
| History snapshots (C++) | After captures the snapshot ring buffer equals `RefBoard`'s stored positions for the last 8 steps; passes duplicate the snapshot; `applySymmetry` transforms all 8 snapshots consistently; planes from a symmetrised board equal symmetrised planes. |
| Features (C++ ↔ Python) | Same random game: C++ `encodeFeatures` at every t equals the Python loader's assembly from the chunk's snapshots, byte for byte, including `z_t` parity. |
| Search, sequential (C++, `FakeEvaluator`, symmetry off, cache off) | Backup signs; terminal handling (two-pass and move-cap leaves, correct sign); first selection at a fresh node = highest prior; PUCT prefers higher Q at equal N; convergence to a forced win in a 5×5 capture position; π sums to 1; tree reuse keeps child statistics and frees siblings; resignation decision uses root perspective; `visitCount == 1 + ΣN` after every simulation; superko-illegal moves never expanded (path hashes respected). **Game-ending moves (§5.4.10):** a pass that lets the opponent end the game at a loss gets `Q = −1` from its first visit (the child's raw network value is kept, its `value()` is +1) and is never chosen, while the plain search passes on the same fixture (the blind spot, reproduced with fewer simulations than the opponent has replies); a winning pass after the opponent's pass is chosen despite a `1e-4` prior, with `rootValue() = +1` and `rootNetValue()` unchanged, while the plain search never explores it; one move before the cap every edge carries its exact value, every simulation ends at a terminal child, the losing pass has `N = 0`; the 3-ply ladder still converges with the rule on; invariants hold throughout. |
| Multi-threaded drivers (M4a, implemented; `test_selfplay.cpp` "threaded self-play …", "threads = 1 …", `test_match.cpp` "threaded match …") | T = 1 is the first-version code path: bit-identical chunks and identical summaries — timing and throughput fields (`seconds`, `eval_seconds`, `evals_per_s`, `positions_per_s`) and the two new configuration fields (`threads`, `max_batch`) excluded — to the M3b driver on the same seed (self-play and match). T ∈ {2, 5} with the deterministic `FakeEvaluator`, including a configuration with `max_batch` smaller than one thread's share (e.g. G = 8, T = 2, B = 3: every forward has ≤ B requests, rounds are split and their threads wait for all parts): every game, matched by `game_seed`, equals the same seed's record in a standalone sequential run (moves, root visit counts, `root_value`, `root_max_q`, `z`, termination); `visitCount == 1 + ΣN` and `Σ count == root_total_visits` after every move in every thread; chunk game counts and `next_chunk_id` exact; the evaluation thread never mixes the two sides of a match in one forward; one failed batch is retried with identical requests and leaves no Pending node or reservation in any thread; two failures wake every waiting thread, every thread exits, only complete chunks are published. Pipeline step 0 (`selfplay.processes = 3`, fake self-play binary; **implemented**, `tests/test_pipeline_workers.py`): the plan holds one record per worker before launch; chunk ids are unique and inside each worker's block; a run where two workers have published chunks and the third fails is terminated, and on restart every worker resumes its own remaining quota from the plan — final game count, manifest and `next_chunk_id` correct, no file overwritten, no game dispatched twice; seeds differ per worker; N = 1 takes the existing single-process code path (plan, seed and `next_chunk_id` unchanged from M4, so a resumed run is unaffected). |
| Driver profile (`test_selfplay.cpp` "driver profile …", `test_torch_eval.cpp` "TorchEvaluator timings …", `tests/test_pipeline_workers.py`) | Histogram bins: 1 → 0, 2–3 → 1, 1,024 → 10, the last bin open. For T = 1, T > 1 with B = G and T > 1 with B smaller than a thread's share: the histogram and the timeline each hold every forward and every request exactly once, a bin's requests lie within its bounds, every timeline bin averages between 1 and G games in flight; every time bucket is ≥ 0; the search buckets fit in T · wall; T = 1: collect + commit + callbacks + evaluator call ≤ wall, one round per forward, no waiting; T > 1: evaluation idle + call + handback ≤ wall, rounds per forward = rounds when no round is split. `TorchEvaluator` timings count calls, rows and padded rows (CPU: no padding; CUDA graphs: every call padded to its bucket, warm-up included). The pipeline passes a per-launch `--profile-out` path (single process and per worker) and logs the profile. Not tested: the values of the timings (machine-dependent). |
| Pipeline, overlapped evaluation (implemented, §6.5.1; `tests/test_pipeline_async.py` with `tests/fake_match.py`, `tests/pipeline_driver.py`; CSV in `test_strength.py`) | `async_ladder`: ladder, fit files, ratings and strength state equal a sequential run; a running ladder job delays nothing; drain empties the lane. `async_gate` with scripted outcomes: self-play models follow the recorded effective iterations; gate incumbents are the best after the previous decision; one run, 4 one-iteration runs and 2 × 2 give the same model sequence. Restart at every recorded point (incl. ladder enqueued but gate not cleared; fit written but not collected) ends in the same state, `best.json`, ladder, fits, `ratings.csv`, reports, with no duplicate events or jobs. Pipeline or supervisor killed by force during a ladder match: no process of the run survives the pipeline, a job is relaunched only when its lock / Job Object is free, matches writing one report never overlap; EOF instead of `go` exits the supervisor; a second pipeline on the same run refuses to start. Equal effective iterations after a switch: the later gate's candidate plays; none recorded: the initial model. Failing job stops and resumes. Switching off settles pending work; both off: today's behaviour. CSV regeneration idempotent, legacy rows kept once, a re-fitted legacy iteration only with its new rows, interrupted migration redone; a fit publication interrupted after the fit file is completed at collection. Pipeline killed during self-play: the driver dies with it, the restart dispatches only afterwards; `--exec` passes output and exit code through. |
| Search, batched (C++) | **K = 1 across games**: every game in a batched run equals its standalone sequential run (deterministic `FakeEvaluator`, one RNG stream per game). **K > 1**: pending set has no duplicate nodes; collisions counted and leave no reservations; backup signs/counts correct; commit restores invariants; after a simulated evaluator failure no node is PENDING, no reservation remains, completed (terminal) simulations are retained and `simulations_this_move` equals the number of completed simulations. Tree reuse: `root_total_visits` after a reused move equals inherited + S. |
| Symmetry | Planes and π mapped by the same table in C++ and Python; **coordinate mapping** tested with a `FakeEvaluator` whose policy is a known function of the point (forward transform then inverse transform is the identity; a policy peaked at point p under symmetry s maps back to p). No test assumes the real network is equivariant — CNN + FC heads are not. |
| NN backend, CUDA fast path (M4a step 2, implemented; `test_torch_eval.cpp`) | For every combination of `channels_last` × `cuda_graphs` × {fp32, fp16} the fixture model passes the M2 parity checks against the PyTorch reference (fp32 max-abs 1e-4; fp16 KL < 1e-3, value 1e-2, no underflow to 0); channels-last is reported active with fp16 only. Fast path against the plain path on the same positions: max-abs policy and value difference < 1e-4 (fp32) / 2e-3 (fp16) for batch sizes on a bucket and between buckets (1, 5, 16, 33, 100), for the 3 warm-up calls and the replays; replays of the same input are bit-identical. Static buffers do not leak: once the bucket is captured, batch A, then a smaller batch B in the same bucket, then A again gives A's previous replay bit for bit, and A and B equal their evaluation on a fresh plain evaluator within the tolerance. Buckets interleaved in one evaluator; two evaluators in one process (the match case). `bucketFor`: monotone, ≥ the batch, < 2× the batch, ≤ 1.5× from 32 on, 0 above 4,096. **Optional, on a real model** (`MANGO_TEST_MODEL_DIR=<model dir>`; run on `0026` on 2026-09-28): 1,024 positions from random games, batches 1–1,024; fp32 every variant < 1e-4 against the plain path; fp16 every variant within max(2e-3, 2× the plain path's own difference between a batch of 200 and the same rows one by one). Not tested: the fallback after a failed capture (cannot be provoked deterministically; the first build hit it for real — capture on the default stream — and the run continued on the plain path). |
| NN backend, exclusive captures (`test_torch_eval.cpp`) | Two evaluators on two threads at once, interleaving 8 buckets for 20 rounds: every result within 2e-3 of a plain evaluator, graphs still enabled, all 8 buckets captured in each. An evaluator warms up and captures all 16 buckets up to 1,024 while another replays batch 512 and runs plain forwards of ever new shapes without pause: every capture succeeds (without the exclusive lock this test failed 5 of 5 runs). |
| NN backend (M2, both platforms) | Fixture model loads and runs on CUDA (fp16 & fp32), MPS (fp32), CPU (fp32). fp32: max-abs 1e-4 on policy and value. **fp16**: KL(p_fp32 ‖ p_fp16) < 1e-3 per row, value abs diff < 1e-2, and **no action with p_fp32 > 1e-4 has p_fp16 = 0** (guards the underflow the fp32-softmax design is meant to prevent). Batch of 1 equals batch of 64 row-wise. |
| Chunk format | Byte-exact parse and re-serialisation of `mgo2_fixture.bin` in C++ and Python; C++ write → Python read round-trip and vice versa on random games; `.tmp` ignored; header offsets and `header_size` validated; `model_id` zero-padding/truncation; holdout tag; per move `Σ count == root_total_visits` and, with tree reuse **off**, `root_total_visits == S`; with tree reuse on, `root_total_visits ≥ S`. |
| Model versions | Export round-trip; immutable directory + `best.json` pointer update order; trace at batch 8 works at batch 1 and 64. |
| Training (Python) | One-step overfit decreases loss; symmetry augmentation keeps π normalised; step cap (§6.2) computed correctly for small windows; LR schedule boundaries; learner checkpoint restore reproduces the next step bit-for-bit on CPU; holdout games never appear in training samples. |
| Match | Unique-trajectory count is reported (diagnostic, no threshold); pair scores computed correctly on a fixture with a decisive pair, a split pair and a double loss; bootstrap interval reproducible under a fixed seed; promotion rule is the point estimate. Ladder: opening file generated once and reused; Bradley–Terry MAP fit on a fixture with a fully separated player returns finite ratings and flags the separation; a disconnected fixture is reported unrated. **M4 tests:** the fit recovers simulated ratings within 40 Elo (2,000 games per pairing) with the anchor at 0, mirrored results give the same fit, draws count ½, a zero-information player has the analytic prior+data SE; ladder bookkeeping with a fake `mango_match` (opponent selection = anchors + most recent neighbours, one opening file, per-pairing seeds stable across restarts, reports reused on restart, a lost `ladder.json` record rebuilt from its report, cross-run round robin); `openingsFromJson` round-trips and rejects occupied/out-of-range/malformed openings; the random anchor plays legal games (pass only when no board move is legal), is seed-deterministic and loses to a score-aware searcher; resign selection (§5.4.7): winner's moves only, `max(v, max Q)`, draws excluded, the largest `t` under 5 %, `r_t = t` is not a false positive, < 100 games or no qualifying `t` → disabled; counterfactual label errors count only games whose *first* dip is the eventual winner's (loser-first, draws, `max Q` rescues and untriggered games do not count); GTP driver: vertex conversion, score parsing, a random-vs-random match refereed by `mango_gtp` (legal, reproducible, report shape); pipeline: the selfplay plan carries `v_resign` and the flag reaches `mango_selfplay`, the strength phase adds the initial model and the candidate, a restarted strength phase replays nothing, eviction deletes exactly the chunks outside the window and keeps their manifest entries; resign selection scope "latest" reads only the previous iteration's chunks and the false-positive target is passed through; the monitor writes the fixed-set columns and freezes the fixed holdout chunk. Rating diagnostics: the opening-adjusted fit recovers simulated ratings (±40 Elo) and opening terms (±70 Elo) with the anchor at 0 and keeps two equal players together on a colour-decided opening set; predicted-vs-observed rows and `z` on hand-built cases; prior sensitivity flags a non-monotone order; `pair_structure` / `opening_balance` on hand-built reports; `write_holdout_games` keeps holdout games only. |
| GUI (`test_gui.py`) | Parsing of `showboard` and `mango-analyze` (top-down rows, point indices, pass, "no analysis" replies); the controller against a scripted fake engine: turn taking in both modes, the model moving first when the human is white, illegal moves rejected by the engine leave the position unchanged, undo retracts one move (analysis) or the pair (play) and re-opens a finished game, two passes and the move cap end the game with the engine's score, resignation by the side to move and a `resign` reply from the engine, analysis truncated to top-k and cleared by a move, a changed budget restarts the engine; with the binaries built, a game against the random `mango_gtp` player where the drawn grid equals the engine's board; the window itself is smoke-tested (a click plays a stone and triggers the analysis overlay) when a display exists. |
| End-to-end | `pipeline.py --smoke` (5×5 config): finishes in < 2 minutes on CPU; runs on both platforms. |
| Acceleration profile (§13, each behind its switch) | PCR: with a seeded game the sequence of full/reduced decisions is reproducible, reduced searches run `reduced_simulations` new simulations with no root noise and `search_kind = 0` is written; the loader excludes exactly those positions (`"drop"`) or masks their policy term (`"value_only"`). Forced playouts: with a `FakeEvaluator` whose prior gives one child `P = 0.5`, that child has at least `floor(sqrt(2 · 0.5 · N))` visits after `N` simulations even when its Q is −1; a child with prior `1e-6` that PUCT never selects stays at `N = 0` (no forcing of unvisited children). Pruning: a fixture root with known N/P/Q prunes to the hand-computed counts, the played move survives, tree statistics are untouched, `Σ count ≤ root_total_visits`. Ownership: `final_ownership` in the chunk equals `areaOwnership` of the final snapshot (C++ writer vs `RefBoard`); the loader's `ownership_t` sign follows `z_t`'s parity rule; **augmentation**: an asymmetric fixture position (distinct ownership at all 8 images) is checked under every symmetry — the transformed ownership map equals `areaOwnership` of the transformed board, and planes/π/ownership were transformed by the same index table; resigned games yield `aux_t = 0` and contribute nothing to `L_own`/`L_score` (loss unchanged when their targets are randomised). Pruning uses frozen Q and the paper's budget: a fixture where `W/N` recomputation would change the outcome prunes to the hand-computed counts only under the frozen rule, and a bad child that was never forced (its visits came from PUCT) is still reduced by up to `floor(n_forced)`. Step cap under PCR: the bound uses the index size (a window whose chunks report 4× more positions than policy-target positions yields the cap for the smaller number); an index of size 0 skips training and re-exports the incumbent. PCR `"drop"`: reduced positions are absent from the dataset index and game weights equal indexed counts. Output tuple: a model with all heads exports 4 outputs, `model.json.outputs` matches, and the C++ evaluator loads it with unchanged policy/value parity while rejecting a model whose first two outputs are not `policy_logits`/`value`. Global pooling: the module is equivariant under the 8 symmetries (numerically, on random input), and a model with the extra heads exports and loads in C++ with unchanged policy/value parity. `accel.gating = false`: pipeline promotes without a match and `state.json` records it as such. Growing window: the formula is checked at `n = w₀` and `n = 10·w₀` against hand-computed values. |

---

## 10. Performance: hypotheses to be measured in M3b

Nothing in this section is an acceptance criterion. M3b measures and reports:

- games/s and positions/s of the full self-play loop (board copy + selection + encoding + transfer + forward + softmax + backup + chunk writing), at G = 128, K = 1, S = 200, fp16.
- **Actual average game length** of early self-play (random-ish networks on 9×9 with dead stones not removed can run far longer than the ~70 moves assumed here, up to the 162-move cap).
- Forward time per batch on the GPU versus wall-clock per step, to see whether the CPU side is the bottleneck.
- The same numbers for the 19×19 config at a handful of games, to replace the estimate in §8.1 with a measurement.

Only after these numbers exist do we decide among: K > 1 (§5.4.5), multiple threads, a shared batching queue, or a hand-written backend. **Decided 2026-09-25 from the R0 numbers** (self-play 350 positions/s at G = 128; the evaluator call — transfer, forward, softmax, copy-back — takes 49 % of the wall time at an average batch of 119, the single search thread the rest; the GPU's own busy fraction is not measured; 8 cores available): multiple search threads with a shared queue — M4a, §5.5.1. Threads shorten the search time between evaluator calls; they do not enlarge batches (G is the global concurrency, one pending leaf per game), so the gain is bounded by the evaluator share until G is raised. K > 1 stays off (it is the other way to raise the batch size and it changes the search; §5.4.5). **Outcome (M4a, 2026-09-25):** the queue does not enlarge batches at a fixed G (they shrink, rounds arrive out of phase), the gain came from G = 1024 with T = 8: 820 positions/s, evaluation thread 97 % busy — the next bottleneck is the forward itself (§11 M4a).

**Measured 2026-09-22 (M3b, RTX 5080, LibTorch fp16 unless noted, random networks, `runs/throughput`).** Self-play numbers are the full loop (board copy + selection + encoding + transfer + forward + softmax + backup + chunk writing); forward-only numbers are an eager PyTorch benchmark of the same network and give the ceiling.

| Run | games | avg length | avg batch | evals/s | positions/s | eval share of wall |
|---|---|---|---|---|---|---|
| 9×9 6×64, S = 200, G = 128, CUDA fp16 | 128 | 86 | 67 | 51,000 | 265 | 70 % |
| 9×9, G = 32, CUDA fp16 | 64 | 79 | 21 | 23,100 | 120 | 87 % |
| 9×9, G = 1, CUDA fp16 | 2 | 127 | 1 | 1,470 | 7.7 | 99 % |
| 9×9, G = 32, CPU fp32 (16 games) | 16 | 86 | 8.4 | 5,800 | 30 | 97 % |
| 19×19 20×128, S = 800, G = 8, CUDA fp16 | 8 | 438 | 6 | 3,200 | 4.1 | 94 % |
| forward only, 9×9: batch 1 / 32 / 128 / 256 | | | | 830 / 26,600 / 109,000 / 237,000 | | |
| forward only, 19×19: batch 8 / 64 / 128 | | | | 2,100 / 12,100 / 14,600 | | |

Readings:
- **Batching is the whole game on the GPU**: G = 128 is 35× G = 1. The 9×9 average batch (67) is below G because the single wave of 128 games drains at the end; a 2,000-game iteration keeps the batch near G. At 265 positions/s the 9×9 default iteration (2,000 games ≈ 172k positions) takes **≈ 11 minutes** of self-play; the 20-pair gate match with the batched `mango_match` is a fraction of that.
- **The CPU side is now 30 % of the wall clock at G = 128** (70 % inside `evaluate`), and forward-only at batch 128 is 2× the achieved evaluator rate, so the next 9×9 gains are in the C++ side (feature encoding, board copies, the softmax loop) and in overlapping transfer with compute, not in a new backend. K > 1 (M3c) is not needed for 9×9.
- **19×19 confirms §8.1**: at G = 64 the ceiling is ≈ 12k evals/s, so the `19x19.json` budget (5,000 games × ~440 positions × 800 simulations ≈ 1.8·10⁹ evaluations per iteration) is **≈ 40 hours per iteration** on this GPU. The 19×19 configuration stays a "runs end to end" configuration (M6), as designed.
- A 9×9 CPU-only run (Apple Silicon without MPS, or a CI box) is ≈ 9× slower than the GPU at G = 32; usable for smoke tests, not for training.

---

## 11. Milestones

| M | Deliverable | Acceptance |
|---|---|---|
| M0 | Environment (done on Windows): venv, PyTorch 2.14+cu130, MSVC 2026 + LibTorch smoke test, setup scripts | smoke binary runs a CUDA conv |
| M1 | **Done 2026-09-18 (Windows).** `cpp/core`: `Board` with snapshots, `GameHistory`, rules, features, symmetry, SGF, Zobrist, `RefBoard`; `mango_gtp` playing random legal moves; C++ tests | rules/history/feature/symmetry tests pass; differential fuzz (5/7/9 full, 13/19 sampled; `MANGO_FUZZ_SCALE` multiplies) passes at 5×; scripted GTP session verified (GoGui not yet tried) |
| M2 | **Done 2026-09-18 on Windows; MPS pending.** `model.py`, `export.py` (logits out, batch-8 trace), model versions; `TorchEvaluator` with fp32 softmax; fixture `cpp/tests/fixtures/model_5x5_v1` + `python/tests/make_fixture.py` | CPU fp32, CUDA fp32 and CUDA fp16 match the PyTorch reference; the MPS case is compiled and **skipped until run on the Mac** |
| M3a | **Done 2026-09-18.** Sequential search: K = 1, cache off, resignation off, symmetry off; collect/commit protocol implemented (K > 1 untested); `MctsPlayer` for GTP; fake-evaluator tests: first selection, backup signs, terminal (two-pass and cap), PUCT ordering, 3-ply ladder convergence, tree reuse and visit accounting, resignation perspective, superko along the path, invariants after every simulation | all sequential search tests pass |
| **M3a′** | **Implemented 2026-09-22 (Windows); acceptance partly met, see below.** `cpp/selfplay` (MGO2 writer/reader with the optional `record_extras` fields and both fixtures, sequential `GameRunner`, `mango_selfplay`), `cpp/match` (`playMatch`, random opening set, colour-swapped pairs, bootstrap interval, `mango_match`), Python `chunk.py`/`data.py`/`train.py`/`state.py`/`pipeline.py`/`known_outcome.py`, `configs/5x5-smoke.json`; cross-language fixtures `parity_5x5.*` and `symmetry_5x5.json`. Run `runs/m3a-5x5`: 60 iterations, ~35 s each with self-play on CPU (batch-1 GPU inference is 2.5× slower until M3b). **5×5 sequential end-to-end learning check**: chunk writer/reader (MGO2 including the optional `record_extras` fields and both fixtures, §5.6; the accel features themselves stay off), `chunk.py`, minimal `train.py`/`export.py`/loop with the sequential engine, K = 1, no batching, no throughput work; a few dozen iterations on `5x5-smoke.json` | the loop-level contracts are exercised (z parity, π symmetry, snapshot assembly, holdout) and learning is shown two ways: (1) the value head has the correct sign on ≥ 95% of the **last two positions of held-out two-pass games** (their area score is the game result; a hand-built settled-position set is reported as a diagnostic); (2) the final model beats the **frozen initial model** in a colour-swapped match on the fixed opening set with a bootstrap interval excluding 0.5. Held-out losses are recorded. | **Result:** (2) met — the final best model beat the frozen initial model 40–0 (mean pair score 1.0, interval [1.0, 1.0]); held-out value MSE fell 1.06 → 0.23, policy CE 3.26 → 1.68, 22 of 60 candidates promoted. (1) **not met on the hand-built settled-template set**: 78 % overall; 100 % on every position with |score| ≥ 12.5, 69 % at 1.5 points, 33 % at 2.5 points — fully settled boards with 1–2 point margins never occur in 5×5 self-play (black wins 72 % of games, games end by two passes after ~27 moves), so the set is out of distribution. On the last two positions of held-out two-pass games (on-distribution, exact area score) the sign accuracy is 97.6 % (84 positions), reported as `holdout_terminal` in `check.json` but not part of the pass rule. **Decision (2026-09-22, review):** criterion (1) is defined on those positions — sign accuracy ≥ 95 % on the last two positions of held-out two-pass games — and the settled-template set stays in `check.json` as a diagnostic (it is a board-size-bound test: on 5×5 with komi 7.5 stronger play makes close games rarer, not more common). With that definition M3a′ **passes** (97.6 % / 40–0). |
| M3b | **Done 2026-09-22 (Windows).** `BatchedSelfplay` (G games in flight, K = 1, one evaluator call per step, failed-batch retry per §5.4.5) behind `mango_selfplay --games-in-flight`; `playMatch` with G game slots and one evaluator call per side per step (`mango_match --games-in-flight`); `SelfplayGame` state machine shared by the sequential and batched drivers; timing in the self-play summary; §10 numbers measured. Atomic chunk publish, SGF and seeds were delivered in M3a′. | Tests: every game of a batched run equals its standalone sequential run (symmetry on and off; evaluations counted per game), a batch that fails once is retried and leaves no trace, two failures abort; match results identical for 1, 3 and 10 games in flight; chunks validated by Python (M3a′). Numbers reported in §10: 9×9 self-play 265 positions/s at G = 128 (≈ 11 min per default iteration), 19×19 ≈ 40 h per full-budget iteration. |
| M3c | (only if M3b numbers require it) K > 1 within a game, search-time symmetry on, resignation with auto threshold | batched protocol tests pass with collisions > 0 |
| M4 | **Implemented 2026-09-23 (Windows); first 9×9 run done; ladder re-measured 2026-09-24 with §5.4.10 — acceptance met on the re-measured ladder; sweep pending.** `train.py`/`data.py`/held-out monitor were delivered in M3a′ (the memmap index of §6.3 is deferred: parsing a 20k-game 9×9 window takes ≈ 9 s and holds ≈ 300 MB, so it is an M6 optimisation); M4 adds the automatic resign threshold (`resign.py`, `mango_selfplay --resign-threshold`, selection scope/target knobs, per-iteration diagnostics), chunk eviction, `strength.py` (frozen ladder, random anchor in `mango_match`, fixed opening file, Bradley–Terry MAP fit with intervals, opening-adjusted fit, predicted-vs-observed and prior-sensitivity diagnostics, `crossrun` for sweeps), `gtp.py` (external GTP anchor driver, refereed by `mango_gtp`), the `strength` pipeline phase, the fixed validation set, `--hours` budgets, `configs/9x9.json` with `resign_auto`, `scripts/run_report.py`, `scripts/sweep_9x9.py`. **Run `runs/9x9-r0`** (R0, the AGZ profile, seed 1; report `docs/RUN_9x9_R0.md`): 15 iterations in 3 h 16 min (≈ 347 s self-play, ≈ 12 s training at the 1,000-step cap, ≈ 200 s gate, ≈ 235 s ladder per iteration); 13 of 15 candidates promoted. **Robust findings:** every ladder entry beats its predecessor head-to-head (58–74 %); iteration 15 beats iteration 1 88–12 and the random anchor 99–1 on the fixed openings; the ladder order is monotone under every prior width tried; held-out policy CE 4.34 → 2.97, value MSE 0.87 → 0.72. **Original measurement (plain search) invalid:** the single-scale model misfit (mean \|z\| 3.5, max 49) because strong models lost ≈ 10 % of their games to iteration 1 through early two-pass endings (the §5.4.10 blind spot); opening/colour effects < 130 Elo. **Re-measured with §5.4.10** (same models, openings, seeds; `strength_plain_search/` keeps the original): no early two-pass loss remains (47 → 3 such games in 60 matches, none lost), iteration 15 beats iteration 1 100–0, mean \|z\| 0.90 (max 3.0), order monotone under σ = 200–1,400, first/last intervals 216 [156, 277] vs 1,925 [1,818, 2,032] at σ = 350. **Still not a finding:** the absolute level (1,541 → 2,413 for σ = 200 → ∞): identified through the neighbour matches but shrunk ≈ 500 Elo by the σ = 350 prior, which the interval does not show; graded opponents near the top would tighten it and make runs comparable (M4b). **Open issues** (§5.4.7, `docs/RUN_9x9_R0.md`): resignation false positives of 5–12 % on the new model (knobs added, control run without resignation not yet done); early-pass blind spot of the search against passing opponents (fixed in §5.4.10 on 2026-09-24 with tests; the R0 ladder has not been re-measured with the fix yet); the monitor could not judge generalisation (fixed validation set added for future runs). **Sweep §8.2 not run** (tooling ready; ≥ 8 GPU-hours, budget to be decided). **External anchor measured (2026-09-24):** GNU Go 3.8 level 10 (`--chinese-rules --capture-all-dead --play-out-aftermath`, refereed by `mango_gtp`) beats iteration 15 82–18 at 200 simulations (≈ −260 Elo) and 70–30 at 800 on the ladder openings; 54 of the 82 losses are whole-board margins produced by Mango passing at every turn once behind (an artefact of the margin, not of the result: 44 of them were already ≥ 10 points behind at the first pass); ≤ 10 losses came from passing early on an open board while ahead — a self-play-learnt habit (a pass made while ahead is answered by a pass in self-play) that §5.4.10 does not cover because the first pass is not game-ending. GNU Go added to the R0 ladder as anchor (c) of §6.6. | Elo on the frozen ladder rises over ≥ 10 iterations with non-overlapping intervals; held-out curves logged. **Status: met on the ladder re-measured with §5.4.10 (2026-09-24)** — rise over 15 iterations, first/last intervals disjoint, and the fit now describes the results (mean \|z\| 0.90); it was not met on the original plain-search measurement, whose fit was invalid (mean \|z\| 3.5). Caveat carried into M4b: the interval widths do not include the prior dependence of the absolute scale (separated top entries). |
| **M4a** | **Single-GPU throughput: multi-threaded self-play and match drivers (planned 2026-09-25, reviewed the same day; §5.5.1, §5.9).** Measured on R0 (iteration 21): self-play 2,000 games = 358 s, of which the evaluator call takes 177 s (49 %; transfer + forward + softmax + copy-back, not a GPU busy fraction) and the single search thread the rest, at an average batch of 119 with G = 128. Two steps: **(0)** `selfplay.processes`: N `mango_selfplay` processes per iteration, each the first-version driver with its own G (total concurrency N·G), per-worker plan records persisted before launch, restart resumes each worker's own quota from the plan, a failed worker stops the others (§5.5.1). **(1)** `BatchedSelfplay` and `playMatch` with T search threads and one evaluation thread over a request queue (up to `max_batch` requests per forward, a thread's round split across forwards when needed), G the global concurrency, T = 1 the first-version code path unchanged, N = 1 likewise for step 0; the match's evaluation thread owns both evaluators and never mixes the sides in one forward. Threads do not enlarge batches (one pending leaf per game); the expected gain is the search share of the wall time, larger batches need a larger G. Out of scope: K > 1, PCR and every other §13 switch (M4b); CUDA graphs / TensorRT (only once the evaluator is the bottleneck, §10). | **Numbers, 9×9, S = 200, fp16, RTX 5080, reported for both (a) the same total concurrency as R0 (G = 128) and (b) each driver's best configuration (G, T, N free, memory permitting):** self-play ≥ 700 positions/s (2× R0's 350) in (b), the (a) number reported next to it; the 200-pair gate ≤ 90 s (R0: ≈ 180 s). The 2× is the target, not a derived number: the shared queue alone is bounded by the evaluator share (≈ 2× at most), beyond that only a larger G helps. **Tests:** §9 "Multi-threaded drivers" row — T = 1 bit-identical to M3b; T > 1 per-game equality by `game_seed` under the deterministic evaluator (records, not chunk bytes); invariants per thread; failure protocol (one retry identical, two failures wake and stop every thread); match buckets per side; step 0 restart with a failed worker. **Record:** the R0 continuation notes the iteration from which self-play used processes or threads (the seed dispatch changes which games are played, not their distribution). **Step 2 (planned and implemented 2026-09-28): the CUDA fast path of the evaluator, §5.3** — channels-last parameters and input (fp16), one CUDA graph per batch bucket. **Numbers (same benchmark as step 1: model of the day, 900 games, T = 8, G = 1024 or the best configuration; nsys before/after for kernels and launches per forward and the GPU busy fraction):** self-play ≥ 1,600 positions/s (2× step 1's 820); the 200-pair gate ≤ 90 s (the criterion step 1 left open). **Tests:** §9 "NN backend, CUDA fast path". **Measured 2026-09-28 (step 2, model `0026`, 900 games, T = 8, G = 1024, fp16, wall time incl. start-up; the plain path measured the same day):** plain 725 positions/s → channels-last only 897 (1.24×) → CUDA graphs only 1,213 (1.67×) → **both 1,338 (1.84×)**; other shapes with both on: T8 G512 1,266, T4 G1024 1,167, T8 G2048 1,156 (900 games cannot fill 2,048 slots). **≥ 1,600 positions/s not met.** **Gate (200 pairs, `0026` vs `0025`):** plain T4 G400 107 s → fast path **T4 G400 43 s**, T8 G400 42 s, T2 G400 50 s — **≤ 90 s met.** nsys, plain → fast path: kernels per forward 81 → 42, transposes 24 % → 0 % of the kernel time, launch calls per forward 81 → 1 (25.4 s = 37 % of the run → 1.6 s = 4 %), GPU busy 30 % → 55 %, GPU time per forward 407 → 351 µs (the padding to a bucket and slower fused BN kernels on channels-last tensors take back part of what the transposes cost). What is left of a forward's ≈ 640 µs on the evaluation thread is ≈ 350 µs of GPU execution and ≈ 290 µs of CPU work between forwards (plane copies, the fp32 softmax, handing results back), so the next steps — folding BN into the convolutions (BN/ReLU kernels are ≈ 35 % of the remaining kernel time), a cheaper softmax/hand-back — are outside this step. **Step 1 done 2026-09-25:** **Measured 2026-09-25 (step 1, 9×9, S = 200, fp16, RTX 5080, model `0022`, one process, 900 games per configuration, wall time incl. start-up; `max_batch` = G):** (a) same total concurrency G = 128: T = 1 **293**, T = 2 **324**, T = 4 311, T = 8 291 positions/s — the threads shorten the search share but the rounds arrive out of phase, so the average batch halves (110 → 55) and the evaluation thread is busy 96–98 % of the wall time: at G = 128 the driver is evaluator-bound and threads gain nothing; (b) larger G: T4 G256 543 (batch 99), T4 G512 595 (125), T8 G512 690 (158), **T8 G1024 820 positions/s** (batch 206, 164k evals/s, evaluation thread 97 % busy) — **≥ 700 met** (2.8× the T = 1 number of the same day, 2.3× R0's 350). **Gate (200 pairs, `0022` vs `0021`):** T1 G128 189 s (R0's ≈ 180), T4 G128 251 s (slower: batches per side halve), T4 G400 **117 s**, T8 G400 128 s — **≤ 90 s not met**: with 400 fixed games the per-side batch is ≤ 200 and shrinks as games finish, and the evaluation thread is already saturated, so the queue cannot take the gate below ≈ 115 s; a faster forward per batch (CUDA graphs / TensorRT, §10) or a different gate size would be needed, neither is part of M4a. With the real fp16 network T > 1 is not bit-reproducible (batch shapes change rounding): the four gate runs scored 0.655, 0.630, 0.662, 0.637 for the same seed, all inside the interval. R0 continues from iteration 24 with `threads` 8, `games_in_flight` 1024, `processes` 1, `eval.threads` 4 (`docs/RUN_9x9_R0.md`). **Step 0 done 2026-09-25:** **Measured 2026-09-25 (step 0, 9×9, S = 200, fp16, RTX 5080, model `0022`, 900 games per configuration, wall time incl. process start-up):** 1×G128 299 positions/s (avg batch 112; R0's 350 was measured on a 2,000-game iteration, whose tail — fewer games in flight at the end — weighs less); (a) same total concurrency 3×G43 = 129: **232** (avg batch 38, the evaluator call 86 % of each process's wall time — splitting a fixed G over processes only shrinks the batches); (b) 2×G128 430, 3×G128 485, **4×G128 501** (avg batch 85, 100k evals/s — at the forward-only ceiling of ≈ 109k at batch 128, §10). So step 0 gives 1.7× at 4× the concurrency and the GPU is now evaluator-bound at batch ≈ 90: **≥ 700 is not met at step 0** (as expected — it is the baseline for step 1), and step 1 only reaches it if the shared queue produces batches ≥ 256 (ceiling 237k evals/s), i.e. G ≥ 512 in one process. |
| **M4b** | **Single-GPU acceleration ablation on 9×9 (§13)**: implement the `accel` switches (§5.4.9, §6.1, §6.2, §8); runs R0 (AGZ profile, the M4 run) and R1–R5 each enabling one technique in the order PCR, aux heads, forced playouts + pruning, global pooling, gating off; R6 all on; equal GPU-hour budget per run, same seeds, same frozen ladder and opening set | a report `docs/ABLATION_9x9.md` with ladder Elo (with intervals) at equal GPU-hours and at equal games for every run, self-play throughput per run, and held-out curves; every deviation listed in §12 D11–D16 is either shown to help on one GPU or recorded as not helping — no acceptance threshold on the direction of the result |
| M5 | Full macOS run of the smoke pipeline by the user | `pipeline.py --smoke` passes on the Mac |
| M6 | 19×19 **runs end to end**; performance work; README | `configs/19x19-smoke.json` (20 games, 50 train steps, 4 eval pairs per iteration, full-size 20×128 network) completes ≥ 2 iterations without format/memory failure; throughput of the full `19x19.json` budget measured on a handful of games and documented against §8.1; **no strength claim** |

---

## 12. Deviations from the paper (intentional, documented)

| ID | Deviation | Reason / switch |
|---|---|---|
| D1 | Scale: smaller network, fewer simulations, smaller window and batch, one GPU instead of 64 GPU workers for training and TPUs for self-play | Hardware. Ratios kept where they matter (ε, τ schedule, gating threshold, L2 weight, move cap = 2·N²). |
| D2 | `c_puct` is a constant (1.5) — a starting point to be swept with FPU (§8.2) | AGZ does not state its value. |
| D3 | Dirichlet α scaled with board size (0.03·361/N²) | 0.03 is only meaningful on 19×19. |
| D4 | Training π = normalised raw visits at every move; paper stores temperature-adjusted π (one-hot after the cutoff) | Richer target; raw counts are stored so `store_pi = "temperature"` restores the paper's behaviour in the loader. |
| D5 | Resignation disabled in the first version; when enabled the threshold is selected automatically as in the paper | Simplicity first. |
| D6 | NN cache off by default; search-time symmetry on | Cache keyed on full input is a later optimisation. |
| D7 | Batched search abandons colliding descents instead of blocking | Simplicity; documented in §5.4.5. |
| D8 | Rules named "Tromp–Taylor scoring with suicide prohibited"; original Tromp–Taylor permits suicide | Matches common AGZ reimplementations. |
| D9 | L2 regularisation applied to conv/linear weights only; paper regularises all parameters (c‖θ‖²) | Standard practice (BN affine and biases excluded); a config flag `l2_all_params` restores the paper's form. |
| D10 | Training steps per iteration capped by window size (0.25 samples per position per iteration); 5% of games held out | Prevents multi-epoch training on the tiny early windows; the paper's window was never small. |
| D11 | Playout cap randomization (§5.4.9) | `accel.playout_cap.enabled`, **off**. KataGo 2019 §3.1: reduced-search positions are not recorded; the value head gains from more games per GPU-hour. `reduced_positions = "value_only"` is a further, separately flagged experiment. |
| D12 | Forced playouts and policy-target pruning (§5.4.9) | `accel.forced_playouts`, `accel.policy_target_pruning`, **off**. KataGo 2019 §3.2. |
| D13 | Auxiliary ownership and score heads/targets (§6.1, §6.2) | `accel.aux.*`, **off**. KataGo 2019 §3.3; the score head is a scalar simplification. |
| D14 | Global pooling in the trunk (§6.1) | `accel.global_pooling`, **off**. KataGo 2019 §3.4. |
| D15 | No gating: always promote (AlphaZero) | `accel.gating = false`, default **true** (AGZ). |
| D16 | Growing training window (§8) | `accel.window = "growing"`, default `"fixed"` (AGZ used a fixed 500k-game window). |
| D17 | Gating matches use the ladder's opening rule (uniformly random legal moves, k = `eval.opening_moves`, seeded, deduplicated) instead of §5.8's openings sampled from the incumbent's policy | Simpler and reproducible without a policy pass; revisit if gating matches show low trajectory diversity (the report counts unique trajectories). |
| D18 | Game-ending moves are resolved exactly in the search (§5.4.10): exact values for a second pass and for moves at the cap, used as a lower bound at expansion and in place of the FPU value in selection | `search.resolve_terminal_moves`, default **true**. The paper's search has no such rule; without it the R0 models passed while behind by komi against an early-passing opponent (§11 M4). `false` restores the plain search. |
| D19 | With `pipeline.async_gate` (§6.5.1, off by default) a promoted candidate plays self-play from the iteration after next, one iteration later than in our sequential v1 | Overlaps the gate with the next self-play (−21 % for the two, measured); closer to the paper's asynchronous evaluation, which has no iteration boundary at all. Implemented 2026-09-28. |

---

## 13. Single-GPU acceleration profile (`accel`)

**Why a profile and not a redesign.** The project's first deliverable is a faithful AGZ reproduction (§1); its second is to find out how much of AGZ's compute the later literature lets one GPU do without. Both need the same infrastructure. Every technique below is therefore a **config switch, off by default**, so that `9x9.json` unchanged is the AGZ profile (§8), and M4b measures each switch against that baseline on the same frozen ladder (§6.6) at equal GPU-hours. Nothing here changes the M1–M4 contracts except the optional chunk fields (§5.6), which are added now so that the format never needs a version bump for M4b.

**Systems-level work is not a deviation** and is not gated by a switch: multi-game batching (M3b), the NN cache (D6), fp16 inference (done), AMP/`channels_last`/`torch.compile` in training (M4), and later a CUDA-Graphs or TensorRT backend if §10 shows launch overhead dominating on the small 9×9 network.

| Technique | Source | Expected effect on one GPU | Where specified |
|---|---|---|---|
| Playout cap randomization | Wu, *Accelerating Self-Play Learning in Go* (2019), §3.1 | Largest single gain in KataGo's ablation: ~3–4× more games per GPU-hour, so the value head sees many more independent outcomes while policy targets keep the full search depth | §5.4.9, §5.6, D11 |
| Auxiliary ownership + score targets | KataGo 2019 §3.3 | Large sample-efficiency gain, mostly on the value head; costs N²+4 bytes per game | §6.1, §6.2, §5.6, D13 |
| Forced playouts + policy-target pruning | KataGo 2019 §3.2 | Modest; makes root exploration honest and keeps the Dirichlet noise out of π | §5.4.9, D12 |
| Global pooling | KataGo 2019 §3.4 | Small on 9×9, expected larger on 19×19 (komi/global-count awareness) | §6.1, D14 |
| No gating | Silver et al., AlphaZero (2017/2018); KataGo | Saves the evaluation matches (~20 % of an iteration at 9×9 defaults) at the risk of promoting a regression | §6.5, D15 |
| Growing window | KataGo 2019 §4 | Small windows early (fast forgetting of random play), large later | §8, D16 |

**Ablation protocol (M4b).** Runs `R0` (all off, = the M4 run), `R1` PCR, `R2` R1 + aux heads, `R3` R2 + forced playouts and pruning, `R4` R3 + global pooling, `R5` R4 + gating off, `R6` R5 + growing window. Cumulative rather than one-at-a-time, because the KataGo ablation showed the techniques interact (PCR without the aux targets weakens the value head). Every run: the same `run_seed`, the same GPU-hour budget (measured, not games), the same frozen ladder, opening file and anchors. Reported per run: ladder Elo with intervals against GPU-hours **and** against games, self-play positions/s, held-out losses, average game length, resignation false-positive rate. The report is descriptive: a technique that does not help on 9×9 at this scale stays in the code, off, with the number that says so.

**Deferred (listed so they are not re-litigated):**
- *Gumbel AlphaZero* (Danihelka et al., 2022: Gumbel-Top-k root sampling with sequential halving, policy-improvement guarantee at 16–32 simulations). Largest potential gain but replaces root selection and the policy target; candidate for a later profile once M4b exists as its baseline.
- *Reanalyze* (Schrittwieser et al., MuZero, 2020): re-search stored positions with the current network to refresh π; inference-only compute. After M4b.
- *Concurrent self-play and training on one GPU* (KataGo's asynchronous loop). At 9×9 defaults training is ≈ 10 % of an iteration, so the upper bound on the gain is that 10 %, against a rewrite of the restart protocol (§6.5). Reconsider for 19×19 where the training share grows.
- *TensorRT / CUDA Graphs backend*, *K > 1 per game*: performance work, decided by §10 numbers (M3c, M6).
- *Score utility in MCTS, multiple rule sets, variable komi, handicap*: KataGo features outside the reproduction's scope (§1).

---

## 14. Decisions taken on the v1 open questions

| Q | Decision |
|---|---|
| Q1 threading | One evaluator, one thread, G games, K = 1. Measure in M3b before adding anything. |
| Q2 FPU | Q = 0 (paper). Parent-value FPU (`Q_init = v(s)`) is an experiment flag, swept in M4. |
| Q3 gating | Keep > 55% point-estimate gating (paper), 200 pairs on 9×9, pair-based CI reported. `--no-gating` for later comparison. |
| Q4 dependencies | Vendor `doctest.h` and `nlohmann/json.hpp`, pin versions, keep licences in-tree. |
| Q5 move cap | Included from the start (2·N²), termination reason recorded per game and its frequency reported per iteration. |
| Q6 chunk encoding | Superseded: game-based MGO2 format (snapshots + moves + sparse visits) from the start (§5.6). |
| Q7 SGFs | Save all during development; sampling becomes a config option later. |

---

## 15. Change log v1 → v2 (response to the first review)

| Review point | Change |
|---|---|
| NN cache key wrong | Key = (model_id, full 17-plane input); value = raw pre-mask output; per-evaluator; off by default. |
| Batched MCTS needs a pending protocol | Node states UNEXPANDED/PENDING/EXPANDED/TERMINAL; reservation, collision (abandon), commit, failure rollback, debug invariants. K = 1 first. |
| Evaluation matches may repeat | Separate evaluators/caches/trees per player; search-time symmetry; opening set with colour-swapped pairs; unique-trajectory count; pair-based Wilson CI. |
| Value perspectives / initial selection | Invariants 1–6; Q = 0 for unvisited edges; explicit tie-break; parent-value FPU defined as `v(s)`. |
| Two execution models | Sequential phases; `state.json`; learner checkpoint separate from best model. |
| File contracts | Atomic `.tmp` → rename; header provenance; termination reason; eviction only between phases; immutable model version dirs + `best.json`. |
| Backend contract, earlier MPS check | TorchScript pinned and its deprecation acknowledged; eval() before trace, InferenceMode, dtype cast, planes ownership; MPS test moved to M2. |
| Strength criterion misleading | Frozen ladder + anchors + Bradley–Terry Elo; promotion log labelled as gating events; throughput as hypotheses. |
| Paper-faithfulness corrections | Q = 0 init; 722-move cap; automatic resign threshold; temperature-adjusted stored π (D4); search-time symmetry; block counting convention; 64 GPU workers. |
| Rules naming, runtime size vs model, union-find liberties | "TT scoring with suicide prohibited"; passes exempt from superko; size-specific models; exact liberty bitsets. |

## 16. Change log v2 → v3 (response to the second review)

| # | Review point | Change |
|---|---|---|
| 1 | 8-step history cannot be derived from the move list (captures) | §5.1.1: `Board` keeps a ring buffer of the last 8 stone configurations (2 bitboards each, 768 B), pushed on every `play` incl. passes; `applySymmetry` transforms them; features encoded from snapshots only. History tests added (§9). |
| 2 | 19×19 data volume | §5.6: new game-based **MGO2** format from the start — packed 2-bit snapshots + moves + sparse root visit counts + result; planes/z/π derived in the loader (no rules engine in Python). ≈ 150 B (9×9) / 500 B (19×19) per position; 19×19 window ≈ 12.5 GB instead of 190 GB. |
| 3 | 19×19 compute reality | §1 non-goal, §8.1 arithmetic (10⁹ evals/iteration, 30–55 h/iteration), M6 acceptance = runs end to end, explicitly no strength claim. |
| 4 | Early over-fitting | §6.2 step cap `0.25 samples per position per iteration`; §5.6/§6.3/§6.6 5% held-out games (by seed) with per-iteration held-out value MSE / policy CE monitor; D10. |
| 5 | Gating statistical power | §5.8/§8: 9×9 uses 200 pairs (400 games); rule written explicitly as point estimate > 0.55 (paper), CI reported only. |
| 6 | L2 deviation undocumented; weight-decay factor | D9 added; §6.2 notes grad `2cθ` and the `2e-4` equivalence. |
| 7 | fp16 softmax underflow | §5.3/§6.1/§6.4: model exports logits; evaluator does fp32 softmax on the CPU; fp16 parity test uses KL + a no-underflow check (§9). |
| 8 | Superko cost and the pass trap | §5.1.1/§5.1.3: `GameHistory` with hash set owned by the game; search carries only path hashes (linear scan); non-capturing hash by one XOR; capture detection O(1) via liberty bitsets; every placement checked; the "captures only" shortcut explicitly forbidden and covered by a test. |
| 9 | Trace at batch 1 | §6.4: trace at batch 8, verify at 1 and 64. |
| 10 | Memory estimate | §5.4.2 table: 9×9 ≈ 70–140 MB, 19×19 ≈ 0.6–1.2 GB at the default G. |
| 11 | Learning signal too late | New milestone **M3a′**: 5×5 sequential end-to-end learning check before batching (§11), with `configs/5x5-smoke.json`. |
| 12 | c_puct = 1.5 with Q = 0 FPU | §5.4.3 describes the asymmetry; §8.2 defines the M4 sweep; D2 reworded. |
| — | Small items | `model_id` zero-padded/truncated (§5.6, §6.4, test); GTP `undo`/`clear_board`/`boardsize`/`komi` discard the tree (§5.4.8, §5.7); self-play `run_seed` in the chunk header and per-game `game_seed` in the game record and SGF (§5.5, §5.6). |

## 17. Change log v3 → v4 (response to the third review)

| # | Review point | Change |
|---|---|---|
| 1 | Tree-reuse visits vs simulation budget; u16 counts | §5.4.4: `simulations_this_move` (new, completed; the budget) vs `root_total_visits` (incl. inherited; the π source); `budget_includes_inherited = false` flag; **u32** counts in tree, file and loader. Chunk test now `Σ count == root_total_visits`, `== S` only with reuse off. |
| 2 | "No collision ⇒ equals sequential" is false for K > 1; failure rollback contradiction | §5.4.5: guarantees restated — K = 1 across games is exactly sequential (tested per game with per-game RNG streams); K > 1 tests check counts, signs, reservations and states only. Failure handling: clear pending/reservations, keep completed simulations, continue from the actual count; no transactional rollback. |
| 3 | MGO2 header was not 64 B; padding; temperature one-hot ties | §5.6: exact 128-byte header with per-field offsets, `header_size`, reserved bytes, field-by-field serialisation on both sides, byte-exact cross-language fixture; game record offsets fixed; `temperature_moves` and `simulations_per_move` in the header; `store_pi = "temperature"` uses the stored played move after the cutoff (§5.4.6). |
| 4 | Wilson on fractional pair scores; unregularised BT; ladder openings drift | §5.8: mean pair score + bootstrap percentile interval over pairs; unique trajectories diagnostic only. §6.6: BT fitted by MAP with Gaussian prior, connectivity and complete-separation checks; ladder opening set generated once per run and versioned. |
| 5 | Resign-threshold data not stored | §5.6: `root_value` and `root_max_q` stored for every move of every game; §5.4.7: precise false-positive definition, denominator, ≥ 100 no-resign games, range [−0.99, −0.50], else disabled. |
| 6 | Phase idempotence not guaranteed | §6.5: per-phase *plan* persisted before the phase starts (fixed `target_global_step`, self-play task id + chunk prefix + target games, candidate id, match seed, promotion decision); reconciliation rules on restart; train-to-target instead of train-for-N. |
| 7 | Komi vs model input | §6.4: `komi`, `rules_id`, `move_cap`, `feature_schema` in `model.json`; engine/trainer validate consistency; GTP `komi` rejects unsupported values; draws defined (`result = 0`, `z = 0`) for integer komi. |
| 8 | Three over-strong tests | §9: symmetry test = coordinate mapping with a known-function `FakeEvaluator`, no equivariance assumption; unique-trajectory ratio is a diagnostic; M3a′ acceptance = value sign on known-outcome positions + colour-swapped match vs the frozen initial model with a bootstrap interval. |
| — | Minor | §5.4.2 "×2" marked as an estimate, peak node count measured; §6.3 mmap + `num_workers = 0` first; §6.6 holdout rise is an alarm, not a verdict; §11/§4 `19x19-smoke.json` for M6; §5.1.1 replay-vs-snapshot wording corrected. |

## 18. Change log v4 → v5 (single-GPU acceleration profile)

| # | Topic | Change |
|---|---|---|
| 1 | Acceleration profile | New §13: motivation, technique table with sources, M4b ablation protocol, deferred list. All switches off by default; `9x9.json` unchanged is the AGZ profile. |
| 2 | Search hooks | §5.4.9: playout cap randomization, forced playouts, policy-target pruning specified against the §5.4 contract (only the stored target is pruned; tree and `root_total_visits` untouched). |
| 3 | MGO2 optional fields | §5.6: header `record_extras` bitmask (offset 100), `reduced_simulations` (102), `full_search_prob` (104); per-move `search_kind` after the moves array; per-game `final_ownership` at the end of the record; `Σ count ≤ root_total_visits` when pruned; second fixture with all extras present. Implemented in M3a′ so the format never needs a version bump. |
| 4 | Model and loss | §6.1: optional global pooling, ownership head, scalar score head, listed in `model.json`; §6.2: masked policy loss and auxiliary terms. |
| 5 | Config, tests, milestones, deviations | §8 `accel` section; §9 test row; §11 M4b; §12 D11–D16. Sections 13–16 renumbered to 14–17. |
| 6 | Goals | §1: goals 6 (single GPU for inference and training, efficiency as a baseline property) and 7 (measure what the later literature removes, via switches); non-goals reworded: KataGo-as-product, Gumbel/reanalyze, concurrency on the one GPU, human data only as an evaluation set. |
| 7 | v5 review (5 points) | (1) §6.3: the same symmetry is applied to planes, π and the ownership map; asymmetric 8-fold fixture in §9. (2) §5.6/§6.2: `aux_t = 0` for resigned games, auxiliary losses averaged over unmasked positions only. (3) §6.1/§5.3/§6.4: positional output tuple `(logits, value[, ownership[, score]])`, `model.json.outputs`, C++ checks length and the first two names. (4) §5.4.9/§6.2/§6.3/§8: PCR follows the paper — reduced-search positions are not training samples (`reduced_positions = "drop"`); `"value_only"` kept as a separately flagged experiment with CE averaged over policy-target positions. (5) §5.4.9: pruning uses frozen `Q`, frozen `N_tot`, an explicit decrement loop and the played-move floor (the removal budget was corrected in row 8). |
| 8 | v5 second review (3 points) | (1) §5.4.9: forced playouts require `0 < N(a)` — unvisited children are never forced; test with a `1e-6` prior. (2) §5.4.9: pruning budget restored to the paper's `floor(n_forced(c))` at the frozen total; the realised-forced-count `F(c)` variant is dropped. (3) §6.2/§6.3/§6.5: step cap denominator = size of the built dataset index (holdout and PCR-filtered), stored in the phase plan; zero positions skip training and re-export the incumbent. |
| 9 | M3a′ done | §11 M3a′ row: implementation, run results, criterion (1) redefined on held-out terminal positions (settled templates diagnostic); §12 D17 random gating openings. |
| 10 | M3b done | §11 M3b row; §10 measured throughput table and readings (batching 35× at G = 128; CPU side 30 % of wall; 19×19 ≈ 40 h/iteration confirms §8.1; K > 1 not needed for 9×9). |
| 11 | M3b review (2 points) | §5.4.5: a failed batch is retried with the same requests (no re-collect, no RNG consumption; test with symmetry on); the batched driver reports games that are over before their first move (move cap 0) instead of throwing. |
| 12 | M4 implemented | §5.5 `--resign-threshold`; §5.8 random anchor, `--openings-file` / `--write-openings`; §6.5 `strength` phase, `v_resign` and eviction in the selfplay plan, strength resumption rule; §6.6 implementation paragraph (files, anchors, neighbour rule, GTP referee scoring, fit, `crossrun`); §8 pipeline-only keys, 9×9 resign auto; §8.2 sweep tooling; §9 M4 tests; §11 M4 row with the `runs/9x9-r0` results (`docs/RUN_9x9_R0.md`); the monitor's training sample is a seeded random subset (§6.6). |
| 13 | M4 review (3 points) | (1) `strength.py`: a stored match report is reused only if it names the same model ids, seed, simulation budget and opening set (`report_matches`); `crossrun` resolves labels to model ids and records them, so a run that trained further gets its matches replayed. (2) `pipeline.py`: a self-play plan written before resign selection existed is migrated (threshold from `state.v_resign`, default −1) instead of failing on restart. (3) §5.8: JSON move lists encode pass as `N²` everywhere (opening files and the report), round-trip test with a pass opening. |
| 14 | M4 run review (4 points) | (1) §6.6: the rating fit is read with diagnostics — predicted vs observed residuals, prior sensitivity, an opening-adjusted fit — and the report quotes order and head-to-head results as findings, Elo only with σ and residuals; the R0 misfit is traced to early two-pass endings against iteration 1, not to the opening set. (2) §5.4.7: realised false-positive rate 5–12 % recorded as an open issue; `resign_select_scope` / `resign_fpr_target` knobs (defaults unchanged); control run without resignation listed as pending. (3) §6.6: the held-out/train comparison is declared insufficient to judge generalisation; fixed validation set (`training.fixed_holdout_iteration`). (4) Training time corrected (≈ 12 s per 1,000 steps, 9.4 s mean); `train_stats.seconds` recorded. §11 M4 row rewritten with robust findings, non-findings and open issues. |
| 15 | Pass blind spot (M4 review, user-approved fix) | §5.4.10: game-ending moves (second pass, moves at the cap) get exact values at expansion — a minimax lower bound on the node value and the exact value in place of FPU for unvisited edges; `search.resolve_terminal_moves` (default true), D18; `rootValue()`/`root_value` use the bounded value (§5.4.7/§5.6 note); three search tests plus the plain-search tests pinned to `false`. The proposal to re-state the M4 acceptance criterion was withdrawn: §11 M4 says "not met". |
| 16 | R0 ladder re-measured | `python -m mango.strength remeasure` (archives the ladder to `strength_<tag>/`, replays every match on the same openings and seeds with the current engine, test with a fake match). §11 M4: acceptance met on the re-measured ladder, original measurement recorded as invalid; `docs/RUN_9x9_R0.md` rewritten with the plain → resolved comparison. |
| 17 | Re-measurement review (3 points + 1) | Report/§11: consecutive head-to-head range corrected to 58–92 % (individual pairs moved); the top ratings are identified through the neighbour matches, not "set by the prior" — σ = 10⁶ gives 2,413, σ = 350 shrinks to 1,925 (§6.6 wording); the paper's false-positive rate is a control quantity, the label-error estimate is the counterfactual first-dip rate (`counterfactual_false_resign_rate`, logged by the pipeline, 8.8 % pooled on R0); "15 models" → 14 models + the random anchor. |
| 18 | Human interface | §6.7 `gui.py`: a tkinter window over `mango_gtp` (GTP, §5.7) for playing against a model and for analysing a game the human plays alone (`mango-analyze` overlay); no rules or search in Python, the engine is the authority; §3/§4/§9 updated. |
| 19 | GNU Go anchor | §6.6 anchor (c) realised on R0 with GNU Go 3.8 level 10 (`docs/RUN_9x9_R0.md`, GNU Go section): iteration 15 is ≈ 260 Elo below GNU Go level 10 at 200 simulations and ≈ 150 below at 800, so the ladder's 1,925 Elo over random is not an engine-level scale; whole-board margins explained (pass-at-every-turn once behind); premature-pass weakness documented as an open issue for later runs (not a search bug: the first pass is valued by the network). GTP matches parallelised (`eval.gtp_workers`, `play_gtp_match(workers=…)`, test) — but only 1.6× on one GPU (batch-1 inference from four processes), 12 min per 100 games, so after five iterations with GNU Go in the ladder (iterations 17–21, ≈ 47 % of each iteration) the user removed it from training: GNU Go is measured on demand by `scripts/vs_gnugo.py` (`external_reports`, `fit_with_external`, tests), the R0 ladder entry and its 8 matches were moved to `strength/gnugo/`. |
| 20 | M4a planned | §5.5.1 multi-threaded self-play / match drivers (T search threads + 1 evaluation thread, shared queue, per-game contracts unchanged) with a multi-process pipeline step 0; §5.9, §8 keys, §9 tests, §10 decision recorded, §11 M4a row with numeric acceptance (≥ 700 positions/s, gate ≤ 90 s) and tests. Not implemented yet. |
| 21 | M4a review (4 points) | (1) G is the global concurrency, one pending leaf per game, so threads do not enlarge batches; the benchmark reports the same-total-G and best-configuration numbers separately and the 2× is a target, not a derived gain; "GPU half idle" replaced by the measured quantity (evaluator call share 49 %). (2) T = 1 keeps the first-version code path (bit-identical); T > 1 is specified as per-game equality by `game_seed` under the deterministic evaluator, not chunk-byte equality, and is not claimed reproducible with the real fp16 network. (3) Step 0 gets a recovery protocol: per-worker plan records (quota, seed, chunk-id block) persisted before launch, restart resumes each worker from the plan, a failed worker terminates the others; the test requires a restart after a partial failure. (4) The match driver has one evaluation thread owning both evaluators with per-side buckets; the second-failure global cancel (abort flag, broadcast, `SearchTree::abort` per pending leaf, join, nothing published) is written out. |
| 22 | M4a review, second round (2 + 1) | Queue defined at request granularity: the evaluation thread takes ≤ `max_batch` requests FIFO, a thread's round may span several forwards and the thread waits for all of them (test with B smaller than one thread's share); step 0 with N = 1 is the existing pipeline path (seed, `next_chunk_id` unchanged, resumed runs unaffected), the per-worker plan applies for N ≥ 2 only; "bit-identical" for T = 1 excludes the timing and throughput summary fields. |
| 23 | M4a step 0 implemented | `selfplay.processes` in the pipeline (§5.5.1): per-worker plan records persisted before launch, restart resumes each worker's own block, a failed worker terminates the others, per-worker logs, merged summaries; N = 1 is the untouched single-process path. Tests with a fake self-play driver (§9 row). Benchmark on R0 `0022` (§11 M4a): 1×G128 299 positions/s, 3×G43 232 (same total G: smaller batches, slower), 4×G128 501 at 100k evals/s = the batch-128 forward ceiling; **≥ 700 not met at step 0**, step 1 needs batches ≥ 256 to get there. `configs/9x9.json`: `processes` 4; R0 uses it from iteration 24 (iteration 23's plan predates it). |
| 24 | M4a step 1 implemented | `EvalQueue` (request-granularity FIFO queue, rounds, one retry with identical requests, abort on the second failure or any thread error, stop), `BatchedSelfplay` and `playMatch` with `threads` / `max_batch` (T = 1 the M3b code paths), one evaluation thread per driver (match: owns both evaluators, one forward per side), `--threads` / `--max-batch`, config keys `selfplay.threads`, `selfplay.max_batch`, `eval.threads`, summary fields `threads` / `max_batch`; tests per the §9 row (per-game equality by seed for T ∈ {2, 3, 5} incl. B < ceil(G/T), invariants after every move, identical retry, two failures abort every thread and free every node, match identical for T ∈ {1, 2, 3, 4, 8} with side-specific evaluators). Measured (§11 M4a): same G = 128 gives nothing (batches halve, evaluator-bound); T8 G1024 **820 positions/s, ≥ 700 met**; gate best 117 s (T4 G400), **≤ 90 s not met** — the fixed 400 games cap the per-side batch, a faster forward would be needed. §5.5 9×9 defaults G = 1024 / T = 8 (`configs/9x9.json`: `threads` 8, `games_in_flight` 1024, `processes` 1, `eval.threads` 4); R0 from iteration 24. |
| 25 | Step 0 review (2 points) | `_run_workers` puts launch, wait and exit under one cleanup path (every started worker terminated and waited for, log handles closed, on any exception); a resumed multi-process self-play phase records whole-iteration counts rebuilt from the published chunks (`selfplay_counts_from_chunks`), the relaunch under `launch`, the launch-only fields under `incomplete`; N = 1 keeps its M4 behaviour (summary of the last launch). Tests: failed second launch cleans up; restart totals; restart with nothing left. |
| 26 | M4a step 2 planned and implemented | §5.3 CUDA fast path of `TorchEvaluator`: channels-last parameters and input (evaluator-side, `model.pt` unchanged), one CUDA graph per batch bucket with static input/output tensors, capture after 3 warm-up forwards, plain-path fallback; §8 keys `inference.channels_last` / `inference.cuda_graphs`; §9 test row; §11 numbers (≥ 1,600 positions/s, gate ≤ 90 s). Motivated by the nsys profile of 2026-09-25: 78 launches per forward, GPU busy 17–20 %, 29–30 % of GPU time in layout transposes. Implemented the same day (`TorchEvaluator` options `channelsLast`, `cudaGraphs`, `graphWarmup`; `InferenceConfig`; `--no-channels-last` / `--no-cuda-graphs`). Two findings changed the plan: a graph has to be captured on a non-default stream; channels-last is applied with fp16 only, because the NHWC fp32 kernels are TF32-class (1.4·10⁻³ on `0026`) and fp32 must stay IEEE. Measured: 725 → 1,338 positions/s (**≥ 1,600 not met**), gate 107 → 43 s (**≤ 90 s met**, closing the criterion M4a step 1 left open); launches per forward 81 → 1, GPU busy 30 % → 55 %. |
| 27 | Driver profile | Measurement before further self-play work (§5.5.1 "Driver profile"): search-thread and evaluation-thread time buckets, rounds per forward, batch histogram, 1-second timeline, `TorchEvaluator::timings()`; summary field `profile`, `mango_selfplay --profile-out`, the pipeline's `logs/selfplay_profile/`. |
| 28 | Two fixes; item 2 reverted | Graph captures exclusive against all other evaluators' GPU work (a capture invalidated by a concurrent forward was observed); result buffers owned by the drivers (latent use-after-free after an abort, self-play and match). Pinned transfers and several evaluation threads measured (no gain / +4–6 %) and reverted. |
| 29 | Overlapped evaluation designed | §6.5.1: measured self-play + gate concurrently 75.5 s against 95.0 s in sequence (−21 %), two self-play processes +5 %. `pipeline.async_ladder` (ladder jobs in one background lane, no semantic change) and `pipeline.async_gate` (phases selfplay → settle → train → monitor → export → launch_eval; a promotion takes effect one iteration later, deviation D19); background jobs are processes recorded in `state.background` before launch, polled by the main thread, relaunched on restart; drain at the end of a run. Awaiting review. |
| 30 | Overlapped-evaluation review (2 P1 + 2 P2) | (1) Decision and effect separated: `promotions` with `effective_selfplay_iteration` (i+1 sync, i+2 async, fixed at gate launch); self-play plans read it, a drain can decide early without changing data — segmented and continuous runs give the same model sequence. (2) Process lifecycle: Windows Job Object with kill-on-close for every process of the run (supervisor starts on a `go` line after assignment), POSIX process groups plus parent-death checks; a run lock and per-job locks so no job runs twice; tested with a forced kill during a ladder match. (3) `settle` as an idempotent commit keyed by (iteration, event, candidate) and ladder iteration, re-run from its first step; fault test with the ladder enqueued but the gate not cleared. (4) Ratings history keyed by fit iteration (`strength/fits/*.json`), `ratings.csv` regenerated atomically, legacy rows preserved once. |
| 31 | Overlapped-evaluation review, second round (3 boundaries) | Promotions keyed by `(gate_iteration, candidate)`, chosen by the largest `(effective_selfplay_iteration, gate_iteration)`, initial model when none — equal effective iterations after a switch are resolved by the later gate. Job locks cover the writers: POSIX `flock` descriptor inherited by every child (released only when the last holder exits), Windows nested Job Object per job (relaunch only when empty); supervisors wait for their children; `go` handshake: EOF or failed assignment means exit. Legacy CSV: an iteration with a fit file takes only the fit's rows, others keep their legacy rows; the migration is a `.tmp` copy plus rename, redone after an interruption. Tests added for supervisor death, the EOF handshake, equal effective iterations and a re-fitted legacy iteration. |
| 32 | Overlapped evaluation implemented | §6.5.1 approved and implemented: `mango/proc.py` (Job Objects, locks, polled subprocesses), `mango/bgjob.py` (supervisor), phases `settle` / `launch_eval`, promotion list with keyed migration (reproduces R0's 72 self-play models), fits published by iteration with a regenerated `ratings.csv`, `--async-ladder` / `--async-gate`, `logs/phase_times.jsonl`. Test 4(b) clarified: a killed supervisor stops the run after its tree is gone, the restart runs the job again (as test 5). A leftover copy of the pre-review items 1–2 of *Process lifecycle* removed. POSIX branch not run (Windows only). |
| 33 | Overlapped-evaluation acceptance: not met | Iterations 73–82 from R0 copies, drain included: sequential 137.1 s, `async_ladder` 125.1 s (0.91×, bound 0.85×), both switches 115.5 s (0.84×, bound 0.75×). Ladder share small (few promotions), overlap contention (gate +10 %, self-play +32 % with `async_gate`), drain 159 s; the copies diverged (different promotions and ladder workloads). Bounds kept; next step for review. |
| 34 | Overlapped evaluation measured with 10×128 | Pretrained on R0's window; 73–82: sequential 396.0 s, `async_ladder` 314.2 s (0.79×, bound 0.85× met as measured), both 310.1 s (0.78×, bound 0.75× not met). The sequential run had 7 ladder entries, the others 5; with equal ladder work ≈ 0.87× and 0.86× — neither bound met. `async_gate` gains ≈ 1 % over `async_ladder` at this size (self-play +51 % under the gate). |
| 35 | Defaults: `async_ladder` on, `async_gate` off | User decision on the measured results (bounds not met, not re-stated). Runs without a `pipeline` section, R0 included, use the ladder lane from their next iteration. |
| 36 | Overlapped-evaluation review, third round (P1 + P2) | (P1) POSIX foreground subprocesses (self-play, synchronous matches) had no parent-death protection: they now run under `mango.bgjob --exec` in their own process group with the inherited `jobs/foreground.lock`, which a restarted pipeline must acquire (signal the recorded groups, wait 60 s, else fail) before dispatching; test: pipeline killed during self-play. (P2) The fit file was taken as completion although `ratings.json` / `ratings.csv` are written after it: collection now completes the publication (`finish_fit_publication`); test: last ladder job interrupted after its fit file (fails without the fix). POSIX branch still not run. |
