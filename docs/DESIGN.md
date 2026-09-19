# Mango: AlphaGo Zero Reproduction — Design Document

Status: **v4 — contracts fixed after third review; implementation of M1–M3a approved and started**
Date: 2026-09-17 (v1–v4)
Target paper: Silver et al., *Mastering the game of Go without human knowledge*, Nature 550, 354–359 (2017) — "AlphaGo Zero" (AGZ).

Change logs: §14 (v1 → v2), §15 (v2 → v3), §16 (v3 → v4).

---

## 1. Goals and non-goals

### Goals

1. A faithful, readable reproduction of the AlphaGo Zero algorithm: one residual network with policy and value heads, PUCT tree search with no rollouts, and a closed loop of self-play → training → gated evaluation.
2. Runs on two platforms from one code base: **Windows + NVIDIA CUDA** (dev machine: RTX 5080, CUDA 13.2, MSVC 2026) and **macOS Apple Silicon + Metal (MPS)**.
3. Board size is a **runtime** parameter of the engine (5 ≤ N ≤ 19). A trained model is nevertheless **size-specific** (both heads contain fully connected layers whose width depends on N²); the engine validates the model's declared size against the requested board size. The first milestone is a complete 9×9 loop; 19×19 is the same code with a different config and its own model.
4. Every component is testable in isolation: rules, search, network export, data format.
5. **Search correctness and evaluation validity are the primary risks** and get the most design and test attention (§§5.4, 5.5, 5.8, 9).

### Non-goals (for now)

- Hand-written CUDA / Metal inference kernels. LibTorch is the inference backend; a custom backend can be added later behind the same C++ interface.
- KataGo-style extensions (ownership head, score head, multiple rule sets, variable komi, analysis features).
- Distributed self-play across machines. Single machine, single GPU.
- Human game data (SL bootstrap) — that is AlphaGo 2016, not AGZ.
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
│   GTP front-end ──► GoGui / Sabaki / gogui-twogtp                              │
└────────────────────────────────────────────────────────────────────────────────┘
                 ▲ immutable model version (model.pt + model.json)      │ *.mgo chunks (atomic publish)
                 │                                                      ▼
┌────────────────────────────── Python (python/mango/) ──────────────────────────┐
│   model.py (ResNet)  ─►  train.py (loss, SGD)  ─►  export.py (TorchScript)     │
│   data.py (chunk reader: assembles planes from snapshots, symmetries, holdout) │
│   pipeline.py (sequential: selfplay → train → export → gate → promote)         │
│   strength.py (frozen-checkpoint ladder + external anchors → Elo)              │
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
- The exported model returns **logits and value**; the evaluator computes the **softmax in fp32** (logits are cast to fp32 first) so that fp16 execution cannot underflow small priors to exactly 0. Legal-move masking and renormalisation are done by the search, never by the evaluator, because legality depends on history the 17 planes do not contain.
- The evaluator is not required to be thread-safe; one evaluator is used from one thread.

**`TorchEvaluator`**

- Loads a **model version directory** (§6.4): `model.pt` (TorchScript) + `model.json` (`model_id`, board size, plane count, feature schema, residual-block and filter counts, export dtype, git hash, export time). Selects the device at runtime: `--device auto` → CUDA if `torch::cuda::is_available()`, else MPS if `torch::mps::is_available()`, else CPU.
- Runs under `c10::InferenceMode`; the module is in eval mode as exported (BN uses running statistics). On CUDA the module is converted to fp16 by default and the input tensor is cast to the module's dtype; fp32 on MPS and CPU; `--fp32` forces fp32 everywhere for parity checks.
- Input planes are copied into a host tensor `[B,17,N,N]` (uint8), moved to the device, converted to the model dtype; one forward call per batch returns `(logits [B,N²+1], value [B])`; logits are brought back as fp32 and soft-maxed on the CPU (a few µs per row).
- **TorchScript is deprecated upstream** but still shipped and loadable in the pinned PyTorch 2.14. We pin to it deliberately for the prototype. **Known risk (observed in M2):** PyTorch 2.14 emits `FutureWarning: torch.jit.trace is not supported in Python 3.14+ and may break` on the dev machine's Python 3.14 venv; tracing and loading currently work and the round-trip tests pass. If tracing breaks on a future PyTorch, the fallback is a Python 3.12 venv for the export step (the C++ loader does not depend on the Python version). `torch.export` / AOTInductor are *not* drop-in replacements for `torch::jit::load` and would require a different C++ loader; that is a separate future task. The exact deployment path (trace in eval mode → save → `torch::jit::load` → forward under InferenceMode on CUDA and on MPS) is verified by `test_torch_eval` on both platforms in **M2**.

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
- **Failure (first version: no transactional rollback).** If `evaluate` throws, every PENDING node in the batch reverts to `UNEXPANDED` and all reservations are removed. Simulations that completed *within the same step without the network* (terminal backups) are **kept** — their visits are already committed and are valid. Lazily created `UNEXPANDED` children with `N = 0` are also kept; they are harmless. `simulations_this_move` reflects only completed simulations, so the search simply continues from the actual count. The exception propagates; the driver retries the step once, then aborts the process loudly. Restoring the exact pre-batch tree would require journaling every modification and is not attempted.
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

**Data for the automatic threshold.** Every move of every game stores `root_value = v(s₀)` and `root_max_q` in the chunk (§5.6), so `r_t` is available offline without re-running any search.

**Selection rule (as in the paper, made precise).** After each iteration, over the **no-resign games of the current window**: for a candidate threshold `t`, a game is a *false positive* if its eventual winner had `r_t < t` at any of their moves. `false_positive_rate(t) = (#false-positive games) / (#no-resign games)`. `v_resign` is the largest `t` in the allowed range `[−0.99, −0.50]` (step 0.01) with `false_positive_rate(t) < 0.05`. If fewer than **100 no-resign games** are in the window, or no `t` in the range qualifies, resignation stays **disabled** for the next iteration. The chosen `t`, the sample size and the rate are logged per iteration.

#### 5.4.8 Tree reuse

After a move is played the chosen child becomes the root, keeping its subtree and statistics. Root noise is re-applied to the new root's priors. The tree is **never shared between two different players/models** (§5.8). **GTP `undo`, `clear_board`, `boardsize` and `komi` discard the whole tree** (tree reuse only ever follows a forward move).

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

With K = 1 per game there can be no within-game collision, so the batch is exactly the set of games with a pending leaf. Defaults for 9×9: `G = 128`, `S = 200`. After M3b measures throughput (§10) we add, in this order and only if needed: K > 1 per game (§5.4.5), multiple threads each with its own evaluator, a shared cross-thread batching queue.

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
| 100 | 28 | u8[28] | reserved (zero) |

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
| — | T·8 | f32, f32 | per move: `root_value` v(s₀), `root_max_q` (both in the mover's perspective; §5.4.7) |
| — | variable | | per move: `u32 root_total_visits; u16 nnz; nnz × (u16 action, u32 count)` — root visit counts including inherited visits; Σ count == root_total_visits |

Derived by the loader, never stored: planes for position `t` come from snapshots `t, t−1, …, t−7` (zeros before 0) split into mover/opponent by parity (**black moves at even t**: no handicap, and passes alternate turns); `z_t = result · (+1 if t even else −1)`; `π_t` from the visit counts with the chosen `store_pi` rule (§5.4.6).

**Cross-language fixture.** `tests/fixtures/mgo2_fixture.bin` is a hand-constructed chunk with two short 5×5 games and known byte content; both the C++ reader and `chunk.py` must parse it to the same values, and both writers must reproduce it byte for byte from the same in-memory games.

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

### 5.9 Threading model summary (first version)

| Component | Threads |
|---|---|
| Self-play | 1 thread, 1 evaluator, G games, K = 1 (§5.5). |
| Match | 1 thread, 2 evaluators, G game pairs. |
| GTP | Single game, 1 evaluator, K = 1 (K > 1 once §5.4.5 is tested). |
| Training | Python; DataLoader workers for chunk reading and augmentation. |

Self-play, training and evaluation run **sequentially** in the first version (§6.5).

---

## 6. Python training (`python/mango`)

### 6.1 Model (`model.py`)

`AGZNet(board_size, res_blocks, filters, planes=17)` exactly as in §2: initial convolutional block **plus** `res_blocks` residual blocks (config counts residual blocks only), policy head, value head. Forward returns `(logits, v)` in both training and export; softmax is the evaluator's job (§5.3).

### 6.2 Loss, optimiser and how much to train per iteration (`train.py`)

- `loss = mse(v, z) + cross_entropy(logits, π) + c · Σ‖θ‖²` with `c = 1e-4` as an explicit term over conv and linear **weights** (BN affine parameters and biases excluded — deviation D9). Implementation note: the gradient of `c‖θ‖²` is `2cθ`; if this is ever replaced by `SGD(weight_decay=…)` the equivalent value is `2e-4`, not `1e-4`.
- SGD, momentum 0.9, LR schedule by global step from the config (9×9 default: 0.01 for 0–30k, 0.001 for 30k–60k, 0.0001 after). Batch 256.
- **Steps per iteration are bounded by the data.** `steps = min(train_steps, ceil(samples_per_position × window_positions / batch))` with `samples_per_position = 0.25` per iteration by default. On iteration 1 (≈ 2,000 games ≈ 140k positions) that is ≈ 140 steps rather than 1,000, so early networks never see the tiny first window for multiple epochs; at steady state (window 20k games ≈ 1.4M positions) the cap is inactive and each position is sampled ≈ 0.25 × 10 iterations ≈ 2.5 times over its lifetime. The paper's value head over-fitting on consecutive positions of the same game (noted in AlphaGo 2016) is the reason for this cap and for the held-out monitor (§6.6).
- Mixed precision (`torch.autocast` + `GradScaler`) on CUDA; fp32 on MPS.
- **Learner checkpoint** (`learner/ckpt_<step>.pt`): model weights, optimizer state, GradScaler state, global step, RNG states (torch, CUDA, NumPy, Python), the config hash. Distinct from the best self-play model: a failed gate never touches the learner state, and training always resumes from the latest learner checkpoint.

### 6.3 Data pipeline (`data.py`, `chunk.py`)

- The **replay manifest** (`replay/manifest.json`) lists chunk files in creation order with `chunk_id`, `model_id`, game count and position count. The window is the most recent `window_games` games by this manifest (9×9 default 20,000).
- `ChunkDataset` **memory-maps** each chunk in the window (`np.memmap`, read-only) and builds a small index of (game offset, T, holdout flag) once; the OS page cache is then shared by all readers, so a 19×19 window of ~17 GB is never duplicated per process. Sampling is uniform over **training positions** (holdout games excluded): pick a game with probability ∝ T, then t uniform in [0, T). For the sampled (game, t) it unpacks snapshots t…t−7, builds the 17 planes and `z_t`, builds `π_t` from the visit counts, applies one of the 8 symmetries to planes and to the board part of π (pass untouched), and yields `(planes float32, π, z)`. The first version runs with `num_workers = 0` (assembly is cheap NumPy indexing); workers are added only if the GPU is measurably starved, and then they share the maps rather than copy them.
- `HoldoutDataset` iterates every position of the holdout games once (no symmetry), for the monitor in §6.6.
- **Windows file locking.** Chunks are only ever deleted by the pipeline process, between phases, when no trainer process is alive (§6.5). The trainer never deletes; DataLoader workers are torn down at the end of the training phase. Eviction of chunks that fall out of the window happens at the start of the next self-play phase.
- Symmetry index tables are precomputed per board size and shared with the C++ `applySymmetry` via a fixture test.

### 6.4 Export and model versions (`export.py`, `docs/MODEL_FORMAT.md`)

- `model.eval()` is required before tracing; export runs `torch.jit.trace` on CPU, fp32, input `[B,17,N,N]` float32, **traced at batch 8** and checked at batch 1 and 64 (tracing at batch 1 risks specialising the batch dimension into view/reshape ops). The exported graph returns `(logits, value)`.
- **Metadata carries the game contract.** `model.json` contains, besides the fields in §5.3: `komi`, `rules_id`, `move_cap`, `feature_schema`, `board_size`. The model has no komi input, so its value head is only meaningful for the komi and rules it was trained under. The engine refuses to run a model whose `board_size`, `feature_schema` or `rules_id` differ from the run config, and refuses (or warns with `--allow-komi-mismatch`) when `komi` differs; the trainer refuses chunks whose komi/rules differ from the learner's. GTP `komi` accepts only the model's komi and answers `? unsupported komi <x>; this model was trained with <k>` otherwise. If a run is configured with an integer komi, draws are possible: `result = 0` and `z = 0` for every position of a drawn game.
- A **model version** is an immutable directory `models/<model_id>/` with `model.pt`, `model.json` and `weights.pt` (eager state dict for the ladder / re-export). `model_id = <iteration:04d>-<8 hex of sha256 over the parameter bytes>` (the digest is over the state dict, not over `model.pt`, whose zip serialisation is not guaranteed byte-stable; ≤ 31 chars, fits the zero-padded `char[32]` header field). Directories are written to `models/.tmp-<id>-<pid>` and renamed into place; re-exporting identical weights is a no-op.
- `best.json` holds the current best `model_id`; it is updated last, after the version directory exists.
- Round-trip tests: Python `torch.jit.load` vs eager, and C++ `test_torch_eval` on the same fixture (CUDA fp16/fp32, MPS fp32, CPU fp32) with stored reference numbers (§9 thresholds).

### 6.5 Orchestration (`pipeline.py`, sequential)

```
runs/<name>/
  config.json  state.json  models/<model_id>/  best.json  learner/ckpt_*.pt
  replay/manifest.json  replay/chunk_*.mgo  sgf/<iteration>/  matches/<iteration>.json
  strength/ladder.json  monitor/holdout.csv  logs/
```

**Restart protocol.** `state.json` is the single authority and is rewritten atomically **before a phase starts** (with the phase's *plan*) and again when it ends. Other files (checkpoints, model directories, match reports, `best.json`) are outputs that the restart logic **reconciles against the plan**, never trusts on their own — atomic writes of individual files do not give cross-file transactions.

```
state.json = {
  run_seed, iteration, best_model_id, v_resign,
  phase: "selfplay" | "train" | "monitor" | "export" | "gate" | "promote" | "done",
  plan: {                                   # written before the phase starts; fixed until it ends
    selfplay: { task_id: "<iteration>", model_id, target_games, chunk_prefix: "chunk_<iteration>_" },
    train:    { input_ckpt, start_global_step, target_global_step },
    export:   { input_ckpt, candidate_model_id },
    gate:     { candidate_model_id, incumbent_model_id, report: "matches/<iteration>.json", match_seed },
    promote:  { candidate_model_id, decision: null | true | false }
  }
}
```

Resumption rules per phase:

- **selfplay**: count games in *published* chunks whose name starts with `chunk_prefix`; run self-play for `target_games − existing` more games (new chunk ids continue the sequence). `.tmp` chunks are deleted.
- **train**: load the newest learner checkpoint whose `global_step ≤ target_global_step` (this may be newer than `input_ckpt` if a checkpoint was saved before the crash); train **until `global_step == target_global_step`**, never "for N more steps". If a checkpoint already reaches the target, the phase is complete.
- **export**: if `models/<candidate_model_id>/` exists and validates, done; otherwise re-export from `input_ckpt` (deterministic, same id).
- **gate**: if the report exists and names the same candidate/incumbent/seed, done; otherwise re-run the match (seeded → same openings).
- **promote**: `decision` is computed from the report and written to the plan first; then `best.json` is written; then the phase is marked done. On restart: if `best.json` already equals the candidate and `decision == true`, done; if `decision` is recorded but `best.json` differs, redo the pointer write; if `decision` is null, recompute it from the report.

Training steps per iteration are therefore expressed as a **fixed target step** (`target_global_step = start + bounded_steps`, §6.2), so a crash after a checkpoint but before the state update can neither repeat nor skip work.

```
iteration i:
  1. selfplay:  mango_selfplay --model models/<best> --seed <run_seed, i>  until games_per_iteration new games are published
  2. train:     bounded steps (§6.2) from the window, resuming from learner/ckpt_*.pt; save new learner ckpt
  3. monitor:   held-out value MSE and policy CE (§6.6) → monitor/holdout.csv
  4. export:    models/<candidate_id>/ (immutable)
  5. gate:      mango_match candidate vs best (§5.8) → matches/<i>.json
  6. promote:   if pair win-rate > 0.55 → best.json = candidate  (logged as a gating event, not as Elo)
  7. strength:  every `ladder_every` iterations run strength.py (§6.6)
```

`--no-gating` (AlphaZero variant) is available for comparison later.

### 6.6 Monitoring and strength measurement (`strength.py`, `monitor/`)

**Held-out monitor (every iteration, cheap, earliest warning).** On the holdout games of the current window (5% of games, never trained on) the trainer reports value MSE and policy cross-entropy, next to the same quantities on a training sample. A held-out value MSE that rises while the training MSE keeps falling is an **alarm**, not a verdict: self-play data and targets drift from iteration to iteration, so a gap between the two curves can also come from distribution shift. The alarm triggers a look at the ladder and at the resign/termination statistics; it does not by itself decide anything. Both curves are recorded per iteration in `monitor/holdout.csv`.

**Strength (frozen ladder).** The promotion log is **not** a strength curve. Strength is measured separately:

- A **frozen ladder**: every promoted model plus every `ladder_every`-th candidate regardless of gating.
- **Anchors** with fixed identity: (a) a uniform-random legal-move player, (b) the first promoted model, (c) an external engine over GTP — GNU Go level 10 on 9×9 (later KataGo with a small net as a stronger anchor).
- **Fixed opening set.** The ladder uses a versioned opening file `strength/openings_v1.json`, generated **once** per run (from uniformly random legal moves, k = 2–4, deduplicated, seeded) and never regenerated, so that strength measurements taken at different times see the same opening distribution. Gating matches (§5.8) may sample openings from the incumbent; the ladder must not.
- Each new ladder entry plays fixed-budget matches (same simulations, no noise, fixed opening set + search symmetry) against the anchors and against 2–3 nearest ladder neighbours. All results go into one table.
- **Rating fit.** Bradley–Terry ratings are fitted by **MAP with a Gaussian prior** on every rating (prior σ = 350 Elo, equivalently a ridge penalty in natural units), with anchor (a) pinned at 0. An unregularised MLE has no finite solution when a player wins or loses every game or when the comparison graph is disconnected, which is exactly the situation early ladders are in. Before fitting, `strength.py` checks that the comparison graph is **connected** (otherwise the disconnected component is reported unrated) and reports **complete separation** (any player with all wins or all losses) so those ratings are read as prior-dominated. Intervals come from the Laplace approximation at the MAP or from a bootstrap over games.
- This fitted curve is what is reported as "strength". "Beats random 100%" is a sanity check only. The M4 acceptance criterion is Elo growth on this ladder over ≥ 10 iterations with non-overlapping intervals between the first and last entries.

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
| resign threshold / no-resign fraction | disabled | disabled at first; then auto / 10% | auto / 10% | auto / 10% |
| komi | 7.5 | 7.5 | 7.5 | 7.5 |
| games per iteration | 100 | 2,000 | 5,000 | 25,000 |
| window (games) | 1,000 | 20,000 | 100,000 | 500,000 |
| batch / max train steps per iteration | 64 / 100 | 256 / 1,000 | 256 / 2,000 | 2048 / 1,000 |
| samples per position per iteration (cap) | 0.25 | 0.25 | 0.25 | — |
| holdout games | 5% | 5% | 5% | — |
| eval pairs (games) / gate | 20 (40) / > 55% | 200 (400) / > 55% | 100 (200) / > 55% | 400 games / > 55% |
| G games in flight × K leaves | 16 × 1 | 128 × 1 | 64 × 1 | — |
| NN cache | off | off | off | (not described) |

### 8.1 Compute arithmetic for 19×19 (why it is a "runs end to end" config)

Per iteration: 5,000 games × ~250 positions × 800 simulations = **10⁹ network evaluations**. A 20×128 network on one RTX 5080 will plausibly reach 5–10k evaluations/s end to end (to be measured), i.e. **30–55 hours per iteration**, and a meaningful 19×19 run needs hundreds of iterations. The 19×19 config therefore proves that the code path (formats, memory, pipeline) works at full size; it is **not** expected to produce a strong player on this hardware, and M6's acceptance is phrased accordingly (§11). Reaching real 19×19 strength would need a much faster backend, more GPUs, and/or a reduced budget (fewer sims, smaller net), which is out of scope.

### 8.2 First parameter sweep (M4)

Once the 9×9 loop runs, sweep jointly: `c_puct ∈ {0.8, 1.1, 1.5, 2.5}` × FPU ∈ {Q = 0, parent value}, judged on the frozen ladder (§6.6) at equal wall-clock budget.

---

## 9. Testing strategy

| Layer | Tests |
|---|---|
| Rules (C++) | Captures, multi-chain captures with shared liberties (merge = OR), suicide, simple ko, positional superko incl. triple ko and the **pass-then-non-capturing-repeat** case, pass exempt from superko, pass/pass end, move cap end, area scoring with seki and dame, Zobrist consistency, **fuzz vs `RefBoard`** for 10k random games (legality, liberties, score, hash). |
| History snapshots (C++) | After captures the snapshot ring buffer equals `RefBoard`'s stored positions for the last 8 steps; passes duplicate the snapshot; `applySymmetry` transforms all 8 snapshots consistently; planes from a symmetrised board equal symmetrised planes. |
| Features (C++ ↔ Python) | Same random game: C++ `encodeFeatures` at every t equals the Python loader's assembly from the chunk's snapshots, byte for byte, including `z_t` parity. |
| Search, sequential (C++, `FakeEvaluator`, symmetry off, cache off) | Backup signs; terminal handling (two-pass and move-cap leaves, correct sign); first selection at a fresh node = highest prior; PUCT prefers higher Q at equal N; convergence to a forced win in a 5×5 capture position; π sums to 1; tree reuse keeps child statistics and frees siblings; resignation decision uses root perspective; `visitCount == 1 + ΣN` after every simulation; superko-illegal moves never expanded (path hashes respected). |
| Search, batched (C++) | **K = 1 across games**: every game in a batched run equals its standalone sequential run (deterministic `FakeEvaluator`, one RNG stream per game). **K > 1**: pending set has no duplicate nodes; collisions counted and leave no reservations; backup signs/counts correct; commit restores invariants; after a simulated evaluator failure no node is PENDING, no reservation remains, completed (terminal) simulations are retained and `simulations_this_move` equals the number of completed simulations. Tree reuse: `root_total_visits` after a reused move equals inherited + S. |
| Symmetry | Planes and π mapped by the same table in C++ and Python; **coordinate mapping** tested with a `FakeEvaluator` whose policy is a known function of the point (forward transform then inverse transform is the identity; a policy peaked at point p under symmetry s maps back to p). No test assumes the real network is equivariant — CNN + FC heads are not. |
| NN backend (M2, both platforms) | Fixture model loads and runs on CUDA (fp16 & fp32), MPS (fp32), CPU (fp32). fp32: max-abs 1e-4 on policy and value. **fp16**: KL(p_fp32 ‖ p_fp16) < 1e-3 per row, value abs diff < 1e-2, and **no action with p_fp32 > 1e-4 has p_fp16 = 0** (guards the underflow the fp32-softmax design is meant to prevent). Batch of 1 equals batch of 64 row-wise. |
| Chunk format | Byte-exact parse and re-serialisation of `mgo2_fixture.bin` in C++ and Python; C++ write → Python read round-trip and vice versa on random games; `.tmp` ignored; header offsets and `header_size` validated; `model_id` zero-padding/truncation; holdout tag; per move `Σ count == root_total_visits` and, with tree reuse **off**, `root_total_visits == S`; with tree reuse on, `root_total_visits ≥ S`. |
| Model versions | Export round-trip; immutable directory + `best.json` pointer update order; trace at batch 8 works at batch 1 and 64. |
| Training (Python) | One-step overfit decreases loss; symmetry augmentation keeps π normalised; step cap (§6.2) computed correctly for small windows; LR schedule boundaries; learner checkpoint restore reproduces the next step bit-for-bit on CPU; holdout games never appear in training samples. |
| Match | Unique-trajectory count is reported (diagnostic, no threshold); pair scores computed correctly on a fixture with a decisive pair, a split pair and a double loss; bootstrap interval reproducible under a fixed seed; promotion rule is the point estimate. Ladder: opening file generated once and reused; Bradley–Terry MAP fit on a fixture with a fully separated player returns finite ratings and flags the separation; a disconnected fixture is reported unrated. |
| End-to-end | `pipeline.py --smoke` (5×5 config): finishes in < 2 minutes on CPU; runs on both platforms. |

---

## 10. Performance: hypotheses to be measured in M3b

Nothing in this section is an acceptance criterion. M3b measures and reports:

- games/s and positions/s of the full self-play loop (board copy + selection + encoding + transfer + forward + softmax + backup + chunk writing), at G = 128, K = 1, S = 200, fp16.
- **Actual average game length** of early self-play (random-ish networks on 9×9 with dead stones not removed can run far longer than the ~70 moves assumed here, up to the 162-move cap).
- Forward time per batch on the GPU versus wall-clock per step, to see whether the CPU side is the bottleneck.
- The same numbers for the 19×19 config at a handful of games, to replace the estimate in §8.1 with a measurement.

Only after these numbers exist do we decide among: K > 1 (§5.4.5), multiple threads, a shared batching queue, or a hand-written backend.

---

## 11. Milestones

| M | Deliverable | Acceptance |
|---|---|---|
| M0 | Environment (done on Windows): venv, PyTorch 2.14+cu130, MSVC 2026 + LibTorch smoke test, setup scripts | smoke binary runs a CUDA conv |
| M1 | **Done 2026-09-18 (Windows).** `cpp/core`: `Board` with snapshots, `GameHistory`, rules, features, symmetry, SGF, Zobrist, `RefBoard`; `mango_gtp` playing random legal moves; C++ tests | rules/history/feature/symmetry tests pass; differential fuzz (5/7/9 full, 13/19 sampled; `MANGO_FUZZ_SCALE` multiplies) passes at 5×; scripted GTP session verified (GoGui not yet tried) |
| M2 | **Done 2026-09-18 on Windows; MPS pending.** `model.py`, `export.py` (logits out, batch-8 trace), model versions; `TorchEvaluator` with fp32 softmax; fixture `cpp/tests/fixtures/model_5x5_v1` + `python/tests/make_fixture.py` | CPU fp32, CUDA fp32 and CUDA fp16 match the PyTorch reference; the MPS case is compiled and **skipped until run on the Mac** |
| M3a | **Done 2026-09-18.** Sequential search: K = 1, cache off, resignation off, symmetry off; collect/commit protocol implemented (K > 1 untested); `MctsPlayer` for GTP; fake-evaluator tests: first selection, backup signs, terminal (two-pass and cap), PUCT ordering, 3-ply ladder convergence, tree reuse and visit accounting, resignation perspective, superko along the path, invariants after every simulation | all sequential search tests pass |
| **M3a′** | **5×5 sequential end-to-end learning check**: chunk writer (MGO2), `chunk.py`, minimal `train.py`/`export.py`/loop with the sequential engine, K = 1, no batching, no throughput work; a few dozen iterations on `5x5-smoke.json` | the loop-level contracts are exercised (z parity, π symmetry, snapshot assembly, holdout) and learning is shown two ways: (1) the value head on a fixed set of **known-outcome 5×5 positions** (near-terminal positions with an unambiguous area score) has the correct sign on ≥ 95% of them; (2) the final model beats the **frozen initial model** in a colour-swapped match on the fixed opening set with a bootstrap interval excluding 0.5. Held-out losses are recorded. |
| M3b | Multi-game batching (K = 1 across G games), pending protocol, atomic chunk publish, SGF, seeds; `mango_match` with opening set + pair statistics; **throughput measurement** (§10) | batched == sequential when collision-free; chunks validated by Python; numbers reported |
| M3c | (only if M3b numbers require it) K > 1 within a game, search-time symmetry on, resignation with auto threshold | batched protocol tests pass with collisions > 0 |
| M4 | Full `train.py` (step cap, AMP, learner ckpt), `data.py`, `pipeline.py`, `strength.py`, held-out monitor; first 9×9 run; sweep §8.2 | Elo on the frozen ladder rises over ≥ 10 iterations with non-overlapping intervals; held-out curves logged |
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

---

## 13. Decisions taken on the v1 open questions

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

## 14. Change log v1 → v2 (response to the first review)

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

## 15. Change log v2 → v3 (response to the second review)

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

## 16. Change log v3 → v4 (response to the third review)

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
