# First 9×9 run: `runs/9x9-r0` (M4, AGZ profile R0)

**Date:** 2026-09-23, 13:21–16:37 (3 h 16 min wall clock for 15 iterations), Windows, RTX 5080, LibTorch fp16 inference, PyTorch AMP training. Revised 2026-09-24 after review (Elo model misfit, resignation, monitoring, timings) and after the ladder was **re-measured with the search fix of DESIGN §5.4.10** (`python -m mango.strength remeasure --tag plain_search`; the original ladder is kept in `strength_plain_search/`).
**Config:** `configs/9x9.json` as of the run (the AGZ profile of DESIGN §8): 6×64 network, 200 simulations, 2,000 games per iteration, window 20,000 games, gate 200 pairs > 0.55, `resign_auto` (window scope, 5 % target), ladder 50 pairs, `ladder_every` 5, neighbours 3. Run seed 1. Self-play and gating used the plain search of §5.4.4; the ladder below was played with `resolve_terminal_moves` on (same models, same openings, same seeds).
**Generated tables:** `python scripts/run_report.py runs/9x9-r0` (everything from "Frozen ladder" on); the fits are recomputed from `strength/ladder.json` and the match reports at generation time.

## What this run shows

**Against the M4 acceptance criterion** ("Elo on the frozen ladder rises over ≥ 10 iterations with non-overlapping intervals; held-out curves logged"): **met on the re-measured ladder.** Every model entry is rated above its predecessor under every prior width tried (σ = 200–1,400 Elo); the first and last entries do not overlap (initial model 216 [156, 277], iteration 15: 1,925 [1,818, 2,032] at σ = 350); and the single-scale model now describes the results (mean standardised residual |z| = 0.90 over the 60 matches, max 3.0; ≤ 1.35 at every σ). Held-out curves are in `monitor/holdout.csv`.

It was **not met on the original measurement** (plain search): the same criterion's intervals came from a fit with mean |z| = 3.5 (max 49), because the strong models lost ≈ 10 % of their games to iteration 1 through early two-pass endings (reading 1). That measurement was invalid, not the run.

**What the Elo numbers still are not:** a prior-free scale. The top entries score 100–0 against both anchors, but the comparison graph is strongly connected through the neighbour matches, so the ratings are identified by the data: the unregularised fit converges (iteration 15: 2,413, SE 78, at σ = 10⁶). The σ = 350 prior shrinks the top by ≈ 500 Elo (1,925), and iteration 15 is 1,541 / 1,925 / 2,232 / 2,360 / 2,413 at σ = 200 / 350 / 700 / 1,400 / 10⁶ — the Laplace interval does not show that dependence. Order, non-overlap and head-to-head results are prior-robust; a single number is not, and is quoted with its σ. Graded opponents near the top (e.g. reduced-simulation copies of a model as anchors) would tighten the estimate and make runs comparable (M4b).

**Robust findings:** every ladder entry beats its predecessor head-to-head (58–92 %); iteration 15 beats iteration 1 100–0 and the random anchor 100–0 on the fixed openings; 13 of 15 candidates passed the gate (iterations 2 and 14 rejected at 0.515 and 0.532); every gate and ladder match had 100 % unique trajectories; held-out policy cross-entropy fell 4.34 → 2.97 (uniform = 4.41) and held-out value MSE 0.87 → 0.72 on the moving window.

**Not supported:** "no generalisation gap" (the run's training-sample curve was the oldest chunk of the window, reading 5; the window and the targets move, so held-out loss alone is not a fixed-task measure) and "resignation works" (realised false-positive rate 5–12 %, reading 3).

## Re-measurement with the search fix (§5.4.10)

The same 14 models and the random anchor, the same 50 fixed openings, the same per-pairing seeds; only the search resolves game-ending moves exactly (a pass answered by a pass, moves at the cap).

| Match (A vs B) | Score plain → resolved | Early two-pass games (lost by A) plain → resolved |
|---|---|---|
| `0015` vs `0001` | 0.88 → 1.00 | 13 (12) → 0 |
| `0013` vs `0001` | 0.89 → 1.00 | 11 (11) → 0 |
| `0012` vs `0001` | 0.92 → 1.00 | 8 (8) → 0 |
| `0010` vs `0001` | 0.96 → 1.00 | 4 (4) → 0 |
| `0009` vs `0001` | 0.96 → 1.00 | 4 (4) → 0 |
| `0005` vs `0001` | 0.97 → 1.00 | 3 (3) → 0 |
| `0003` vs `0001` | 0.90 → 0.89 | 1 (1) → 0 |
| all 60 matches | — | 47 → 3 (none lost) |

| Fit (plain Bradley–Terry, σ = 350) | plain search | resolved |
|---|---|---|
| mean \|z\| / max \|z\| | 3.46 / 48.8 | 0.90 / 3.0 |
| iteration 15 | 1,645 [1,548, 1,741] | 1,925 [1,818, 2,032] |
| iteration 1 | 356 [294, 418] | 288 [226, 350] |
| initial model | 113 [55, 170] | 216 [156, 277] |
| iteration 15 at σ = 200 / 700 / 1,400 / 10⁶ | 1,373 / 1,828 / 1,893 / 1,917 | 1,541 / 2,232 / 2,360 / 2,413 |

Every consecutive entry still scores above 50 % head-to-head (58–92 % after, 58–91 % before), though individual pairs moved (0008 vs 0007 0.80 → 0.65, 0011 vs 0010 0.60 → 0.71, 0015 vs 0013 0.72 → 0.67); the systematic change is in the long-range matches, which now agree with the chain. The initial model gained (vs random 0.71 → 0.87) and iteration 1 lost ground (vs the initial model 0.74 → 0.64): both were playing pass endings badly in different ways. Mean game length against iteration 1 rose by 0–11 moves (the games are now played out).

## Readings

1. **The misfit came from early two-pass endings against iteration 1 — a search blind spot, now fixed.** Against `0001` the strong models lost only games that ended by two consecutive passes before move 40 with a 0.5–2.5-point margin: iteration 15 had 13 such games and lost 12 (9 as black, passing first at moves 16–27 while behind by komi; 3 as white, accepting `0001`'s pass), iteration 13 lost 11, iteration 12 lost 8 — and no other game to `0001`. No other opponent (random included) produced early two-pass games, and iteration 15's own 2,000 self-play games contained one. Mechanism: with the plain search the pass edge is valued by the network at the position after the pass and by the replies searched from there; the opponent's answering pass, which ends the game at an exact loss, has a tiny prior and is never visited within the budget, so the value head's near-komi misjudgement (the same weakness M3a′ found on 1.5/2.5-point positions) decides. DESIGN §5.4.10 gives game-ending moves their exact value at expansion (a minimax lower bound on the node) and in selection; with it, no early two-pass loss remains (table above). The fix applies to self-play as well from the next run on; this run's training data was generated without it.
2. **The opening set is not the cause.** With a per-opening colour term the fit changes little (before the fix mean |z| 2.7 vs 3.5; after it 0.90 either way), no opening has |black advantage| > 210 Elo, black won 48.9 % (plain) / 47.9 % (resolved) of all decided ladder games.
3. **Resignation lags the model.** With `resign_auto` the threshold moved from −0.50 (iteration 2, first selection from 200 no-resign games) to −0.68 (iteration 15, ≈ 2,000 games). Selected on the window (older models' games) it under-protects the sharper new model: realised false-positive rates 5–12 % ("FP rate" column; 26/231 = 11.3 % at iteration 15, 23/187 = 12.3 % at iteration 4). That rate is the paper's control quantity (fraction of no-resign games whose eventual winner was ever below the threshold), not a label-error rate: the game would end at the *first* dip by *either* player, and usually the loser dips first. The counterfactual estimate (`counterfactual_false_resign_rate`: over the no-resign games, the first player below the threshold resigns; mislabelled iff that player went on to win) gives 3–13 % per iteration and 178 / 2,018 = 8.8 % pooled over iterations 5–15 (the iterations whose chunks are still on disk) of the games resignation would have ended. On this run almost every no-resign game triggers and the two rates nearly coincide; in general they need not. Replaying the selection on the stored games with `resign_select_scope = "latest"` (previous iteration only, ≈ 200 games) would have given lower realised rates in 7 of 10 iterations (iteration 15: 0.113 → 0.048; 13: 0.086 → 0.041) and a 2.5 % target 0.5–5.5 % everywhere. Both knobs exist now (defaults unchanged). What the wrong labels cost in strength is unknown: a control run with resignation off (≈ 2× self-play time) is the pending experiment. 70–85 % of games ended by resignation from iteration 2 on; average game length fell from 77 to ≈ 50 moves.
4. **Time budget.** Per iteration: self-play ≈ 347 s, gate ≈ 201 s, ladder ≈ 234 s (when run), training ≈ 12 s once the 1,000-step cap was active (iteration 8 on; 2–8 s before, 9.4 s mean; 11,773 steps in total). The learning rate stayed at 0.01 (`lr_step1 = 30,000`). The window was full at iteration 11; eviction removed 8 chunks per iteration from iteration 12. The re-measurement (60 matches of 100 games) took 55 min.
5. **The "Train-sample v_mse" column of this run is not comparable across iterations.** During the run the monitor's training sample was the first 4,096 positions of the index — the oldest chunk in the window, from an older model — which is why it jumps (0.21 at iteration 11). Since the review the sample is a seeded uniform random subset (`ChunkDataset.subset_entries`) and a fixed validation set (`training.fixed_holdout_iteration`, the frozen holdout games of one iteration) is evaluated every iteration; both apply to future runs only, the "Fixed-set" column is empty here.
6. **Per-iteration ladder log lines are not a curve.** The pipeline logs the new entry's rating at the fit made when it joined (iteration 7: 1,713; iteration 13: 1,663); every later match re-positions everything. `ratings.csv` keeps every fit; a report refits the whole ladder.

## Caveats

- The run was restarted once, at the start of iteration 2, to pass `--games-in-flight 128` to the match programs (the gate of iteration 1 took 576 s at the default 32; later gates ≈ 170–200 s). The restart protocol resumed the self-play phase from its plan; only the chunk in progress was regenerated.
- The match reports of the original ladder (`strength_plain_search/`) were written before the pass-encoding fix (pass = −1 in `games_detail`); the analysis tools use the result and termination fields only. The re-measured reports use the N² encoding.
- No external engine anchor (GNU Go / KataGo are not installed on this machine); every scale here is relative to the random-move anchor.
- One seed, one run: the §8.2 sweep (c_puct × FPU at equal wall-clock) is prepared (`scripts/sweep_9x9.py`, `python -m mango.strength crossrun`) but was not run; at ≥ 1 GPU-hour per configuration it is an 8+ hour job whose budget is a user decision.

Config: board 9×9, komi 7.5, 200 sims/move, 6×64 net, 2000 games/iteration, window 20000 games, gate 200 pairs > 0.55, ladder 50 pairs. Iterations completed: 15. Best model: `0015-4da2388b`.

## Frozen ladder

Two fits of the same matches, random anchor = 0, prior σ = 350 Elo, ±1.96 SE (Laplace). **Plain**: game-level Bradley–Terry (DESIGN §6.6). **Opening-adjusted**: the same model with a per-opening colour-advantage term (logit P(black wins) = r_black − r_white + o_k), which absorbs what the fixed random opening set does to the results. Both intervals are conditional on their model and prior; see the diagnostics below before reading either as an absolute scale.

| Entry | It | Promoted | Plain Elo | Plain 95 % | Adjusted Elo | Adjusted 95 % | Games | Flags |
|---|---|---|---|---|---|---|---|---|
| `random` |  |  | 0 | [0, 0] | 0 | [0, 0] | 1400 |  |
| `0000-8ebc3d52` | 0 | no | 216 | [156, 277] | 221 | [160, 282] | 500 |  |
| `0001-b8b3486c` | 1 | yes | 288 | [226, 350] | 295 | [233, 358] | 1400 |  |
| `0003-82ecb6e4` | 3 | yes | 549 | [478, 620] | 563 | [491, 634] | 600 |  |
| `0004-9c2c69ea` | 4 | yes | 802 | [723, 882] | 821 | [741, 901] | 700 |  |
| `0005-8aed0cac` | 5 | yes | 992 | [908, 1076] | 1016 | [932, 1101] | 800 |  |
| `0006-eb0b1d36` | 6 | yes | 1094 | [1008, 1180] | 1121 | [1034, 1208] | 800 |  |
| `0007-cca28adf` | 7 | yes | 1309 | [1219, 1400] | 1343 | [1251, 1435] | 800 |  |
| `0008-26e2377a` | 8 | yes | 1428 | [1335, 1521] | 1465 | [1371, 1560] | 800 |  |
| `0009-16a40970` | 9 | yes | 1519 | [1424, 1614] | 1560 | [1463, 1657] | 800 |  |
| `0010-12c7fe30` | 10 | yes | 1582 | [1485, 1678] | 1625 | [1527, 1724] | 800 |  |
| `0011-3fcfb756` | 11 | yes | 1706 | [1607, 1805] | 1754 | [1653, 1855] | 800 |  |
| `0012-0cce97c6` | 12 | yes | 1734 | [1633, 1835] | 1784 | [1680, 1887] | 700 |  |
| `0013-bb09a99e` | 13 | yes | 1823 | [1719, 1926] | 1876 | [1770, 1982] | 600 |  |
| `0015-4da2388b` | 15 | yes | 1925 | [1818, 2032] | 1983 | [1872, 2093] | 500 |  |

### Prior sensitivity

Final entry, first model entry and iteration-1 entry under different prior widths; "monotone" = every model entry rated above its predecessor; mean |z| = mean standardised residual of the fit's own predictions against the observed match scores (≈ 1 for a well-specified model). σ = 1e6 is effectively the unregularised maximum likelihood; a finite value there means the data identify the rating and the smaller σ only shrink it.

| Model | σ (Elo) | Final | Iteration 1 | Initial | Monotone | mean \|z\| | max \|z\| |
|---|---|---|---|---|---|---|---|
| plain | 200 | 1541 [1457, 1626] | 174 | 118 | yes | 1.35 | 4.5 |
| plain | 350 | 1925 [1818, 2032] | 288 | 216 | yes | 0.90 | 3.0 |
| plain | 700 | 2232 [2100, 2365] | 385 | 301 | yes | 0.74 | 4.3 |
| plain | 1400 | 2360 [2214, 2507] | 427 | 338 | yes | 0.73 | 5.4 |
| plain | 1000000 | 2413 [2260, 2566] | 445 | 354 | yes | 0.74 | 6.0 |
| opening-adjusted | 200 | 1573 [1487, 1659] | 176 | 118 | yes | 1.14 | 4.4 |
| opening-adjusted | 350 | 1983 [1872, 2093] | 295 | 221 | yes | 0.90 | 4.2 |
| opening-adjusted | 700 | 2319 [2182, 2457] | 399 | 311 | yes | 0.85 | 7.7 |
| opening-adjusted | 1400 | 2462 [2310, 2615] | 445 | 352 | yes | 0.87 | 9.9 |
| opening-adjusted | 1000000 | 2522 [2362, 2682] | 465 | 369 | yes | 0.89 | 10.9 |

### Predicted vs observed (plain fit, σ = 350), largest residuals first

| A | B | Games | Predicted score of A | Observed | z |
|---|---|---|---|---|---|
| `0007-cca28adf` | `0006-eb0b1d36` | 100 | 0.776 | 0.900 | +3.0 |
| `0010-12c7fe30` | `0007-cca28adf` | 100 | 0.827 | 0.940 | +3.0 |
| `0004-9c2c69ea` | `0003-82ecb6e4` | 100 | 0.811 | 0.920 | +2.8 |
| `0001-b8b3486c` | `random` | 100 | 0.840 | 0.930 | +2.5 |
| `0008-26e2377a` | `0001-b8b3486c` | 100 | 0.999 | 0.990 | -2.3 |
| `0000-8ebc3d52` | `random` | 100 | 0.776 | 0.870 | +2.2 |
| `0009-16a40970` | `0007-cca28adf` | 100 | 0.770 | 0.860 | +2.1 |
| `0003-82ecb6e4` | `random` | 100 | 0.959 | 1.000 | +2.1 |
| `0003-82ecb6e4` | `0001-b8b3486c` | 100 | 0.818 | 0.890 | +1.9 |
| `0006-eb0b1d36` | `0004-9c2c69ea` | 100 | 0.843 | 0.910 | +1.8 |
| `0004-9c2c69ea` | `0001-b8b3486c` | 100 | 0.951 | 0.990 | +1.8 |
| `0005-8aed0cac` | `0004-9c2c69ea` | 100 | 0.749 | 0.820 | +1.6 |

### Predicted vs observed (opening-adjusted fit, σ = 350), per colour assignment, largest first

| Black | White | Games | Predicted P(black) | Observed | z |
|---|---|---|---|---|---|
| `0001-b8b3486c` | `0008-26e2377a` | 50 | 0.001 | 0.020 | +4.2 |
| `0005-8aed0cac` | `0008-26e2377a` | 50 | 0.061 | 0.180 | +3.5 |
| `0000-8ebc3d52` | `random` | 50 | 0.736 | 0.920 | +3.0 |
| `0010-12c7fe30` | `0007-cca28adf` | 50 | 0.797 | 0.960 | +2.9 |
| `0001-b8b3486c` | `random` | 50 | 0.809 | 0.960 | +2.8 |
| `0006-eb0b1d36` | `0004-9c2c69ea` | 50 | 0.813 | 0.960 | +2.7 |
| `0007-cca28adf` | `0006-eb0b1d36` | 50 | 0.737 | 0.900 | +2.7 |
| `0001-b8b3486c` | `0000-8ebc3d52` | 50 | 0.552 | 0.720 | +2.4 |
| `0003-82ecb6e4` | `0004-9c2c69ea` | 50 | 0.161 | 0.040 | -2.4 |
| `0008-26e2377a` | `0005-8aed0cac` | 50 | 0.910 | 1.000 | +2.2 |
| `0007-cca28adf` | `0009-16a40970` | 50 | 0.195 | 0.080 | -2.1 |
| `0010-12c7fe30` | `0013-bb09a99e` | 50 | 0.167 | 0.060 | -2.1 |

### Opening colour-advantage terms (opening-adjusted fit)

50 openings; 1 with |black advantage| > 200 Elo (11). Largest:

| Opening | Black advantage (Elo) | SE | Games |
|---|---|---|---|
| 11 | -207 | 52 | 120 |
| 48 | -176 | 51 | 120 |
| 19 | -146 | 51 | 120 |
| 40 | +146 | 51 | 120 |
| 8 | -146 | 51 | 120 |
| 34 | -131 | 51 | 120 |
| 22 | -117 | 51 | 120 |
| 9 | -102 | 51 | 120 |
| 26 | -102 | 51 | 120 |
| 35 | -102 | 51 | 120 |

### Opening-set balance and pair structure

Black won 0.479 of the decided ladder games (0.5 = balanced set). 551 of 3000 pairs were split (the same colour won both games); 0 of 50 openings were split in at least half of their pairs: [].

| A | B | Pairs | Both won | Split | Both lost | Draw pairs | Black wins | Score |
|---|---|---|---|---|---|---|---|---|
| `0000-8ebc3d52` | `random` | 50 | 37 | 13 | 0 | 0 | 0.55 | 0.870 |
| `0001-b8b3486c` | `random` | 50 | 43 | 7 | 0 | 0 | 0.53 | 0.930 |
| `0001-b8b3486c` | `0000-8ebc3d52` | 50 | 22 | 20 | 8 | 0 | 0.58 | 0.640 |
| `0003-82ecb6e4` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0003-82ecb6e4` | `0001-b8b3486c` | 50 | 39 | 11 | 0 | 0 | 0.47 | 0.890 |
| `0003-82ecb6e4` | `0000-8ebc3d52` | 50 | 42 | 7 | 1 | 0 | 0.47 | 0.910 |
| `0004-9c2c69ea` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0004-9c2c69ea` | `0001-b8b3486c` | 50 | 49 | 1 | 0 | 0 | 0.51 | 0.990 |
| `0004-9c2c69ea` | `0003-82ecb6e4` | 50 | 42 | 8 | 0 | 0 | 0.46 | 0.920 |
| `0004-9c2c69ea` | `0000-8ebc3d52` | 50 | 49 | 1 | 0 | 0 | 0.51 | 0.990 |
| `0005-8aed0cac` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0005-8aed0cac` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0005-8aed0cac` | `0004-9c2c69ea` | 50 | 33 | 16 | 1 | 0 | 0.42 | 0.820 |
| `0005-8aed0cac` | `0003-82ecb6e4` | 50 | 45 | 5 | 0 | 0 | 0.53 | 0.950 |
| `0005-8aed0cac` | `0000-8ebc3d52` | 50 | 48 | 2 | 0 | 0 | 0.50 | 0.980 |
| `0006-eb0b1d36` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0006-eb0b1d36` | `0001-b8b3486c` | 50 | 49 | 1 | 0 | 0 | 0.51 | 0.990 |
| `0006-eb0b1d36` | `0005-8aed0cac` | 50 | 23 | 24 | 3 | 0 | 0.50 | 0.700 |
| `0006-eb0b1d36` | `0004-9c2c69ea` | 50 | 42 | 7 | 1 | 0 | 0.55 | 0.910 |
| `0006-eb0b1d36` | `0003-82ecb6e4` | 50 | 47 | 3 | 0 | 0 | 0.53 | 0.970 |
| `0007-cca28adf` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0007-cca28adf` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0007-cca28adf` | `0006-eb0b1d36` | 50 | 40 | 10 | 0 | 0 | 0.50 | 0.900 |
| `0007-cca28adf` | `0005-8aed0cac` | 50 | 41 | 9 | 0 | 0 | 0.49 | 0.910 |
| `0007-cca28adf` | `0004-9c2c69ea` | 50 | 48 | 2 | 0 | 0 | 0.50 | 0.980 |
| `0008-26e2377a` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0008-26e2377a` | `0001-b8b3486c` | 50 | 49 | 1 | 0 | 0 | 0.51 | 0.990 |
| `0008-26e2377a` | `0007-cca28adf` | 50 | 22 | 21 | 7 | 0 | 0.45 | 0.650 |
| `0008-26e2377a` | `0006-eb0b1d36` | 50 | 37 | 12 | 1 | 0 | 0.50 | 0.860 |
| `0008-26e2377a` | `0005-8aed0cac` | 50 | 41 | 9 | 0 | 0 | 0.59 | 0.910 |
| `0009-16a40970` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0009-16a40970` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0009-16a40970` | `0008-26e2377a` | 50 | 15 | 29 | 6 | 0 | 0.49 | 0.590 |
| `0009-16a40970` | `0007-cca28adf` | 50 | 36 | 14 | 0 | 0 | 0.44 | 0.860 |
| `0009-16a40970` | `0006-eb0b1d36` | 50 | 43 | 7 | 0 | 0 | 0.43 | 0.930 |
| `0010-12c7fe30` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0010-12c7fe30` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0010-12c7fe30` | `0009-16a40970` | 50 | 16 | 26 | 8 | 0 | 0.42 | 0.580 |
| `0010-12c7fe30` | `0008-26e2377a` | 50 | 24 | 23 | 3 | 0 | 0.35 | 0.710 |
| `0010-12c7fe30` | `0007-cca28adf` | 50 | 44 | 6 | 0 | 0 | 0.52 | 0.940 |
| `0011-3fcfb756` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0011-3fcfb756` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0011-3fcfb756` | `0010-12c7fe30` | 50 | 25 | 21 | 4 | 0 | 0.39 | 0.710 |
| `0011-3fcfb756` | `0009-16a40970` | 50 | 34 | 13 | 3 | 0 | 0.45 | 0.810 |
| `0011-3fcfb756` | `0008-26e2377a` | 50 | 33 | 14 | 3 | 0 | 0.42 | 0.800 |
| `0012-0cce97c6` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0012-0cce97c6` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0012-0cce97c6` | `0011-3fcfb756` | 50 | 15 | 31 | 4 | 0 | 0.39 | 0.610 |
| `0012-0cce97c6` | `0010-12c7fe30` | 50 | 22 | 27 | 1 | 0 | 0.35 | 0.710 |
| `0012-0cce97c6` | `0009-16a40970` | 50 | 28 | 20 | 2 | 0 | 0.44 | 0.760 |
| `0013-bb09a99e` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0013-bb09a99e` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0013-bb09a99e` | `0012-0cce97c6` | 50 | 16 | 30 | 4 | 0 | 0.40 | 0.620 |
| `0013-bb09a99e` | `0011-3fcfb756` | 50 | 21 | 26 | 3 | 0 | 0.40 | 0.680 |
| `0013-bb09a99e` | `0010-12c7fe30` | 50 | 35 | 14 | 1 | 0 | 0.40 | 0.840 |
| `0015-4da2388b` | `random` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0015-4da2388b` | `0001-b8b3486c` | 50 | 50 | 0 | 0 | 0 | 0.50 | 1.000 |
| `0015-4da2388b` | `0013-bb09a99e` | 50 | 22 | 23 | 5 | 0 | 0.37 | 0.670 |
| `0015-4da2388b` | `0012-0cce97c6` | 50 | 32 | 15 | 3 | 0 | 0.45 | 0.790 |
| `0015-4da2388b` | `0011-3fcfb756` | 50 | 26 | 22 | 2 | 0 | 0.46 | 0.740 |

## Iterations

| It | v_resign | Games | Positions | Avg len | Resigned | FP rate | Self-play s | Steps | Train s | Train loss (v / p) | Held-out v_mse | Held-out p_ce | Train-sample v_mse | Fixed-set v_mse | Gate score [CI] | Promoted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.00 | 2000 | 153801 | 76.9 | 0 | — | 395 | 143 | 2.0 | 0.850 / 4.354 | 0.871 | 4.337 | 0.699 | — | 0.775 [0.73, 0.81] | yes |
| 2 | -0.50 | 2000 | 148639 | 74.3 | 1360 | 0.050 | 361 | 281 | 3.7 | 0.765 / 4.315 | 0.804 | 4.297 | 1.058 | — | 0.515 [0.46, 0.56] | no |
| 3 | -0.50 | 2000 | 149415 | 74.7 | 1392 | 0.077 | 365 | 420 | 4.9 | 0.681 / 4.185 | 0.756 | 4.030 | 1.049 | — | 0.860 [0.82, 0.89] | yes |
| 4 | -0.50 | 2000 | 130198 | 65.1 | 1635 | 0.123 | 342 | 540 | 6.3 | 0.723 / 3.888 | 0.847 | 3.770 | 1.180 | — | 0.935 [0.91, 0.96] | yes |
| 5 | -0.52 | 2000 | 132977 | 66.5 | 1735 | 0.094 | 350 | 662 | 7.8 | 0.735 / 3.680 | 0.772 | 3.639 | 1.067 | — | 0.855 [0.82, 0.89] | yes |
| 6 | -0.54 | 2000 | 149381 | 74.7 | 1467 | 0.086 | 393 | 800 | 10.5 | 0.743 / 3.596 | 0.775 | 3.590 | 0.914 | — | 0.632 [0.58, 0.68] | yes |
| 7 | -0.57 | 2000 | 136690 | 68.3 | 1400 | 0.096 | 366 | 927 | 11.0 | 0.747 / 3.548 | 0.791 | 3.547 | 0.991 | — | 0.775 [0.73, 0.82] | yes |
| 8 | -0.60 | 2000 | 148349 | 74.2 | 1557 | 0.027 | 391 | 1000 | 11.6 | 0.753 / 3.496 | 0.785 | 3.503 | 0.828 | — | 0.772 [0.73, 0.81] | yes |
| 9 | -0.59 | 2000 | 114852 | 57.4 | 1536 | 0.075 | 315 | 1000 | 11.8 | 0.745 / 3.468 | 0.810 | 3.468 | 0.947 | — | 0.583 [0.54, 0.63] | yes |
| 10 | -0.60 | 2000 | 125060 | 62.5 | 1645 | 0.041 | 355 | 1000 | 11.9 | 0.735 / 3.427 | 0.745 | 3.423 | 0.947 | — | 0.670 [0.63, 0.71] | yes |
| 11 | -0.60 | 2000 | 122082 | 61.0 | 1621 | 0.065 | 340 | 1000 | 11.8 | 0.715 / 3.366 | 0.734 | 3.361 | 0.213 | — | 0.685 [0.64, 0.73] | yes |
| 12 | -0.61 | 2000 | 117057 | 58.5 | 1706 | 0.086 | 345 | 1000 | 11.9 | 0.728 / 3.269 | 0.753 | 3.279 | 0.295 | — | 0.593 [0.54, 0.64] | yes |
| 13 | -0.64 | 2000 | 99777 | 49.9 | 1654 | 0.086 | 290 | 1000 | 11.8 | 0.742 / 3.162 | 0.731 | 3.167 | 0.901 | — | 0.675 [0.63, 0.72] | yes |
| 14 | -0.66 | 2000 | 102062 | 51.0 | 1690 | 0.096 | 294 | 1000 | 11.8 | 0.727 / 3.060 | 0.744 | 3.058 | 0.838 | — | 0.532 [0.48, 0.58] | no |
| 15 | -0.68 | 2000 | 104064 | 52.0 | 1696 | 0.113 | 297 | 1000 | 11.7 | 0.720 / 2.973 | 0.717 | 2.972 | 0.709 | — | 0.677 [0.63, 0.72] | yes |
