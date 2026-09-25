# First 9×9 run: `runs/9x9-r0` (M4, AGZ profile R0)

**Date:** 2026-09-23, 13:21–16:37 (3 h 16 min wall clock for 15 iterations), Windows, RTX 5080, LibTorch fp16 inference, PyTorch AMP training. Revised 2026-09-24 after review (Elo model misfit, resignation, monitoring, timings) and after the ladder was **re-measured with the search fix of DESIGN §5.4.10** (`python -m mango.strength remeasure --tag plain_search`; the original ladder is kept in `strength_plain_search/`).
**Config:** `configs/9x9.json` as of the run (the AGZ profile of DESIGN §8): 6×64 network, 200 simulations, 2,000 games per iteration, window 20,000 games, gate 200 pairs > 0.55, `resign_auto` (window scope, 5 % target), ladder 50 pairs, `ladder_every` 5, neighbours 3. Run seed 1. Self-play and gating used the plain search of §5.4.4; the ladder below was played with `resolve_terminal_moves` on (same models, same openings, same seeds).
**Generated tables:** `python scripts/run_report.py runs/9x9-r0` (everything from "Frozen ladder" on); the fits are recomputed from `strength/ladder.json` and the match reports at generation time.

## What this run shows

**Against the M4 acceptance criterion** ("Elo on the frozen ladder rises over ≥ 10 iterations with non-overlapping intervals; held-out curves logged"): **met on the re-measured ladder.** Every model entry is rated above its predecessor under every prior width tried (σ = 200–1,400 Elo); the first and last entries do not overlap (initial model 216 [156, 277], iteration 15: 1,925 [1,818, 2,032] at σ = 350); and the single-scale model now describes the results (mean standardised residual |z| = 0.90 over the 60 matches, max 3.0; ≤ 1.35 at every σ). Held-out curves are in `monitor/holdout.csv`.

It was **not met on the original measurement** (plain search): the same criterion's intervals came from a fit with mean |z| = 3.5 (max 49), because the strong models lost ≈ 10 % of their games to iteration 1 through early two-pass endings (reading 1). That measurement was invalid, not the run.

**What the Elo numbers still are not:** a prior-free scale. The top entries score 100–0 against both anchors, but the comparison graph is strongly connected through the neighbour matches, so the ratings are identified by the data: the unregularised fit converges (iteration 15: 2,413, SE 78, at σ = 10⁶). The σ = 350 prior shrinks the top by ≈ 500 Elo (1,925), and iteration 15 is 1,541 / 1,925 / 2,232 / 2,360 / 2,413 at σ = 200 / 350 / 700 / 1,400 / 10⁶ — the Laplace interval does not show that dependence. Order, non-overlap and head-to-head results are prior-robust; a single number is not, and is quoted with its σ. Graded opponents near the top (e.g. reduced-simulation copies of a model as anchors) would tighten the estimate and make runs comparable (M4b).

**Robust findings:** every ladder entry beats its predecessor head-to-head (58–92 %); iteration 15 beats iteration 1 100–0 and the random anchor 100–0 on the fixed openings; 13 of 15 candidates passed the gate (iterations 2 and 14 rejected at 0.515 and 0.532); every gate and ladder match had 100 % unique trajectories; held-out policy cross-entropy fell 4.34 → 2.97 (uniform = 4.41) and held-out value MSE 0.87 → 0.72 on the moving window.

**External reference (added 2026-09-24):** against GNU Go 3.8 level 10, iteration 15 scores 18–82 at the ladder budget (200 simulations, ≈ −260 Elo) and 30–70 at 800 simulations (≈ −150 Elo); see the GNU Go section. The ladder's 1,925 Elo is a scale over random moves, not a human or engine level.

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

## Against GNU Go 3.8, level 10 (external anchor, 2026-09-24)

GNU Go 3.8 (the prebuilt Windows build from gnugo.baduk.org, kept outside the repository in `build/gnugo/`) run as `gnugo --mode gtp --level 10 --chinese-rules --capture-all-dead --play-out-aftermath`, so that it keeps playing until every dead stone is captured and every neutral point filled — the only way a Tromp–Taylor referee (a third `mango_gtp`, no dead-stone judgement) scores its games fairly. Iteration 15 (`0015`) played the ladder's 50 fixed openings with both colours (`play_gtp_match`, DESIGN §6.6), first at the ladder budget and then at four times it:

| Mango budget | Result (W–L) | as black / as white | mean pair score [95 % bootstrap] | Elo vs GNU Go | pairs 2–0 / 1–1 / 0–2 | whole-board losses |
|---|---|---|---|---|---|---|
| 200 simulations | **18–82** | 10/50 · 8/50 | 0.18 [0.10, 0.26] | −263 [−382, −182] | 3 / 12 / 35 | 54 |
| 800 simulations | **30–70** | 11/50 · 19/50 | 0.30 [0.21, 0.39] | −147 [−230, −78] | 4 / 22 / 24 | 41 |

(`strength/gnugo/L10_s200.json`, `L10_s800.json`; Elo from the pair score, `400 log10(p / (1 − p))`.) All 200 games ended by two passes; no resignations, no illegal moves.

**The margins are an artefact, the results are not.** "Whole-board" losses (73.5 or 88.5 points, i.e. every Mango stone captured) are 54 of the 82 losses at 200 simulations. Replaying every game with the referee shows what happens: once the value head judges the game lost, Mango passes at every turn; GNU Go, which does not pass while dead stones remain, captures everything. At Mango's first pass it was already ≥ 10 points behind (Tromp–Taylor count, dead stones counted alive) in 44 of those 54 games, < 10 behind in 7, ahead in 3. So the whole-board margins inflate how badly the games were lost, not how many were lost.

**A real weakness underneath: passing early on an open board.** Mango's first pass came with ≥ 20 empty points in 49 of the 100 games at 200 simulations (median first pass at move 86 with 19 empty points; earliest at move 34 with 49 empty points), and in 23 of those it was ahead on the count at that moment. It went on to lose 10 of these 23 (e.g. game 47: white, pass at move 38 with 46 empty points and every white group unsettled, final 73.5-point loss). In self-play both sides are the same network: a pass made while ahead is answered by a pass, the game ends, the label confirms the pass — so the value head has learnt that "ahead ⇒ passing is safe" and the policy prior on pass is high in won positions. §5.4.10 only makes the *game-ending* pass exact; a first pass against an opponent who plays on is valued by the network. Self-play should correct this over time (the losing side's search finds that continuing beats passing, which then punishes the early pass), but not by iteration 15. Bound on its cost here: ≤ 10 of the 82 losses at 200 simulations; the other losses were decided on the board.

**Budget helps but does not close the gap.** Four times the search moves the score from 0.18 to 0.30 (≈ +115 Elo, intervals overlapping at the edges), mostly as white (8 → 19 of 50), and reduces whole-board losses (54 → 41) and early passes (49 → 38 games with ≥ 20 empty points at the first pass). At 800 simulations 26 of the 38 open-board first passes were made while ahead and 23 of those games were won.

**Reading.** On 9×9 the best model of this run is clearly below GNU Go level 10 at the ladder budget — roughly 260 Elo at 200 simulations, 150 at 800 — which puts the ladder's 1,925 Elo over random into perspective: it is a scale relative to uniformly random moves, not to any human or engine reference. GNU Go is now the external anchor DESIGN §6.6 asked for.

**GNU Go on the ladder scale.** Added to `ladder.json` as the GTP anchor and played, with the ladder's own derived seeds and the same 50 openings, against three entries: `0001` (first promoted) 0–100, `0008` 1–99, `0015` 17–83 (a second, independently seeded sample of the 18–82 above). The Bradley–Terry MAP fit (σ = 350, random = 0) then rates **GNU Go 3.8 level 10 at 2,123 [1,994, 2,251]** against **1,877 [1,773, 1,980] for iteration 15** — 246 Elo apart, in line with the direct-match estimate; the whole ladder shifted down slightly with the three new matches (iteration 15 was 1,925 before). GNU Go's own interval is wide because it is separated from two of its three opponents (0–100, 1–99 carry almost no information) and is held up by the prior and the one informative match. Iterations 17–21 played it inside the pipeline (results in the Continuation section); from iteration 22 GNU Go is measured on demand only (`scripts/vs_gnugo.py`), and its ladder-scale rating comes from that script's refit rather than from `ladder.json`.

## Continuation (from 2026-09-24 evening)

The run was resumed from its state (iteration 16, self-play phase, best model `0015`) with `--hours 8` after the GNU Go anchor matches. Two things differ from iterations 1–15 and are part of the record: (1) self-play now uses the §5.4.10 search (game-ending moves resolved exactly), i.e. the training data from iteration 16 on comes from a slightly different search than before — DESIGN §5.4.10 said "from the next run on"; the user chose to continue this run instead, so iterations ≤ 15 and ≥ 16 are not one homogeneous data-generating process; (2) for iterations 17–21 the ladder had GNU Go 3.8 level 10 as an anchor and every new entry played 100 games against it (`0017` 29–71, `0018` 20–80, `0019` 27–73, `0020` 31–69, `0021` see `strength/gnugo/`); that cost ≈ 12 min of a 26-min iteration (one game per process at batch-1 inference; four parallel trios gave only 1.6×), so from iteration 22 the pipeline no longer plays GNU Go: the entry and its 8 matches were moved out of `ladder.json` into `strength/gnugo/` (`ladder_matches.json`, the reports) and the ladder fit is again the plain one. GNU Go is measured on demand with `scripts/vs_gnugo.py`, which also refits the ladder with the stored GNU Go reports as an extra player. The window (20,000 games) still contains iterations 6–15 data at the restart.

## Caveats

- The run was restarted once, at the start of iteration 2, to pass `--games-in-flight 128` to the match programs (the gate of iteration 1 took 576 s at the default 32; later gates ≈ 170–200 s). The restart protocol resumed the self-play phase from its plan; only the chunk in progress was regenerated.
- The match reports of the original ladder (`strength_plain_search/`) were written before the pass-encoding fix (pass = −1 in `games_detail`); the analysis tools use the result and termination fields only. The re-measured reports use the N² encoding.
- The ladder scale is relative to the random-move anchor; the GNU Go section above is the only external reference (KataGo is not installed). GNU Go's own games were scored by Mango's referee under Tromp–Taylor rules with `--capture-all-dead`, so its passes are late; the same rule applied to Mango is what produces the whole-board margins.
- One seed, one run: the §8.2 sweep (c_puct × FPU at equal wall-clock) is prepared (`scripts/sweep_9x9.py`, `python -m mango.strength crossrun`) but was not run; at ≥ 1 GPU-hour per configuration it is an 8+ hour job whose budget is a user decision.

Config: board 9×9, komi 7.5, 200 sims/move, 6×64 net, 2000 games/iteration, window 20000 games, gate 200 pairs > 0.55, ladder 50 pairs. Iterations completed: 15. Best model: `0015-4da2388b`.

## Frozen ladder

Two fits of the same matches, random anchor = 0, prior σ = 350 Elo, ±1.96 SE (Laplace). **Plain**: game-level Bradley–Terry (DESIGN §6.6). **Opening-adjusted**: the same model with a per-opening colour-advantage term (logit P(black wins) = r_black − r_white + o_k), which absorbs what the fixed random opening set does to the results. Both intervals are conditional on their model and prior; see the diagnostics below before reading either as an absolute scale.

| Entry | It | Promoted | Plain Elo | Plain 95 % | Adjusted Elo | Adjusted 95 % | Games | Flags |
|---|---|---|---|---|---|---|---|---|
| `random` |  |  | 0 | [0, 0] | 0 | [0, 0] | 1400 |  |
| `gnugo-3.8-L10` |  |  | 2123 | [1994, 2251] | 2183 | [2051, 2315] | 300 |  |
| `0000-8ebc3d52` | 0 | no | 207 | [147, 266] | 211 | [151, 271] | 500 |  |
| `0001-b8b3486c` | 1 | yes | 277 | [216, 338] | 284 | [223, 345] | 1500 |  |
| `0003-82ecb6e4` | 3 | yes | 533 | [464, 603] | 546 | [476, 616] | 600 |  |
| `0004-9c2c69ea` | 4 | yes | 781 | [704, 858] | 798 | [720, 876] | 700 |  |
| `0005-8aed0cac` | 5 | yes | 966 | [885, 1048] | 989 | [907, 1071] | 800 |  |
| `0006-eb0b1d36` | 6 | yes | 1066 | [983, 1150] | 1091 | [1007, 1176] | 800 |  |
| `0007-cca28adf` | 7 | yes | 1277 | [1189, 1365] | 1308 | [1219, 1398] | 800 |  |
| `0008-26e2377a` | 8 | yes | 1392 | [1302, 1483] | 1427 | [1336, 1519] | 900 |  |
| `0009-16a40970` | 9 | yes | 1483 | [1390, 1575] | 1521 | [1427, 1615] | 800 |  |
| `0010-12c7fe30` | 10 | yes | 1544 | [1450, 1638] | 1584 | [1489, 1680] | 800 |  |
| `0011-3fcfb756` | 11 | yes | 1665 | [1569, 1762] | 1710 | [1612, 1809] | 800 |  |
| `0012-0cce97c6` | 12 | yes | 1693 | [1595, 1791] | 1739 | [1639, 1839] | 700 |  |
| `0013-bb09a99e` | 13 | yes | 1781 | [1680, 1881] | 1830 | [1727, 1933] | 600 |  |
| `0015-4da2388b` | 15 | yes | 1877 | [1773, 1980] | 1929 | [1823, 2036] | 600 |  |

### Prior sensitivity

Final entry, first model entry and iteration-1 entry under different prior widths; "monotone" = every model entry rated above its predecessor; mean |z| = mean standardised residual of the fit's own predictions against the observed match scores (≈ 1 for a well-specified model). σ = 1e6 is effectively the unregularised maximum likelihood; a finite value there means the data identify the rating and the smaller σ only shrink it.

| Model | σ (Elo) | Final | Iteration 1 | Initial | Monotone | mean \|z\| | max \|z\| |
|---|---|---|---|---|---|---|---|
| plain | 200 | 1472 [1392, 1552] | 160 | 106 | yes | 1.42 | 4.7 |
| plain | 350 | 1877 [1773, 1980] | 277 | 207 | yes | 0.91 | 3.1 |
| plain | 700 | 2210 [2080, 2340] | 379 | 296 | yes | 0.72 | 4.1 |
| plain | 1400 | 2353 [2208, 2498] | 425 | 336 | yes | 0.70 | 5.3 |
| plain | 1000000 | 2413 [2260, 2565] | 445 | 354 | yes | 0.71 | 6.0 |
| opening-adjusted | 200 | 1500 [1418, 1581] | 162 | 107 | yes | 1.16 | 4.6 |
| opening-adjusted | 350 | 1929 [1823, 2036] | 284 | 211 | yes | 0.90 | 3.8 |
| opening-adjusted | 700 | 2293 [2158, 2428] | 393 | 306 | yes | 0.84 | 7.4 |
| opening-adjusted | 1400 | 2453 [2301, 2604] | 443 | 350 | yes | 0.86 | 9.8 |
| opening-adjusted | 1000000 | 2520 [2361, 2679] | 465 | 369 | yes | 0.87 | 11.0 |

### Predicted vs observed (plain fit, σ = 350), largest residuals first

| A | B | Games | Predicted score of A | Observed | z |
|---|---|---|---|---|---|
| `0010-12c7fe30` | `0007-cca28adf` | 100 | 0.823 | 0.940 | +3.1 |
| `0007-cca28adf` | `0006-eb0b1d36` | 100 | 0.771 | 0.900 | +3.1 |
| `0004-9c2c69ea` | `0003-82ecb6e4` | 100 | 0.806 | 0.920 | +2.9 |
| `0001-b8b3486c` | `random` | 100 | 0.831 | 0.930 | +2.6 |
| `0000-8ebc3d52` | `random` | 100 | 0.767 | 0.870 | +2.4 |
| `0009-16a40970` | `0007-cca28adf` | 100 | 0.765 | 0.860 | +2.2 |
| `0003-82ecb6e4` | `random` | 100 | 0.956 | 1.000 | +2.2 |
| `0008-26e2377a` | `0001-b8b3486c` | 100 | 0.998 | 0.990 | -2.1 |
| `0006-eb0b1d36` | `0004-9c2c69ea` | 100 | 0.838 | 0.910 | +2.0 |
| `0003-82ecb6e4` | `0001-b8b3486c` | 100 | 0.814 | 0.890 | +2.0 |
| `0004-9c2c69ea` | `0001-b8b3486c` | 100 | 0.948 | 0.990 | +1.9 |
| `0005-8aed0cac` | `0004-9c2c69ea` | 100 | 0.744 | 0.820 | +1.7 |

### Predicted vs observed (opening-adjusted fit, σ = 350), per colour assignment, largest first

| Black | White | Games | Predicted P(black) | Observed | z |
|---|---|---|---|---|---|
| `0001-b8b3486c` | `0008-26e2377a` | 50 | 0.001 | 0.020 | +3.8 |
| `0005-8aed0cac` | `0008-26e2377a` | 50 | 0.064 | 0.180 | +3.3 |
| `0000-8ebc3d52` | `random` | 50 | 0.725 | 0.920 | +3.1 |
| `0010-12c7fe30` | `0007-cca28adf` | 50 | 0.791 | 0.960 | +3.0 |
| `0001-b8b3486c` | `random` | 50 | 0.799 | 0.960 | +2.9 |
| `0006-eb0b1d36` | `0004-9c2c69ea` | 50 | 0.807 | 0.960 | +2.8 |
| `0007-cca28adf` | `0006-eb0b1d36` | 50 | 0.732 | 0.900 | +2.7 |
| `0001-b8b3486c` | `0000-8ebc3d52` | 50 | 0.549 | 0.720 | +2.5 |
| `0003-82ecb6e4` | `0004-9c2c69ea` | 50 | 0.165 | 0.040 | -2.4 |
| `0008-26e2377a` | `0005-8aed0cac` | 50 | 0.905 | 1.000 | +2.3 |
| `0007-cca28adf` | `0009-16a40970` | 50 | 0.199 | 0.080 | -2.1 |
| `0010-12c7fe30` | `0013-bb09a99e` | 50 | 0.171 | 0.060 | -2.1 |

### Opening colour-advantage terms (opening-adjusted fit)

50 openings; 0 with |black advantage| > 200 Elo (). Largest:

| Opening | Black advantage (Elo) | SE | Games |
|---|---|---|---|
| 11 | -198 | 51 | 126 |
| 48 | -183 | 50 | 126 |
| 19 | -154 | 50 | 126 |
| 40 | +140 | 50 | 126 |
| 8 | -140 | 50 | 126 |
| 34 | -126 | 50 | 126 |
| 9 | -111 | 50 | 126 |
| 22 | -111 | 50 | 126 |
| 12 | -97 | 49 | 126 |
| 35 | -97 | 49 | 126 |

### Opening-set balance and pair structure

Black won 0.479 of the decided ladder games (0.5 = balanced set). 565 of 3150 pairs were split (the same colour won both games); 0 of 50 openings were split in at least half of their pairs: [].

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
| `0001-b8b3486c` | `gnugo-3.8-L10` | 50 | 0 | 0 | 50 | 0 | 0.50 | 0.000 |
| `0008-26e2377a` | `gnugo-3.8-L10` | 50 | 0 | 1 | 49 | 0 | 0.51 | 0.010 |
| `0015-4da2388b` | `gnugo-3.8-L10` | 50 | 2 | 13 | 35 | 0 | 0.43 | 0.170 |

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
