"""Markdown report of a run: ladder ratings (plain and opening-adjusted Bradley-Terry
fits), model diagnostics (predicted vs observed, prior sensitivity, opening balance),
gating history, held-out curves, resign thresholds and phase timings.

    python scripts/run_report.py runs/9x9-r0 [--out docs/RUN_9x9_R0.md]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mango.state import read_json  # noqa: E402
from mango.strength import (Ladder, fit_bradley_terry_openings, match_residuals_openings,  # noqa: E402
                            opening_balance, pair_structure, predicted_vs_observed, prior_sensitivity)

SIGMAS = (200.0, 350.0, 700.0, 1400.0, 1e6)  # 1e6 ~ the unregularised MLE


def fmt(x, nd=0):
    return "—" if x is None else f"{x:.{nd}f}"


def interval(r):
    return f"[{fmt(r['low'])}, {fmt(r['high'])}]" if r["rated"] else "—"


def ladder_sections(run: Path, cfg: dict, st: dict) -> list[str]:
    ladder = Ladder(run, cfg, ".", st["run_seed"])
    entries = ladder.data["entries"]
    order = ladder.model_order()
    players = [e["name"] for e in entries]
    results = ladder.results()
    plain = ladder.fit(record=False)
    games = ladder.games()
    adj = fit_bradley_terry_openings(players, games) if games else None
    lines = ["## Frozen ladder", "",
             "Two fits of the same matches, random anchor = 0, prior σ = 350 Elo, ±1.96 SE (Laplace). "
             "**Plain**: game-level Bradley–Terry (DESIGN §6.6). **Opening-adjusted**: the same model with a per-opening "
             "colour-advantage term (logit P(black wins) = r_black − r_white + o_k), which absorbs what the fixed random "
             "opening set does to the results. Both intervals are conditional on their model and prior; see the "
             "diagnostics below before reading either as an absolute scale.", "",
             "| Entry | It | Promoted | Plain Elo | Plain 95 % | Adjusted Elo | Adjusted 95 % | Games | Flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        r = plain["ratings"][e["name"]]
        a = adj["ratings"][e["name"]] if adj else None
        flags = ([] if r["rated"] else ["unrated"]) + (["separated"] if r["separated"] else [])
        it = "" if e.get("iteration") is None else str(e["iteration"])
        prom = "" if e["kind"] != "model" else ("yes" if e.get("promoted") else "no")
        lines.append(f"| `{e['name']}` | {it} | {prom} | {fmt(r['elo'])} | {interval(r)} | "
                     f"{fmt(a['elo']) if a else '—'} | {interval(a) if a else '—'} | {r['games']} | {', '.join(flags)} |")
    lines.append("")

    # --- prior sensitivity (both models)
    lines += ["### Prior sensitivity", "",
              "Final entry, first model entry and iteration-1 entry under different prior widths; \"monotone\" = every "
              "model entry rated above its predecessor; mean |z| = mean standardised residual of the fit's own "
              "predictions against the observed match scores (≈ 1 for a well-specified model). σ = 1e6 is "
              "effectively the unregularised maximum likelihood; a finite value there means the data identify "
              "the rating and the smaller σ only shrink it.", "",
              "| Model | σ (Elo) | Final | Iteration 1 | Initial | Monotone | mean \\|z\\| | max \\|z\\| |",
              "|---|---|---|---|---|---|---|---|"]
    first, it1, last = order[0], (order[1] if len(order) > 1 else order[0]), order[-1]
    for s in prior_sensitivity(players, results, order, SIGMAS):
        lines.append(f"| plain | {s['sigma']:.0f} | {fmt(s['elo'][last])} [{fmt(s['low'][last])}, {fmt(s['high'][last])}] | "
                     f"{fmt(s['elo'][it1])} | {fmt(s['elo'][first])} | {'yes' if s['monotone'] else 'no'} | "
                     f"{s['mean_abs_z']:.2f} | {s['max_abs_z']:.1f} |")
    if games:
        for sigma in SIGMAS:
            f = fit_bradley_terry_openings(players, games, prior_sigma_elo=sigma)
            elos = [f["ratings"][p]["elo"] for p in order]
            mono = all(e is not None for e in elos) and all(elos[i] < elos[i + 1] for i in range(len(elos) - 1))
            res = match_residuals_openings(f)
            mz = sum(abs(x["z"]) for x in res) / len(res) if res else 0.0
            lines.append(f"| opening-adjusted | {sigma:.0f} | {fmt(f['ratings'][last]['elo'])} "
                         f"[{fmt(f['ratings'][last]['low'])}, {fmt(f['ratings'][last]['high'])}] | "
                         f"{fmt(f['ratings'][it1]['elo'])} | {fmt(f['ratings'][first]['elo'])} | {'yes' if mono else 'no'} | "
                         f"{mz:.2f} | {max((abs(x['z']) for x in res), default=0.0):.1f} |")
    lines.append("")

    # --- predicted vs observed (plain fit), worst first
    rows = sorted(predicted_vs_observed(plain, results), key=lambda r: -abs(r["z"]))
    lines += ["### Predicted vs observed (plain fit, σ = 350), largest residuals first", "",
              "| A | B | Games | Predicted score of A | Observed | z |", "|---|---|---|---|---|---|"]
    for r in rows[:12]:
        lines.append(f"| `{r['a']}` | `{r['b']}` | {r['games']} | {r['predicted']:.3f} | {r['observed']:.3f} | {r['z']:+.1f} |")
    lines.append("")
    if adj:
        rows2 = sorted(match_residuals_openings(adj), key=lambda r: -abs(r["z"]))
        lines += ["### Predicted vs observed (opening-adjusted fit, σ = 350), per colour assignment, largest first", "",
                  "| Black | White | Games | Predicted P(black) | Observed | z |", "|---|---|---|---|---|---|"]
        for r in rows2[:12]:
            lines.append(f"| `{r['black']}` | `{r['white']}` | {r['games']} | {r['predicted']:.3f} | {r['observed']:.3f} | {r['z']:+.1f} |")
        lines.append("")
        # opening terms
        terms = sorted(adj["openings"].items(), key=lambda kv: -abs(kv[1]["black_elo"]))
        big = [k for k, v in terms if abs(v["black_elo"]) > 200.0]
        lines += ["### Opening colour-advantage terms (opening-adjusted fit)", "",
                  f"{len(terms)} openings; {len(big)} with |black advantage| > 200 Elo "
                  f"({', '.join(str(k) for k in big[:20])}{'…' if len(big) > 20 else ''}). Largest:", "",
                  "| Opening | Black advantage (Elo) | SE | Games |", "|---|---|---|---|"]
        for k, v in terms[:10]:
            lines.append(f"| {k} | {v['black_elo']:+.0f} | {v['se_elo']:.0f} | {v['games']} |")
        lines.append("")

    # --- opening balance + pair structure
    reports = [ladder.report_of(m) for m in ladder.data["matches"]]
    reports = [r for r in reports if r and "games_detail" in r]
    if reports:
        ob = opening_balance(reports)
        lines += ["### Opening-set balance and pair structure", "",
                  f"Black won {ob['black_win_rate']:.3f} of the decided ladder games (0.5 = balanced set). "
                  f"{ob['colour_decided_pairs']} of {ob['pairs']} pairs were split (the same colour won both games); "
                  f"{len(ob['colour_decided_openings'])} of {len(ob['openings'])} openings were split in at least half of "
                  f"their pairs: {ob['colour_decided_openings']}.", "",
                  "| A | B | Pairs | Both won | Split | Both lost | Draw pairs | Black wins | Score |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, rep in zip(ladder.data["matches"], [ladder.report_of(m) for m in ladder.data["matches"]]):
            if not rep or "games_detail" not in rep:
                continue
            ps = pair_structure(rep)
            lines.append(f"| `{m['a']}` | `{m['b']}` | {ps['pairs']} | {ps['both_won']} | {ps['split']} | {ps['both_lost']} | "
                         f"{ps['draw_pairs']} | {fmt(ps['black_wins'], 2)} | {m['mean_pair_score']:.3f} |")
        lines.append("")
    return lines


def iteration_section(run: Path, st: dict) -> list[str]:
    hold = {}
    hp = run / "monitor" / "holdout.csv"
    if hp.exists():
        with open(hp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                hold[int(r["iteration"])] = r
    events = {e["iteration"]: e for e in st.get("events", []) if e.get("event") == "gate"}
    gates = {}
    for p in sorted((run / "matches").glob("*.json")):
        j = read_json(p)
        if j and "mean_pair_score" in j:
            gates[int(p.stem)] = j
    sp = st.get("selfplay_stats", {})
    tr = st.get("train_stats", {})
    # Training seconds: from train_stats (new runs) or the log line "train: done in Xs".
    train_s = {}
    log = run / "logs" / "pipeline.log"
    if log.exists():
        for line in open(log, encoding="utf-8"):
            m = re.search(r"\[it (\d+)\] train: done in ([\d.]+)s", line)
            if m:
                train_s[int(m.group(1))] = float(m.group(2))
    lines = ["## Iterations", "",
             "| It | v_resign | Games | Positions | Avg len | Resigned | FP rate | Self-play s | Steps | Train s | "
             "Train loss (v / p) | Held-out v_mse | Held-out p_ce | Train-sample v_mse | Fixed-set v_mse | Gate score [CI] | Promoted |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for it in range(1, st["iteration"]):
        s = sp.get(str(it), {})
        t = tr.get(str(it), {})
        h = hold.get(it, {})
        g = gates.get(it)
        term = s.get("terminations", {})
        fpr = (s.get("resign_false_positive") or {}).get("false_positive_rate")
        gate = f"{g['mean_pair_score']:.3f} [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}]" if g else "—"
        prom = events.get(it, {}).get("promoted")
        loss = f"{t['value']:.3f} / {t['policy']:.3f}" if t else "—"
        secs = t.get("seconds", train_s.get(it))
        fixed = h.get("fixed_value_mse")
        fixed = fmt(float(fixed), 3) if fixed not in (None, "", "nan") else "—"
        lines.append(f"| {it} | {fmt(s.get('resign_threshold'), 2)} | {s.get('games', '—')} | {s.get('positions', '—')} | "
                     f"{fmt(s.get('avg_game_length'), 1)} | {term.get('resign', '—')} | {fmt(fpr, 3)} | {fmt(s.get('seconds'))} | "
                     f"{fmt(t.get('steps')) if t else '—'} | {fmt(secs, 1)} | {loss} | "
                     f"{fmt(float(h['holdout_value_mse']), 3) if h else '—'} | {fmt(float(h['holdout_policy_ce']), 3) if h else '—'} | "
                     f"{fmt(float(h['train_value_mse']), 3) if h else '—'} | {fixed} | {gate} | "
                     f"{'' if prom is None else ('yes' if prom else 'no')} |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = read_json(run / "config.json")
    st = read_json(run / "state.json")
    lines = [f"# Run report: `{run.name}`", ""]
    lines.append(f"Config: board {cfg['board']['size']}×{cfg['board']['size']}, komi {cfg['board']['komi']}, "
                 f"{cfg['search']['simulations']} sims/move, {cfg['training']['res_blocks']}×{cfg['training']['filters']} net, "
                 f"{cfg['selfplay']['games_per_iteration']} games/iteration, window {cfg['training']['window_games']} games, "
                 f"gate {cfg['eval']['pairs']} pairs > {cfg['eval']['gate_threshold']}, ladder {cfg['eval']['ladder_pairs']} pairs. "
                 f"Iterations completed: {st['iteration'] - 1}. Best model: `{st['best_model_id']}`.")
    lines.append("")
    if (run / "strength" / "ladder.json").exists():
        lines += ladder_sections(run, cfg, st)
    lines += iteration_section(run, st)
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
