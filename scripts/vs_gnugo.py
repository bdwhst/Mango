"""Best model (or any model) of a run against GNU Go over GTP, on demand (DESIGN 6.6, anchor (c)).

The pipeline never plays GNU Go; run this when you want the external reference:

    python scripts/vs_gnugo.py --run runs/9x9-r0                       # best model, 50 pairs, config sims
    python scripts/vs_gnugo.py --run runs/9x9-r0 --model runs/9x9-r0/models/0015-4da2388b --sims 800
    python scripts/vs_gnugo.py --run runs/9x9-r0 --rate                # only refit: GNU Go on the ladder scale

Games use the run's fixed ladder openings (both colours), Mango's referee (`mango_gtp`,
Tromp-Taylor), and GNU Go with `--chinese-rules --capture-all-dead --play-out-aftermath` so it
captures dead stones instead of passing. Reports go to runs/<run>/strength/gnugo/. With --rate
(default on after a match) every report there that used the ladder's openings and budget is
fitted together with the ladder (`fit_with_external`), placing GNU Go on the ladder's Elo scale.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mango.config import load_config  # noqa: E402
from mango.gtp import play_gtp_match  # noqa: E402
from mango.gui import find_bin_dir, model_metadata, resolve_model  # noqa: E402
from mango.state import derive_seed, read_json  # noqa: E402
from mango.strength import (RANDOM, effective_move_cap, exe_path, external_reports, fit_with_external,  # noqa: E402
                            format_ratings, hash_name, load_openings, match_key)


def default_gnugo() -> Path:
    exe = "gnugo.exe" if platform.system() == "Windows" else "gnugo"
    return ROOT / "build" / "gnugo" / exe


def elo_of(p: float) -> float:
    return 400.0 * math.log10(p / (1.0 - p)) if 0.0 < p < 1.0 else math.copysign(float("inf"), p - 0.5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mango model vs GNU Go over GTP, on the run's ladder openings")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", help="model directory (default: the run's best model)")
    ap.add_argument("--gnugo", help=f"gnugo executable (default: {default_gnugo()})")
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=None, help="openings, each played with both colours (default: eval.ladder_pairs)")
    ap.add_argument("--sims", type=int, default=None, help="Mango simulations per move (default: search.eval_simulations)")
    ap.add_argument("--workers", type=int, default=None, help="parallel engine trios (default: eval.gtp_workers)")
    ap.add_argument("--bin", help="directory with mango_gtp")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None, help="Mango search seed (default: the ladder's seed for this pairing)")
    ap.add_argument("--rate", action="store_true", help="no match; only refit the ladder with the stored GNU Go reports")
    ap.add_argument("--no-rate", action="store_true", help="play the match but skip the ladder refit")
    args = ap.parse_args(argv)

    run = Path(args.run)
    cfg = load_config(run / "config.json")
    n, komi = int(cfg["board"]["size"]), float(cfg["board"]["komi"])
    sims = int(args.sims or cfg["search"]["eval_simulations"])
    pairs = int(args.pairs or cfg["eval"]["ladder_pairs"])
    workers = int(args.workers or cfg["eval"].get("gtp_workers", 1))
    gnugo = Path(args.gnugo) if args.gnugo else default_gnugo()
    name_g = f"gnugo-3.8-L{args.level}"
    out_dir = run / "strength" / "gnugo"
    out_dir.mkdir(parents=True, exist_ok=True)
    openings_path = run / "strength" / "openings_v1.json"

    if not args.rate:
        bin_dir = Path(args.bin) if args.bin else find_bin_dir()
        if bin_dir is None:
            print("mango_gtp not found (build with scripts/build.ps1 -Torch, or --bin)", file=sys.stderr)
            return 2
        if not gnugo.exists():
            print(f"GNU Go not found at {gnugo}: download the 3.8 Windows build into build/gnugo/ or pass --gnugo",
                  file=sys.stderr)
            return 2
        if not openings_path.exists():
            print(f"no ladder openings at {openings_path} (the run has not reached its first strength phase)", file=sys.stderr)
            return 2
        model_dir = resolve_model(run, args.model)
        meta = model_metadata(model_dir)
        model_id = str(meta["model_id"])
        state = read_json(run / "state.json")
        seed = args.seed if args.seed is not None else derive_seed(int(state["run_seed"]), hash_name(match_key(model_id, name_g)))
        mango_cmd = [exe_path(bin_dir, "mango_gtp"), "--model", model_dir, "--config", run / "config.json", "--device",
                     args.device, "--seed", seed, "--sims", sims]
        gnugo_cmd = [gnugo, "--mode", "gtp", "--level", args.level, "--chinese-rules", "--capture-all-dead",
                     "--play-out-aftermath"]
        referee = [exe_path(bin_dir, "mango_gtp"), "--size", n, "--komi", komi]
        ops = load_openings(openings_path, pairs)
        print(f"{model_id} ({sims} sims) vs {name_g}: {len(ops)} openings x 2 colours, {workers} workers ...", flush=True)
        t0 = time.time()
        r = play_gtp_match(mango_cmd, gnugo_cmd, referee, n, komi, ops, effective_move_cap(cfg), model_id, name_g,
                           seed=seed, workers=workers)
        r["simulations"] = sims
        r["seconds"] = time.time() - t0
        r["level"] = args.level
        out = out_dir / f"{model_id}__{name_g}_s{sims}_p{len(ops)}.json"
        out.write_text(json.dumps(r, indent=1), encoding="utf-8")
        p, (lo, hi) = r["mean_pair_score"], r["ci95"]
        print(f"{r['a']} vs {r['b']}: {r['wins_a']}-{r['losses_a']}-{r['draws_a']} in {r['games']} games, "
              f"mean pair score {p:.3f} [{lo:.2f}, {hi:.2f}], Elo vs GNU Go {elo_of(p):+.0f} [{elo_of(lo):+.0f}, {elo_of(hi):+.0f}], "
              f"{r['seconds']:.0f} s -> {out}")
        if args.no_rate:
            return 0

    ladder = read_json(run / "strength" / "ladder.json")
    if ladder is None:
        print("no ladder yet; nothing to rate against", file=sys.stderr)
        return 0
    expected = load_openings(openings_path, int(cfg["eval"]["ladder_pairs"])) if openings_path.exists() else None
    reports = external_reports(out_dir, name_g, expected, int(cfg["search"]["eval_simulations"]))
    if not reports:
        print(f"no {name_g} report at the ladder's openings and budget under {out_dir}; nothing to rate")
        return 0
    fit = fit_with_external(ladder, reports, name_g)
    entries = [e for e in ladder["entries"] if e["name"] != RANDOM]
    order = [{"name": RANDOM}, {"name": name_g}] + entries
    print(f"ladder fit with {name_g} ({len(reports)} matches at the ladder's openings and {cfg['search']['eval_simulations']} sims):")
    print(format_ratings(fit, order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
