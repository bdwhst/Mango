"""First 9x9 parameter sweep (docs/DESIGN.md section 8.2): c_puct x FPU at equal wall-clock.

    python scripts/sweep_9x9.py --bin build/windows-cuda/Release --root runs/sweep --hours 1.0
    python scripts/sweep_9x9.py --bin build/windows-cuda/Release --root runs/sweep --compare

Each combination gets its own run directory (restart-safe like any run) and the same
run seed; `--hours` is the budget per run. `--compare` plays the final best models of
every finished run round-robin plus the random anchor on one opening set and fits one
Bradley-Terry table (`<root>/crossrun/crossrun.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mango.config import load_config  # noqa: E402
from mango.pipeline import Pipeline  # noqa: E402
from mango.state import read_json  # noqa: E402
from mango.strength import cross_run  # noqa: E402

C_PUCT = [0.8, 1.1, 1.5, 2.5]
FPU = ["zero", "parent"]


def run_name(c: float, fpu: str) -> str:
    return f"cpuct{c:g}_fpu-{fpu}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--root", default="runs/sweep")
    ap.add_argument("--config", default="configs/9x9.json")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare", action="store_true", help="only compare the runs' best models")
    ap.add_argument("--pairs", type=int, default=50)
    args = ap.parse_args()
    root = Path(args.root)
    base = load_config(args.config)
    if not args.compare:
        for c in C_PUCT:
            for fpu in FPU:
                cfg = json.loads(json.dumps(base))
                cfg["search"]["c_puct"] = c
                cfg["search"]["fpu"] = fpu
                p = Pipeline(root / run_name(c, fpu), cfg, args.bin, args.device, None, args.seed)
                p.log(f"sweep: c_puct {c} fpu {fpu}, budget {args.hours} h")
                p.run_for_seconds(args.hours * 3600.0)
    models = {}
    for c in C_PUCT:
        for fpu in FPU:
            best = read_json(root / run_name(c, fpu) / "best.json")
            if best:
                models[run_name(c, fpu)] = root / run_name(c, fpu) / "models" / best["model_id"]
    if len(models) < 2:
        print("fewer than two runs have a best model; nothing to compare")
        return 1
    cross_run(args.bin, args.config, root / "crossrun", models, args.pairs, args.seed, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
