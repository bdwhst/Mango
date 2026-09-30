"""Runs a pipeline in its own process with the fake self-play and match drivers, for the
tests that kill the pipeline or a supervisor (DESIGN 6.5.1, test 4).

    python pipeline_driver.py --run DIR --iterations N [--config-json JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # python/

from mango.config import merge_config  # noqa: E402
from mango.pipeline import Pipeline  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--config-json", default=None)
    args = ap.parse_args(argv)
    Pipeline._selfplay_command = lambda self: [sys.executable, str(HERE / "fake_selfplay.py")]  # type: ignore[method-assign]
    Pipeline._match_command = lambda self: [sys.executable, str(HERE / "fake_match.py")]  # type: ignore[method-assign]
    cfg = merge_config(json.loads(args.config_json)) if args.config_json else None
    p = Pipeline(args.run, cfg, args.run, device="cpu", selfplay_device="cpu", quiet=True)
    try:
        p.run_iterations(args.iterations)
    finally:
        p.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
