"""A scripted stand-in for mango_match (DESIGN 6.5.1 tests): accepts the gate's and the
ladder's command lines, writes a valid report and records every launch. A control file
<run>/fake_match.json (run = the directory of --config) scripts it:

    {"gate": {"<candidate iteration>": mean pair score, ...},   # default 0.5
     "sleep": {"gate": s, "ladder": s},                         # before the report is written
     "fail": {"gate": exit code, "ladder": exit code, "gate_<iteration>": exit code}}

A gate match is one without --openings-file; the candidate's iteration is the prefix of
its model id. Every launch writes <run>/fake_match_log/<ns>_<pid>.json with its pid, the
report path, the start time and, when it completes, the end time.

    python fake_match.py --a DIR|random --b DIR|random --config C --pairs P --seed S --out R ...
    python fake_match.py --write-openings F --config C --pairs P --openings K --seed S
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def name(arg: str) -> str:
    return "random" if arg == "random" else Path(arg).name


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    config = Path(opts["config"])
    run = config.parent
    cfg = json.loads(config.read_text(encoding="utf-8"))
    pairs = int(opts["pairs"])
    if "write-openings" in opts:
        write_json(Path(opts["write-openings"]), [[k % 25, (k + 7) % 25] for k in range(pairs)])
        return 0
    control = {}
    if (run / "fake_match.json").exists():
        control = json.loads((run / "fake_match.json").read_text(encoding="utf-8"))
    kind = "ladder" if "openings-file" in opts else "gate"
    a, b = name(opts["a"]), name(opts["b"])
    out = Path(opts["out"])
    log_dir = run / "fake_match_log"
    log_dir.mkdir(exist_ok=True)
    record = {"pid": os.getpid(), "kind": kind, "a": a, "b": b, "out": str(out), "start": time.time(), "end": None}
    record_path = log_dir / f"{time.time_ns()}_{os.getpid()}.json"
    write_json(record_path, record)
    deadline = time.time() + float(control.get("sleep", {}).get(kind, 0.0))
    while time.time() < deadline:
        time.sleep(0.05)
    iteration = int(a.split("-")[0]) if kind == "gate" else None
    fail = control.get("fail", {})
    code = fail.get(kind) or (fail.get(f"gate_{iteration}") if kind == "gate" else None)
    if code:
        sys.stderr.write(f"fake_match: failing on purpose ({kind})\n")
        return int(code)
    score = 0.5
    if kind == "gate":
        score = float(control.get("gate", {}).get(str(iteration), 0.5))
    games = 2 * pairs
    wins = int(round(score * games))
    report = {"a": a, "b": b, "seed": int(opts["seed"]), "pairs": pairs, "games": games, "wins_a": wins,
              "losses_a": games - wins, "draws_a": 0, "mean_pair_score": score, "ci95": [score - 0.1, score + 0.1],
              "unique_trajectories": games, "simulations": int(cfg["search"]["eval_simulations"])}
    if kind == "ladder":
        report["openings"] = json.loads(Path(opts["openings-file"]).read_text(encoding="utf-8"))[:pairs]
    write_json(out, report)
    record["end"] = time.time()
    write_json(record_path, record)
    print(json.dumps({"games": games}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
