"""A stand-in for mango_selfplay used by the multi-process pipeline tests (DESIGN 5.5.1
step 0): accepts the driver's command line, publishes valid chunks of synthetic games and
prints the same JSON summary line. A control file <run>/fake_selfplay.json makes a worker
(identified by its --chunk-id-start) fail after publishing some chunks — once the files
named in "wait_for" exist — or hang after publishing everything with a .tmp file left
behind. Every launch writes one record to <run>/fake_dispatch/ (one file per launch, so
concurrent workers cannot clobber each other's record).

    python fake_selfplay.py --games N --out DIR --chunk-prefix P --chunk-id-start K --seed S ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/

from mango.chunk import Chunk, ChunkHeader, GameRecord, pack_snapshot, write_chunk  # noqa: E402


def make_game(n: int, seed: int, T: int, result: int) -> GameRecord:
    """A synthetic game (as tests/test_data.make_game, without importing torch)."""
    nn = n * n
    g = GameRecord(game_seed=seed, result=result, score=float(result) * 3.5)
    cells = np.zeros((T + 1, nn), np.int8)
    moves = []
    for t in range(T):
        cells[t + 1] = cells[t]
        cells[t + 1, t % nn] = 1 if t % 2 == 0 else 2
        moves.append(t % nn)
    g.snapshots = np.stack([pack_snapshot(c, n) for c in cells])
    g.moves = np.array(moves, np.uint16)
    g.root_value = np.linspace(-0.5, 0.5, T).astype(np.float32)
    g.root_max_q = np.zeros(T, np.float32)
    g.root_total_visits = np.full(T, 10, np.uint32)
    for t in range(T):
        g.visit_actions.append(np.array([moves[t], nn], np.uint16))
        g.visit_counts.append(np.array([7, 3], np.uint32))
    return g


def main(argv: list[str]) -> int:
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    games = int(opts["games"])
    out = Path(opts["out"])
    prefix = opts["chunk-prefix"]
    chunk_id = int(opts["chunk-id-start"])
    seed = int(opts["seed"])
    run = out.parent
    cfg = json.loads(Path(opts["config"]).read_text(encoding="utf-8"))
    n = int(cfg["board"]["size"])
    per_chunk = max(1, int(cfg["selfplay"]["chunk_games"]))
    model_id = Path(opts.get("model", "fake")).name
    control = {}
    if (run / "fake_selfplay.json").exists():
        control = json.loads((run / "fake_selfplay.json").read_text(encoding="utf-8"))
    dispatch_dir = run / "fake_dispatch"
    dispatch_dir.mkdir(exist_ok=True)
    (dispatch_dir / f"{time.time_ns()}_{os.getpid()}.json").write_text(
        json.dumps({"seed": seed, "chunk_id_start": chunk_id, "games": games, "pid": os.getpid()}), encoding="utf-8")
    fail = control.get("fail") if control.get("fail", {}).get("chunk_id_start") == chunk_id else None
    hang = control.get("hang") if control.get("hang", {}).get("chunk_id_start") == chunk_id else None

    published: list[str] = []
    positions = 0
    start = 0
    while start < games:
        count = min(per_chunk, games - start)
        h = ChunkHeader(board_size=n, komi=float(cfg["board"].get("komi", 7.5)), model_id=model_id, chunk_id=chunk_id,
                        move_cap=2 * n * n)
        recs = [make_game(n, seed=(seed * 1000 + start + j) & 0xFFFFFFFFFFFFFFFF, T=4 + j % 3,
                          result=1 if j % 2 == 0 else -1) for j in range(count)]
        positions += sum(g.T for g in recs)
        name = f"{prefix}{chunk_id:06d}.mgo"
        write_chunk(out / name, Chunk(h, recs))
        published.append(name)
        chunk_id += 1
        start += count
        if fail is not None and len(published) >= int(fail.get("after_chunks", 1)):
            # Fail only once the files the test expects from the other workers exist, so the
            # scenario "two workers have published, the third fails" is deterministic.
            deadline = time.time() + 30
            while not all((out / f).exists() for f in fail.get("wait_for", [])) and time.time() < deadline:
                time.sleep(0.05)
            sys.stderr.write("fake_selfplay: failing on purpose\n")
            return int(fail.get("exit", 3))
    if hang is not None:
        if hang.get("tmp", True):
            (out / f"{prefix}{chunk_id:06d}.mgo.tmp").write_bytes(b"partial")
        sys.stderr.write("fake_selfplay: hanging on purpose\n")
        time.sleep(float(hang.get("seconds", 60)))
        return 0
    summary = {
        "games": games, "positions": positions, "evaluations": positions * 8, "batches": positions,
        "avg_batch": 8.0, "retries": 0, "seconds": 0.01, "eval_seconds": 0.005, "evals_per_s": 100.0,
        "positions_per_s": 50.0, "games_in_flight": int(cfg["selfplay"].get("games_in_flight", 1)),
        "avg_game_length": positions / games if games else 0.0, "black_wins": (games + 1) // 2,
        "terminations": {"two_passes": games, "resign": 0, "move_cap": 0},
        "resign_threshold": float(opts.get("resign-threshold", -1.0)), "chunks": published,
        "next_chunk_id": chunk_id, "model_id": model_id, "device": "fake", "iteration": int(opts.get("iteration", 0)),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
