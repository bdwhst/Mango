"""Multi-process self-play in the pipeline (DESIGN 5.5.1 step 0, 9 "Multi-threaded
drivers" row, pipeline part): per-worker plan records persisted before launch, unique
chunk ids inside each worker's block, a failed worker stops the others, restart resumes
every worker's own remaining quota from the plan, seeds differ per worker, and
selfplay.processes = 1 keeps the single-process path. Uses tests/fake_selfplay.py instead
of the C++ driver, so no build is needed."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from mango.config import merge_config
from mango.pipeline import Pipeline, merge_selfplay_summaries, split_selfplay_workers
from mango.state import derive_seed, read_json

FAKE = Path(__file__).resolve().parent / "fake_selfplay.py"

CFG = {
    "board": {"size": 5},
    "search": {"simulations": 8, "resign_auto": False},
    "selfplay": {"games_per_iteration": 10, "chunk_games": 2, "save_sgf": False, "processes": 3},
    "training": {"res_blocks": 1, "filters": 8, "batch_size": 8, "max_steps_per_iteration": 1, "window_games": 100},
}


@pytest.fixture
def fake_driver(monkeypatch):
    monkeypatch.setattr(Pipeline, "_selfplay_command", lambda self: [sys.executable, str(FAKE)])


def _dispatches(run: Path) -> list[dict]:
    """Launch records of the fake driver, in launch order (one file per launch)."""
    files = sorted(run.glob("fake_dispatch/*.json"), key=lambda q: int(q.name.split("_")[0]))
    return [json.loads(q.read_text(encoding="utf-8")) for q in files]


def _chunk_ids(run: Path) -> list[int]:
    return sorted(int(p.name.split("_")[-1].split(".")[0]) for p in run.glob("replay/*.mgo"))


def test_split_selfplay_workers_blocks_and_seeds():
    ws = split_selfplay_workers(10, 3, chunk_id_start=7, seed=12345)
    assert [w["games"] for w in ws] == [4, 3, 3]
    assert [(w["chunk_id_start"], w["chunk_id_end"]) for w in ws] == [(7, 11), (11, 14), (14, 17)]
    assert [w["k"] for w in ws] == [0, 1, 2]
    assert all(w["chunk_id_end"] - w["chunk_id_start"] == w["games"] for w in ws)
    assert [w["seed"] for w in ws] == [derive_seed(12345, k) for k in range(3)]
    assert len({w["seed"] for w in ws}) == 3
    with pytest.raises(ValueError):
        split_selfplay_workers(10, 1, 1, 1)


def test_merge_selfplay_summaries_sums_counts_and_uses_wall_time():
    a = {"games": 4, "positions": 40, "evaluations": 320, "batches": 40, "retries": 1, "eval_seconds": 1.0,
         "games_in_flight": 8, "black_wins": 2, "terminations": {"two_passes": 3, "resign": 1, "move_cap": 0},
         "chunks": ["c1", "c2"], "next_chunk_id": 3, "seconds": 2.0, "positions_per_s": 20.0,
         "resign_threshold": -0.9, "model_id": "m", "device": "fake", "iteration": 1}
    b = {"games": 6, "positions": 20, "evaluations": 80, "batches": 10, "retries": 0, "eval_seconds": 0.5,
         "games_in_flight": 8, "black_wins": 3, "terminations": {"two_passes": 6, "resign": 0, "move_cap": 0},
         "chunks": ["c5"], "next_chunk_id": 6, "seconds": 1.0, "positions_per_s": 20.0}
    m = merge_selfplay_summaries({1: b, 0: a}, seconds=4.0, processes=2)
    assert (m["games"], m["positions"], m["evaluations"], m["batches"], m["retries"]) == (10, 60, 400, 50, 1)
    assert m["avg_batch"] == 8.0 and m["avg_game_length"] == 6.0
    assert m["seconds"] == 4.0 and m["positions_per_s"] == 15.0 and m["evals_per_s"] == 100.0
    assert m["eval_seconds"] == 1.5 and m["games_in_flight"] == 16 and m["processes"] == 2
    assert m["terminations"] == {"two_passes": 9, "resign": 1, "move_cap": 0}
    assert m["chunks"] == ["c1", "c2", "c5"] and m["next_chunk_id"] == 6 and m["black_wins"] == 5
    assert [w["k"] for w in m["workers"]] == [0, 1]
    assert (m["resign_threshold"], m["model_id"], m["device"], m["iteration"]) == (-0.9, "m", "fake", 1)


def test_processes_plan_failed_worker_and_restart(tmp_path, fake_driver):
    run = tmp_path / "run"
    p = Pipeline(run, merge_config(CFG), tmp_path, device="cpu", selfplay_device="cpu", quiet=True)
    it_seed = derive_seed(p.state["run_seed"], 1)
    expected = split_selfplay_workers(10, 3, 1, it_seed)  # blocks [1,5) [5,8) [8,11)
    # Worker 1 publishes everything, leaves a .tmp and hangs; worker 2 publishes one chunk
    # (2 games) and exits 3 once workers 0 and 1 have published; worker 0 completes.
    (run / "fake_selfplay.json").write_text(json.dumps({
        "hang": {"chunk_id_start": 5, "tmp": True, "seconds": 60},
        "fail": {"chunk_id_start": 8, "after_chunks": 1, "exit": 3,
                 "wait_for": ["chunk_0001_000002.mgo", "chunk_0001_000006.mgo", "chunk_0001_000007.mgo.tmp"]},
    }), encoding="utf-8")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="worker 2 failed with exit code 3"):
        p.phase_selfplay()
    assert time.time() - t0 < 20, "the hanging worker was not terminated"

    # The plan (with the per-worker records) was persisted before launch and survives the failure.
    st = read_json(run / "state.json")
    assert st["phase"] == "selfplay"
    plan = st["plan"]["selfplay"]
    assert plan["processes"] == 3 and plan["workers"] == expected
    first = _dispatches(run)
    assert sorted((d["chunk_id_start"], d["games"]) for d in first) == [(1, 4), (5, 3), (8, 3)]
    assert {d["seed"] for d in first} == {w["seed"] for w in expected}
    assert len({d["seed"] for d in first}) == 3
    assert _chunk_ids(run) == [1, 2, 5, 6, 8]
    assert list(run.glob("replay/*.tmp")), "the hanging worker's partial chunk should still be there"
    for k in range(3):
        assert (run / "logs" / f"selfplay_{k}.log").exists()
    before = {q.name: q.read_bytes() for q in run.glob("replay/*.mgo")}

    # Restart: a fresh Pipeline re-reads the plan; only worker 2's remaining quota is dispatched.
    (run / "fake_selfplay.json").write_text("{}", encoding="utf-8")
    p2 = Pipeline(run, None, tmp_path, device="cpu", selfplay_device="cpu", quiet=True)
    p2.phase_selfplay()
    st = read_json(run / "state.json")
    assert st["phase"] == "train" and st["next_chunk_id"] == 11
    second = _dispatches(run)[len(first):]
    assert [(d["chunk_id_start"], d["games"], d["seed"]) for d in second] == [(9, 1, expected[2]["seed"])]
    assert not list(run.glob("replay/*.tmp"))
    assert _chunk_ids(run) == [1, 2, 5, 6, 8, 9]
    for name, data in before.items():
        assert (run / "replay" / name).read_bytes() == data, f"{name} was overwritten"
    manifest = read_json(run / "replay" / "manifest.json")
    assert sorted(c["chunk_id"] for c in manifest["chunks"]) == [1, 2, 5, 6, 8, 9]
    assert sum(c["games"] for c in manifest["chunks"]) == 10
    for c in manifest["chunks"]:
        assert any(w["chunk_id_start"] <= c["chunk_id"] < w["chunk_id_end"] for w in expected)
    # The iteration's counts are the whole iteration's (rebuilt from the published chunks);
    # the resumed launch's own counts and timing are kept aside and flagged as launch-only.
    stats = st["selfplay_stats"]["1"]
    assert stats["processes"] == 3 and stats["resumed"] is True
    assert stats["games"] == 10 and stats["positions"] == sum(c["positions"] for c in manifest["chunks"])
    assert sorted(stats["chunks"]) == sorted(c["path"] for c in manifest["chunks"])
    assert stats["terminations"]["two_passes"] == 10
    assert stats["black_wins"] == sum((c["games"] + 1) // 2 for c in manifest["chunks"])  # the fake: black wins even games
    assert stats["launch"]["games"] == 1 and [w["k"] for w in stats["launch"]["workers"]] == [2]
    assert "seconds" in stats["incomplete"] and "evaluations" in stats["incomplete"]
    assert stats["avg_game_length"] == round(stats["positions"] / 10, 2)

    # A restart with nothing left to play launches no worker and still records the totals.
    st["phase"] = "selfplay"
    st["plan"] = {"selfplay": plan}
    (run / "state.json").write_text(json.dumps(st), encoding="utf-8")
    p3 = Pipeline(run, None, tmp_path, device="cpu", selfplay_device="cpu", quiet=True)
    p3.phase_selfplay()
    assert len(_dispatches(run)) == len(first) + len(second)
    st = read_json(run / "state.json")
    stats = st["selfplay_stats"]["1"]
    assert stats["games"] == 10 and stats["resumed"] is True and stats["launch"] == {} and stats["incomplete"] == []
    assert st["phase"] == "train" and st["next_chunk_id"] == 11


def test_run_workers_cleans_up_after_a_failed_launch(tmp_path, fake_driver, monkeypatch):
    """P1 of the step-0 review: a Popen failure (or an interrupt) while launching must not
    leave the already started workers running, and closes their log handles."""
    import mango.pipeline as pl

    run = tmp_path / "run"
    p = Pipeline(run, merge_config(CFG), tmp_path, device="cpu", selfplay_device="cpu", quiet=True)
    # Worker 0 (block start 1) would hang for a minute; the second launch fails.
    (run / "fake_selfplay.json").write_text(json.dumps({"hang": {"chunk_id_start": 1, "tmp": False, "seconds": 60}}),
                                            encoding="utf-8")
    real_popen = pl.subprocess.Popen
    started: list = []

    class RecordingPopen(real_popen):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            if len(started) == 1:
                raise OSError("simulated launch failure")
            super().__init__(*args, **kwargs)
            self.errlog = kwargs["stderr"]
            started.append(self)

    monkeypatch.setattr(pl.subprocess, "Popen", RecordingPopen)
    t0 = time.time()
    with pytest.raises(OSError, match="simulated launch failure"):
        p.phase_selfplay()
    assert time.time() - t0 < 20
    assert len(started) == 1
    assert started[0].poll() is not None, "the first worker was not terminated and waited for"
    assert started[0].errlog.closed
    assert read_json(run / "state.json")["phase"] == "selfplay"  # the plan survives for the restart


def test_processes_one_keeps_the_single_process_path(tmp_path, fake_driver):
    cfg = merge_config(CFG)
    cfg["selfplay"]["processes"] = 1
    run = tmp_path / "run"
    p = Pipeline(run, cfg, tmp_path, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_selfplay()
    st = read_json(run / "state.json")
    plan = st["plan"]["selfplay"]
    assert "workers" not in plan and "processes" not in plan
    assert plan["seed"] == derive_seed(st["run_seed"], 1) and plan["chunk_id_start"] == 1
    assert _dispatches(run) == [{**_dispatches(run)[0], "seed": plan["seed"], "chunk_id_start": 1, "games": 10}]
    assert _chunk_ids(run) == [1, 2, 3, 4, 5] and st["next_chunk_id"] == 6
    assert (run / "logs" / "selfplay.log").exists()
    assert "processes" not in st["selfplay_stats"]["1"]
