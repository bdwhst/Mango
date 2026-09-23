"""Pipeline phases and restart protocol (DESIGN 6.5) on a tiny 5x5 run. Needs the C++
executables (mango_selfplay, mango_match); skipped when they are not built."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pytest

from mango.config import merge_config
from mango.pipeline import Pipeline
from mango.state import read_json

ROOT = Path(__file__).resolve().parents[2]


def _bin_dir() -> Path | None:
    env = os.environ.get("MANGO_BIN_DIR")
    candidates = [Path(env)] if env else []
    candidates += [ROOT / "build" / "windows-cuda" / "Release", ROOT / "build" / "windows" / "Release",
                   ROOT / "build" / "macos-mps", ROOT / "build" / "macos", ROOT / "build" / "linux"]
    exe = "mango_selfplay" + (".exe" if platform.system() == "Windows" else "")
    for c in candidates:
        if (c / exe).exists():
            return c
    return None


TINY = {
    "board": {"size": 5},
    "search": {"simulations": 8, "eval_simulations": 8, "temperature_moves": 4, "search_symmetry": False},
    "selfplay": {"games_per_iteration": 6, "chunk_games": 4, "save_sgf": True},
    "training": {"res_blocks": 1, "filters": 8, "batch_size": 8, "max_steps_per_iteration": 5, "window_games": 100},
    "eval": {"pairs": 2, "opening_moves": 2},
}


@pytest.fixture
def bin_dir():
    b = _bin_dir()
    if b is None:
        pytest.skip("C++ executables not built")
    return b


def test_pipeline_runs_two_iterations_and_survives_restarts(tmp_path, bin_dir):
    run = tmp_path / "run"
    p = Pipeline(run, merge_config(TINY), bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    st = read_json(run / "state.json")
    assert st["iteration"] == 1 and st["phase"] == "selfplay"
    initial = st["best_model_id"]
    assert (run / "models" / initial / "model.pt").exists()
    assert read_json(run / "best.json")["model_id"] == initial

    # Iteration 1, phase by phase, re-creating the Pipeline object each time (a restart).
    for expected in ["selfplay", "train", "monitor", "export", "gate", "promote"]:
        p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
        assert p.state["phase"] == expected
        p.run_phase()
    st = read_json(run / "state.json")
    assert st["iteration"] == 2 and st["phase"] == "selfplay" and st["plan"] == {}
    manifest = read_json(run / "replay" / "manifest.json")
    assert [c["games"] for c in manifest["chunks"]] == [4, 2]
    assert sum(c["positions"] for c in manifest["chunks"]) > 0
    assert (run / "matches" / "0001.json").exists()
    assert (run / "monitor" / "holdout.csv").exists()
    assert st["global_step"] >= 1
    assert len(list((run / "sgf" / "0001").glob("*.sgf"))) == 6
    assert len(st["events"]) == 1 and st["events"][0]["event"] == "gate"

    # Self-play resumption: delete the last published chunk of iteration 2 mid-way and
    # check that only the missing games are regenerated (chunk ids continue).
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_selfplay()
    chunks = sorted((run / "replay").glob("chunk_0002_*.mgo"))
    assert len(chunks) == 2
    chunks[-1].unlink()
    manifest = read_json(run / "replay" / "manifest.json")
    manifest["chunks"] = [c for c in manifest["chunks"] if c["path"] != chunks[-1].name]
    (run / "replay" / "manifest.json").write_text(__import__("json").dumps(manifest))
    st = read_json(run / "state.json")
    st["phase"] = "selfplay"  # pretend the crash happened before the phase ended
    (run / "state.json").write_text(__import__("json").dumps(st))
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_selfplay()
    chunks = sorted((run / "replay").glob("chunk_0002_*.mgo"))
    assert len(chunks) == 2  # one regenerated, not three
    manifest = read_json(run / "replay" / "manifest.json")
    assert sum(c["games"] for c in manifest["chunks"] if c["path"].startswith("chunk_0002_")) == 6

    # Train resumption: the target step is fixed in the plan; a second run does nothing.
    p.phase_train()
    plan = read_json(run / "state.json")["plan"]["train"]
    target = plan["target_global_step"]
    assert target == plan["start_global_step"] + min(5, -(-plan["window_train_positions"] * 0.25 // 8))
    st = read_json(run / "state.json")
    st["phase"] = "train"
    (run / "state.json").write_text(__import__("json").dumps(st))
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_train()
    assert read_json(run / "state.json")["global_step"] == target
    assert (run / "learner" / f"ckpt_{target}.pt").exists()

    # Export is idempotent (same weights -> same id); gate report reuse; promote decision recorded.
    p.phase_monitor()
    p.phase_export()
    cand = read_json(run / "state.json")["plan"]["export"]["candidate_model_id"]
    p.phase_gate()
    report = read_json(run / "matches" / "0002.json")
    assert report["a"] == cand
    mtime = (run / "matches" / "0002.json").stat().st_mtime_ns
    st = read_json(run / "state.json")
    st["phase"] = "gate"
    (run / "state.json").write_text(__import__("json").dumps(st))
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_gate()
    assert (run / "matches" / "0002.json").stat().st_mtime_ns == mtime  # not re-run
    p.phase_promote()
    st = read_json(run / "state.json")
    assert st["iteration"] == 3
    ev = st["events"][-1]
    assert ev["candidate"] == cand and isinstance(ev["promoted"], bool)
    assert (read_json(run / "best.json")["model_id"] == cand) == ev["promoted"]

    # The learning check runs end to end (no threshold asserted on a tiny run).
    result = p.check(pairs=2)
    assert 0.0 <= result["known_outcome"]["best"]["accuracy"] <= 1.0
    assert result["match_vs_initial"]["pairs"] == 2
    assert "accuracy" in result["holdout_terminal"] and "per_margin" in result["known_outcome"]["best"]
    assert (run / "check.json").exists()
