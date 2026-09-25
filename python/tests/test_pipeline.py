"""Pipeline phases and restart protocol (DESIGN 6.5) on a tiny 5x5 run, including the
M4 additions: automatic resign threshold in the self-play plan, chunk eviction, and the
strength phase (frozen ladder). Needs the C++ executables (mango_selfplay, mango_match);
skipped when they are not built."""

from __future__ import annotations

import os
import platform
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
    "search": {"simulations": 8, "eval_simulations": 8, "temperature_moves": 4, "search_symmetry": False,
               "resign_auto": True},
    "selfplay": {"games_per_iteration": 6, "chunk_games": 4, "save_sgf": True},
    "training": {"res_blocks": 1, "filters": 8, "batch_size": 8, "max_steps_per_iteration": 5, "window_games": 100,
                 "fixed_holdout_iteration": 1},
    "eval": {"pairs": 2, "opening_moves": 2, "ladder_every": 1, "ladder_pairs": 2, "ladder_neighbours": 2},
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
    for expected in ["selfplay", "train", "monitor", "export", "gate", "promote", "strength"]:
        p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
        assert p.state["phase"] == expected
        if expected == "selfplay":
            p.run_phase()
            plan = read_json(run / "state.json")["plan"]["selfplay"]
            # resign_auto with an empty window: disabled, recorded in the plan and passed on.
            assert plan["v_resign"] == -1.0 and plan["resign_selection"]["auto"] is True
            assert plan["resign_selection"]["threshold"] is None and plan["resign_selection"]["samples"] == 0
            log = (run / "logs" / "pipeline.log").read_text(encoding="utf-8")
            assert "--resign-threshold -1.0" in log
            assert read_json(run / "state.json")["selfplay_stats"]["1"]["resign_threshold"] == -1.0
            continue
        p.run_phase()
    st = read_json(run / "state.json")
    assert st["iteration"] == 2 and st["phase"] == "selfplay" and st["plan"] == {}
    # Strength phase (ladder_every = 1): the initial model and the candidate were added,
    # each played the random anchor, and the candidate played the initial model.
    ladder = read_json(run / "strength" / "ladder.json")
    assert [e["kind"] for e in ladder["entries"]] == ["random", "model", "model"]
    assert ladder["entries"][1]["name"] == initial and ladder["entries"][1]["iteration"] == 0
    cand1 = ladder["entries"][2]["name"]
    assert sorted((m["a"], m["b"]) for m in ladder["matches"]) == sorted([(initial, "random"), (cand1, "random"),
                                                                           (cand1, initial)])
    assert all(m["pairs"] == 2 for m in ladder["matches"])
    assert (run / "strength" / "openings_v1.json").exists()
    assert (run / "strength" / "ratings.csv").exists()
    ratings = read_json(run / "strength" / "ratings.json")["ratings"]
    assert ratings["random"]["elo"] == 0.0 and ratings[cand1]["rated"]
    assert st["strength"]["1"]["model_id"] == cand1 and st["strength"]["1"]["matches"] == 3
    manifest = read_json(run / "replay" / "manifest.json")
    assert [c["games"] for c in manifest["chunks"]] == [4, 2]
    assert sum(c["positions"] for c in manifest["chunks"]) > 0
    assert (run / "matches" / "0001.json").exists()
    assert (run / "monitor" / "holdout.csv").exists()
    import csv as _csv
    with open(run / "monitor" / "holdout.csv", newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1 and {"fixed_positions", "fixed_value_mse", "fixed_policy_ce"} <= set(rows[0])
    fixed = run / "monitor" / "fixed_holdout.mgo"
    # Frozen iff iteration 1 had a holdout game (6 games: either way is legitimate).
    assert fixed.exists() == (int(rows[0]["fixed_positions"]) > 0)
    assert "seconds" in st["train_stats"]["1"]
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
    # ... in a version whose self-play plan had no resign fields (resignation always off).
    del st["plan"]["selfplay"]["v_resign"]
    del st["plan"]["selfplay"]["resign_selection"]
    st["v_resign"] = -1.0
    (run / "state.json").write_text(__import__("json").dumps(st))
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_selfplay()
    plan = read_json(run / "state.json")["plan"]["selfplay"]
    assert plan["v_resign"] == -1.0 and plan["resign_selection"] is None  # migrated, not KeyError
    assert "resuming an older self-play plan; resign threshold -1" in (run / "logs" / "pipeline.log").read_text(encoding="utf-8")
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
    assert st["iteration"] == 2 and st["phase"] == "strength"
    ev = st["events"][-1]
    assert ev["candidate"] == cand and isinstance(ev["promoted"], bool)
    assert (read_json(run / "best.json")["model_id"] == cand) == ev["promoted"]
    # Strength: the new candidate plays random, the first promoted model (if any) and its
    # neighbours; a restart of the phase replays nothing (reports reused).
    p.phase_strength()
    ladder = read_json(run / "strength" / "ladder.json")
    n_matches = len(ladder["matches"])
    assert ladder["entries"][-1]["name"] == cand and ladder["entries"][-1]["iteration"] == 2
    assert n_matches == 3 + 3  # + cand vs random, cand vs cand1 (first promoted or neighbour), cand vs initial
    mtimes = {p_.name: p_.stat().st_mtime_ns for p_ in (run / "strength" / "matches").glob("*.json")}
    st = read_json(run / "state.json")
    assert st["iteration"] == 3 and st["phase"] == "selfplay"
    st["iteration"] = 2  # pretend the crash happened after the last match, before the phase ended
    st["phase"] = "strength"
    st["plan"] = {"promote": {"candidate_model_id": cand, "decision": ev["promoted"]},
                  "strength": {"add": True, "model_id": cand, "promoted": ev["promoted"]}}
    (run / "state.json").write_text(__import__("json").dumps(st))
    p = Pipeline(run, None, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.phase_strength()
    assert len(read_json(run / "strength" / "ladder.json")["matches"]) == n_matches
    assert {p_.name: p_.stat().st_mtime_ns for p_ in (run / "strength" / "matches").glob("*.json")} == mtimes
    assert read_json(run / "state.json")["iteration"] == 3

    # Resign selection scope: "latest" reads only the previous iteration's chunks, "window"
    # the whole window; the false-positive target is passed through.
    import mango.pipeline as pl
    seen = {}

    def fake_select(games, max_fpr=0.05, **kw):
        seen["seeds"] = sorted(g.game_seed for g in games)
        seen["max_fpr"] = max_fpr
        return {"threshold": None, "no_resign_games": 0, "samples": len(games), "false_positive_rate": None}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pl, "select_resign_threshold", fake_select)
    try:
        from mango.chunk import read_chunk
        p.cfg["search"]["resign_select_scope"] = "latest"
        p.cfg["search"]["resign_fpr_target"] = 0.025
        sel = p._select_resign()
        latest = sorted(g.game_seed for q in sorted((run / "replay").glob("chunk_0002_*.mgo")) for g in read_chunk(q).games)
        assert seen["seeds"] == latest and len(latest) == 6 and seen["max_fpr"] == 0.025
        assert sel["scope"] == "latest" and sel["fpr_target"] == 0.025
        p.cfg["search"]["resign_select_scope"] = "window"
        p._select_resign()
        assert len(seen["seeds"]) == 12  # both iterations
        p.cfg["search"]["resign_select_scope"] = "bogus"
        with pytest.raises(ValueError):
            p._select_resign()
        p.cfg["search"]["resign_select_scope"] = "window"
        p.cfg["search"]["resign_fpr_target"] = 0.05
    finally:
        monkeypatch.undo()

    # Eviction: with a 4-game window the two iteration-1 chunks fall out (files deleted,
    # manifest entries kept and marked), and nothing else is touched on a second call.
    p.cfg["training"]["window_games"] = 4
    assert p._evict_old_chunks() == 2
    manifest = read_json(run / "replay" / "manifest.json")
    assert [c.get("evicted", False) for c in manifest["chunks"]] == [True, True, False, False]
    assert not any((run / "replay" / c["path"]).exists() for c in manifest["chunks"] if c.get("evicted"))
    assert all((run / "replay" / c["path"]).exists() for c in manifest["chunks"] if not c.get("evicted"))
    assert [q.name for q in p.window_chunks()] == [c["path"] for c in manifest["chunks"][2:]]
    assert p._evict_old_chunks() == 0
    p.cfg["training"]["window_games"] = 100

    # A zero budget still completes exactly one whole iteration (DESIGN 8.2 budgets).
    assert p.run_for_seconds(0) == 1
    st = read_json(run / "state.json")
    assert st["iteration"] == 4 and st["phase"] == "selfplay" and st["plan"] == {}

    # The learning check runs end to end (no threshold asserted on a tiny run).
    result = p.check(pairs=2)
    assert 0.0 <= result["known_outcome"]["best"]["accuracy"] <= 1.0
    assert result["match_vs_initial"]["pairs"] == 2
    assert "accuracy" in result["holdout_terminal"] and "per_margin" in result["known_outcome"]["best"]
    assert (run / "check.json").exists()
