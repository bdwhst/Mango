"""Overlapped evaluation (DESIGN 6.5.1; 9 row "Pipeline, overlapped evaluation"):
pipeline.async_ladder and pipeline.async_gate. Uses tests/fake_selfplay.py and the scripted
match driver tests/fake_match.py (no build needed); the last test runs the real
executables when they are built.

Numbering follows the design's test list: 1 ladder lane, 2 one-iteration delay and
segmented runs, 3 restart at every crash point, 4 forced termination, 5 a failing job,
6 switching and the promotion list, 7 (ratings.csv) is in test_strength.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mango.config import merge_config
from mango.pipeline import Pipeline, add_promotion, promotions_from_events, select_selfplay_model
from mango.proc import WINDOWS, RunLockError, pid_alive
from mango.state import read_json, write_json_atomic

HERE = Path(__file__).resolve().parent
FAKE_SELFPLAY = HERE / "fake_selfplay.py"
FAKE_MATCH = HERE / "fake_match.py"
DRIVER = HERE / "pipeline_driver.py"

CFG = {
    "board": {"size": 5},
    "search": {"simulations": 8, "eval_simulations": 8, "resign_auto": False},
    "selfplay": {"games_per_iteration": 6, "chunk_games": 3, "save_sgf": False},
    "training": {"res_blocks": 1, "filters": 8, "batch_size": 8, "max_steps_per_iteration": 1, "window_games": 100},
    "eval": {"pairs": 2, "ladder_every": 1, "ladder_pairs": 2, "ladder_neighbours": 1, "opening_moves": 2},
}


class Crash(Exception):
    pass


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(Pipeline, "_selfplay_command", lambda self: [sys.executable, str(FAKE_SELFPLAY)])
    monkeypatch.setattr(Pipeline, "_match_command", lambda self: [sys.executable, str(FAKE_MATCH)])


def config(async_gate: bool, async_ladder: bool, **eval_over) -> dict:
    cfg = json.loads(json.dumps(CFG))
    cfg["pipeline"] = {"async_gate": async_gate, "async_ladder": async_ladder}
    cfg["eval"].update(eval_over)
    return cfg


def new_run(root: Path, name: str, cfg: dict, control: dict | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    write_json_atomic(run / "fake_match.json", control or {})
    p = open_pipeline(run, cfg)
    p.close()
    return run


def open_pipeline(run: Path, cfg: dict | None = None) -> Pipeline:
    return Pipeline(run, merge_config(cfg) if cfg else None, run, device="cpu", selfplay_device="cpu", quiet=True)


def run_to(run: Path, iterations: int) -> None:
    p = open_pipeline(run)
    try:
        p.run_iterations(iterations)
    finally:
        p.close()


def selfplay_models(run: Path) -> dict[int, str]:
    st = read_json(run / "state.json")
    return {int(k): v["model_id"] for k, v in st["selfplay_stats"].items()}


def candidates(run: Path) -> dict[int, str]:
    return {int(m.name.split("-")[0]): m.name for m in (run / "models").iterdir() if not m.name.startswith("0000-")}


def snapshot(run: Path) -> dict:
    """Everything an interrupted run must reproduce (timings excluded)."""
    st = read_json(run / "state.json")
    files = {}
    rels = ["best.json", "strength/ladder.json", "strength/ratings.json", "strength/ratings.csv"]
    rels += sorted(str(p.relative_to(run)).replace("\\", "/") for p in run.glob("matches/*.json"))
    rels += sorted(str(p.relative_to(run)).replace("\\", "/") for p in run.glob("strength/fits/*.json"))
    rels += sorted(str(p.relative_to(run)).replace("\\", "/") for p in run.glob("strength/matches/*.json"))
    for rel in rels:
        files[rel] = (run / rel).read_text(encoding="utf-8") if (run / rel).exists() else None
    keys = ("iteration", "phase", "best_model_id", "promotions", "events", "strength", "background", "global_step",
            "next_chunk_id")
    return {"state": {k: st.get(k) for k in keys}, "selfplay_models": selfplay_models(run), "files": files}


def match_records(run: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "fake_match_log").glob("*.json"))]


def assert_no_concurrent_writers(records: list[dict], killed_before: float | None = None) -> None:
    """Per report: the matches writing it never overlap. A record without an end was
    killed; it must have started before `killed_before` (the moment its tree was known
    dead) and every later writer of that report must start after it."""
    by_out: dict[str, list[dict]] = {}
    for r in records:
        by_out.setdefault(r["out"], []).append(r)
    for out, rs in by_out.items():
        rs.sort(key=lambda r: r["start"])
        for prev, nxt in zip(rs, rs[1:]):
            if prev["end"] is None:
                assert killed_before is not None and prev["start"] < killed_before <= nxt["start"], out
            else:
                assert prev["end"] <= nxt["start"], out


# --- 6: the promotion list ------------------------------------------------------------------

def test_selfplay_model_is_chosen_by_effective_then_gate_iteration():
    promos: list[dict] = []
    assert select_selfplay_model(promos, 5, "init") == "init"
    assert add_promotion(promos, 3, "c3", 5)  # asynchronous gate of iteration 3
    assert add_promotion(promos, 4, "c4", 5)  # synchronous gate of iteration 4 after the switch
    assert not add_promotion(promos, 3, "c3", 5)  # keyed
    assert len(promos) == 2
    assert select_selfplay_model(promos, 4, "init") == "init"
    assert select_selfplay_model(promos, 5, "init") == "c4"
    assert select_selfplay_model(list(reversed(promos)), 5, "init") == "c4"  # not the list order
    add_promotion(promos, 1, "c1", 2)
    assert select_selfplay_model(promos, 3, "init") == "c1"
    assert select_selfplay_model(promos, 99, "init") == "c4"


def test_promotion_list_migration_is_keyed_and_checked(tmp_path, fakes):
    run = new_run(tmp_path, "run", config(False, False))
    st = read_json(run / "state.json")
    init = st["initial_model_id"]
    events = [{"iteration": 1, "event": "gate", "candidate": "0001-a", "promoted": True},
              {"iteration": 1, "event": "gate", "candidate": "0001-a", "promoted": True},  # duplicated by a restart
              {"iteration": 2, "event": "gate", "candidate": "0002-b", "promoted": False},
              {"iteration": 3, "event": "train_skipped"},
              {"iteration": 3, "event": "gate", "candidate": "0003-c", "promoted": True}]
    assert promotions_from_events(events) == [
        {"gate_iteration": 1, "candidate": "0001-a", "effective_selfplay_iteration": 2},
        {"gate_iteration": 3, "candidate": "0003-c", "effective_selfplay_iteration": 4}]
    del st["promotions"], st["background"]
    st["events"], st["best_model_id"] = events, "0003-c"
    write_json_atomic(run / "state.json", st)
    p = open_pipeline(run)
    assert p.state["promotions"] == promotions_from_events(events)
    assert p.state["background"] == {"gate": None, "ladder": []}
    assert [p.selfplay_model(j) for j in (1, 2, 3, 4)] == [init, "0001-a", "0001-a", "0003-c"]
    p.close()
    # Events that do not end at best_model_id: the run stops instead of changing the model.
    st = read_json(run / "state.json")
    del st["promotions"]
    st["best_model_id"] = "0002-b"
    write_json_atomic(run / "state.json", st)
    with pytest.raises(RuntimeError, match="cannot rebuild the promotion list"):
        open_pipeline(run)


# --- 1: the ladder lane ---------------------------------------------------------------------

def test_async_ladder_equals_the_sequential_ladder_and_delays_nothing(tmp_path, fakes):
    control = {"gate": {"1": 0.9, "2": 0.3, "3": 0.9}}
    seq = new_run(tmp_path, "seq", config(False, False), control)
    run_to(seq, 3)
    control_slow = {**control, "sleep": {"ladder": 1.0}}
    asy = new_run(tmp_path, "async", config(False, True), control_slow)
    run_to(asy, 3)
    a, s = snapshot(asy), snapshot(seq)
    assert a["state"]["strength"] == s["state"]["strength"] and set(a["state"]["strength"]) == {"1", "2", "3"}
    for rel in s["files"]:
        if rel.startswith("strength/"):
            assert a["files"][rel] == s["files"][rel], rel
    assert a == s  # everything else too: the ladder is measurement only
    assert read_json(asy / "state.json")["background"] == {"gate": None, "ladder": []}  # drained
    # A sleeping ladder job delayed nothing: iteration 2 started its self-play while the
    # ladder job of iteration 1 was still playing.
    ladder1_end = max(r["end"] for r in match_records(asy) if r["kind"] == "ladder" and "0001-" in r["a"])
    starts = sorted(int(q.name.split("_")[0]) / 1e9 for q in (asy / "fake_dispatch").glob("*.json"))
    assert starts[1] < ladder1_end
    times = [json.loads(line) for line in (asy / "logs" / "phase_times.jsonl").read_text().splitlines()]
    assert any("ladder_0001" in t["overlapped"] for t in times if t["iteration"] == 2)
    assert "background: launched ladder_0001" in (asy / "logs" / "pipeline.log").read_text(encoding="utf-8")


# --- 2: the one-iteration delay; segmented runs -----------------------------------------------

def test_async_gate_delays_promotion_by_one_iteration_and_segmented_runs_agree(tmp_path, fakes):
    control = {"gate": {"1": 0.3, "2": 0.9, "3": 0.3, "4": 0.9}}
    cfg = config(True, True)
    whole = new_run(tmp_path, "whole", cfg, control)
    run_to(whole, 4)
    init = read_json(whole / "state.json")["initial_model_id"]
    c = candidates(whole)
    # Promotion of iteration i plays from i + 2.
    assert selfplay_models(whole) == {1: init, 2: init, 3: init, 4: c[2]}
    st = read_json(whole / "state.json")
    assert st["promotions"] == [{"gate_iteration": 2, "candidate": c[2], "effective_selfplay_iteration": 4},
                                {"gate_iteration": 4, "candidate": c[4], "effective_selfplay_iteration": 6}]
    # Every gate's incumbent is the best after the previous decision.
    incumbents = {i: read_json(whole / "matches" / f"{i:04d}.json")["b"] for i in range(1, 5)}
    assert incumbents == {1: init, 2: init, 3: c[2], 4: c[2]}
    assert read_json(whole / "best.json")["model_id"] == c[4]  # the drain settled gate 4
    assert st["background"] == {"gate": None, "ladder": []}
    assert [(e["iteration"], e["promoted"]) for e in st["events"]] == [(1, False), (2, True), (3, False), (4, True)]
    assert set(st["strength"]) == {"1", "2", "3", "4"}
    # Segmented: 4 x 1 and 2 x 2 iterations, each segment drained.
    ones = new_run(tmp_path, "ones", cfg, control)
    for k in range(1, 5):
        run_to(ones, k)
    twos = new_run(tmp_path, "twos", cfg, control)
    run_to(twos, 2)
    run_to(twos, 4)
    assert snapshot(ones) == snapshot(whole)
    assert snapshot(twos) == snapshot(whole)


# --- 3: restart at every crash point ----------------------------------------------------------

CRASH_POINTS = [
    ("launch_eval:recorded", 1),   # the gate recorded, not launched
    ("phase:selfplay", 2),         # the gate of iteration 1 running
    ("settle:report", 1),          # between the report and the recorded decision
    ("settle:decision", 1),        # between the decision and best.json
    ("settle:promotion", 1),       # between the promotion and best.json
    ("settle:event", 2),
    ("settle:strength", 1),        # the ladder job enqueued and saved, the gate not cleared
    ("phase:train", 2),            # inside a ladder job with a mango_match running
    ("collect:read", 1),           # a fit file written, the job not collected
    ("settle:report", 3),          # inside the final drain
]


@pytest.mark.parametrize("point,occurrence", CRASH_POINTS)
def test_restart_after_a_crash_reaches_the_uninterrupted_state(tmp_path, fakes, monkeypatch, point, occurrence):
    control = {"gate": {"1": 0.9, "2": 0.3, "3": 0.9}, "sleep": {"gate": 1.0, "ladder": 0.5}}
    cfg = config(True, True)
    ref = new_run(tmp_path, "ref", cfg, control)
    run_to(ref, 3)
    run = new_run(tmp_path, "run", cfg, control)
    seen = {"n": 0, "armed": True}

    def crash_point(self, name):
        if seen["armed"] and name == point:
            seen["n"] += 1
            if seen["n"] == occurrence:
                seen["armed"] = False
                raise Crash(name)

    monkeypatch.setattr(Pipeline, "_crash_point", crash_point)
    with pytest.raises(Crash):
        run_to(run, 3)
    killed_before = time.time()  # run_iterations stopped every background tree before re-raising
    for r in match_records(run):
        assert not pid_alive(r["pid"])
    run_to(run, 3)
    assert snapshot(run) == snapshot(ref)
    st = read_json(run / "state.json")
    keys = [(e["iteration"], e["event"], e.get("candidate")) for e in st["events"]]
    assert len(keys) == len(set(keys))
    assert_no_concurrent_writers(match_records(run), killed_before)


# --- 5: a failing job -------------------------------------------------------------------------

def test_a_failing_gate_job_stops_the_run_and_the_restart_resumes_it(tmp_path, fakes):
    # Gate 2 fails while ladder job 1 is still playing (its matches sleep).
    control = {"gate": {"1": 0.9, "2": 0.9}, "sleep": {"ladder": 3.0}, "fail": {"gate_2": 5}}
    run = new_run(tmp_path, "run", config(True, True), control)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="background job gate_0002 failed with exit code 1"):
        run_to(run, 3)
    assert time.time() - t0 < 60
    assert "failed with exit code 5" in (run / "logs" / "match_bg.log").read_text(encoding="utf-8")
    st = read_json(run / "state.json")
    assert st["background"]["gate"]["iteration"] == 2 and st["background"]["gate"]["decision"] is None
    assert st["background"]["ladder"][0]["iteration"] == 1
    records = match_records(run)
    assert any(r["kind"] == "ladder" and r["end"] is None for r in records)  # the ladder job was stopped
    for r in records:
        assert not pid_alive(r["pid"])
    write_json_atomic(run / "fake_match.json", {**control, "fail": {}})
    run_to(run, 3)
    st = read_json(run / "state.json")
    c = candidates(run)
    assert [p["candidate"] for p in st["promotions"]] == [c[1], c[2]]
    assert st["background"] == {"gate": None, "ladder": []} and set(st["strength"]) == {"1", "2", "3"}


# --- 6: switching -----------------------------------------------------------------------------

def test_switching_async_gate_off_settles_the_pending_gate_first(tmp_path, fakes):
    control = {"gate": {"1": 0.9, "2": 0.9, "3": 0.3}}
    run = new_run(tmp_path, "run", config(True, False), control)
    p = open_pipeline(run)
    while p.state["iteration"] == 1:  # iteration 1 asynchronous, without a drain
        p.run_phase()
    assert p.state["background"]["gate"]["effective_selfplay_iteration"] == 3
    p.set_switches(async_gate=False)
    p.run_iterations(3)
    p.close()
    st = read_json(run / "state.json")
    c = candidates(run)
    init = st["initial_model_id"]
    # Both promotions take effect in iteration 3; the later gate (2, synchronous) wins.
    assert st["promotions"] == [{"gate_iteration": 1, "candidate": c[1], "effective_selfplay_iteration": 3},
                                {"gate_iteration": 2, "candidate": c[2], "effective_selfplay_iteration": 3}]
    assert selfplay_models(run) == {1: init, 2: init, 3: c[2]}
    assert read_json(run / "matches" / "0002.json")["b"] == c[1]  # gated against the settled incumbent
    phases = [json.loads(line)["phase"] for line in (run / "logs" / "phase_times.jsonl").read_text().splitlines()]
    assert phases == ["selfplay", "train", "monitor", "export", "launch_eval",
                      "selfplay", "settle", "train", "monitor", "export", "gate", "promote", "strength",
                      "selfplay", "train", "monitor", "export", "gate", "promote", "strength"]


# --- 4: forced termination --------------------------------------------------------------------

def start_driver(run: Path, iterations: int, cfg: dict | None = None) -> subprocess.Popen:
    args = [sys.executable, str(DRIVER), "--run", str(run), "--iterations", str(iterations)]
    if cfg is not None:
        args += ["--config-json", json.dumps(cfg)]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(run.parent / f"{run.name}_driver.log", "a"))


def wait_for(cond, timeout: float, what: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = cond()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def ladder_match_running(run: Path):
    recs = [r for r in match_records(run) if r["kind"] == "ladder" and r["end"] is None] if (run / "fake_match_log").exists() else []
    return recs[0] if recs else None


@pytest.mark.parametrize("victim", ["pipeline", "supervisor"])
def test_forced_termination_leaves_no_process_and_no_second_writer(tmp_path, victim):
    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(run / "fake_match.json", {"sleep": {"ladder": 60.0}})
    driver = start_driver(run, 2, config(False, True))
    try:
        rec = wait_for(lambda: ladder_match_running(run), 120, "a ladder match")
        # A second pipeline on the same run refuses to start while this one is alive.
        with pytest.raises(RunLockError):
            Pipeline(run, None, run, device="cpu", selfplay_device="cpu", quiet=True)
        st = read_json(run / "state.json")
        supervisor = st["background"]["ladder"][0]["pid"]
        assert pid_alive(supervisor) and pid_alive(rec["pid"])
        if victim == "pipeline":
            driver.kill()  # TerminateProcess / SIGKILL: no handler runs
            driver.wait()
        else:
            os.kill(supervisor, 9)
            assert wait_for(lambda: driver.poll() is not None, 60, "the pipeline to stop the run") is not None
            assert driver.returncode != 0
        wait_for(lambda: not pid_alive(supervisor) and not pid_alive(rec["pid"]), 10, "every process of the run to exit")
        killed_before = time.time()
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()
    assert read_json(run / "state.json")["background"]["ladder"][0]["iteration"] == 1
    write_json_atomic(run / "fake_match.json", {})
    driver = start_driver(run, 2)
    assert driver.wait(timeout=300) == 0
    st = read_json(run / "state.json")
    assert st["background"] == {"gate": None, "ladder": []} and set(st["strength"]) == {"1", "2"}
    assert_no_concurrent_writers(match_records(run), killed_before)


def test_supervisor_without_go_exits_without_starting_a_child(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(merge_config(CFG)), encoding="utf-8")
    spec = {"kind": "gate", "job_id": "gate_0001", "run": str(run), "log": "match_bg.log",
            "command": [sys.executable, str(FAKE_MATCH), "--a", "0001-a", "--b", "0000-b", "--config", str(run / "config.json"),
                        "--pairs", "1", "--seed", "1", "--out", str(run / "r.json")]}
    write_json_atomic(run / "spec.json", spec)
    from mango.proc import child_env
    for stdin in (b"", b"no\n"):
        proc = subprocess.run([sys.executable, "-m", "mango.bgjob", "--spec", str(run / "spec.json"), "--parent-pid",
                               str(os.getpid())], input=stdin, env=child_env(), capture_output=True, timeout=60)
        assert proc.returncode == 2
    assert not (run / "fake_match_log").exists() and not (run / "r.json").exists()
    proc = subprocess.run([sys.executable, "-m", "mango.bgjob", "--spec", str(run / "spec.json"), "--parent-pid",
                           str(os.getpid())], input=b"go\n", env=child_env(), capture_output=True, timeout=60)
    assert proc.returncode == 0 and (run / "r.json").exists()


@pytest.mark.skipif(not WINDOWS, reason="Job Objects are Windows-only")
def test_job_object_kills_its_tree_when_closed(tmp_path):
    from mango.proc import JobObject, RunContainer

    c = RunContainer()
    job = c.new_job()
    parent = subprocess.Popen([sys.executable, "-c",
                               "import subprocess, sys, time; "
                               "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                               "print(p.pid, flush=True); time.sleep(60)"], stdout=subprocess.PIPE, text=True)
    c.adopt(parent, job)
    child = int(parent.stdout.readline())
    # >= 2: a venv's python.exe may be a launcher that runs the interpreter as its child.
    assert job.active_processes() >= 2 and isinstance(job, JobObject)
    job.terminate()
    assert job.wait_empty(10) and not pid_alive(child)
    parent.wait()
    # Kill on close of the run-level object.
    p2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    c.adopt(p2)
    t0 = time.time()
    c.close()
    p2.wait(timeout=10)  # it would sleep for 60 s
    assert time.time() - t0 < 10


# --- real executables ---------------------------------------------------------------------------

def test_async_pipeline_with_the_real_executables(tmp_path):
    from test_pipeline import TINY, _bin_dir

    bin_dir = _bin_dir()
    if bin_dir is None:
        pytest.skip("C++ executables not built")
    cfg = merge_config({**TINY, "pipeline": {"async_gate": True, "async_ladder": True}})
    p = Pipeline(tmp_path / "async", cfg, bin_dir, device="cpu", selfplay_device="cpu", quiet=True)
    p.run_iterations(2)
    p.close()
    st = read_json(tmp_path / "async" / "state.json")
    assert st["background"] == {"gate": None, "ladder": []}
    assert set(st["strength"]) == {"1", "2"} and len(st["events"]) == 2
    assert (tmp_path / "async" / "matches" / "0002.json").exists()
    assert (tmp_path / "async" / "strength" / "fits" / "0002.json").exists()
    # Iteration 2 plays the initial model whatever gate 1 decided (one-iteration delay).
    assert st["selfplay_stats"]["2"]["model_id"] == st["initial_model_id"]
    for q in st["promotions"]:
        assert q["effective_selfplay_iteration"] == q["gate_iteration"] + 2
    log = (tmp_path / "async" / "logs" / "pipeline.log").read_text(encoding="utf-8")
    assert "background: launched gate_0001" in log and "background: gate_0001 finished" in log


# --- review 2026-09-30 -------------------------------------------------------------------------

def test_forced_termination_during_selfplay_leaves_no_driver_and_no_double_dispatch(tmp_path):
    """P1: the pipeline's foreground subprocesses (self-play here) die with it — Windows by
    the run-level Job Object, POSIX by the --exec supervisor and the foreground lock — and
    the restart dispatches the phase only once they are gone."""
    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(run / "fake_selfplay.json", {"hang": {"chunk_id_start": 1, "after_chunks": 1, "seconds": 120}})
    driver = start_driver(run, 1, config(False, True))
    try:
        disp = wait_for(lambda: sorted((run / "fake_dispatch").glob("*.json")) if (run / "fake_dispatch").exists() else None,
                        120, "the self-play launch")
        sp_pid = json.loads(disp[0].read_text(encoding="utf-8"))["pid"]
        assert pid_alive(sp_pid)
        driver.kill()
        driver.wait()
        wait_for(lambda: not pid_alive(sp_pid), 10, "the self-play driver to exit with the pipeline")
        killed_before = time.time()
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()
    write_json_atomic(run / "fake_selfplay.json", {})
    driver = start_driver(run, 1)
    assert driver.wait(timeout=300) == 0
    launches = sorted(int(q.name.split("_")[0]) / 1e9 for q in (run / "fake_dispatch").glob("*.json"))
    assert len(launches) == 2 and launches[1] > killed_before
    assert read_json(run / "state.json")["iteration"] == 2


def test_exec_mode_passes_output_and_exit_code_through(tmp_path):
    from mango.proc import child_env

    for code in (0, 7):
        proc = subprocess.run([sys.executable, "-m", "mango.bgjob", "--exec", "--parent-pid", str(os.getpid()), "--",
                               sys.executable, "-c", f"print('hello'); raise SystemExit({code})"],
                              env=child_env(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == code and proc.stdout.strip() == "hello"


def test_collection_completes_an_interrupted_fit_publication(tmp_path, fakes, monkeypatch):
    """P2: the fit file is written before ratings.json and ratings.csv; a ladder job
    interrupted in between is collected by its fit file, and the collection completes the
    publication (here the last job of the run, so no later fit would repair it)."""
    control = {"gate": {"1": 0.9, "2": 0.3}}
    cfg = config(False, True)
    ref = new_run(tmp_path, "ref", cfg, control)
    run_to(ref, 2)
    run = new_run(tmp_path, "run", cfg, control)
    seen = {"n": 0}

    def crash_point(self, name):
        if name == "collect:read":
            seen["n"] += 1
            if seen["n"] == 2:
                raise Crash(name)

    monkeypatch.setattr(Pipeline, "_crash_point", crash_point)
    with pytest.raises(Crash):
        run_to(run, 2)
    monkeypatch.undo()
    monkeypatch.setattr(Pipeline, "_selfplay_command", lambda self: [sys.executable, str(FAKE_SELFPLAY)])
    monkeypatch.setattr(Pipeline, "_match_command", lambda self: [sys.executable, str(FAKE_MATCH)])
    # The state a job leaves when it is interrupted right after writing its fit file.
    (run / "strength" / "ratings.json").unlink()
    (run / "strength" / "ratings.csv").unlink()
    assert (run / "strength" / "fits" / "0002.json").exists()
    assert read_json(run / "state.json")["background"]["ladder"][0]["iteration"] == 2
    run_to(run, 2)
    assert snapshot(run) == snapshot(ref)
