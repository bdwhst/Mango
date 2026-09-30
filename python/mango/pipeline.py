"""Sequential orchestration (docs/DESIGN.md section 6.5).

runs/<name>/
  config.json  state.json  models/<model_id>/  best.json  learner/ckpt_*.pt
  replay/manifest.json  replay/chunk_*.mgo  sgf/<iteration>/  matches/<iteration>.json
  strength/ (ladder, openings, ratings)  monitor/holdout.csv  logs/  check.json

Phases per iteration: selfplay, train, monitor, export, gate, promote, strength.
state.json is rewritten atomically before a phase starts (with the phase's plan) and
when it ends; every phase reconciles its outputs against the plan on restart. With
search.resign_auto the self-play plan carries the v_resign selected from the window's
no-resign games (DESIGN 5.4.7); the strength phase adds every promoted model and every
eval.ladder_every-th candidate to the frozen ladder (DESIGN 6.6). With selfplay.processes
>= 2 an iteration's games are split over N mango_selfplay processes from per-worker plan
records persisted before launch (DESIGN 5.5.1 step 0); N = 1 is the single-process path.

Overlapped evaluation (DESIGN 6.5.1): with pipeline.async_ladder the strength step runs
as background ladder jobs (one at a time, in iteration order); with pipeline.async_gate
an iteration's phases are selfplay, settle, train, monitor, export, launch_eval — the
gate of iteration i runs as a background job while iteration i+1 plays its games and is
settled after that self-play. Which model plays self-play in iteration j is decided by
the recorded promotions (`effective_selfplay_iteration`), never by wall-clock order.
Background jobs are supervisor processes (mango/bgjob.py) in a kill-on-close Job Object
(Windows) or their own process group (POSIX); only this process writes state.json and
best.json.

    python -m mango.pipeline --run runs/smoke --config configs/5x5-smoke.json \
        --bin build/windows-cuda/Release --iterations 30 [--device cuda] [--check] \
        [--async-ladder on|off] [--async-gate on|off]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from .chunk import EXTRAS_SEARCH_KIND, read_chunk, read_header
from .config import effective_move_cap, load_config, merge_config
from .data import write_holdout_games
from .export import export_model_version, load_eager_model, make_model_id
from .known_outcome import known_positions, value_sign_accuracy
from .proc import (WINDOWS, FileLock, ForegroundGuard, RunContainer, acquire_run_lock, child_env, release_run_lock,
                   run_polled, terminate_group)
from .resign import measured_false_positive_rate, select_resign_threshold
from .state import derive_seed, read_json, write_json_atomic
from .strength import Ladder, finish_fit_publication, fit_path, format_ratings, ladder_step, strength_record
from .train import (build_model, build_optimizer, build_window_dataset, bounded_steps, evaluate_dataset,
                    load_checkpoint, make_scaler, save_checkpoint, train_until)

# Sequential iteration: selfplay, train, monitor, export, gate, promote, strength. With
# async_gate: selfplay, settle, train, monitor, export, launch_eval. `settle` also runs in
# a sequential iteration that follows an asynchronous one (a gate is pending).
PHASES = ["selfplay", "settle", "train", "monitor", "export", "gate", "promote", "strength", "launch_eval"]
LAST_PHASES = ("strength", "launch_eval")


def select_selfplay_model(promotions: list[dict[str, Any]], iteration: int, initial_model_id: str) -> str:
    """The self-play model of an iteration (DESIGN 6.5.1): among the promotions with
    effective_selfplay_iteration <= iteration, the one with the largest
    (effective_selfplay_iteration, gate_iteration) — never the list order; with none, the
    run's initial model."""
    eligible = [p for p in promotions if int(p["effective_selfplay_iteration"]) <= iteration]
    if not eligible:
        return initial_model_id
    return max(eligible, key=lambda p: (int(p["effective_selfplay_iteration"]), int(p["gate_iteration"])))["candidate"]


def add_promotion(promotions: list[dict[str, Any]], gate_iteration: int, candidate: str, effective: int) -> bool:
    """Adds a promotion keyed by (gate_iteration, candidate) unless present."""
    if any(int(p["gate_iteration"]) == gate_iteration and p["candidate"] == candidate for p in promotions):
        return False
    promotions.append({"gate_iteration": int(gate_iteration), "candidate": candidate,
                       "effective_selfplay_iteration": int(effective)})
    return True


def promotions_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The promotion list of a run older than it (every gate was synchronous: e = i + 1)."""
    promotions: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("event") == "gate" and ev.get("promoted"):
            add_promotion(promotions, int(ev["iteration"]), ev["candidate"], int(ev["iteration"]) + 1)
    return promotions


def pick_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_selfplay_workers(target_games: int, processes: int, chunk_id_start: int, seed: int) -> list[dict[str, Any]]:
    """Per-worker records of a multi-process self-play plan (DESIGN 5.5.1 step 0): worker k
    plays `games` games with seed derive_seed(iteration seed, k) and chunk ids in
    [chunk_id_start, chunk_id_end), one id per game (chunks hold >= 1 game). Persisted
    before any process starts; a restart never re-splits."""
    if processes < 2:
        raise ValueError("a worker plan needs at least two processes")
    base, extra = divmod(int(target_games), processes)
    workers: list[dict[str, Any]] = []
    next_id = int(chunk_id_start)
    for k in range(processes):
        games = base + (1 if k < extra else 0)
        workers.append({"k": k, "games": games, "seed": derive_seed(seed, k), "chunk_id_start": next_id,
                        "chunk_id_end": next_id + games})
        next_id += games
    return workers


# Summary fields that describe one launch of the driver(s) and cannot be rebuilt for the
# launches a restart interrupted (DESIGN 5.5.1 step 0: listed under `incomplete`).
LAUNCH_ONLY_FIELDS = ("seconds", "evaluations", "batches", "avg_batch", "retries", "eval_seconds", "evals_per_s",
                      "positions_per_s", "games_in_flight", "workers")

TERMINATION_NAMES = {0: "two_passes", 1: "resign", 2: "move_cap"}


def selfplay_counts_from_chunks(paths: list[Path]) -> dict[str, Any]:
    """The per-iteration counts the driver summary reports, rebuilt from the published
    chunks (every game record carries its result and termination)."""
    games = positions = black_wins = 0
    terminations = {name: 0 for name in TERMINATION_NAMES.values()}
    for p in sorted(paths):
        chunk = read_chunk(p)
        games += chunk.header.num_games
        positions += chunk.header.num_positions
        for g in chunk.games:
            black_wins += 1 if g.result > 0 else 0
            name = TERMINATION_NAMES.get(int(g.termination), str(int(g.termination)))
            terminations[name] = terminations.get(name, 0) + 1
    return {"games": games, "positions": positions, "black_wins": black_wins, "terminations": terminations,
            "avg_game_length": round(positions / games, 2) if games else 0.0, "chunks": [p.name for p in sorted(paths)]}


def merge_selfplay_summaries(summaries: dict[int, dict[str, Any]], seconds: float, processes: int) -> dict[str, Any]:
    """One summary for the phase from the per-worker mango_selfplay summaries: counts summed,
    `seconds` the wall time of the phase, rates from those, `games_in_flight` the total
    concurrency N*G (DESIGN 5.5.1)."""
    parts = [summaries[k] for k in sorted(summaries)]

    def total(key: str) -> int:
        return sum(int(s.get(key, 0)) for s in parts)

    terminations: dict[str, int] = {}
    for s in parts:
        for name, count in s.get("terminations", {}).items():
            terminations[name] = terminations.get(name, 0) + int(count)
    games, positions, evaluations, batches = total("games"), total("positions"), total("evaluations"), total("batches")
    merged: dict[str, Any] = {
        "games": games,
        "positions": positions,
        "evaluations": evaluations,
        "batches": batches,
        "avg_batch": round(evaluations / batches, 2) if batches else 0.0,
        "retries": total("retries"),
        "seconds": round(seconds, 1),
        "eval_seconds": round(sum(float(s.get("eval_seconds", 0.0)) for s in parts), 1),
        "evals_per_s": round(evaluations / seconds, 1) if seconds > 0 else 0.0,
        "positions_per_s": round(positions / seconds, 1) if seconds > 0 else 0.0,
        "games_in_flight": total("games_in_flight"),
        "processes": processes,
        "avg_game_length": round(positions / games, 2) if games else 0.0,
        "black_wins": total("black_wins"),
        "terminations": terminations,
        "chunks": [c for s in parts for c in s.get("chunks", [])],
        "next_chunk_id": max(int(s.get("next_chunk_id", 0)) for s in parts),
        "workers": [{"k": k, "games": int(summaries[k].get("games", 0)), "positions": int(summaries[k].get("positions", 0)),
                     "seconds": summaries[k].get("seconds"), "positions_per_s": summaries[k].get("positions_per_s")}
                    for k in sorted(summaries)],
    }
    for key in ("resign_threshold", "model_id", "device", "iteration"):
        if key in parts[0]:
            merged[key] = parts[0][key]
    return merged


class Pipeline:
    def __init__(self, run_dir: str | Path, config: dict[str, Any] | None, bin_dir: str | Path, device: str | None = None,
                 selfplay_device: str | None = None, run_seed: int | None = None, quiet: bool = False):
        self.run = Path(run_dir)
        self.bin = Path(bin_dir)
        self.device = pick_device(device)
        self.selfplay_device = selfplay_device or (self.device.type if self.device.type != "mps" else "mps")
        self.quiet = quiet
        self.run.mkdir(parents=True, exist_ok=True)
        # One pipeline per run (DESIGN 6.5.1): refused if another process holds the lock.
        self._lock_key: str | None = acquire_run_lock(self.run / ".pipeline.lock")
        self.container = RunContainer()
        # POSIX: the foreground subprocesses' parent-death protection and lock (proc.ForegroundGuard).
        self.fg = ForegroundGuard(self.run, log=lambda m: print(m, file=sys.stderr, flush=True))
        self._bg: dict[str, dict[str, Any]] = {}  # running background jobs by job id
        self._bg_started = False
        self._overlap: set[str] = set()
        for sub in ("models", "learner", "replay", "sgf", "matches", "monitor", "logs"):
            (self.run / sub).mkdir(exist_ok=True)
        cfg_path = self.run / "config.json"
        if cfg_path.exists():
            self.cfg = merge_config(read_json(cfg_path))
        else:
            if config is None:
                raise ValueError("a config is required to create a run")
            self.cfg = merge_config(config)
            write_json_atomic(cfg_path, self.cfg)
        self.state_path = self.run / "state.json"
        self.state = read_json(self.state_path)
        if self.state is None:
            self._init_run(run_seed if run_seed is not None else 1)
        self._migrate_state()

    def close(self) -> None:
        """Stops every background job of this pipeline and releases the run lock."""
        try:
            self._terminate_all()
        finally:
            self.container.close()
            if getattr(self, "fg", None) is not None:
                self.fg.release()
            if self._lock_key is not None:
                release_run_lock(self._lock_key)
                self._lock_key = None

    def __del__(self) -> None:
        try:
            if getattr(self, "_lock_key", None) is not None:
                self.close()
        except Exception:
            pass

    def _migrate_state(self) -> None:
        """Keys added by DESIGN 6.5.1, rebuilt once for a run started by an older version:
        the promotion list from the recorded gate events (keyed, so a repeat adds nothing;
        it must reproduce best_model_id, else the run stops rather than change which model
        plays) and an empty background record."""
        changed = False
        if "promotions" not in self.state:
            promotions = promotions_from_events(self.state.get("events", []))
            latest = select_selfplay_model(promotions, 1 << 30, self.state["initial_model_id"])
            if latest != self.state["best_model_id"]:
                raise RuntimeError(f"cannot rebuild the promotion list: the gate events end at {latest}, "
                                   f"best_model_id is {self.state['best_model_id']}")
            self.state["promotions"] = promotions
            changed = True
        if "background" not in self.state:
            self.state["background"] = {"gate": None, "ladder": []}
            changed = True
        if changed:
            self.save_state()

    def _crash_point(self, name: str) -> None:
        """Test seam: the restart tests raise here to simulate a crash between two steps."""

    def _config_mode(self) -> dict[str, bool]:
        pl = self.cfg["pipeline"]
        return {"async_gate": bool(pl["async_gate"]), "async_ladder": bool(pl["async_ladder"])}

    def _mode(self) -> dict[str, bool]:
        """The switches of the current iteration: read from the config when its self-play
        plan is written and kept in the plan (a switch changes at iteration boundaries)."""
        return self.state["plan"].get("mode") or self._config_mode()

    def set_switches(self, async_ladder: bool | None = None, async_gate: bool | None = None) -> None:
        """Changes pipeline.async_* in the run's config.json (from the next iteration on)."""
        cfg = read_json(self.run / "config.json")
        section = cfg.setdefault("pipeline", {})
        if async_ladder is not None:
            section["async_ladder"] = bool(async_ladder)
        if async_gate is not None:
            section["async_gate"] = bool(async_gate)
        write_json_atomic(self.run / "config.json", cfg)
        self.cfg = merge_config(cfg)
        self.log(f"pipeline switches: {self._config_mode()} (from the next iteration boundary)")

    def _add_event(self, event: dict[str, Any]) -> bool:
        """Events are keyed by (iteration, event, candidate): a phase re-run after a crash
        adds nothing twice."""
        key = (event["iteration"], event["event"], event.get("candidate"))
        if any((e["iteration"], e["event"], e.get("candidate")) == key for e in self.state["events"]):
            return False
        self.state["events"].append(event)
        return True

    def selfplay_model(self, iteration: int) -> str:
        return select_selfplay_model(self.state["promotions"], iteration, self.state["initial_model_id"])

    # --- helpers ---------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [it {self.state.get('iteration', 0) if self.state else 0}] {msg}"
        if not self.quiet:
            print(line, flush=True)
        with open(self.run / "logs" / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def exe(self, name: str) -> Path:
        p = self.bin / (name + (".exe" if platform.system() == "Windows" else ""))
        if not p.exists():
            raise FileNotFoundError(f"missing executable {p}")
        return p

    def model_dir(self, model_id: str) -> Path:
        return self.run / "models" / model_id

    @property
    def n(self) -> int:
        return int(self.cfg["board"]["size"])

    def _adopt(self, proc: subprocess.Popen) -> None:
        """A foreground subprocess: into the run's container (Windows), its process group
        recorded for a restart (POSIX)."""
        self.container.adopt(proc)
        self.fg.record(proc)

    def _run_subprocess(self, args: list[Any], log_name: str) -> str:
        """Runs one command in the run's container, polling the background jobs while it
        waits (DESIGN 6.5.1); stderr to logs/<log_name>."""
        self.log("exec " + " ".join(str(a) for a in args))
        cmd, kwargs = self.fg.wrap(args)
        rc, out = run_polled(cmd, self.run / "logs" / log_name, poll=self._poll_background, started=self._adopt,
                             popen_kwargs=kwargs)
        if rc != 0:
            raise RuntimeError(f"{args[0]} failed with exit code {rc}; see logs/{log_name}")
        return out

    def _run_workers(self, launches: list[tuple[int, list[Any]]], log_prefix: str) -> dict[int, str]:
        """Runs the commands concurrently (stderr to logs/<prefix>_<k>.log) and returns their
        stdout by k. If one exits non-zero the others are terminated, then RuntimeError.
        Whatever interrupts the launch or the wait (a failed Popen, KeyboardInterrupt) leaves
        no worker behind: every started process is terminated and waited for, every log
        handle closed."""
        procs: dict[int, subprocess.Popen[str]] = {}
        outputs: dict[int, str] = {}
        logs = []
        readers: list[threading.Thread] = []
        failed: tuple[int, int] | None = None
        try:
            for k, args in launches:
                self.log(f"exec [worker {k}] " + " ".join(str(a) for a in args))
                errlog = open(self.run / "logs" / f"{log_prefix}_{k}.log", "a", encoding="utf-8")
                logs.append(errlog)
                cmd, kwargs = self.fg.wrap(args)
                procs[k] = subprocess.Popen([str(a) for a in cmd], stdout=subprocess.PIPE, stderr=errlog, text=True,
                                            **kwargs)
                self._adopt(procs[k])

            def read(k: int) -> None:
                outputs[k] = procs[k].stdout.read()  # type: ignore[union-attr]

            readers = [threading.Thread(target=read, args=(k,), daemon=True) for k in procs]
            for t in readers:
                t.start()
            running = set(procs)
            while running:
                for k in sorted(running):
                    rc = procs[k].poll()
                    if rc is None:
                        continue
                    running.discard(k)
                    if rc != 0 and failed is None:
                        failed = (k, rc)
                        self.log(f"worker {k} failed with exit code {rc}; terminating {len(running)} running worker(s)")
                        for j in running:
                            procs[j].terminate()
                if running:
                    self._poll_background()
                    time.sleep(0.1)
        finally:
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            for p in procs.values():
                if p.poll() is None:
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            for t in readers:
                t.join()
            for errlog in logs:
                errlog.close()
        if failed is not None:
            raise RuntimeError(f"{launches[0][1][0]} worker {failed[0]} failed with exit code {failed[1]}; see "
                               f"logs/{log_prefix}_{failed[0]}.log (the other workers were terminated; restart resumes "
                               f"every worker from the plan)")
        return outputs

    # --- initialisation --------------------------------------------------------------------

    def _init_run(self, run_seed: int) -> None:
        torch.manual_seed(run_seed)
        model = build_model(self.cfg)
        model_dir = export_model_version(model, self.run / "models", iteration=0, komi=self.cfg["board"]["komi"],
                                         move_cap=effective_move_cap(self.cfg))
        model_id = model_dir.name
        optimizer = build_optimizer(model.to(self.device), self.cfg)
        save_checkpoint(self.run / "learner" / "ckpt_0.pt", model, optimizer, make_scaler(self.device), 0, self.cfg)
        write_json_atomic(self.run / "best.json", {"model_id": model_id, "iteration": 0})
        write_json_atomic(self.run / "replay" / "manifest.json", {"chunks": []})
        self.state = {
            "run_seed": run_seed,
            "iteration": 1,
            "best_model_id": model_id,
            "initial_model_id": model_id,
            "v_resign": -1.0,
            "phase": "selfplay",
            "plan": {},
            "next_chunk_id": 1,
            "global_step": 0,
            "events": [],
            "promotions": [],
            "background": {"gate": None, "ladder": []},
        }
        self.save_state()
        self.log(f"initialised run with model {model_id} on {self.device}")

    # --- manifest --------------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        return read_json(self.run / "replay" / "manifest.json", {"chunks": []})

    def _published_chunks(self, prefix: str) -> list[Path]:
        return sorted(self.run.glob(f"replay/{prefix}*.mgo"))

    def _update_manifest(self, prefix: str, iteration: int) -> None:
        m = self.manifest()
        known = {c["path"] for c in m["chunks"]}
        for p in self._published_chunks(prefix):
            rel = p.name
            if rel in known:
                continue
            h = read_header(p)
            policy_positions = h.num_positions
            if h.record_extras & EXTRAS_SEARCH_KIND:
                policy_positions = int(sum(int(g.search_kind.sum()) for g in read_chunk(p).games))
            m["chunks"].append({"path": rel, "chunk_id": h.chunk_id, "model_id": h.model_id, "games": h.num_games,
                                "positions": h.num_positions, "policy_positions": policy_positions,
                                "iteration": iteration})
        m["chunks"].sort(key=lambda c: c["chunk_id"])
        write_json_atomic(self.run / "replay" / "manifest.json", m)

    def available_chunks(self) -> list[dict[str, Any]]:
        """Manifest entries whose file still exists (evicted chunks stay listed)."""
        return [c for c in self.manifest()["chunks"] if not c.get("evicted")]

    def window_chunks(self) -> list[Path]:
        """Most recent window_games games by manifest order (whole chunks, newest last)."""
        m = self.available_chunks()
        want = int(self.cfg["training"]["window_games"])
        chosen: list[str] = []
        games = 0
        for c in reversed(m):
            if games >= want:
                break
            chosen.append(c["path"])
            games += int(c["games"])
        return [self.run / "replay" / p for p in reversed(chosen)]

    def _evict_old_chunks(self) -> int:
        """Deletes chunk files that fell out of the window (DESIGN 6.3: between phases,
        with no trainer alive); their manifest entries are kept and marked evicted."""
        if not self.cfg["selfplay"].get("evict_old_chunks", True):
            return 0
        keep = {p.name for p in self.window_chunks()}
        m = self.manifest()
        evicted = 0
        for c in m["chunks"]:
            if c.get("evicted") or c["path"] in keep:
                continue
            p = self.run / "replay" / c["path"]
            if p.exists():
                p.unlink()
            c["evicted"] = True
            evicted += 1
        if evicted:
            write_json_atomic(self.run / "replay" / "manifest.json", m)
            self.log(f"evicted {evicted} chunk(s) that fell out of the window")
        return evicted

    def _resign_selection_chunks(self) -> list[Path]:
        """Chunks the resign selection reads: the window (DESIGN 5.4.7, the paper) or,
        with search.resign_select_scope = "latest", only the previous iteration's."""
        scope = self.cfg["search"].get("resign_select_scope", "window")
        if scope == "window":
            return self.window_chunks()
        if scope == "latest":
            it = int(self.state["iteration"])
            return self._published_chunks(f"chunk_{it - 1:04d}_") if it > 1 else []
        raise ValueError(f"unknown search.resign_select_scope {scope!r}")

    def _select_resign(self) -> dict[str, Any]:
        """v_resign for the next self-play phase (DESIGN 5.4.7): from the no-resign games
        of the selection scope when search.resign_auto, else the configured constant."""
        s = self.cfg["search"]
        if not s.get("resign_auto", False):
            return {"threshold": float(s["resign_threshold"]), "auto": False}
        games = [g for p in self._resign_selection_chunks() for g in read_chunk(p).games]
        sel = select_resign_threshold(games, max_fpr=float(s.get("resign_fpr_target", 0.05)))
        sel["auto"] = True
        sel["scope"] = s.get("resign_select_scope", "window")
        sel["fpr_target"] = float(s.get("resign_fpr_target", 0.05))
        return sel

    # --- phases ----------------------------------------------------------------------------

    def _begin(self, phase: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Persists the plan before the phase starts, unless one exists (restart)."""
        if self.state["phase"] == phase and self.state["plan"].get(phase):
            self.log(f"resuming phase {phase}")
            return self.state["plan"][phase]
        self.state["phase"] = phase
        self.state["plan"][phase] = plan
        self.save_state()
        return plan

    def _next_phase(self, phase: str) -> str:
        if phase == "selfplay":
            return "settle" if self.state["background"]["gate"] is not None else "train"
        if phase == "export":
            return "launch_eval" if self._mode()["async_gate"] else "gate"
        return {"settle": "train", "train": "monitor", "monitor": "export", "gate": "promote",
                "promote": "strength"}[phase]

    def _end(self, phase: str) -> None:
        if phase in LAST_PHASES:
            self.state["iteration"] += 1
            self.state["plan"] = {}
            self.state["phase"] = "selfplay"
        else:
            self.state["phase"] = self._next_phase(phase)
        self.save_state()

    def phase_selfplay(self) -> None:
        it = self.state["iteration"]
        if not self.state["plan"].get("selfplay"):
            self._evict_old_chunks()
            resign = self._select_resign()
            v_resign = resign["threshold"] if resign["threshold"] is not None else -1.0
            if resign.get("auto"):
                rate = resign["false_positive_rate"]
                self.log(f"resign: v_resign {v_resign:g} from {resign['samples']} no-resign games "
                         f"(false-positive rate {rate if rate is None else f'{rate:.3f}'}, {resign['no_resign_games']} tagged)")
        else:
            resign, v_resign = None, None
        if "mode" not in self.state["plan"]:
            self.state["plan"]["mode"] = self._config_mode()  # the switches of this iteration (DESIGN 6.5.1)
        new_plan = {
            "task_id": str(it),
            # The recorded promotions decide, not best_model_id (DESIGN 6.5.1; equal to it
            # in a sequential run).
            "model_id": self.selfplay_model(it),
            "target_games": int(self.cfg["selfplay"]["games_per_iteration"]),
            "chunk_prefix": f"chunk_{it:04d}_",
            "chunk_id_start": int(self.state["next_chunk_id"]),
            "seed": derive_seed(self.state["run_seed"], it),
            "v_resign": v_resign,
            "resign_selection": resign,
        }
        processes = int(self.cfg["selfplay"].get("processes", 1))
        if processes >= 2:
            # DESIGN 5.5.1 step 0: per-worker records, persisted with the plan before launch.
            new_plan["processes"] = processes
            new_plan["workers"] = split_selfplay_workers(new_plan["target_games"], processes, new_plan["chunk_id_start"],
                                                         new_plan["seed"])
        plan = self._begin("selfplay", new_plan)
        if "v_resign" not in plan:
            # Plan written by a version without resign selection (resignation was always
            # off): keep the threshold that phase was started with and persist it.
            plan["v_resign"] = float(self.state.get("v_resign", -1.0))
            plan["resign_selection"] = None
            self.save_state()
            self.log(f"resuming an older self-play plan; resign threshold {plan['v_resign']:g}")
        self.state["v_resign"] = plan["v_resign"]
        for tmp in self.run.glob("replay/*.tmp"):
            tmp.unlink()
        if plan.get("workers"):
            self._selfplay_workers(plan, it)
        else:
            self._selfplay_single(plan, it)
        self._update_manifest(plan["chunk_prefix"], it)
        if plan["v_resign"] is not None and plan["v_resign"] > -1.0:
            # Diagnostic: how often this iteration's eventual winners dipped below v_resign.
            games = [g for p in self._published_chunks(plan["chunk_prefix"]) for g in read_chunk(p).games]
            fpr = measured_false_positive_rate(games, plan["v_resign"])
            self.state.setdefault("selfplay_stats", {}).setdefault(str(it), {})["resign_false_positive"] = fpr
            rate = fpr["false_positive_rate"]
            cf = fpr["counterfactual"]
            cf_rate = cf["rate_of_triggered"] if cf else None
            self.log(f"resign: measured false-positive rate {rate if rate is None else f'{rate:.3f}'} on "
                     f"{fpr['samples']} no-resign games of this iteration; counterfactual label errors "
                     f"{cf['mislabelled'] if cf else 0}/{cf['triggered'] if cf else 0} of the games it would have ended "
                     f"({cf_rate if cf_rate is None else f'{cf_rate:.3f}'})")
        if plan.get("workers"):
            self.state["next_chunk_id"] = int(plan["workers"][-1]["chunk_id_end"])  # every block lies below it
        else:
            ids = [c["chunk_id"] for c in self.manifest()["chunks"]]
            self.state["next_chunk_id"] = (max(ids) + 1) if ids else 1
        self._end("selfplay")

    def _selfplay_command(self) -> list[Any]:
        """The self-play executable (tests substitute a fake driver here)."""
        return [self.exe("mango_selfplay")]

    def _selfplay_args(self, plan: dict[str, Any], it: int, games: int, chunk_id_start: int, seed: int,
                       worker: int | None = None) -> list[Any]:
        # Driver profile with its per-second timeline (DESIGN 5.5.1); a relaunch overwrites it.
        profile = self.run / "logs" / "selfplay_profile" / (f"{it:04d}.json" if worker is None else f"{it:04d}_k{worker}.json")
        return self._selfplay_command() + [
            "--model", self.model_dir(plan["model_id"]), "--config", self.run / "config.json", "--games", games, "--out",
            self.run / "replay", "--chunk-prefix", plan["chunk_prefix"], "--chunk-id-start", chunk_id_start, "--seed", seed,
            "--sgf-dir", self.run / "sgf" / f"{it:04d}", "--iteration", it, "--device", self.selfplay_device,
            "--resign-threshold", plan["v_resign"], "--profile-out", profile]

    def _selfplay_single(self, plan: dict[str, Any], it: int) -> None:
        """selfplay.processes = 1: one process for the games not yet published (M4 path)."""
        existing = 0
        max_id = plan["chunk_id_start"] - 1
        for p in self._published_chunks(plan["chunk_prefix"]):
            h = read_header(p)
            existing += h.num_games
            max_id = max(max_id, h.chunk_id)
        remaining = plan["target_games"] - existing
        if remaining <= 0:
            return
        args = self._selfplay_args(plan, it, remaining, max_id + 1, plan["seed"])
        t0 = time.time()
        out = self._run_subprocess(args, "selfplay.log")
        summary = json.loads(out.strip().splitlines()[-1])
        summary["seconds"] = round(time.time() - t0, 1)
        self.log(f"selfplay: {summary['games']} games, {summary['positions']} positions, "
                 f"avg length {summary['avg_game_length']:.1f}, {summary['terminations']}, "
                 f"resign {plan['v_resign']:g}, {summary['seconds']}s")
        pr = summary.get("profile")
        if pr:
            ev = pr.get("evaluator", {})
            self.log(f"selfplay profile: evaluation thread busy {pr['eval_busy']:.2f} ({ev.get('us_per_call', 0):.0f} us per "
                     f"call, avg batch {summary.get('avg_batch', 0):.0f}, {pr['rounds_per_batch']:.2f} rounds per forward), "
                     f"search threads busy {pr['search_busy']:.2f}")
        self.state.setdefault("selfplay_stats", {})[str(it)] = summary

    def _selfplay_workers(self, plan: dict[str, Any], it: int) -> None:
        """selfplay.processes >= 2 (DESIGN 5.5.1 step 0): every worker of the plan plays its
        own remaining quota — its `games` minus the games in the published chunks of its id
        block — from max(published id in its block) + 1, with its own seed, concurrently."""
        headers = [read_header(p) for p in self._published_chunks(plan["chunk_prefix"])]
        launches: list[tuple[int, list[Any]]] = []
        for w in plan["workers"]:
            mine = [h for h in headers if w["chunk_id_start"] <= h.chunk_id < w["chunk_id_end"]]
            remaining = int(w["games"]) - sum(h.num_games for h in mine)
            if remaining <= 0:
                continue
            start = max((h.chunk_id for h in mine), default=w["chunk_id_start"] - 1) + 1
            launches.append((int(w["k"]), self._selfplay_args(plan, it, remaining, start, int(w["seed"]), int(w["k"]))))
        resumed = bool(headers)  # a restart: earlier launches of this phase published these
        summary: dict[str, Any] = {"processes": len(plan["workers"])}
        if launches:
            t0 = time.time()
            outputs = self._run_workers(launches, "selfplay")
            summaries = {k: json.loads(out.strip().splitlines()[-1]) for k, out in outputs.items()}
            summary = merge_selfplay_summaries(summaries, time.time() - t0, len(plan["workers"]))
            per_worker = ", ".join(f"{w['k']}: {w['games']}g {w['positions_per_s']:.0f}pos/s" for w in summary["workers"]
                                   if w["positions_per_s"] is not None)
            self.log(f"selfplay: {summary['games']} games, {summary['positions']} positions, "
                     f"avg length {summary['avg_game_length']:.1f}, {summary['terminations']}, "
                     f"resign {plan['v_resign']:g}, {summary['seconds']}s over {len(launches)} of {len(plan['workers'])} "
                     f"processes ({summary['positions_per_s']:.0f} positions/s; {per_worker})")
        if resumed:
            # The launch above (if any) played only what the interrupted launches had not
            # published: the iteration's counts come from every published chunk, the
            # launch's own counts and timing are kept aside, and the fields that only the
            # interrupted launches could have supplied are flagged.
            launch = {k: summary[k] for k in ("games", "positions") + LAUNCH_ONLY_FIELDS if k in summary}
            summary.update(selfplay_counts_from_chunks(self._published_chunks(plan["chunk_prefix"])))
            summary["resumed"] = True
            summary["launch"] = launch
            summary["incomplete"] = [k for k in LAUNCH_ONLY_FIELDS if k in summary]
            self.log(f"selfplay: resumed phase — iteration totals {summary['games']} games, {summary['positions']} "
                     f"positions from the published chunks; this launch {launch.get('games', 0)} games; timing and "
                     f"evaluator fields cover this launch only")
        self.state.setdefault("selfplay_stats", {})[str(it)] = summary

    def _latest_ckpt(self, max_step: int | None = None) -> tuple[Path, int]:
        best: tuple[Path, int] | None = None
        for p in self.run.glob("learner/ckpt_*.pt"):
            m = re.match(r"ckpt_(\d+)\.pt$", p.name)
            if not m:
                continue
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            if best is None or step > best[1]:
                best = (p, step)
        if best is None:
            raise FileNotFoundError("no learner checkpoint")
        return best

    def phase_train(self) -> None:
        it = self.state["iteration"]
        if not self.state["plan"].get("train"):
            ckpt, step = self._latest_ckpt()
            window = [str(p.name) for p in self.window_chunks()]
            ds = build_window_dataset([self.run / "replay" / w for w in window], self.cfg, seed=derive_seed(it, 7))
            steps = bounded_steps(self.cfg, ds.train_positions)
            plan = self._begin("train", {
                "input_ckpt": ckpt.name, "start_global_step": step, "target_global_step": step + steps,
                "window": window, "window_train_positions": ds.train_positions, "window_games": ds.window.num_games,
            })
        else:
            plan = self._begin("train", {})
            ds = None
        target = plan["target_global_step"]
        ckpt, step = self._latest_ckpt(max_step=target)
        if plan["window_train_positions"] == 0:
            self.log("train: no training positions in the window; phase skipped")
            self._add_event({"iteration": it, "event": "train_skipped"})
        elif step < target:
            if ds is None:
                ds = build_window_dataset([self.run / "replay" / w for w in plan["window"]], self.cfg, seed=derive_seed(it, 7))
            model = build_model(self.cfg).to(self.device)
            optimizer = build_optimizer(model, self.cfg)
            scaler = make_scaler(self.device)
            load_checkpoint(ckpt, model, optimizer, scaler, self.device)
            self.log(f"train: {plan['window_games']} games / {plan['window_train_positions']} positions, "
                     f"steps {step} -> {target} on {self.device}")
            t0 = time.time()
            stats = train_until(model, optimizer, scaler, ds, self.cfg, step, target, self.device,
                                ckpt_path=self.run / "learner" / f"ckpt_{target}.pt", log=self.log)
            save_checkpoint(self.run / "learner" / f"ckpt_{target}.pt", model, optimizer, scaler, target, self.cfg)
            stats["seconds"] = round(time.time() - t0, 1)
            self.log(f"train: done in {time.time() - t0:.1f}s, mean loss {stats.get('total', float('nan')):.4f} "
                     f"(v {stats.get('value', float('nan')):.4f} p {stats.get('policy', float('nan')):.4f})")
            self.state.setdefault("train_stats", {})[str(it)] = stats
        self.state["global_step"] = target
        self._end("train")

    def _fixed_holdout(self) -> Path | None:
        """The fixed validation set (DESIGN 6.6): the holdout games of iteration
        training.fixed_holdout_iteration, frozen once into monitor/fixed_holdout.mgo. None
        when disabled or not (yet) available."""
        k = int(self.cfg["training"].get("fixed_holdout_iteration", 0))
        if k <= 0:
            return None
        path = self.run / "monitor" / "fixed_holdout.mgo"
        if path.exists():
            return path
        if int(self.state["iteration"]) < k:
            return None
        chunks = self._published_chunks(f"chunk_{k:04d}_")
        if not chunks:
            self.log(f"monitor: fixed holdout set from iteration {k} cannot be built (chunks gone)")
            return None
        n = write_holdout_games(chunks, path)
        if n == 0:
            self.log(f"monitor: iteration {k} has no holdout games; fixed validation set is empty")
            return None
        self.log(f"monitor: froze {n} holdout games of iteration {k} as the fixed validation set")
        return path

    def phase_monitor(self) -> None:
        it = self.state["iteration"]
        plan = self._begin("monitor", {"step": self.state["global_step"], "window": self.state["plan"]["train"]["window"]})
        csv_path = self.run / "monitor" / "holdout.csv"
        done = False
        fields_on_disk = None
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields_on_disk = reader.fieldnames
                done = any(int(r["iteration"]) == it for r in reader)
        if not done:
            paths = [self.run / "replay" / w for w in plan["window"]]
            model = build_model(self.cfg).to(self.device)
            ckpt, _ = self._latest_ckpt(max_step=plan["step"])
            load_checkpoint(ckpt, model, None, None, self.device, restore_rng=False)
            hold = build_window_dataset(paths, self.cfg, seed=0, holdout=True, symmetry=False)
            train = build_window_dataset(paths, self.cfg, seed=0, holdout=False, symmetry=False)
            h = evaluate_dataset(model, hold, self.device)
            tr = evaluate_dataset(model, train, self.device, max_positions=4096, seed=derive_seed(it, 11))
            fixed_path = self._fixed_holdout()
            fx = {"positions": 0, "value_mse": float("nan"), "policy_ce": float("nan")}
            if fixed_path is not None:
                fixed = build_window_dataset([fixed_path], self.cfg, seed=0, holdout=True, symmetry=False)
                fx = evaluate_dataset(model, fixed, self.device)
            row = {"iteration": it, "step": plan["step"], "holdout_positions": h["positions"],
                   "holdout_value_mse": h["value_mse"], "holdout_policy_ce": h["policy_ce"],
                   "train_positions": tr["positions"], "train_value_mse": tr["value_mse"], "train_policy_ce": tr["policy_ce"],
                   "fixed_positions": fx["positions"], "fixed_value_mse": fx["value_mse"], "fixed_policy_ce": fx["policy_ce"]}
            new = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                # A CSV started by an older version keeps its columns (extra fields are dropped).
                w = csv.DictWriter(f, fieldnames=list(fields_on_disk or row.keys()), extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
            self.log(f"monitor: holdout v_mse {h['value_mse']:.4f} p_ce {h['policy_ce']:.4f} ({h['positions']} pos); "
                     f"train sample v_mse {tr['value_mse']:.4f} p_ce {tr['policy_ce']:.4f}"
                     + (f"; fixed set v_mse {fx['value_mse']:.4f} p_ce {fx['policy_ce']:.4f} ({fx['positions']} pos)"
                        if fixed_path is not None else ""))
        self._end("monitor")

    def phase_export(self) -> None:
        it = self.state["iteration"]
        ckpt, step = self._latest_ckpt(max_step=self.state["global_step"])
        model = build_model(self.cfg)
        load_checkpoint(ckpt, model, None, None, torch.device("cpu"), restore_rng=False)
        plan = self._begin("export", {"input_ckpt": ckpt.name, "candidate_model_id": make_model_id(it, model)})
        d = self.model_dir(plan["candidate_model_id"])
        if not (d / "model.json").exists():
            export_model_version(model, self.run / "models", iteration=it, komi=self.cfg["board"]["komi"],
                                 move_cap=effective_move_cap(self.cfg), model_id=plan["candidate_model_id"])
            self.log(f"export: {plan['candidate_model_id']} from {ckpt.name} (step {step})")
        self._end("export")

    def _match_command(self) -> list[Any]:
        """The mango_match executable (tests substitute a scripted driver here)."""
        return [self.exe("mango_match")]

    def _gate_args(self, plan: dict[str, Any]) -> list[Any]:
        return self._match_command() + [
            "--a", self.model_dir(plan["candidate_model_id"]), "--b", self.model_dir(plan["incumbent_model_id"]),
            "--config", self.run / "config.json", "--pairs", int(self.cfg["eval"]["pairs"]), "--seed", plan["match_seed"],
            "--out", self.run / plan["report"], "--device", self.selfplay_device, "--games-in-flight",
            int(self.cfg["selfplay"]["games_in_flight"])]

    def _gate_plan(self, it: int) -> dict[str, Any]:
        return {"candidate_model_id": self.state["plan"]["export"]["candidate_model_id"],
                "incumbent_model_id": self.state["best_model_id"], "report": f"matches/{it:04d}.json",
                "match_seed": derive_seed(self.state["run_seed"], 1000 + it)}

    def _gate_report(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        """The gate report if it exists and names the plan's candidate, incumbent and seed."""
        report = read_json(self.run / plan["report"])
        valid = (report is not None and report.get("a") == plan["candidate_model_id"]
                 and report.get("b") == plan["incumbent_model_id"] and report.get("seed") == plan["match_seed"])
        return report if valid else None

    def _write_skipped_gate(self, plan: dict[str, Any]) -> None:
        write_json_atomic(self.run / plan["report"], {"a": plan["candidate_model_id"], "b": plan["incumbent_model_id"],
                                                      "seed": plan["match_seed"], "skipped": True})

    def _gate_decision(self, report: dict[str, Any]) -> bool:
        if report.get("skipped"):
            return True
        return bool(report["mean_pair_score"] > float(self.cfg["eval"]["gate_threshold"]))

    def _log_gate(self, report: dict[str, Any], seconds: float) -> None:
        if report.get("skipped"):
            return
        self.log(f"gate: candidate mean pair score {report['mean_pair_score']:.3f} "
                 f"[{report['ci95'][0]:.3f}, {report['ci95'][1]:.3f}], unique {report['unique_trajectories']}/"
                 f"{report['games']}, {seconds:.1f}s")

    def phase_gate(self) -> None:
        it = self.state["iteration"]
        plan = self._begin("gate", self._gate_plan(it))
        if self._gate_report(plan) is None:
            if not self.cfg["eval"]["gating"]:
                self._write_skipped_gate(plan)
            else:
                t0 = time.time()
                self._run_subprocess(self._gate_args(plan), "match.log")
                self._log_gate(read_json(self.run / plan["report"]), time.time() - t0)
        self._end("gate")

    def phase_promote(self) -> None:
        it = self.state["iteration"]
        gate = self.state["plan"]["gate"]
        plan = self._begin("promote", {"candidate_model_id": gate["candidate_model_id"], "decision": None})
        if plan["decision"] is None:
            plan["decision"] = self._gate_decision(read_json(self.run / gate["report"]))
            self.save_state()
        if plan["decision"]:
            # A synchronous gate: the candidate plays from the next iteration on (DESIGN 6.5.1).
            if add_promotion(self.state["promotions"], it, plan["candidate_model_id"], it + 1):
                self.save_state()
            write_json_atomic(self.run / "best.json", {"model_id": plan["candidate_model_id"], "iteration": it})
            self.state["best_model_id"] = plan["candidate_model_id"]
        self._add_event({"iteration": it, "event": "gate", "candidate": plan["candidate_model_id"],
                         "promoted": plan["decision"]})
        self.log(f"promote: {'promoted' if plan['decision'] else 'rejected'} {plan['candidate_model_id']}")
        self._end("promote")

    def _ladder_add(self, iteration: int, promoted: bool) -> bool:
        every = int(self.cfg["eval"]["ladder_every"])
        return bool(promoted) or (every > 0 and iteration % every == 0)

    def phase_strength(self) -> None:
        """Frozen ladder (DESIGN 6.6): every promoted model and every ladder_every-th
        candidate is added and plays the anchors and its nearest neighbours — inline, or
        queued for the ladder lane with pipeline.async_ladder (6.5.1)."""
        it = self.state["iteration"]
        promote = self.state["plan"]["promote"]
        plan = self._begin("strength", {"add": self._ladder_add(it, promote["decision"]),
                                        "model_id": promote["candidate_model_id"], "promoted": bool(promote["decision"])})
        if plan["add"]:
            self._strength_step(it, plan["model_id"], plan["promoted"])
        self._end("strength")

    def _strength_step(self, iteration: int, model_id: str, promoted: bool) -> None:
        if self._mode()["async_ladder"]:
            self._enqueue_ladder(iteration, model_id, promoted)
            return
        # Inline: the ladder files have one writer, and entries arrive in iteration order.
        self._drain_ladder()
        ladder = Ladder(self.run, self.cfg, self.bin, self.state["run_seed"], self.selfplay_device,
                        runner=self._run_subprocess, log=self.log, match_command=self._match_command())
        t0 = time.time()
        fit = ladder_step(ladder, self.state["initial_model_id"], model_id, iteration, promoted)
        self._record_strength(iteration, fit, ladder.data, model_id, time.time() - t0)

    def _record_strength(self, iteration: int, fit: dict[str, Any], ladder_data: dict[str, Any], model_id: str,
                         seconds: float) -> None:
        rec = strength_record(fit, ladder_data, model_id)
        self.state.setdefault("strength", {})[str(iteration)] = rec
        self.log(f"strength: {model_id} rated {rec['elo']:.0f} [{rec['low']:.0f}, {rec['high']:.0f}] Elo "
                 f"({'separated, ' if rec['separated'] else ''}{rec['games']} games; ladder {rec['entries']} "
                 f"entries, {rec['matches']} matches, {seconds:.0f}s)")
        self.log("strength ladder:\n" + format_ratings(fit, ladder_data["entries"]))

    # --- overlapped evaluation (DESIGN 6.5.1) ------------------------------------------------

    def phase_launch_eval(self) -> None:
        """async_gate: records the gate of this iteration's candidate (against the incumbent
        now, effective from iteration + 2), launches it in the background and ends the
        iteration."""
        it = self.state["iteration"]
        bg = self.state["background"]
        if bg["gate"] is None:
            g = {"iteration": it, "plan": self._gate_plan(it), "effective_selfplay_iteration": it + 2, "decision": None}
            if not self.cfg["eval"]["gating"]:
                self._write_skipped_gate(g["plan"])
                g["decision"] = True
            bg["gate"] = g
            self.save_state()
        elif int(bg["gate"]["iteration"]) != it:
            raise RuntimeError(f"a gate of iteration {bg['gate']['iteration']} is still pending in iteration {it}")
        self._crash_point("launch_eval:recorded")
        self._ensure_gate_job(bg["gate"])
        self._end("launch_eval")

    def phase_settle(self) -> None:
        self._settle()
        self._end("settle")

    def _settle(self) -> None:
        """Completes the pending gate (DESIGN 6.5.1 "Settle"): each step idempotent by a
        stable key, re-run from the first step after any crash."""
        g = self.state["background"]["gate"]
        if g is None:
            return
        i = int(g["iteration"])
        plan = g["plan"]
        cand = plan["candidate_model_id"]
        # 1. the decision from the report (waiting for the job if needed)
        if g["decision"] is None:
            report = self._gate_report(plan)
            if report is None:
                self._ensure_gate_job(g)
                self._wait_job(f"gate_{i:04d}")
                report = self._gate_report(plan)
                if report is None:
                    raise RuntimeError(f"gate job {i} finished without a valid report {plan['report']}")
            self._crash_point("settle:report")
            g["decision"] = self._gate_decision(report)
            self.save_state()
        self._crash_point("settle:decision")
        # 2. the promotion (keyed), then best.json
        if g["decision"]:
            if add_promotion(self.state["promotions"], i, cand, int(g["effective_selfplay_iteration"])):
                self.save_state()
            self._crash_point("settle:promotion")
            write_json_atomic(self.run / "best.json", {"model_id": cand, "iteration": i})
            self.state["best_model_id"] = cand
        # 3. the gating event (keyed)
        self._add_event({"iteration": i, "event": "gate", "candidate": cand, "promoted": g["decision"]})
        self.save_state()
        if g["decision"]:
            self.log(f"promote: promoted {cand} (gate of iteration {i}; self-play from iteration "
                     f"{g['effective_selfplay_iteration']})")
        else:
            self.log(f"promote: rejected {cand} (gate of iteration {i})")
        self._crash_point("settle:event")
        # 4. the strength step of candidate i (queued keyed by iteration, or inline)
        if self._ladder_add(i, g["decision"]):
            self._strength_step(i, cand, bool(g["decision"]))
        self._crash_point("settle:strength")
        # 5. done
        self.state["background"]["gate"] = None
        self.save_state()

    def _enqueue_ladder(self, iteration: int, model_id: str, promoted: bool) -> None:
        queue = self.state["background"]["ladder"]
        if any(int(j["iteration"]) == iteration for j in queue) or str(iteration) in self.state.get("strength", {}):
            return
        queue.append({"iteration": iteration, "model_id": model_id, "promoted": bool(promoted)})
        self.save_state()
        self.log(f"background: ladder job {iteration} queued ({len(queue)} in the lane)")
        self._pump_ladder()

    def _ensure_gate_job(self, g: dict[str, Any]) -> None:
        job_id = f"gate_{int(g['iteration']):04d}"
        if job_id in self._bg or g["decision"] is not None or self._gate_report(g["plan"]) is not None:
            return
        spec = {"kind": "gate", "log": "match_bg.log", "command": [str(a) for a in self._gate_args(g["plan"])]}
        self._launch_job(job_id, spec, g)

    def _pump_ladder(self) -> None:
        """Starts the head of the ladder queue when the lane is idle; a head whose fit file
        exists (the job finished, the collection did not) is collected instead."""
        queue = self.state["background"]["ladder"]
        while queue and not any(k.startswith("ladder_") for k in self._bg):
            head = queue[0]
            if fit_path(self.run / "strength", head["iteration"]).exists():
                self._collect_ladder(head)
                continue
            spec = {"kind": "ladder", "log": "ladder_job.log", "run_seed": self.state["run_seed"], "bin": str(self.bin),
                    "device": self.selfplay_device, "match_command": [str(a) for a in self._match_command()],
                    "initial_model_id": self.state["initial_model_id"], "model_id": head["model_id"],
                    "iteration": int(head["iteration"]), "promoted": bool(head["promoted"])}
            self._launch_job(f"ladder_{int(head['iteration']):04d}", spec, head)
            break

    def _collect_ladder(self, head: dict[str, Any]) -> None:
        it = int(head["iteration"])
        # The fit file is the job's completion mark, but it is written before ratings.json
        # and ratings.csv: complete (idempotently) whatever an interruption left unwritten.
        fit = finish_fit_publication(self.run / "strength", it)
        ladder_data = read_json(self.run / "strength" / "ladder.json")
        self._crash_point("collect:read")
        self._record_strength(it, fit, ladder_data, head["model_id"], float(head.get("seconds", 0.0)))
        queue = self.state["background"]["ladder"]
        if queue and int(queue[0]["iteration"]) == it:
            queue.pop(0)
        self.save_state()

    def _job_lock(self, job_id: str) -> FileLock:
        return FileLock(self.run / "jobs" / f"{job_id}.lock")

    def _acquire_job_lock(self, job_id: str, record: dict[str, Any]) -> FileLock:
        """POSIX relaunch rule: the job lock must be free (every process of an earlier copy
        of the job has exited). If it is held, the recorded process group is signalled and
        the lock awaited for up to 60 s; then the run fails rather than start a second writer."""
        lock = self._job_lock(job_id)
        if lock.try_acquire():
            return lock
        pgid = record.get("pgid")
        self.log(f"background: {job_id} is still held by process group {pgid}; signalling it")
        if pgid:
            terminate_group(int(pgid))
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if lock.try_acquire():
                return lock
            time.sleep(0.2)
        raise RuntimeError(f"background job {job_id}: its lock is still held after 60 s (process group {pgid}); "
                           f"stop those processes and restart")

    def _launch_job(self, job_id: str, spec: dict[str, Any], record: dict[str, Any]) -> None:
        """Starts a supervisor (python -m mango.bgjob) for a recorded job: placed in the
        run's container and its own nested Job Object (Windows) or its own process group
        holding the job lock (POSIX) before it reads `go` and starts work."""
        jobs = self.run / "jobs"
        jobs.mkdir(exist_ok=True)
        spec = {**spec, "job_id": job_id, "run": str(self.run.resolve())}
        spec_path = jobs / f"{job_id}.json"
        write_json_atomic(spec_path, spec)
        args = [sys.executable, "-m", "mango.bgjob", "--spec", str(spec_path), "--parent-pid", str(os.getpid())]
        kwargs: dict[str, Any] = {}
        lock = None
        if not WINDOWS:
            lock = self._acquire_job_lock(job_id, record)
            args += ["--lock-fd", str(lock.fileno())]
            kwargs = {"pass_fds": (lock.fileno(),), "start_new_session": True}
        errlog = open(self.run / "logs" / spec["log"], "a", encoding="utf-8")
        job = self.container.new_job()
        try:
            try:
                proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=errlog, stderr=errlog, text=True,
                                        env=child_env(), **kwargs)
            finally:
                if lock is not None:
                    lock.release()  # the supervisor holds its inherited copy from here on
            try:
                self.container.adopt(proc, job)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        except BaseException:
            errlog.close()
            if job is not None:
                job.close()
            raise
        try:
            proc.stdin.write("go\n")  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass  # the supervisor already exited; the next poll reports it
        record["pid"] = proc.pid
        record["pgid"] = None if WINDOWS else proc.pid
        self.save_state()
        self._bg[job_id] = {"proc": proc, "job": job, "errlog": errlog, "t0": time.time(), "record": record,
                            "log": spec["log"]}
        self._overlap.add(job_id)
        self.log(f"background: launched {job_id} (pid {proc.pid})")

    def _wait_tree(self, job_id: str, rt: dict[str, Any], stop: bool) -> None:
        """Returns when no process of the job is alive (DESIGN 6.5.1: nothing reads or
        relaunches a job before that). `stop` terminates the tree first."""
        if WINDOWS:
            job = rt["job"]
            if stop or job.active_processes() > 0:
                job.terminate()
            if not job.wait_empty(60.0):
                raise RuntimeError(f"background job {job_id}: processes still alive 60 s after termination")
        else:
            if stop:
                terminate_group(rt["proc"].pid)
            lock = self._job_lock(job_id)
            deadline = time.time() + 60.0
            while not lock.try_acquire():
                if time.time() > deadline:
                    raise RuntimeError(f"background job {job_id}: processes still alive 60 s after termination")
                terminate_group(rt["proc"].pid)
                time.sleep(0.2)
            lock.release()
        try:
            rt["proc"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            rt["proc"].kill()
            rt["proc"].wait()
        rt["errlog"].close()
        if rt["job"] is not None:
            rt["job"].close()

    def _terminate_all(self) -> None:
        """Stops every background job of this pipeline and waits for their trees."""
        for job_id in list(self._bg):
            rt = self._bg.pop(job_id)
            try:
                self._wait_tree(job_id, rt, stop=True)
                self.log(f"background: stopped {job_id}")
            except Exception as e:  # keep stopping the others
                self.log(f"background: stopping {job_id} failed: {e}")

    def _poll_background(self) -> None:
        """Collects finished jobs and starts the next ladder job (at phase boundaries and
        while the main thread waits for a subprocess). A job that failed stops the run:
        every background job is terminated, the state keeps it for the restart."""
        for job_id in list(self._bg):
            self._overlap.add(job_id)
            rt = self._bg[job_id]
            rc = rt["proc"].poll()
            if rc is None:
                continue
            del self._bg[job_id]
            self._wait_tree(job_id, rt, stop=False)
            seconds = time.time() - rt["t0"]
            if rc != 0:
                self.log(f"background: {job_id} failed with exit code {rc} after {seconds:.1f}s; stopping the run")
                self._terminate_all()
                raise RuntimeError(f"background job {job_id} failed with exit code {rc}; see logs/{rt['log']} "
                                   f"(the restart resumes it)")
            self.log(f"background: {job_id} finished ({seconds:.1f}s)")
            if job_id.startswith("gate_"):
                report = self._gate_report(rt["record"]["plan"])
                if report is not None:
                    self._log_gate(report, seconds)
            else:
                rt["record"]["seconds"] = round(seconds, 1)
                self._collect_ladder(rt["record"])
        self._pump_ladder()

    def _wait_job(self, job_id: str) -> None:
        while job_id in self._bg:
            self._poll_background()
            time.sleep(0.1)

    def _drain_ladder(self) -> None:
        self._start_background()
        while self.state["background"]["ladder"] or any(k.startswith("ladder_") for k in self._bg):
            self._poll_background()
            time.sleep(0.1)

    def _start_background(self) -> None:
        """Once per pipeline: relaunch the recorded jobs of an interrupted run (a pending
        gate without decision and valid report; the head of the ladder queue)."""
        if self._bg_started:
            return
        self._bg_started = True
        g = self.state["background"]["gate"]
        if g is not None:
            self._ensure_gate_job(g)
        self._pump_ladder()

    def drain(self) -> None:
        """End of --iterations / --hours (DESIGN 6.5.1): settles a pending gate (its
        promotion keeps its recorded effective iteration), then empties the ladder lane."""
        self._start_background()
        self._settle()
        self._drain_ladder()

    def run_phase(self) -> None:
        self._start_background()
        phase, it = self.state["phase"], self.state["iteration"]
        self._overlap = set(self._bg)
        self._crash_point("phase:" + phase)
        t0 = time.time()
        getattr(self, "phase_" + phase)()
        seconds = time.time() - t0
        overlapped = sorted(self._overlap)
        with open(self.run / "logs" / "phase_times.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"iteration": it, "phase": phase, "seconds": round(seconds, 2),
                                "overlapped": overlapped, "end": round(time.time(), 2)}) + "\n")
        if overlapped:
            self.log(f"phase {phase}: {seconds:.1f}s, overlapped: {', '.join(overlapped)}")
        self._poll_background()

    def run_iterations(self, iterations: int) -> None:
        """Runs until iteration `iterations` has completed all its phases, then drains the
        background work. Any exception stops every background job first."""
        try:
            while self.state["iteration"] <= iterations:
                self.run_phase()
            self.drain()
        except BaseException:
            self._terminate_all()
            raise

    def run_for_seconds(self, budget: float) -> int:
        """Equal-wall-clock budgets (DESIGN 8.2): runs whole iterations until `budget`
        seconds have elapsed in this call; the iteration in progress is always completed,
        then the background work is drained. Returns the number of iterations completed."""
        t0 = time.time()
        done = 0
        try:
            while True:
                start = self.state["iteration"]
                while self.state["iteration"] == start:
                    self.run_phase()
                done += 1
                if time.time() - t0 >= budget:
                    break
            self.drain()
        except BaseException:
            self._terminate_all()
            raise
        return done

    # --- M3a' learning check ---------------------------------------------------------------

    @torch.no_grad()
    def _terminal_sign_accuracy(self, model, max_chunks: int = 20) -> dict[str, Any]:
        """Acceptance metric: sign(v) vs the game result on the last two positions of holdout
        games that ended by two passes (the area score of those positions is the result; the
        last two moves were passes, so the stones are final). On-distribution, never trained on."""
        from .chunk import TERMINATION_TWO_PASSES

        chunks = [self.run / "replay" / c["path"] for c in self.available_chunks()[-max_chunks:]]
        if not chunks:
            return {"positions": 0, "accuracy": float("nan")}
        ds = build_window_dataset(chunks, self.cfg, seed=0, holdout=True, symmetry=False)
        model.eval()
        ok = tot = 0
        for gi, gd in enumerate(ds.window.games):
            g = gd.record
            if not gd.holdout or g.termination != TERMINATION_TWO_PASSES or g.T < 3:
                continue
            for t in (g.T - 1, g.T - 2):
                planes, _, z, _ = ds.example(gi, t, 0)
                _, v = model(torch.from_numpy(planes[None].astype("float32")).to(self.device))
                ok += int((float(v) > 0) == (z > 0))
                tot += 1
        return {"positions": tot, "accuracy": (ok / tot) if tot else float("nan")}

    def check(self, pairs: int | None = None) -> dict[str, Any]:
        n = self.n
        best_id = self.state["best_model_id"]
        initial_id = self.state["initial_model_id"]
        model = load_eager_model(self.model_dir(best_id)).to(self.device)
        positions = known_positions(n, float(self.cfg["board"]["komi"]))
        acc = value_sign_accuracy(model, positions, n, self.device)
        initial = load_eager_model(self.model_dir(initial_id)).to(self.device)
        acc0 = value_sign_accuracy(initial, positions, n, self.device)
        terminal = self._terminal_sign_accuracy(model)
        report_path = self.run / "check_match.json"
        args = [self.exe("mango_match"), "--a", self.model_dir(best_id), "--b", self.model_dir(initial_id), "--config",
                self.run / "config.json", "--pairs", pairs or int(self.cfg["eval"]["pairs"]), "--seed",
                derive_seed(self.state["run_seed"], 999_999), "--out", report_path, "--device", self.selfplay_device,
                "--games-in-flight", int(self.cfg["selfplay"]["games_in_flight"])]
        self._run_subprocess(args, "check_match.log")
        match = read_json(report_path)
        result = {
            "best_model_id": best_id,
            "initial_model_id": initial_id,
            "iterations_completed": self.state["iteration"] - 1,
            "known_outcome": {"best": acc, "initial": acc0},
            "holdout_terminal": terminal,
            "match_vs_initial": {k: match[k] for k in ("pairs", "games", "mean_pair_score", "ci95", "unique_trajectories",
                                                          "wins_a", "losses_a", "draws_a")},
            # Acceptance (DESIGN 11, M3a'): value sign on held-out two-pass terminal positions
            # >= 95 % and the match interval above 0.5. The settled templates are a diagnostic.
            "pass": bool(terminal["accuracy"] >= 0.95 and match["ci95"][0] > 0.5),
        }
        write_json_atomic(self.run / "check.json", result)
        self.log(f"check: holdout terminal sign accuracy {terminal['accuracy']:.3f} ({terminal['positions']} pos); "
                 f"settled-template sign accuracy {acc['accuracy']:.3f} (initial {acc0['accuracy']:.3f}, diagnostic); "
                 f"match vs initial {match['mean_pair_score']:.3f} [{match['ci95'][0]:.3f}, {match['ci95'][1]:.3f}]; "
                 f"{'PASS' if result['pass'] else 'FAIL'}")
        return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mango sequential pipeline (DESIGN 6.5)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", help="config JSON (required to create a run)")
    ap.add_argument("--bin", required=True, help="directory with mango_selfplay / mango_match")
    ap.add_argument("--iterations", type=int, default=0, help="run until this iteration has completed")
    ap.add_argument("--hours", type=float, default=0.0, help="run whole iterations until this wall-clock budget is used")
    ap.add_argument("--device", default="auto", help="training device: auto | cuda | mps | cpu")
    ap.add_argument("--selfplay-device", default=None, help="device for the C++ engines (default: same)")
    ap.add_argument("--seed", type=int, default=1, help="run seed (new runs only)")
    ap.add_argument("--check", action="store_true", help="run the M3a' learning check at the end")
    ap.add_argument("--check-pairs", type=int, default=None)
    ap.add_argument("--async-ladder", choices=["on", "off"], default=None,
                    help="set pipeline.async_ladder in the run's config (DESIGN 6.5.1; from the next iteration)")
    ap.add_argument("--async-gate", choices=["on", "off"], default=None,
                    help="set pipeline.async_gate in the run's config (DESIGN 6.5.1; from the next iteration)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    p = Pipeline(args.run, cfg, args.bin, args.device, args.selfplay_device, args.seed)
    try:
        if args.async_ladder is not None or args.async_gate is not None:
            p.set_switches(None if args.async_ladder is None else args.async_ladder == "on",
                           None if args.async_gate is None else args.async_gate == "on")
        if args.iterations > 0:
            p.run_iterations(args.iterations)
        if args.hours > 0:
            p.run_for_seconds(args.hours * 3600.0)
        if args.check:
            result = p.check(args.check_pairs)
            print(json.dumps(result, indent=2))
            return 0 if result["pass"] else 3
    finally:
        p.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
