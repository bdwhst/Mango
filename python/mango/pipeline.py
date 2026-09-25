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
eval.ladder_every-th candidate to the frozen ladder (DESIGN 6.6).

    python -m mango.pipeline --run runs/smoke --config configs/5x5-smoke.json \
        --bin build/windows-cuda/Release --iterations 30 [--device cuda] [--check]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .chunk import EXTRAS_SEARCH_KIND, read_chunk, read_header
from .config import effective_move_cap, load_config, merge_config
from .data import write_holdout_games
from .export import export_model_version, load_eager_model, make_model_id
from .known_outcome import known_positions, value_sign_accuracy
from .resign import measured_false_positive_rate, select_resign_threshold
from .state import derive_seed, read_json, write_json_atomic
from .strength import Ladder, format_ratings
from .train import (build_model, build_optimizer, build_window_dataset, bounded_steps, evaluate_dataset,
                    load_checkpoint, make_scaler, save_checkpoint, train_until)

PHASES = ["selfplay", "train", "monitor", "export", "gate", "promote", "strength"]


def pick_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Pipeline:
    def __init__(self, run_dir: str | Path, config: dict[str, Any] | None, bin_dir: str | Path, device: str | None = None,
                 selfplay_device: str | None = None, run_seed: int | None = None, quiet: bool = False):
        self.run = Path(run_dir)
        self.bin = Path(bin_dir)
        self.device = pick_device(device)
        self.selfplay_device = selfplay_device or (self.device.type if self.device.type != "mps" else "mps")
        self.quiet = quiet
        self.run.mkdir(parents=True, exist_ok=True)
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

    def _run_subprocess(self, args: list[str], log_name: str) -> str:
        self.log("exec " + " ".join(str(a) for a in args))
        with open(self.run / "logs" / log_name, "a", encoding="utf-8") as errlog:
            proc = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=errlog, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{args[0]} failed with exit code {proc.returncode}; see logs/{log_name}")
        return proc.stdout

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

    def _end(self, phase: str) -> None:
        nxt = PHASES[(PHASES.index(phase) + 1) % len(PHASES)]
        if phase == PHASES[-1]:
            self.state["iteration"] += 1
            self.state["plan"] = {}
        self.state["phase"] = nxt
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
        plan = self._begin("selfplay", {
            "task_id": str(it),
            "model_id": self.state["best_model_id"],
            "target_games": int(self.cfg["selfplay"]["games_per_iteration"]),
            "chunk_prefix": f"chunk_{it:04d}_",
            "chunk_id_start": int(self.state["next_chunk_id"]),
            "seed": derive_seed(self.state["run_seed"], it),
            "v_resign": v_resign,
            "resign_selection": resign,
        })
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
        existing = 0
        max_id = plan["chunk_id_start"] - 1
        for p in self._published_chunks(plan["chunk_prefix"]):
            h = read_header(p)
            existing += h.num_games
            max_id = max(max_id, h.chunk_id)
        remaining = plan["target_games"] - existing
        if remaining > 0:
            sgf_dir = self.run / "sgf" / f"{it:04d}"
            args = [self.exe("mango_selfplay"), "--model", self.model_dir(plan["model_id"]), "--config",
                    self.run / "config.json", "--games", remaining, "--out", self.run / "replay", "--chunk-prefix",
                    plan["chunk_prefix"], "--chunk-id-start", max_id + 1, "--seed", plan["seed"], "--sgf-dir", sgf_dir,
                    "--iteration", it, "--device", self.selfplay_device, "--resign-threshold", plan["v_resign"]]
            t0 = time.time()
            out = self._run_subprocess(args, "selfplay.log")
            summary = json.loads(out.strip().splitlines()[-1])
            summary["seconds"] = round(time.time() - t0, 1)
            self.log(f"selfplay: {summary['games']} games, {summary['positions']} positions, "
                     f"avg length {summary['avg_game_length']:.1f}, {summary['terminations']}, "
                     f"resign {plan['v_resign']:g}, {summary['seconds']}s")
            self.state.setdefault("selfplay_stats", {})[str(it)] = summary
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
        ids = [c["chunk_id"] for c in self.manifest()["chunks"]]
        self.state["next_chunk_id"] = (max(ids) + 1) if ids else 1
        self._end("selfplay")

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
            self.state["events"].append({"iteration": it, "event": "train_skipped"})
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

    def phase_gate(self) -> None:
        it = self.state["iteration"]
        plan = self._begin("gate", {
            "candidate_model_id": self.state["plan"]["export"]["candidate_model_id"],
            "incumbent_model_id": self.state["best_model_id"],
            "report": f"matches/{it:04d}.json",
            "match_seed": derive_seed(self.state["run_seed"], 1000 + it),
        })
        report_path = self.run / plan["report"]
        report = read_json(report_path)
        valid = (report is not None and report.get("a") == plan["candidate_model_id"]
                 and report.get("b") == plan["incumbent_model_id"] and report.get("seed") == plan["match_seed"])
        if not valid:
            if not self.cfg["eval"]["gating"]:
                write_json_atomic(report_path, {"a": plan["candidate_model_id"], "b": plan["incumbent_model_id"],
                                                "seed": plan["match_seed"], "skipped": True})
            else:
                args = [self.exe("mango_match"), "--a", self.model_dir(plan["candidate_model_id"]), "--b",
                        self.model_dir(plan["incumbent_model_id"]), "--config", self.run / "config.json", "--pairs",
                        int(self.cfg["eval"]["pairs"]), "--seed", plan["match_seed"], "--out", report_path, "--device",
                        self.selfplay_device, "--games-in-flight", int(self.cfg["selfplay"]["games_in_flight"])]
                t0 = time.time()
                self._run_subprocess(args, "match.log")
                report = read_json(report_path)
                self.log(f"gate: candidate mean pair score {report['mean_pair_score']:.3f} "
                         f"[{report['ci95'][0]:.3f}, {report['ci95'][1]:.3f}], unique {report['unique_trajectories']}/"
                         f"{report['games']}, {time.time() - t0:.1f}s")
        self._end("gate")

    def phase_promote(self) -> None:
        it = self.state["iteration"]
        gate = self.state["plan"]["gate"]
        plan = self._begin("promote", {"candidate_model_id": gate["candidate_model_id"], "decision": None})
        if plan["decision"] is None:
            report = read_json(self.run / gate["report"])
            if report.get("skipped"):
                plan["decision"] = True
            else:
                plan["decision"] = bool(report["mean_pair_score"] > float(self.cfg["eval"]["gate_threshold"]))
            self.save_state()
        if plan["decision"]:
            write_json_atomic(self.run / "best.json", {"model_id": plan["candidate_model_id"], "iteration": it})
            self.state["best_model_id"] = plan["candidate_model_id"]
        self.state["events"].append({"iteration": it, "event": "gate", "candidate": plan["candidate_model_id"],
                                     "promoted": plan["decision"]})
        self.log(f"promote: {'promoted' if plan['decision'] else 'rejected'} {plan['candidate_model_id']}")
        self._end("promote")

    def phase_strength(self) -> None:
        """Frozen ladder (DESIGN 6.6): every promoted model and every ladder_every-th
        candidate is added and plays the anchors and its nearest neighbours."""
        it = self.state["iteration"]
        promote = self.state["plan"]["promote"]
        every = int(self.cfg["eval"]["ladder_every"])
        add = bool(promote["decision"]) or (every > 0 and it % every == 0)
        plan = self._begin("strength", {"add": add, "model_id": promote["candidate_model_id"],
                                        "promoted": bool(promote["decision"])})
        if plan["add"]:
            ladder = Ladder(self.run, self.cfg, self.bin, self.state["run_seed"], self.selfplay_device,
                            runner=self._run_subprocess, log=self.log)
            t0 = time.time()
            if not ladder.model_entries():
                ladder.add_model(self.state["initial_model_id"], 0, False)
            ladder.add_model(plan["model_id"], it, plan["promoted"])
            fit = ladder.fit(fit_iteration=it)
            r = fit["ratings"][plan["model_id"]]
            self.state.setdefault("strength", {})[str(it)] = {
                "model_id": plan["model_id"], "elo": r["elo"], "low": r["low"], "high": r["high"], "games": r["games"],
                "separated": r["separated"], "entries": len(ladder.data["entries"]), "matches": len(ladder.data["matches"]),
            }
            self.log(f"strength: {plan['model_id']} rated {r['elo']:.0f} [{r['low']:.0f}, {r['high']:.0f}] Elo "
                     f"({'separated, ' if r['separated'] else ''}{r['games']} games; ladder {len(ladder.data['entries'])} "
                     f"entries, {len(ladder.data['matches'])} matches, {time.time() - t0:.0f}s)")
            self.log("strength ladder:\n" + format_ratings(fit, ladder.data["entries"]))
        self._end("strength")

    def run_phase(self) -> None:
        getattr(self, "phase_" + self.state["phase"])()

    def run_iterations(self, iterations: int) -> None:
        """Runs until iteration `iterations` has completed all its phases."""
        while self.state["iteration"] <= iterations:
            self.run_phase()

    def run_for_seconds(self, budget: float) -> int:
        """Equal-wall-clock budgets (DESIGN 8.2): runs whole iterations until `budget`
        seconds have elapsed in this call; the iteration in progress is always completed.
        Returns the number of iterations completed."""
        t0 = time.time()
        done = 0
        while True:
            start = self.state["iteration"]
            while self.state["iteration"] == start:
                self.run_phase()
            done += 1
            if time.time() - t0 >= budget:
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
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    p = Pipeline(args.run, cfg, args.bin, args.device, args.selfplay_device, args.seed)
    if args.iterations > 0:
        p.run_iterations(args.iterations)
    if args.hours > 0:
        p.run_for_seconds(args.hours * 3600.0)
    if args.check:
        result = p.check(args.check_pairs)
        print(json.dumps(result, indent=2))
        return 0 if result["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
