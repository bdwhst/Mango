"""Frozen ladder and Bradley-Terry ratings (docs/DESIGN.md section 6.6).

runs/<name>/strength/
  openings_v1.json      fixed opening set, generated once per run, never regenerated
  ladder.json           entries (anchors and models) and every match played
  matches/<a>__<b>.json mango_match / GTP-driver reports
  ratings.json          latest fit; ratings.csv: one row per player per fit (history)

Anchors: "random" (uniform-random legal moves, pinned at 0 Elo), the first promoted
model, and optionally an external GTP engine (`eval.gtp_anchor`). Each new entry plays
the anchors and its `eval.ladder_neighbours` nearest ladder neighbours (the most recently
added model entries: entries arrive in time order and strength is expected to be
monotone in time, so they are the nearest in rating), all with `eval.ladder_pairs`
colour-swapped pairs on the fixed opening set, no noise, the eval simulation budget.

Rating fit: Bradley-Terry by MAP with a Gaussian prior (sigma = 350 Elo) on every
rating, anchor pinned at 0; Laplace intervals (+-1.96 SE); the comparison graph must be
connected to the anchor (others are reported unrated); players with all wins or all
losses are flagged "separated" (prior-dominated rating).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import effective_move_cap, load_config
from .export import load_metadata
from .gtp import play_gtp_match
from .state import derive_seed, read_json, write_json_atomic

ELO_PER_NAT = 400.0 / math.log(10.0)
PRIOR_SIGMA_ELO = 350.0
RANDOM = "random"


# --- Bradley-Terry --------------------------------------------------------------------------

@dataclass
class MatchResult:
    a: str
    b: str
    wins_a: int
    losses_a: int
    draws_a: int = 0

    @property
    def games(self) -> int:
        return self.wins_a + self.losses_a + self.draws_a

    @property
    def score_a(self) -> float:
        return self.wins_a + 0.5 * self.draws_a


def fit_bradley_terry(players: list[str], results: list[MatchResult], anchor: str = RANDOM,
                      prior_sigma_elo: float = PRIOR_SIGMA_ELO, max_iter: int = 200, tol: float = 1e-10) -> dict[str, Any]:
    """MAP Bradley-Terry ratings in Elo with Laplace standard errors.

    Returns {"anchor", "converged", "ratings": {name: {elo, se_elo, low, high, games,
    score, rated, separated}}, "unrated": [names not connected to the anchor]}."""
    if anchor not in players:
        raise ValueError(f"anchor {anchor!r} is not among the players")
    idx = {p: i for i, p in enumerate(players)}
    for r in results:
        if r.a not in idx or r.b not in idx:
            raise ValueError(f"result between unknown players {r.a!r} / {r.b!r}")
        if r.a == r.b:
            raise ValueError("a player cannot play itself")
    # Aggregate per unordered pair (i < j): n games, s = score of i.
    agg: dict[tuple[int, int], list[float]] = {}
    games = np.zeros(len(players))
    score = np.zeros(len(players))
    for r in results:
        i, j = idx[r.a], idx[r.b]
        s_i = r.score_a
        if i > j:
            i, j = j, i
            s_i = r.games - r.score_a
        cell = agg.setdefault((i, j), [0.0, 0.0])
        cell[0] += r.games
        cell[1] += s_i
        games[idx[r.a]] += r.games
        games[idx[r.b]] += r.games
        score[idx[r.a]] += r.score_a
        score[idx[r.b]] += r.games - r.score_a
    # Connectivity to the anchor.
    adj: dict[int, set[int]] = {i: set() for i in range(len(players))}
    for (i, j), (n, _) in agg.items():
        if n > 0:
            adj[i].add(j)
            adj[j].add(i)
    connected = {idx[anchor]}
    queue = deque([idx[anchor]])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in connected:
                connected.add(v)
                queue.append(v)
    free = sorted(i for i in connected if i != idx[anchor])
    pos = {i: k for k, i in enumerate(free)}
    sigma = prior_sigma_elo / ELO_PER_NAT
    pairs = [(i, j, n, s) for (i, j), (n, s) in agg.items() if i in connected and j in connected and n > 0]
    r = np.zeros(len(players))

    def objective(rv: np.ndarray) -> float:
        val = 0.0
        for i, j, n, s in pairs:
            d = rv[i] - rv[j]
            # s * log sigmoid(d) + (n - s) * log sigmoid(-d), computed stably
            val += s * (-np.logaddexp(0.0, -d)) + (n - s) * (-np.logaddexp(0.0, d))
        for i in free:
            val -= rv[i] ** 2 / (2 * sigma * sigma)
        return float(val)

    converged = False
    hess = np.zeros((len(free), len(free)))
    for _ in range(max_iter):
        if not free:
            converged = True
            break
        grad = np.zeros(len(free))
        hess[:] = 0.0
        for i, j, n, s in pairs:
            d = r[i] - r[j]
            p = 1.0 / (1.0 + math.exp(-d))
            g = s - n * p
            w = n * p * (1.0 - p)
            if i in pos:
                grad[pos[i]] += g
                hess[pos[i], pos[i]] -= w
            if j in pos:
                grad[pos[j]] -= g
                hess[pos[j], pos[j]] -= w
            if i in pos and j in pos:
                hess[pos[i], pos[j]] += w
                hess[pos[j], pos[i]] += w
        for i in free:
            grad[pos[i]] -= r[i] / (sigma * sigma)
            hess[pos[i], pos[i]] -= 1.0 / (sigma * sigma)
        step = np.linalg.solve(hess, -grad)
        # Damped Newton: halve the step until the (concave) objective does not decrease.
        base = objective(r)
        scale = 1.0
        while True:
            trial = r.copy()
            for i in free:
                trial[i] += scale * step[pos[i]]
            if objective(trial) >= base - 1e-12 or scale < 1e-6:
                break
            scale *= 0.5
        r = trial
        if float(np.max(np.abs(scale * step))) < tol:
            converged = True
            break
    se = np.full(len(players), float("nan"))
    if free:
        cov = np.linalg.inv(-hess)
        for i in free:
            se[i] = math.sqrt(max(float(cov[pos[i], pos[i]]), 0.0))
    se[idx[anchor]] = 0.0
    ratings: dict[str, Any] = {}
    unrated = []
    for p, i in idx.items():
        rated = i in connected
        elo = float(r[i] * ELO_PER_NAT) if rated else None
        se_elo = float(se[i] * ELO_PER_NAT) if rated else None
        ratings[p] = {
            "elo": elo,
            "se_elo": se_elo,
            "low": (elo - 1.96 * se_elo) if rated else None,
            "high": (elo + 1.96 * se_elo) if rated else None,
            "games": int(games[i]),
            "score": float(score[i]),
            "rated": rated,
            "separated": bool(games[i] > 0 and (score[i] == 0.0 or score[i] == games[i])),
        }
        if not rated:
            unrated.append(p)
    return {"anchor": anchor, "converged": converged, "ratings": ratings, "unrated": unrated,
            "prior_sigma_elo": prior_sigma_elo}


# --- ladder -----------------------------------------------------------------------------

Runner = Callable[[list[Any], str], str]


def default_runner(args: list[Any], log_name: str) -> str:
    if not Path(args[0]).exists():
        raise FileNotFoundError(f"missing executable {args[0]}")
    proc = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return proc.stdout


def exe_path(bin_dir: str | Path, name: str) -> Path:
    """The runner (default_runner / Pipeline._run_subprocess) reports a missing binary."""
    return Path(bin_dir) / (name + (".exe" if platform.system() == "Windows" else ""))


def match_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def player_model_id(model_dir: str | Path | None) -> str:
    """What mango_match names a player: the model id from model.json, or "random"."""
    if model_dir is None:
        return RANDOM
    return str(load_metadata(model_dir)["model_id"])


def report_matches(report: dict[str, Any] | None, name_a: str, name_b: str, seed: int, simulations: int,
                   openings: list[list[int]] | None) -> bool:
    """A stored report is reused only if it was produced for the same two players (model
    ids), the same seed, the same simulation budget and the same opening set."""
    if not report or report.get("a") != name_a or report.get("b") != name_b:
        return False
    if int(report.get("seed", -1)) != int(seed):
        return False
    if simulations > 0 and report.get("simulations") not in (None, 0) and int(report["simulations"]) != int(simulations):
        return False
    if openings is not None and [list(map(int, o)) for o in report.get("openings", [])] != openings:
        return False
    return True


def load_openings(path: str | Path, limit: int | None = None) -> list[list[int]]:
    ops = [list(map(int, o)) for o in json.loads(Path(path).read_text(encoding="utf-8"))]
    return ops[:limit] if limit is not None else ops


class Ladder:
    def __init__(self, run_dir: str | Path, cfg: dict[str, Any], bin_dir: str | Path, run_seed: int, device: str = "auto",
                 runner: Runner = default_runner, log: Callable[[str], None] | None = None):
        self.run = Path(run_dir)
        self.cfg = cfg
        self.bin = Path(bin_dir)
        self.run_seed = int(run_seed)
        self.device = device
        self.runner = runner
        self.log = log or (lambda msg: None)
        self.dir = self.run / "strength"
        (self.dir / "matches").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ladder.json"
        self.data = read_json(self.path, None)
        if self.data is None:
            self.data = {"version": 1, "entries": [], "matches": []}
            anchor = cfg["eval"].get("gtp_anchor")
            self.data["entries"].append({"name": RANDOM, "kind": "random", "anchor": "random", "iteration": None})
            if anchor:
                self.data["entries"].append({"name": anchor["name"], "kind": "gtp", "anchor": "gtp", "iteration": None,
                                             "command": list(anchor["command"])})
            self.save()

    # --- files ---
    def save(self) -> None:
        write_json_atomic(self.path, self.data)

    @property
    def n(self) -> int:
        return int(self.cfg["board"]["size"])

    @property
    def komi(self) -> float:
        return float(self.cfg["board"]["komi"])

    @property
    def openings_path(self) -> Path:
        return self.dir / "openings_v1.json"

    def ensure_openings(self) -> Path:
        """Generated once per run (uniformly random legal moves, k = eval.opening_moves)."""
        if not self.openings_path.exists():
            args = [exe_path(self.bin, "mango_match"), "--write-openings", self.openings_path, "--config",
                    self.run / "config.json", "--pairs", int(self.cfg["eval"]["ladder_pairs"]), "--openings",
                    int(self.cfg["eval"]["opening_moves"]), "--seed", derive_seed(self.run_seed, 0x0BE1)]
            self.runner(args, "strength.log")
            self.log(f"strength: wrote the fixed opening set {self.openings_path.name}")
        return self.openings_path

    # --- entries ---
    def entry(self, name: str) -> dict[str, Any] | None:
        for e in self.data["entries"]:
            if e["name"] == name:
                return e
        return None

    def model_entries(self) -> list[dict[str, Any]]:
        return [e for e in self.data["entries"] if e["kind"] == "model"]

    def first_promoted(self) -> dict[str, Any] | None:
        for e in self.model_entries():
            if e.get("promoted"):
                return e
        return None

    def has_match(self, a: str, b: str) -> bool:
        return any({m["a"], m["b"]} == {a, b} for m in self.data["matches"])

    def opponents_for(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Anchors (random, first promoted, GTP) and the nearest neighbours."""
        chosen: list[dict[str, Any]] = []
        names = {entry["name"]}
        for e in self.data["entries"]:
            if e["kind"] in ("random", "gtp") and e["name"] not in names:
                chosen.append(e)
                names.add(e["name"])
        fp = self.first_promoted()
        if fp is not None and fp["name"] not in names:
            chosen.append(fp)
            names.add(fp["name"])
        k = int(self.cfg["eval"]["ladder_neighbours"])
        recent = [e for e in reversed(self.model_entries()) if e["name"] not in names]
        for e in recent[:k]:
            chosen.append(e)
            names.add(e["name"])
        return chosen

    def add_model(self, model_id: str, iteration: int, promoted: bool) -> list[dict[str, Any]]:
        """Adds a model entry (idempotent) and plays its missing matches. Returns the
        matches played now."""
        e = self.entry(model_id)
        if e is None:
            e = {"name": model_id, "kind": "model", "iteration": int(iteration), "promoted": bool(promoted),
                 "model_dir": f"models/{model_id}"}
            self.data["entries"].append(e)
            self.save()
        opponents = self.opponents_for(e)
        played = []
        for opp in opponents:
            if self.has_match(e["name"], opp["name"]):
                continue
            m = self.play(e, opp, iteration)
            self.data["matches"].append(m)
            self.save()
            played.append(m)
            self.log(f"strength: {m['a']} vs {m['b']}: {m['wins_a']}-{m['losses_a']}-{m['draws_a']} "
                     f"(mean pair score {m['mean_pair_score']:.3f})")
        return played

    # --- matches ---
    def _model_dir(self, e: dict[str, Any]) -> Path:
        return self.run / e["model_dir"]

    def play(self, x: dict[str, Any], y: dict[str, Any], iteration: int) -> dict[str, Any]:
        """One match x (A) vs y (B) on the fixed openings; reuses a valid existing report."""
        openings = self.ensure_openings()
        report_rel = f"strength/matches/{match_key(x['name'], y['name'])}.json"
        report_path = self.run / report_rel
        report = read_json(report_path)
        seed = derive_seed(self.run_seed, hash_name(match_key(x["name"], y["name"])))
        pairs = int(self.cfg["eval"]["ladder_pairs"])
        sims = int(self.cfg["search"]["eval_simulations"])
        expected = load_openings(openings, pairs)
        if not report_matches(report, x["name"], y["name"], seed, sims, expected):
            kinds = {x["kind"], y["kind"]}
            if "gtp" in kinds:
                if kinds != {"gtp", "model"}:
                    raise ValueError("a GTP anchor only plays model entries")
                report = self._play_gtp(x, y, openings, seed)
                write_json_atomic(report_path, report)
            else:
                args = [exe_path(self.bin, "mango_match"), "--a", RANDOM if x["kind"] == "random" else self._model_dir(x),
                        "--b", RANDOM if y["kind"] == "random" else self._model_dir(y), "--config", self.run / "config.json",
                        "--openings-file", openings, "--pairs", int(self.cfg["eval"]["ladder_pairs"]), "--seed", seed,
                        "--out", report_path, "--device", self.device, "--games-in-flight",
                        int(self.cfg["selfplay"]["games_in_flight"])]
                self.runner(args, "strength.log")
                report = read_json(report_path)
                if not report_matches(report, x["name"], y["name"], seed, sims, expected):
                    raise RuntimeError(f"unexpected match report {report_path}")
        return {"a": x["name"], "b": y["name"], "pairs": int(report["pairs"]), "wins_a": int(report["wins_a"]),
                "losses_a": int(report["losses_a"]), "draws_a": int(report["draws_a"]),
                "mean_pair_score": float(report["mean_pair_score"]), "ci95": list(report["ci95"]),
                "report": report_rel, "iteration": int(iteration)}

    def _play_gtp(self, x: dict[str, Any], y: dict[str, Any], openings: Path, seed: int) -> dict[str, Any]:
        gtp, model = (x, y) if x["kind"] == "gtp" else (y, x)
        mango_cmd = [exe_path(self.bin, "mango_gtp"), "--model", self._model_dir(model), "--config", self.run / "config.json",
                     "--device", self.device, "--seed", seed]
        referee = [exe_path(self.bin, "mango_gtp"), "--size", self.n, "--komi", self.komi]
        cmd_a, cmd_b = (gtp["command"], mango_cmd) if x is gtp else (mango_cmd, gtp["command"])
        ops = load_openings(openings, int(self.cfg["eval"]["ladder_pairs"]))
        report = play_gtp_match(cmd_a, cmd_b, referee, self.n, self.komi, ops, effective_move_cap(self.cfg), x["name"],
                                y["name"], seed=seed)
        report["simulations"] = int(self.cfg["search"]["eval_simulations"])
        return report

    # --- ratings ---
    def report_of(self, match: dict[str, Any]) -> dict[str, Any] | None:
        return read_json(self.run / match["report"])

    def results(self) -> list[MatchResult]:
        return [MatchResult(m["a"], m["b"], int(m["wins_a"]), int(m["losses_a"]), int(m["draws_a"]))
                for m in self.data["matches"]]

    def games(self) -> list[GameObs]:
        return game_observations(self.data["matches"], self.report_of)

    def fit_openings(self, prior_sigma_elo: float = PRIOR_SIGMA_ELO, opening_sigma_elo: float = PRIOR_SIGMA_ELO) -> dict[str, Any]:
        """The opening-adjusted fit (diagnostic; the ladder's recorded fit is the plain one)."""
        players = [e["name"] for e in self.data["entries"]]
        return fit_bradley_terry_openings(players, self.games(), prior_sigma_elo=prior_sigma_elo,
                                          opening_sigma_elo=opening_sigma_elo)

    def model_order(self) -> list[str]:
        return [e["name"] for e in self.model_entries()]

    def fit(self, fit_iteration: int | None = None, record: bool = True,
            prior_sigma_elo: float = PRIOR_SIGMA_ELO) -> dict[str, Any]:
        players = [e["name"] for e in self.data["entries"]]
        results = self.results()
        fit = fit_bradley_terry(players, results, prior_sigma_elo=prior_sigma_elo)
        fit["fit_iteration"] = fit_iteration
        fit["entries"] = {e["name"]: {k: v for k, v in e.items() if k != "command"} for e in self.data["entries"]}
        fit["residuals"] = predicted_vs_observed(fit, results)
        if record:
            write_json_atomic(self.dir / "ratings.json", fit)
            append_ratings_csv(self.dir / "ratings.csv", fit, self.data["entries"])
        return fit


# --- diagnostics of the rating model (M4 review) ---------------------------------------------

def pair_structure(report: dict[str, Any]) -> dict[str, Any]:
    """Per colour-swapped pair of a match report: both games won by A, both lost, split
    (one each - in a colour-swapped pair a split always means the same colour won both
    games, so splits carry opening/colour information as well as strength), pairs with
    a draw. `black_wins` is the fraction of decided games won by black (opening-set
    balance: 0.5 for a balanced set)."""
    games = report["games_detail"]
    out = {"pairs": 0, "both_won": 0, "both_lost": 0, "split": 0, "draw_pairs": 0}
    black = decided = 0
    for g in games:
        if g["result"] != 0:
            decided += 1
            black += g["result"] > 0
    for i in range(0, len(games) - 1, 2):
        g0, g1 = games[i], games[i + 1]
        out["pairs"] += 1
        if g0["result"] == 0 or g1["result"] == 0:
            out["draw_pairs"] += 1
            continue
        a0 = (g0["result"] > 0) == g0["a_is_black"]
        a1 = (g1["result"] > 0) == g1["a_is_black"]
        if a0 and a1:
            out["both_won"] += 1
        elif not a0 and not a1:
            out["both_lost"] += 1
        else:
            out["split"] += 1
    out["black_wins"] = (black / decided) if decided else None
    return out


@dataclass
class GameObs:
    """One game as the opening-adjusted model sees it."""

    black: str
    white: str
    opening: int      # index into the fixed opening set (report "pair")
    result: float     # 1 black won, 0 white won, 0.5 draw


def game_observations(matches: list[dict[str, Any]],
                      report_of: Callable[[dict[str, Any]], dict[str, Any] | None]) -> list[GameObs]:
    """Every game of every match with a stored report (matches without one are skipped)."""
    out = []
    for m in matches:
        report = report_of(m)
        if report is None or "games_detail" not in report:
            continue
        for g in report["games_detail"]:
            black, white = (m["a"], m["b"]) if g["a_is_black"] else (m["b"], m["a"])
            out.append(GameObs(black, white, int(g["pair"]), 1.0 if g["result"] > 0 else (0.0 if g["result"] < 0 else 0.5)))
    return out


def fit_bradley_terry_openings(players: list[str], games: list[GameObs], anchor: str = RANDOM,
                               prior_sigma_elo: float = PRIOR_SIGMA_ELO, opening_sigma_elo: float = PRIOR_SIGMA_ELO,
                               max_iter: int = 200, tol: float = 1e-10) -> dict[str, Any]:
    """Bradley-Terry with a per-opening colour advantage (the "home advantage" extension):
    logit P(black wins) = r_black - r_white + o_k for opening k. Gaussian priors on the
    ratings (sigma) and the opening terms (opening_sigma), anchor pinned at 0, damped
    Newton, Laplace SE. The opening terms absorb what a fixed random opening set does to
    the results, so the ratings measure strength on balanced positions. Returns the
    same "ratings" structure as fit_bradley_terry plus "openings": {k: {black_elo, se_elo}}
    and per-game predictions ("predicted": [P(black wins)])."""
    if anchor not in players:
        raise ValueError(f"anchor {anchor!r} is not among the players")
    idx = {p: i for i, p in enumerate(players)}
    for g in games:
        if g.black not in idx or g.white not in idx:
            raise ValueError(f"game between unknown players {g.black!r} / {g.white!r}")
        if g.black == g.white:
            raise ValueError("a player cannot play itself")
    n_games = np.zeros(len(players))
    score = np.zeros(len(players))
    adj: dict[int, set[int]] = {i: set() for i in range(len(players))}
    for g in games:
        b, w = idx[g.black], idx[g.white]
        n_games[b] += 1
        n_games[w] += 1
        score[b] += g.result
        score[w] += 1.0 - g.result
        adj[b].add(w)
        adj[w].add(b)
    connected = {idx[anchor]}
    queue = deque([idx[anchor]])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in connected:
                connected.add(v)
                queue.append(v)
    free = sorted(i for i in connected if i != idx[anchor])
    openings = sorted({g.opening for g in games if idx[g.black] in connected and idx[g.white] in connected})
    pos = {i: k for k, i in enumerate(free)}
    opos = {o: len(free) + k for k, o in enumerate(openings)}
    n_par = len(free) + len(openings)
    obs = [g for g in games if idx[g.black] in connected and idx[g.white] in connected]
    X = np.zeros((len(obs), n_par))
    y = np.zeros(len(obs))
    for row, g in enumerate(obs):
        b, w = idx[g.black], idx[g.white]
        if b in pos:
            X[row, pos[b]] += 1.0
        if w in pos:
            X[row, pos[w]] -= 1.0
        X[row, opos[g.opening]] += 1.0
        y[row] = g.result
    sigma = prior_sigma_elo / ELO_PER_NAT
    osigma = opening_sigma_elo / ELO_PER_NAT
    lam = np.concatenate([np.full(len(free), 1.0 / (sigma * sigma)), np.full(len(openings), 1.0 / (osigma * osigma))])
    theta = np.zeros(n_par)

    def objective(t: np.ndarray) -> float:
        z = X @ t
        return float(np.sum(y * (-np.logaddexp(0.0, -z)) + (1.0 - y) * (-np.logaddexp(0.0, z))) - 0.5 * np.sum(lam * t * t))

    converged = n_par == 0
    hess = -np.diag(lam) if n_par else np.zeros((0, 0))
    for _ in range(max_iter if n_par else 0):
        z = X @ theta
        pr = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (y - pr) - lam * theta
        hess = -(X.T * (pr * (1.0 - pr))) @ X - np.diag(lam)
        step = np.linalg.solve(hess, -grad)
        base = objective(theta)
        scale = 1.0
        while True:
            trial = theta + scale * step
            if objective(trial) >= base - 1e-12 or scale < 1e-6:
                break
            scale *= 0.5
        theta = trial
        if float(np.max(np.abs(scale * step))) < tol:
            converged = True
            break
    cov = np.linalg.inv(-hess) if n_par else np.zeros((0, 0))
    ratings: dict[str, Any] = {}
    unrated = []
    for p_, i in idx.items():
        rated = i in connected
        r = float(theta[pos[i]]) if i in pos else 0.0
        se = math.sqrt(max(float(cov[pos[i], pos[i]]), 0.0)) if i in pos else 0.0
        elo = r * ELO_PER_NAT if rated else None
        se_elo = se * ELO_PER_NAT if rated else None
        ratings[p_] = {"elo": elo, "se_elo": se_elo, "low": (elo - 1.96 * se_elo) if rated else None,
                       "high": (elo + 1.96 * se_elo) if rated else None, "games": int(n_games[i]), "score": float(score[i]),
                       "rated": rated,
                       "separated": bool(n_games[i] > 0 and (score[i] == 0.0 or score[i] == n_games[i]))}
        if not rated:
            unrated.append(p_)
    opening_terms = {}
    for o in openings:
        k = opos[o]
        opening_terms[int(o)] = {"black_elo": float(theta[k] * ELO_PER_NAT),
                                 "se_elo": math.sqrt(max(float(cov[k, k]), 0.0)) * ELO_PER_NAT,
                                 "games": int(sum(1 for g in obs if g.opening == o))}
    pred = (1.0 / (1.0 + np.exp(-(X @ theta)))).tolist() if n_par else []
    return {"anchor": anchor, "converged": converged, "ratings": ratings, "unrated": unrated,
            "prior_sigma_elo": prior_sigma_elo, "opening_sigma_elo": opening_sigma_elo, "openings": opening_terms,
            "model": "openings", "observations": [(g.black, g.white, g.opening, g.result) for g in obs], "predicted": pred}


def match_residuals_openings(fit: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicted vs observed per (black, white) pairing under the opening-adjusted fit,
    aggregated over its games: predicted = mean P(black wins), observed = black's score,
    z with the binomial variance of the game-level predictions."""
    agg: dict[tuple[str, str], list[float]] = {}
    for (b, w, _o, r), p in zip(fit["observations"], fit["predicted"]):
        cell = agg.setdefault((b, w), [0.0, 0.0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += r
        cell[2] += p
        cell[3] += p * (1.0 - p)
    rows = []
    for (b, w), (n, s, ps, var) in agg.items():
        z = (s - ps) / math.sqrt(max(var, 1e-12))
        rows.append({"black": b, "white": w, "games": int(n), "predicted": ps / n, "observed": s / n, "z": z})
    return rows


def predicted_vs_observed(fit: dict[str, Any], results: list[MatchResult]) -> list[dict[str, Any]]:
    """Per observation: the fit's predicted score of A, the observed score, and the
    standardised residual z = (obs - pred) / sqrt(pred (1 - pred) / n). |z| >> 2 on many
    rows means the single-scale model does not describe the results."""
    rows = []
    r = fit["ratings"]
    for m in results:
        if m.games == 0 or not r[m.a]["rated"] or not r[m.b]["rated"]:
            continue
        d = (r[m.a]["elo"] - r[m.b]["elo"]) / ELO_PER_NAT
        p = 1.0 / (1.0 + math.exp(-d))
        obs = m.score_a / m.games
        z = (obs - p) / math.sqrt(max(p * (1.0 - p), 1e-12) / m.games)
        rows.append({"a": m.a, "b": m.b, "games": m.games, "predicted": p, "observed": obs, "z": z})
    return rows


def prior_sensitivity(players: list[str], results: list[MatchResult], order: list[str],
                      sigmas: tuple[float, ...] = (200.0, 350.0, 700.0, 1400.0), anchor: str = RANDOM) -> list[dict[str, Any]]:
    """The same results fitted under several prior widths: ratings of the entries in
    `order` (time order), whether that order is monotone, and the mean |z|. Conclusions
    that hold for every sigma are prior-robust; the absolute scale usually is not when
    the top entries only have 100-0 results."""
    out = []
    for sigma in sigmas:
        fit = fit_bradley_terry(players, results, anchor=anchor, prior_sigma_elo=sigma)
        elos = [fit["ratings"][p]["elo"] for p in order]
        rated = [e is not None for e in elos]
        mono = all(rated) and all(elos[i] < elos[i + 1] for i in range(len(elos) - 1))
        res = predicted_vs_observed(fit, results)
        out.append({"sigma": sigma, "elo": dict(zip(order, elos)),
                    "low": {p: fit["ratings"][p]["low"] for p in order}, "high": {p: fit["ratings"][p]["high"] for p in order},
                    "monotone": mono, "converged": fit["converged"],
                    "mean_abs_z": (sum(abs(x["z"]) for x in res) / len(res)) if res else 0.0,
                    "max_abs_z": max((abs(x["z"]) for x in res), default=0.0)})
    return out


def opening_balance(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Across reports played on the same opening set: per opening index, the number of
    pairs played, how often the same colour won both games (colour-decided), and black's
    share of decided games; a summary of openings colour-decided in at least half of
    their pairs."""
    per: dict[int, dict[str, int]] = {}
    for rep in reports:
        games = rep["games_detail"]
        for i in range(0, len(games) - 1, 2):
            g0, g1 = games[i], games[i + 1]
            idx = int(g0["pair"])
            c = per.setdefault(idx, {"pairs": 0, "colour_decided": 0, "black_wins": 0, "decided_games": 0})
            c["pairs"] += 1
            for g in (g0, g1):
                if g["result"] != 0:
                    c["decided_games"] += 1
                    c["black_wins"] += g["result"] > 0
            if g0["result"] != 0 and g1["result"] != 0 and (g0["result"] > 0) == (g1["result"] > 0):
                c["colour_decided"] += 1
    decided = sorted(i for i, c in per.items() if c["pairs"] > 0 and c["colour_decided"] * 2 >= c["pairs"])
    total_pairs = sum(c["pairs"] for c in per.values())
    total_cd = sum(c["colour_decided"] for c in per.values())
    games = sum(c["decided_games"] for c in per.values())
    black = sum(c["black_wins"] for c in per.values())
    return {"openings": per, "colour_decided_openings": decided, "pairs": total_pairs,
            "colour_decided_pairs": total_cd, "black_win_rate": (black / games) if games else None}


def remeasure_ladder(run_dir: str | Path, cfg: dict[str, Any], bin_dir: str | Path, run_seed: int, tag: str,
                     device: str = "auto", runner: Runner = default_runner,
                     log: Callable[[str], None] | None = None) -> Ladder:
    """Replays every ladder match of a run with the current engine (after a search
    change, DESIGN 5.4.10): the old ladder.json, matches/, ratings.json and ratings.csv
    move to strength_<tag>/ (with a copy of the opening file), the same entries are
    re-added in their original order with their iteration and promoted flags, and every
    match is played again on the same fixed openings with the same per-pairing seeds."""
    import shutil

    log = log or (lambda msg: None)
    run = Path(run_dir)
    src = run / "strength"
    old = read_json(src / "ladder.json")
    if old is None:
        raise FileNotFoundError(f"no ladder to re-measure in {src}")
    archive = run / f"strength_{tag}"
    if archive.exists():
        raise FileExistsError(f"{archive} exists; choose another tag")
    archive.mkdir(parents=True)
    for name in ("ladder.json", "ratings.json", "ratings.csv"):
        if (src / name).exists():
            shutil.move(str(src / name), str(archive / name))
    if (src / "matches").exists():
        shutil.move(str(src / "matches"), str(archive / "matches"))
    if (src / "openings_v1.json").exists():
        shutil.copy2(str(src / "openings_v1.json"), str(archive / "openings_v1.json"))
    log(f"strength: archived the previous ladder to {archive.name}")
    ladder = Ladder(run, cfg, bin_dir, run_seed, device, runner=runner, log=log)
    for e in old["entries"]:
        if e["kind"] != "model":
            continue
        ladder.add_model(e["name"], int(e["iteration"]), bool(e.get("promoted")))
    ladder.fit(fit_iteration=None)
    return ladder


def hash_name(s: str) -> int:
    h = 0xCBF29CE484222325
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFFFFFFFFFF


def append_ratings_csv(path: Path, fit: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    fields = ["fit_iteration", "name", "kind", "iteration", "promoted", "elo", "low", "high", "games", "score", "rated",
              "separated"]
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for e in entries:
            r = fit["ratings"][e["name"]]
            w.writerow({"fit_iteration": fit.get("fit_iteration"), "name": e["name"], "kind": e["kind"],
                        "iteration": e.get("iteration"), "promoted": e.get("promoted"), "elo": r["elo"], "low": r["low"],
                        "high": r["high"], "games": r["games"], "score": r["score"], "rated": r["rated"],
                        "separated": r["separated"]})


def format_ratings(fit: dict[str, Any], entries: list[dict[str, Any]] | None = None) -> str:
    rows = []
    order = [e["name"] for e in entries] if entries else list(fit["ratings"].keys())
    for name in order:
        r = fit["ratings"][name]
        if r["rated"]:
            flag = " (separated)" if r["separated"] else ""
            rows.append(f"{name:>20}  {r['elo']:8.1f}  [{r['low']:8.1f}, {r['high']:8.1f}]  games {r['games']:4d}{flag}")
        else:
            rows.append(f"{name:>20}  unrated (not connected to {fit['anchor']})")
    return "\n".join(rows)


# --- cross-run comparison (parameter sweeps, DESIGN 8.2) ----------------------------------

def cross_run(bin_dir: str | Path, config_path: str | Path, out_dir: str | Path, models: dict[str, Path], pairs: int,
              seed: int, device: str = "auto", runner: Runner = default_runner, log: Callable[[str], None] | None = None,
              opening_moves: int | None = None) -> dict[str, Any]:
    """Round robin between the given models plus the random anchor, one opening set,
    one Bradley-Terry fit: the tool for comparing the final models of several runs.

    Labels (the dict keys) name runs, not models: a stored report is reused only if it
    was played by the model ids the labels currently point to, with this seed, budget
    and opening set (a run that trained further gets its matches replayed)."""
    log = log or print
    out = Path(out_dir)
    (out / "matches").mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    sims = int(cfg["search"]["eval_simulations"])
    openings = out / "openings.json"
    if not openings.exists():
        runner([exe_path(bin_dir, "mango_match"), "--write-openings", openings, "--config", config_path, "--pairs", pairs,
                "--openings", opening_moves if opening_moves is not None else int(cfg["eval"]["opening_moves"]),
                "--seed", seed], "crossrun.log")
    expected = load_openings(openings, pairs)
    names = list(models.keys())
    ids = {RANDOM: RANDOM, **{k: player_model_id(v) for k, v in models.items()}}
    players = [RANDOM] + names
    results: list[MatchResult] = []
    matches = []
    pairs_to_play = [(a, b) for i, a in enumerate(players) for b in players[i + 1:]]
    for a, b in pairs_to_play:
        report_path = out / "matches" / f"{match_key(a, b)}.json"
        report = read_json(report_path)
        match_seed = derive_seed(seed, hash_name(match_key(a, b)))
        if not report_matches(report, ids[a], ids[b], match_seed, sims, expected):
            runner([exe_path(bin_dir, "mango_match"), "--a", RANDOM if a == RANDOM else models[a], "--b",
                    RANDOM if b == RANDOM else models[b], "--config", config_path, "--openings-file", openings, "--pairs",
                    pairs, "--seed", match_seed, "--out", report_path, "--device", device,
                    "--games-in-flight", int(cfg["selfplay"]["games_in_flight"])], "crossrun.log")
            report = read_json(report_path)
            if not report_matches(report, ids[a], ids[b], match_seed, sims, expected):
                raise RuntimeError(f"unexpected match report {report_path}: expected {ids[a]} vs {ids[b]}")
        results.append(MatchResult(a, b, report["wins_a"], report["losses_a"], report["draws_a"]))
        matches.append({"a": a, "b": b, "model_a": ids[a], "model_b": ids[b], "wins_a": report["wins_a"],
                        "losses_a": report["losses_a"], "draws_a": report["draws_a"],
                        "mean_pair_score": report["mean_pair_score"], "ci95": report["ci95"]})
        log(f"crossrun: {a} ({ids[a]}) vs {b} ({ids[b]}): {report['wins_a']}-{report['losses_a']}-{report['draws_a']}")
    fit = fit_bradley_terry(players, results)
    fit["matches"] = matches
    fit["models"] = {k: {"dir": str(v), "model_id": ids[k]} for k, v in models.items()}
    fit["simulations"] = sims
    fit["seed"] = seed
    write_json_atomic(out / "crossrun.json", fit)
    log(format_ratings(fit))
    return fit


# --- CLI ----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mango frozen ladder / Bradley-Terry ratings (DESIGN 6.6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_fit = sub.add_parser("fit", help="refit and print a run's ladder")
    s_fit.add_argument("--run", required=True)
    s_add = sub.add_parser("add", help="add a model of a run to its ladder and play its matches")
    s_add.add_argument("--run", required=True)
    s_add.add_argument("--bin", required=True)
    s_add.add_argument("--model-id", required=True)
    s_add.add_argument("--iteration", type=int, required=True)
    s_add.add_argument("--promoted", action="store_true")
    s_add.add_argument("--device", default="auto")
    s_rm = sub.add_parser("remeasure", help="replay every ladder match with the current engine")
    s_rm.add_argument("--run", required=True)
    s_rm.add_argument("--bin", required=True)
    s_rm.add_argument("--tag", required=True, help="the previous ladder moves to strength_<tag>/")
    s_rm.add_argument("--device", default="auto")
    s_x = sub.add_parser("crossrun", help="round robin between models of different runs plus the random anchor")
    s_x.add_argument("--bin", required=True)
    s_x.add_argument("--config", required=True)
    s_x.add_argument("--out", required=True)
    s_x.add_argument("--model", action="append", required=True, help="name=model_dir (repeatable)")
    s_x.add_argument("--pairs", type=int, default=50)
    s_x.add_argument("--seed", type=int, default=1)
    s_x.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if args.cmd == "crossrun":
        models = {}
        for spec in args.model:
            name, _, d = spec.partition("=")
            if not d:
                ap.error(f"--model expects name=dir, got {spec!r}")
            models[name] = Path(d)
        cross_run(args.bin, args.config, args.out, models, args.pairs, args.seed, args.device)
        return 0
    run = Path(args.run)
    cfg = load_config(run / "config.json")
    state = read_json(run / "state.json")
    if args.cmd == "fit":
        ladder = Ladder(run, cfg, ".", state["run_seed"])
        fit = ladder.fit(record=False)
        print(format_ratings(fit, ladder.data["entries"]))
        return 0
    if args.cmd == "remeasure":
        ladder = remeasure_ladder(run, cfg, args.bin, state["run_seed"], args.tag, args.device, log=print)
        print(format_ratings(ladder.fit(record=False), ladder.data["entries"]))
        return 0
    ladder = Ladder(run, cfg, args.bin, state["run_seed"], args.device, log=print)
    ladder.add_model(args.model_id, args.iteration, args.promoted)
    fit = ladder.fit(fit_iteration=args.iteration)
    print(format_ratings(fit, ladder.data["entries"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
