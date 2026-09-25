"""Frozen ladder and Bradley-Terry fit (DESIGN 6.6): MAP fit recovers known ratings,
anchor pinned, prior tames separation, disconnected players unrated, Laplace intervals
shrink with games; ladder bookkeeping with a fake mango_match (opponent selection,
fixed opening file generated once, restart reuse, cross-run round robin)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mango.config import merge_config
from mango.state import read_json, write_json_atomic
from mango.strength import (ELO_PER_NAT, RANDOM, GameObs, Ladder, MatchResult, cross_run, fit_bradley_terry,
                            fit_bradley_terry_openings, format_ratings, game_observations, match_key,
                            match_residuals_openings, opening_balance, pair_structure, predicted_vs_observed,
                            prior_sensitivity, remeasure_ladder, report_matches)


def simulate(true_elo: dict[str, float], pairs: list[tuple[str, str]], games: int, seed: int) -> list[MatchResult]:
    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        p = 1.0 / (1.0 + 10 ** ((true_elo[b] - true_elo[a]) / 400.0))
        w = int(rng.binomial(games, p))
        out.append(MatchResult(a, b, w, games - w, 0))
    return out


def test_fit_recovers_ratings_and_pins_the_anchor():
    true = {RANDOM: 0.0, "m1": 300.0, "m2": 600.0, "m3": 800.0}
    pairs = [(RANDOM, "m1"), ("m1", "m2"), ("m2", "m3"), (RANDOM, "m2"), ("m1", "m3")]
    fit = fit_bradley_terry(list(true), simulate(true, pairs, 2000, 1))
    assert fit["converged"]
    r = fit["ratings"]
    assert r[RANDOM]["elo"] == 0.0 and r[RANDOM]["se_elo"] == 0.0
    for name in ("m1", "m2", "m3"):
        assert r[name]["rated"] and not r[name]["separated"]
        assert abs(r[name]["elo"] - true[name]) < 40.0, name
        assert r[name]["low"] < true[name] < r[name]["high"]
        assert r[name]["se_elo"] < 25.0
    # Order preserved and the interval width shrinks with more games.
    assert r["m1"]["elo"] < r["m2"]["elo"] < r["m3"]["elo"]
    small = fit_bradley_terry(list(true), simulate(true, pairs, 50, 2))["ratings"]
    assert small["m2"]["se_elo"] > r["m2"]["se_elo"] * 3
    # Games and score bookkeeping.
    assert r["m1"]["games"] == 3 * 2000


def test_mirrored_results_give_the_same_fit():
    res = [MatchResult(RANDOM, "a", 10, 30, 0), MatchResult("a", "b", 25, 15, 0)]
    mirrored = [MatchResult("a", RANDOM, 30, 10, 0), MatchResult("b", "a", 15, 25, 0)]
    f1 = fit_bradley_terry([RANDOM, "a", "b"], res)
    f2 = fit_bradley_terry([RANDOM, "a", "b"], mirrored)
    for p in ("a", "b"):
        assert f1["ratings"][p]["elo"] == pytest.approx(f2["ratings"][p]["elo"], abs=1e-6)
        assert f1["ratings"][p]["se_elo"] == pytest.approx(f2["ratings"][p]["se_elo"], abs=1e-6)
    # Draws count half: 20 draws equal 10 wins + 10 losses.
    f3 = fit_bradley_terry([RANDOM, "a"], [MatchResult(RANDOM, "a", 0, 0, 20)])
    f4 = fit_bradley_terry([RANDOM, "a"], [MatchResult(RANDOM, "a", 10, 10, 0)])
    assert f3["ratings"]["a"]["elo"] == pytest.approx(f4["ratings"]["a"]["elo"], abs=1e-6)
    assert f3["ratings"]["a"]["elo"] == pytest.approx(0.0, abs=1e-6)


def test_complete_separation_is_finite_flagged_and_prior_dominated():
    # 'a' wins every game: an MLE would diverge; the MAP stays finite and is flagged.
    fit = fit_bradley_terry([RANDOM, "a"], [MatchResult("a", RANDOM, 40, 0, 0)])
    r = fit["ratings"]["a"]
    assert fit["converged"] and r["separated"] and r["rated"]
    assert 0 < r["elo"] < 5 * 350.0 and math.isfinite(r["se_elo"])
    # More wins push it further, so the number still carries information.
    more = fit_bradley_terry([RANDOM, "a"], [MatchResult("a", RANDOM, 400, 0, 0)])["ratings"]["a"]["elo"]
    assert more > r["elo"]
    # A tighter prior pulls harder.
    tight = fit_bradley_terry([RANDOM, "a"], [MatchResult("a", RANDOM, 40, 0, 0)], prior_sigma_elo=100.0)
    assert tight["ratings"]["a"]["elo"] < r["elo"]
    # Loser side flagged as well.
    fit = fit_bradley_terry([RANDOM, "a"], [MatchResult("a", RANDOM, 0, 40, 0)])
    assert fit["ratings"]["a"]["separated"] and fit["ratings"]["a"]["elo"] < 0


def test_players_not_connected_to_the_anchor_are_unrated():
    res = [MatchResult(RANDOM, "a", 10, 10, 0), MatchResult("x", "y", 12, 8, 0)]
    fit = fit_bradley_terry([RANDOM, "a", "x", "y", "z"], res)
    assert set(fit["unrated"]) == {"x", "y", "z"}
    for p in ("x", "y", "z"):
        assert not fit["ratings"][p]["rated"] and fit["ratings"][p]["elo"] is None
    assert fit["ratings"]["a"]["rated"] and fit["ratings"]["a"]["elo"] == pytest.approx(0.0, abs=1e-6)
    assert "unrated" in format_ratings(fit)
    with pytest.raises(ValueError):
        fit_bradley_terry(["a", "b"], [MatchResult("a", "b", 1, 0, 0)])  # no anchor among the players
    with pytest.raises(ValueError):
        fit_bradley_terry([RANDOM, "a"], [MatchResult("a", "q", 1, 0, 0)])  # unknown player
    with pytest.raises(ValueError):
        fit_bradley_terry([RANDOM, "a"], [MatchResult("a", "a", 1, 0, 0)])  # self-play result


def test_prior_only_player_sits_at_zero_with_the_prior_width():
    # Connected through zero-information games (all draws): rating 0, SE from prior + data.
    fit = fit_bradley_terry([RANDOM, "a"], [MatchResult(RANDOM, "a", 0, 0, 2)])
    r = fit["ratings"]["a"]
    assert r["elo"] == pytest.approx(0.0, abs=1e-9)
    # Prior sigma 350 Elo and 2 games at p = 0.5 (Fisher info 0.5 nats^-2): SE = 1/sqrt(1/sigma^2 + 0.5).
    sigma = 350.0 / ELO_PER_NAT
    expect = ELO_PER_NAT / math.sqrt(1.0 / sigma**2 + 0.5)
    assert r["se_elo"] == pytest.approx(expect, rel=1e-6)


# --- ladder bookkeeping with a fake mango_match ------------------------------------------

class FakeMatch:
    """Stands in for mango_match: writes openings and reports; records the calls."""

    def __init__(self, strengths: dict[str, float]):
        self.strengths = strengths
        self.calls: list[list[str]] = []

    def __call__(self, args, log_name):
        args = [str(a) for a in args]
        self.calls.append(args)
        opt = {args[i]: args[i + 1] for i in range(1, len(args) - 1, 2)}
        if "--write-openings" in opt:
            pairs = int(opt["--pairs"])
            Path(opt["--write-openings"]).write_text(json.dumps([[i, i + 1] for i in range(pairs)]))
            return ""
        a = self.model_id(opt["--a"])
        b = self.model_id(opt["--b"])
        assert Path(opt["--openings-file"]).exists()
        ops = json.loads(Path(opt["--openings-file"]).read_text())[: int(opt["--pairs"])]
        pairs = len(ops)
        p = 1.0 / (1.0 + 10 ** ((self.strengths[b] - self.strengths[a]) / 400.0))
        wins = int(round(p * 2 * pairs))
        write_json_atomic(opt["--out"], {"a": a, "b": b, "seed": int(opt["--seed"]), "simulations": self.sims,
                                         "openings": ops, "pairs": pairs, "games": 2 * pairs,
                                         "wins_a": wins, "losses_a": 2 * pairs - wins, "draws_a": 0,
                                         "mean_pair_score": wins / (2 * pairs), "ci95": [0.0, 1.0]})
        return ""

    sims = 200

    @staticmethod
    def model_id(arg: str) -> str:
        """What mango_match reports: the model id from model.json (the dir name when the
        test did not write one), or "random"."""
        if arg == RANDOM:
            return RANDOM
        meta = Path(arg) / "model.json"
        return json.loads(meta.read_text())["model_id"] if meta.exists() else Path(arg).name


def make_run(tmp_path: Path, cfg_over: dict | None = None) -> tuple[Path, dict]:
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    cfg = merge_config({"board": {"size": 5}, "eval": {"ladder_pairs": 4, "ladder_neighbours": 2, "opening_moves": 2},
                        **(cfg_over or {})})
    write_json_atomic(run / "config.json", cfg)
    return run, cfg


def test_ladder_adds_entries_plays_anchors_and_neighbours_and_reuses_reports(tmp_path):
    run, cfg = make_run(tmp_path)
    strengths = {RANDOM: 0.0, "0000-init": 100.0, "0001-a": 300.0, "0002-b": 450.0, "0003-c": 500.0, "0005-d": 650.0}
    for m in strengths:
        if m != RANDOM:
            (run / "models" / m).mkdir()
    fake = FakeMatch(strengths)
    ladder = Ladder(run, cfg, tmp_path, run_seed=7, runner=fake)
    assert [e["name"] for e in ladder.data["entries"]] == [RANDOM]
    ladder.add_model("0000-init", 0, False)
    assert (run / "strength" / "openings_v1.json").exists()
    opening_calls = [c for c in fake.calls if "--write-openings" in c]
    assert len(opening_calls) == 1
    # Only the random anchor exists yet.
    assert [(m["a"], m["b"]) for m in ladder.data["matches"]] == [("0000-init", RANDOM)]
    ladder.add_model("0001-a", 1, True)   # first promoted: plays random, then the neighbour 0000-init
    ladder.add_model("0002-b", 2, True)   # random, first promoted (0001-a, also a neighbour), 0000-init
    ladder.add_model("0003-c", 3, False)  # random, first promoted, neighbours 0002-b (and 0001-a already chosen) -> + 0000-init
    played = ladder.add_model("0005-d", 5, True)
    opp = sorted((m["b"] for m in played))
    assert opp == sorted([RANDOM, "0001-a", "0003-c", "0002-b"])  # anchors + 2 most recent neighbours
    assert ladder.first_promoted()["name"] == "0001-a"
    # The opening file was generated once and every match used it.
    assert len([c for c in fake.calls if "--write-openings" in c]) == 1
    for c in fake.calls:
        if "--a" in c:
            assert c[c.index("--openings-file") + 1].endswith("openings_v1.json")
            assert c[c.index("--pairs") + 1] == "4"
    # Restart: adding an existing entry plays nothing new and reuses the reports on disk.
    n_calls = len(fake.calls)
    ladder2 = Ladder(run, cfg, tmp_path, run_seed=7, runner=fake)
    assert ladder2.add_model("0005-d", 5, True) == []
    assert len(fake.calls) == n_calls
    # A lost match record (crash between the report write and ladder.json) is rebuilt from the report.
    ladder2.data["matches"] = [m for m in ladder2.data["matches"] if {m["a"], m["b"]} != {"0005-d", RANDOM}]
    ladder2.save()
    again = ladder2.add_model("0005-d", 5, True)
    assert len(again) == 1 and len(fake.calls) == n_calls  # the report was reused, not replayed
    # Ratings: the fit runs over every entry; random pinned; order follows the fake strengths.
    fit = ladder2.fit(fit_iteration=5)
    elo = {k: v["elo"] for k, v in fit["ratings"].items()}
    assert elo[RANDOM] == 0.0
    assert elo["0000-init"] < elo["0001-a"] < elo["0002-b"] < elo["0003-c"] < elo["0005-d"]
    assert (run / "strength" / "ratings.json").exists()
    rows = (run / "strength" / "ratings.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + len(ladder2.data["entries"])
    assert rows[0].startswith("fit_iteration,name,kind,iteration,promoted,elo,low,high,games")
    # Seeds differ per pairing and are stable across restarts (the same pairing -> the same seed).
    seeds = {}
    for c in fake.calls:
        if "--a" in c:
            key = match_key(Path(c[c.index('--a') + 1]).name, c[c.index('--b') + 1] if c[c.index('--b') + 1] == RANDOM
                            else Path(c[c.index('--b') + 1]).name)
            seeds.setdefault(key, set()).add(c[c.index("--seed") + 1])
    assert all(len(s) == 1 for s in seeds.values())
    assert len({next(iter(s)) for s in seeds.values()}) == len(seeds)


def test_gtp_anchor_entry_only_plays_models(tmp_path):
    run, cfg = make_run(tmp_path, {"eval": {"ladder_pairs": 4, "ladder_neighbours": 2, "opening_moves": 2,
                                            "gtp_anchor": {"name": "gnugo", "command": ["gnugo", "--mode", "gtp"]}}})
    ladder = Ladder(run, cfg, tmp_path, run_seed=1, runner=FakeMatch({RANDOM: 0.0}))
    names = [e["name"] for e in ladder.data["entries"]]
    assert names == [RANDOM, "gnugo"]
    assert ladder.entry("gnugo")["command"] == ["gnugo", "--mode", "gtp"]
    with pytest.raises(ValueError):
        ladder.play(ladder.entry("gnugo"), ladder.entry(RANDOM), 0)


def _model_dir(path: Path, model_id: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path / "model.json", {"model_id": model_id})
    return path


def test_cross_run_round_robin_with_random_anchor(tmp_path):
    strengths = {RANDOM: 0.0, "0010-aaaa": 400.0, "0010-bbbb": 550.0, "0020-cccc": 700.0}
    fake = FakeMatch(strengths)
    cfg_path = tmp_path / "cfg.json"
    write_json_atomic(cfg_path, {"board": {"size": 5}})
    models = {"r0": _model_dir(tmp_path / "runA" / "models" / "0010-aaaa", "0010-aaaa"),
              "r1": _model_dir(tmp_path / "runB" / "models" / "0010-bbbb", "0010-bbbb")}
    fit = cross_run(tmp_path, cfg_path, tmp_path / "x", models, pairs=6, seed=3, runner=fake, log=lambda s: None)
    played = {(m["a"], m["b"]) for m in fit["matches"]}
    assert played == {(RANDOM, "r0"), (RANDOM, "r1"), ("r0", "r1")}
    assert {(m["model_a"], m["model_b"]) for m in fit["matches"]} == {(RANDOM, "0010-aaaa"), (RANDOM, "0010-bbbb"),
                                                                     ("0010-aaaa", "0010-bbbb")}
    assert fit["ratings"]["r0"]["elo"] < fit["ratings"]["r1"]["elo"]
    saved = read_json(tmp_path / "x" / "crossrun.json")
    assert saved["models"]["r0"] == {"dir": str(models["r0"]), "model_id": "0010-aaaa"}
    # Re-running reuses the reports.
    n = len(fake.calls)
    cross_run(tmp_path, cfg_path, tmp_path / "x", models, pairs=6, seed=3, runner=fake, log=lambda s: None)
    assert len(fake.calls) == n
    # Run A trained further: the label "r0" now points to a different model id, so its
    # two matches are replayed (the r1-random one is kept) and the fit changes.
    models["r0"] = _model_dir(tmp_path / "runA" / "models" / "0020-cccc", "0020-cccc")
    fit2 = cross_run(tmp_path, cfg_path, tmp_path / "x", models, pairs=6, seed=3, runner=fake, log=lambda s: None)
    assert len(fake.calls) == n + 2
    assert fit2["models"]["r0"]["model_id"] == "0020-cccc"
    assert fit2["ratings"]["r0"]["elo"] > fit2["ratings"]["r1"]["elo"]
    # A different seed, budget or opening set also invalidates a stored report.
    n = len(fake.calls)
    cross_run(tmp_path, cfg_path, tmp_path / "x", models, pairs=6, seed=4, runner=fake, log=lambda s: None)
    assert len(fake.calls) == n + 3
    ops = json.loads((tmp_path / "x" / "openings.json").read_text())
    report = read_json(tmp_path / "x" / "matches" / f"{match_key(RANDOM, 'r1')}.json")
    assert report_matches(report, RANDOM, "0010-bbbb", report["seed"], 200, ops[:6])
    assert not report_matches(report, RANDOM, "0010-bbbb", report["seed"], 100, ops[:6])
    assert not report_matches(report, RANDOM, "0010-bbbb", report["seed"], 200, ops[:5])
    assert not report_matches(report, RANDOM, "0010-bbbb", report["seed"] + 1, 200, ops[:6])
    assert not report_matches(None, RANDOM, "0010-bbbb", 0, 200, ops[:6])


# --- diagnostics of the rating model (M4 review) ------------------------------------------

def _report(pairs: list[tuple[int, int]]) -> dict:
    """games_detail from per-pair results (black-positive result of A's black game, then
    of A's white game); result 0 = draw."""
    games = []
    for i, (r0, r1) in enumerate(pairs):
        games.append({"pair": i, "a_is_black": True, "result": r0, "score": float(r0), "moves": []})
        games.append({"pair": i, "a_is_black": False, "result": r1, "score": float(r1), "moves": []})
    wins = sum(1 for g in games if g["result"] != 0 and (g["result"] > 0) == g["a_is_black"])
    losses = sum(1 for g in games if g["result"] != 0 and (g["result"] > 0) != g["a_is_black"])
    return {"a": "A", "b": "B", "games_detail": games, "wins_a": wins, "losses_a": losses,
            "draws_a": len(games) - wins - losses, "pairs": len(pairs)}


def test_pair_structure_classifies_every_pair():
    # (result of A's black game, result of A's white game), black-positive.
    rep = _report([(+1, -1), (-1, +1), (+1, +1), (-1, -1), (0, +1)])
    ps = pair_structure(rep)
    assert ps["pairs"] == 5 and ps["both_won"] == 1 and ps["both_lost"] == 1 and ps["split"] == 2 and ps["draw_pairs"] == 1
    # In a colour-swapped pair a split means the same colour won both games.
    assert ps["black_wins"] == pytest.approx(5 / 9)
    obs = game_observations([{"a": "A", "b": "B"}], lambda _: rep)
    assert len(obs) == 10
    assert (obs[0].black, obs[0].white, obs[0].opening, obs[0].result) == ("A", "B", 0, 1.0)
    assert (obs[1].black, obs[1].white, obs[1].opening, obs[1].result) == ("B", "A", 0, 0.0)
    assert obs[8].result == 0.5
    assert game_observations([{"a": "A", "b": "B"}], lambda _: None) == []


def _simulate_games(true_elo, openings_black_elo, pairings, pairs_per_match, seed):
    rng = np.random.default_rng(seed)
    games = []
    for a, b in pairings:
        for k in range(pairs_per_match):
            o = k % len(openings_black_elo)
            for black, white in ((a, b), (b, a)):
                d = (true_elo[black] - true_elo[white] + openings_black_elo[o]) / ELO_PER_NAT
                p = 1.0 / (1.0 + math.exp(-d))
                games.append(GameObs(black, white, o, 1.0 if rng.random() < p else 0.0))
    return games


def test_opening_adjusted_fit_separates_strength_from_colour_advantage():
    true = {RANDOM: 0.0, "m1": 300.0, "m2": 500.0, "m3": 650.0}
    ob = [0.0, 400.0, -400.0, 250.0, -250.0, 0.0, 600.0, -600.0, 100.0, -100.0]
    pairings = [(RANDOM, "m1"), ("m1", "m2"), ("m2", "m3"), (RANDOM, "m2"), ("m1", "m3"), (RANDOM, "m3")]
    games = _simulate_games(true, ob, pairings, pairs_per_match=400, seed=5)
    fit = fit_bradley_terry_openings(list(true), games)
    assert fit["converged"] and fit["model"] == "openings"
    r = fit["ratings"]
    assert r[RANDOM]["elo"] == 0.0
    for name in ("m1", "m2", "m3"):
        assert abs(r[name]["elo"] - true[name]) < 40.0, name
        assert r[name]["low"] < true[name] < r[name]["high"]
    for k, adv in enumerate(ob):
        assert abs(fit["openings"][k]["black_elo"] - adv) < 70.0, k
    assert fit["openings"][0]["games"] == 2 * 40 * len(pairings)
    res = match_residuals_openings(fit)
    assert len(res) == 2 * len(pairings)
    assert max(abs(x["z"]) for x in res) < 3.5
    # Equal players on a colour-decided set: ratings stay together, the opening terms are large.
    equal = {RANDOM: 0.0, "x": 0.0}
    decided = _simulate_games(equal, [900.0, -900.0], [(RANDOM, "x")], pairs_per_match=300, seed=9)
    f2 = fit_bradley_terry_openings([RANDOM, "x"], decided)
    assert abs(f2["ratings"]["x"]["elo"]) < 60.0
    assert f2["openings"][0]["black_elo"] > 500.0 and f2["openings"][1]["black_elo"] < -500.0
    # Bookkeeping: separation flag, unrated, errors.
    sep = fit_bradley_terry_openings([RANDOM, "s"], [GameObs("s", RANDOM, 0, 1.0), GameObs(RANDOM, "s", 0, 0.0)])
    assert sep["ratings"]["s"]["separated"] and sep["ratings"]["s"]["rated"] and math.isfinite(sep["ratings"]["s"]["elo"])
    iso = fit_bradley_terry_openings([RANDOM, "s", "t"], [GameObs("s", RANDOM, 0, 1.0)])
    assert iso["unrated"] == ["t"]
    with pytest.raises(ValueError):
        fit_bradley_terry_openings(["s"], [])
    with pytest.raises(ValueError):
        fit_bradley_terry_openings([RANDOM, "s"], [GameObs("s", "s", 0, 1.0)])


def test_predicted_vs_observed_and_prior_sensitivity():
    res = [MatchResult(RANDOM, "a", 10, 90, 0)]
    fit = fit_bradley_terry([RANDOM, "a"], res, prior_sigma_elo=1e6)
    rows = predicted_vs_observed(fit, res)
    assert len(rows) == 1 and rows[0]["observed"] == pytest.approx(0.1)
    assert rows[0]["predicted"] == pytest.approx(0.1, abs=1e-3)  # a wide prior reproduces the MLE
    assert abs(rows[0]["z"]) < 0.05
    # A result the fit cannot explain gets a large |z|: force ratings by hand.
    fake = {"ratings": {RANDOM: {"elo": 0.0, "rated": True}, "a": {"elo": 1289.0, "rated": True}}}
    row = predicted_vs_observed(fake, [MatchResult("a", RANDOM, 88, 12, 0)])[0]
    assert row["predicted"] > 0.999 and row["observed"] == pytest.approx(0.88) and row["z"] < -30
    fake["ratings"]["a"]["rated"] = False
    assert predicted_vs_observed(fake, [MatchResult("a", RANDOM, 88, 12, 0)]) == []
    # Sensitivity: a separated player's rating grows with sigma; monotone order is judged per sigma.
    chain = [MatchResult("m1", RANDOM, 100, 0, 0), MatchResult("m2", "m1", 60, 40, 0), MatchResult("m2", RANDOM, 100, 0, 0)]
    sens = prior_sensitivity([RANDOM, "m1", "m2"], chain, ["m1", "m2"], sigmas=(200.0, 1400.0))
    assert [s["sigma"] for s in sens] == [200.0, 1400.0]
    assert sens[0]["elo"]["m2"] < sens[1]["elo"]["m2"]
    assert all(s["monotone"] for s in sens)
    assert all(s["mean_abs_z"] >= 0 and s["max_abs_z"] >= s["mean_abs_z"] for s in sens)
    swapped = prior_sensitivity([RANDOM, "m1", "m2"], chain, ["m2", "m1"], sigmas=(350.0,))
    assert not swapped[0]["monotone"]


def test_opening_balance_finds_colour_decided_openings():
    r1 = _report([(+1, -1), (+1, +1), (-1, -1)])  # opening 1 black-decided, opening 2 white-decided
    r2 = _report([(+1, -1), (+1, +1), (+1, -1)])  # opening 1 again black-decided, opening 2 not
    ob = opening_balance([r1, r2])
    assert ob["pairs"] == 6 and ob["colour_decided_pairs"] == 3
    assert ob["colour_decided_openings"] == [1, 2]  # 2/2 and 1/2 pairs colour-decided (>= half)
    assert ob["openings"][0]["colour_decided"] == 0
    assert ob["black_win_rate"] == pytest.approx(7 / 12)


def test_ladder_fit_records_residuals_and_opening_fit_falls_back(tmp_path):
    run, cfg = make_run(tmp_path)
    strengths = {RANDOM: 0.0, "0000-init": 100.0, "0001-a": 300.0}
    for m in strengths:
        if m != RANDOM:
            (run / "models" / m).mkdir()
    ladder = Ladder(run, cfg, tmp_path, run_seed=7, runner=FakeMatch(strengths))
    ladder.add_model("0000-init", 0, False)
    ladder.add_model("0001-a", 1, True)
    fit = ladder.fit(record=False)
    assert len(fit["residuals"]) == len(ladder.data["matches"])
    assert ladder.model_order() == ["0000-init", "0001-a"]
    # The fake reports carry no games_detail: no game observations, everything unrated.
    assert ladder.games() == []
    of = ladder.fit_openings()
    assert set(of["unrated"]) == {"0000-init", "0001-a"} and of["converged"]


def test_remeasure_replays_every_match_and_archives_the_old_ladder(tmp_path):
    run, cfg = make_run(tmp_path)
    strengths = {RANDOM: 0.0, "0000-init": 100.0, "0001-a": 300.0, "0002-b": 450.0}
    for m in strengths:
        if m != RANDOM:
            (run / "models" / m).mkdir()
    fake = FakeMatch(strengths)
    ladder = Ladder(run, cfg, tmp_path, run_seed=7, runner=fake)
    ladder.add_model("0000-init", 0, False)
    ladder.add_model("0001-a", 1, True)
    ladder.add_model("0002-b", 2, False)
    ladder.fit(fit_iteration=2)
    old_entries = [dict(e) for e in ladder.data["entries"]]
    old_matches = [(m["a"], m["b"]) for m in ladder.data["matches"]]
    old_seeds = {(m["a"], m["b"]): read_json(run / m["report"])["seed"] for m in ladder.data["matches"]}
    openings = (run / "strength" / "openings_v1.json").read_bytes()
    n_calls = len(fake.calls)
    # The engine "changed": the fake now rates 0002-b much higher.
    fake.strengths["0002-b"] = 900.0
    new = remeasure_ladder(run, cfg, tmp_path, 7, "plain", runner=fake)
    arch = run / "strength_plain"
    assert (arch / "ladder.json").exists() and (arch / "matches").is_dir() and (arch / "ratings.csv").exists()
    assert (arch / "openings_v1.json").read_bytes() == openings
    assert (run / "strength" / "openings_v1.json").read_bytes() == openings  # kept, not regenerated
    assert len([c for c in fake.calls if "--write-openings" in c]) == 1
    assert [dict(e) for e in new.data["entries"]] == old_entries
    assert [(m["a"], m["b"]) for m in new.data["matches"]] == old_matches
    assert len(fake.calls) == n_calls + len(old_matches)  # every match replayed, nothing reused
    for m in new.data["matches"]:
        assert read_json(run / m["report"])["seed"] == old_seeds[(m["a"], m["b"])]
    fit = read_json(run / "strength" / "ratings.json")
    assert fit["ratings"]["0002-b"]["elo"] > read_json(arch / "ratings.json")["ratings"]["0002-b"]["elo"] + 200
    with pytest.raises(FileExistsError):
        remeasure_ladder(run, cfg, tmp_path, 7, "plain", runner=fake)
