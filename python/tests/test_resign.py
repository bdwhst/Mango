"""Automatic resignation threshold (DESIGN 5.4.7): false-positive definition, the
largest qualifying t, the >= 100 games rule, draws excluded, the diagnostic rate."""

from __future__ import annotations

import numpy as np
import pytest

from mango.chunk import GameRecord
from mango.resign import (candidate_thresholds, counterfactual_false_resign_rate, false_positive_rate, first_trigger,
                          measured_false_positive_rate, select_resign_threshold, winner_min_root_value)


def game(result: int, black_r: list[float], white_r: list[float], no_resign: bool = True, max_q_offset: float = 0.0) -> GameRecord:
    """A record whose root values alternate black (even t) / white (odd t)."""
    assert len(black_r) - len(white_r) in (0, 1), "black moves first: |black| == |white| or |white| + 1"
    T = len(black_r) + len(white_r)
    rv = np.zeros(T, np.float32)
    rv[0::2] = black_r
    rv[1::2] = white_r
    g = GameRecord(result=result, no_resign_game=no_resign)
    g.moves = np.zeros(T, np.uint16)
    g.root_value = rv
    g.root_max_q = rv + max_q_offset
    return g


def test_winner_min_uses_the_winners_moves_and_the_max_of_value_and_q():
    # Black wins; black's values dip to -0.8, white's to -0.95: only black's count.
    g = game(+1, [0.1, -0.8, 0.3], [-0.95, -0.9], max_q_offset=0.0)
    assert winner_min_root_value(g) == pytest.approx(-0.8)
    # White wins: white's moves (odd t).
    g = game(-1, [0.1, -0.8, 0.3], [-0.95, -0.9])
    assert winner_min_root_value(g) == pytest.approx(-0.95)
    # r_t = max(v, max Q): a higher max Q rescues the dip.
    g = game(+1, [0.1, -0.8, 0.3], [0.0, 0.0], max_q_offset=0.5)
    assert winner_min_root_value(g) == pytest.approx(-0.3)
    # Draws have no winner.
    assert winner_min_root_value(game(0, [-1.0], [-1.0])) is None
    # A winner without moves (T = 1, white wins) cannot be a false positive.
    assert winner_min_root_value(game(-1, [-1.0], [])) is None


def test_candidate_grid_and_rate():
    ts = candidate_thresholds()
    assert ts[0] == -0.5 and ts[-1] == -0.99 and len(ts) == 50
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))
    mins = np.array([-0.6, -0.7, -0.95, 0.2], np.float32)
    assert false_positive_rate(mins, -0.5) == pytest.approx(0.75)
    assert false_positive_rate(mins, -0.65) == pytest.approx(0.5)  # -0.7 and -0.95 are below
    assert false_positive_rate(mins, -0.99) == pytest.approx(0.0)
    assert false_positive_rate(np.zeros(0, np.float32), -0.5) == 0.0


def test_selects_the_largest_threshold_under_five_percent():
    # 200 no-resign games: 4 winners dipped to -0.75, 6 to -0.62, the rest never below -0.3.
    games = [game(+1, [0.0, -0.75, 0.1], [0.0, 0.0]) for _ in range(4)]
    games += [game(-1, [0.0, 0.0], [-0.62, 0.3]) for _ in range(6)]
    games += [game(+1, [0.0, -0.3, 0.1], [0.0, 0.0]) for _ in range(190)]
    sel = select_resign_threshold(games)
    # t = -0.61: 4 + 6 = 10 / 200 = 5 % (not < 5 %); t = -0.62 excludes the -0.62 group -> 4 / 200 = 2 %.
    assert sel["threshold"] == pytest.approx(-0.62)
    assert sel["false_positive_rate"] == pytest.approx(0.02)
    assert sel["samples"] == 200 and sel["no_resign_games"] == 200
    # With 11 games at -0.75 (11 / 207 = 5.3 % for every t above -0.75) the threshold
    # drops to exactly -0.75: r_t = t is not "below t", so those winners would not resign.
    games2 = [game(+1, [0.0, -0.75, 0.1], [0.0, 0.0]) for _ in range(11)] + games[4:]
    sel2 = select_resign_threshold(games2)
    assert sel2["threshold"] == pytest.approx(-0.75)
    assert sel2["false_positive_rate"] == pytest.approx(0.0)
    # A tighter target (2.5 %) on the first set: 4 / 200 = 2 % qualifies only below -0.62... at
    # t = -0.62 the rate is 2 % (< 2.5 %), so the choice is unchanged; at 1 % it must drop below -0.75.
    assert select_resign_threshold(games, max_fpr=0.025)["threshold"] == pytest.approx(-0.62)
    assert select_resign_threshold(games, max_fpr=0.01)["threshold"] == pytest.approx(-0.75)


def test_disabled_with_too_few_games_or_no_qualifying_threshold():
    few = [game(+1, [0.0], [0.0]) for _ in range(99)]
    sel = select_resign_threshold(few)
    assert sel["threshold"] is None and sel["samples"] == 99
    # Games played WITH resignation allowed do not count, even when there are many.
    tagged_off = [game(+1, [0.0], [0.0], no_resign=False) for _ in range(300)]
    assert select_resign_threshold(tagged_off)["threshold"] is None
    # 100 games where 10 % of winners dip to -1.0: no t in [-0.99, -0.5] is below 5 %.
    bad = [game(+1, [0.0, -1.0], [0.0]) for _ in range(10)] + [game(+1, [0.0], [0.0]) for _ in range(90)]
    sel = select_resign_threshold(bad)
    assert sel["threshold"] is None and sel["false_positive_rate"] == pytest.approx(0.10)
    # Draws are excluded from both the numerator and the denominator.
    draws = [game(0, [-1.0], [-1.0]) for _ in range(50)] + [game(+1, [0.0], [0.0]) for _ in range(99)]
    assert select_resign_threshold(draws)["threshold"] is None  # 99 samples, not 149
    ok = draws + [game(+1, [0.0], [0.0])]
    assert select_resign_threshold(ok)["threshold"] == pytest.approx(-0.5)


def test_measured_rate_is_a_diagnostic_over_the_given_games():
    games = [game(+1, [0.0, -0.9], [0.0]) for _ in range(2)] + [game(+1, [0.0], [0.0]) for _ in range(6)]
    d = measured_false_positive_rate(games, -0.8)
    assert d["samples"] == 8 and d["false_positive_rate"] == pytest.approx(0.25)
    assert measured_false_positive_rate(games, None)["false_positive_rate"] is None


def test_counterfactual_label_error_counts_only_winner_first_triggers():
    t = -0.8
    # Loser (white) dips first at t=1, winner (black) dips later at t=2: the game would have
    # ended correctly with white resigning -> triggered, not mislabelled (but it IS a
    # paper false positive, because the winner dipped at some move).
    loser_first = game(+1, [0.0, -0.9, 0.1], [-0.95, 0.0])
    assert first_trigger(loser_first, t) == (1, False)
    # Winner (black) dips first at t=0: black would have resigned a game it went on to win.
    winner_first = game(+1, [-0.9, 0.2], [0.0, -0.95])
    assert first_trigger(winner_first, t) == (0, True)
    # Nobody dips: not triggered.
    quiet = game(-1, [0.0, 0.0], [0.0, 0.0])
    assert first_trigger(quiet, t) is None
    # A draw has no winner: triggered, never mislabelled.
    draw = game(0, [-0.9], [0.0])
    assert first_trigger(draw, t) == (0, False)
    # r_t = max(v, max Q): a high max Q prevents the trigger.
    rescued = game(+1, [-0.9, 0.0], [0.0, 0.0], max_q_offset=0.5)
    assert first_trigger(rescued, t) is None
    games = [loser_first, winner_first, quiet, draw, rescued, game(+1, [0.0], [0.0], no_resign=False)]
    cf = counterfactual_false_resign_rate(games, t)
    assert cf == {"threshold": t, "samples": 5, "triggered": 3, "mislabelled": 1,
                  "rate_of_triggered": pytest.approx(1 / 3), "rate_of_all": pytest.approx(1 / 5)}
    # The paper's rate over the same games counts loser_first and winner_first (winner dipped): 2 / 4 decided.
    m = measured_false_positive_rate(games, t)
    assert m["samples"] == 4 and m["false_positive_rate"] == pytest.approx(0.5)
    assert m["counterfactual"]["mislabelled"] == 1 and m["counterfactual"]["triggered"] == 3
    assert counterfactual_false_resign_rate(games, None)["rate_of_triggered"] is None
    assert counterfactual_false_resign_rate([], t)["rate_of_all"] is None
