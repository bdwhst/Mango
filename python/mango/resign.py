"""Automatic resignation threshold (docs/DESIGN.md section 5.4.7).

Over the no-resign games of the current window: a game is a false positive for a
threshold t if its eventual winner had r_t = max(v(s0), max_a Q(s0, a)) < t at any of
their moves. v_resign is the largest t in [-0.99, -0.50] (step 0.01) whose false-positive
rate is below 5 %; with fewer than 100 no-resign games, or when no t qualifies,
resignation stays disabled (None). Drawn games have no winner and are excluded from
both the numerator and the denominator.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .chunk import GameRecord

RESIGN_MIN, RESIGN_MAX = -0.99, -0.50
RESIGN_STEP = 0.01
MIN_NO_RESIGN_GAMES = 100
MAX_FALSE_POSITIVE_RATE = 0.05


def winner_min_root_value(g: GameRecord) -> float | None:
    """min over the winner's moves of r_t; None for a draw or a winner without moves."""
    if g.result == 0:
        return None
    start = 0 if g.result > 0 else 1  # black moves at even t
    r = np.maximum(np.asarray(g.root_value, np.float32)[start::2], np.asarray(g.root_max_q, np.float32)[start::2])
    if r.size == 0:
        return None
    return float(r.min())


def false_positive_rate(winner_mins: np.ndarray, t: float) -> float:
    """Fraction of games whose winner dipped below t at some move."""
    if winner_mins.size == 0:
        return 0.0
    return float(np.count_nonzero(winner_mins < t)) / float(winner_mins.size)


def candidate_thresholds() -> list[float]:
    """-0.50, -0.51, ..., -0.99 (largest first)."""
    steps = int(round((RESIGN_MAX - RESIGN_MIN) / RESIGN_STEP))
    return [round(RESIGN_MAX - k * RESIGN_STEP, 2) for k in range(steps + 1)]


def select_resign_threshold(games: Iterable[GameRecord], min_games: int = MIN_NO_RESIGN_GAMES,
                            max_fpr: float = MAX_FALSE_POSITIVE_RATE) -> dict:
    """Returns {"threshold": float | None, "no_resign_games": n, "samples": m,
    "false_positive_rate": rate at the chosen threshold (or at the lowest candidate)}."""
    mins = []
    n_no_resign = 0
    for g in games:
        if not g.no_resign_game:
            continue
        n_no_resign += 1
        m = winner_min_root_value(g)
        if m is not None:
            mins.append(m)
    arr = np.asarray(mins, np.float32)
    out = {"threshold": None, "no_resign_games": n_no_resign, "samples": int(arr.size), "false_positive_rate": None}
    if arr.size < min_games:
        return out
    for t in candidate_thresholds():
        rate = false_positive_rate(arr, t)
        if rate < max_fpr:
            out["threshold"] = t
            out["false_positive_rate"] = rate
            return out
    out["false_positive_rate"] = false_positive_rate(arr, candidate_thresholds()[-1])
    return out


def first_trigger(g: GameRecord, t: float) -> tuple[int, bool] | None:
    """The first move at which the player to move had r_t < t, as (t, mover_is_winner);
    None if nobody dipped below t. A game with resignation enabled would have ended
    there, resigned by that mover, so the label would be wrong iff the mover is the
    eventual winner. Draws have no winner (mover_is_winner is False)."""
    r = np.maximum(np.asarray(g.root_value, np.float32), np.asarray(g.root_max_q, np.float32))
    below = np.flatnonzero(r < t)
    if below.size == 0:
        return None
    t0 = int(below[0])
    mover_black = t0 % 2 == 0
    winner_black = g.result > 0
    return t0, bool(g.result != 0 and mover_black == winner_black)


def counterfactual_false_resign_rate(games: Iterable[GameRecord], threshold: float | None) -> dict:
    """What the paper's false-positive rate does not give: the estimated fraction of
    games that resignation at `threshold` would have ended with the WRONG label. Over
    the no-resign games given, a game is "triggered" if either player dipped below the
    threshold at some move; it is mislabelled if the first player to dip is the eventual
    winner (the game ends there with that player resigning). Returns the counts and
    both rates: per triggered game (the estimate for resigned games) and per game."""
    if threshold is None:
        return {"threshold": None, "samples": 0, "triggered": 0, "mislabelled": 0,
                "rate_of_triggered": None, "rate_of_all": None}
    samples = triggered = mislabelled = 0
    for g in games:
        if not g.no_resign_game:
            continue
        samples += 1
        ft = first_trigger(g, threshold)
        if ft is None:
            continue
        triggered += 1
        mislabelled += ft[1]
    return {"threshold": threshold, "samples": samples, "triggered": triggered, "mislabelled": mislabelled,
            "rate_of_triggered": (mislabelled / triggered) if triggered else None,
            "rate_of_all": (mislabelled / samples) if samples else None}


def measured_false_positive_rate(games: Iterable[GameRecord], threshold: float | None) -> dict:
    """Diagnostic: the paper's false-positive rate of `threshold` on the no-resign games
    given (typically the newest iteration's): the fraction of games whose eventual winner
    was below the threshold at some move. This is the control quantity of 5.4.7, not a
    label-error rate: in most such games the loser dips first and the game would have
    ended correctly. The counterfactual label-error estimate is reported alongside."""
    if threshold is None:
        return {"threshold": None, "samples": 0, "false_positive_rate": None, "counterfactual": None}
    games = list(games)
    mins = [m for g in games if g.no_resign_game for m in [winner_min_root_value(g)] if m is not None]
    arr = np.asarray(mins, np.float32)
    return {"threshold": threshold, "samples": int(arr.size),
            "false_positive_rate": false_positive_rate(arr, threshold) if arr.size else None,
            "counterfactual": counterfactual_false_resign_rate(games, threshold)}
