"""GTP client and a colour-swapped pair match against an external engine (DESIGN 6.6,
the ladder's external anchor: GNU Go, KataGo, ... anything speaking GTP v2).

The Mango side is `mango_gtp --model DIR`; the referee is a third `mango_gtp` process
(random player, no model) that receives every move and answers `final_score`, so games
against external engines are scored with Mango's own rules (Tromp-Taylor area scoring,
DESIGN 5.1) regardless of what the opponent would report. Games end on two consecutive
passes, a resignation, an illegal move (loss for the side that played it) or the move
cap. The report has the same shape as `mango_match`'s JSON.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LETTERS = "ABCDEFGHJKLMNOPQRST"


def move_to_vertex(m: int, n: int) -> str:
    if m == n * n:
        return "pass"
    r, c = divmod(int(m), n)
    return f"{LETTERS[c]}{n - r}"


def vertex_to_move(s: str, n: int) -> int:
    s = s.strip().lower()
    if s == "pass":
        return n * n
    if s == "resign":
        return -1
    col = LETTERS.index(s[0].upper())
    row = int(s[1:])
    if not (1 <= row <= n) or col >= n:
        raise ValueError(f"vertex out of range: {s!r}")
    return (n - row) * n + col


class GtpClient:
    """One engine process; `command(...)` returns the response text (without the '=')
    and raises GtpError on a '?' reply."""

    def __init__(self, cmd: Sequence[str], name: str = ""):
        self.cmd = [str(c) for c in cmd]
        self.name = name or self.cmd[0]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)

    def command(self, line: str) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(line.strip() + "\n")
        self.proc.stdin.flush()
        lines: list[str] = []
        while True:
            out = self.proc.stdout.readline()
            if out == "":
                raise GtpError(f"{self.name}: engine closed its output on {line!r}")
            out = out.rstrip("\r\n")
            if out == "" and lines:
                break
            if out == "":
                continue
            lines.append(out)
        head = lines[0]
        body = "\n".join([head[1:].strip()] + lines[1:]).strip()
        if head.startswith("="):
            return body
        if head.startswith("?"):
            raise GtpError(f"{self.name}: {body}")
        raise GtpError(f"{self.name}: malformed reply {head!r}")

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self) -> "GtpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GtpError(RuntimeError):
    pass


def parse_final_score(s: str) -> float:
    """'B+3.5' -> 3.5, 'W+2' -> -2.0, '0' / 'draw' -> 0.0 (black-positive)."""
    s = s.strip()
    if s in ("0", "draw", "Draw", "jigo"):
        return 0.0
    side, _, num = s.partition("+")
    if side.upper().startswith("B"):
        return float(num)
    if side.upper().startswith("W"):
        return -float(num)
    raise ValueError(f"unrecognised score {s!r}")


@dataclass
class GtpGame:
    pair: int
    a_is_black: bool
    moves: list[int] = field(default_factory=list)
    result: int = 0
    score: float = 0.0
    termination: str = "two_passes"

    def score_for_a(self) -> float:
        if self.result == 0:
            return 0.5
        return 1.0 if (self.result > 0) == self.a_is_black else 0.0


def _setup(engine: GtpClient, n: int, komi: float) -> None:
    engine.command(f"boardsize {n}")
    engine.command("clear_board")
    engine.command(f"komi {komi}")


def play_gtp_game(black: GtpClient, white: GtpClient, referee: GtpClient, n: int, komi: float, opening: Sequence[int],
                  move_cap: int, pair: int, a_is_black: bool) -> GtpGame:
    """One game; `black`/`white` generate moves, `referee` (a mango_gtp random player)
    tracks the position and scores it."""
    for e in (black, white, referee):
        _setup(e, n, komi)
    g = GtpGame(pair=pair, a_is_black=a_is_black)
    colour = "B"
    for m in opening:
        v = move_to_vertex(m, n)
        for e in (black, white, referee):
            e.command(f"play {colour} {v}")
        g.moves.append(int(m))
        colour = "W" if colour == "B" else "B"
    passes = 0
    while len(g.moves) < move_cap:
        mover, other = (black, white) if colour == "B" else (white, black)
        reply = mover.command(f"genmove {colour}")
        if reply.strip().lower() == "resign":
            g.termination = "resign"
            g.result = -1 if colour == "B" else 1
            g.score = float(g.result) * (n * n + komi)
            return g
        m = vertex_to_move(reply, n)
        v = move_to_vertex(m, n)
        try:
            referee.command(f"play {colour} {v}")
        except GtpError:
            # Illegal under Mango's rules (superko, suicide): the side that played it loses.
            g.termination = "illegal"
            g.result = -1 if colour == "B" else 1
            g.score = float(g.result) * (n * n + komi)
            return g
        other.command(f"play {colour} {v}")
        g.moves.append(m)
        passes = passes + 1 if m == n * n else 0
        colour = "W" if colour == "B" else "B"
        if passes >= 2:
            break
    if passes < 2:
        g.termination = "move_cap"
    g.score = parse_final_score(referee.command("final_score"))
    g.result = 1 if g.score > 0 else (-1 if g.score < 0 else 0)
    return g


def bootstrap_mean_interval(values: Sequence[float], resamples: int = 10000, seed: int = 1) -> tuple[float, float]:
    arr = np.asarray(values, np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(resamples, arr.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def play_gtp_match(cmd_a: Sequence[str], cmd_b: Sequence[str], referee_cmd: Sequence[str], n: int, komi: float,
                   openings: Sequence[Sequence[int]], move_cap: int, name_a: str, name_b: str, seed: int = 1) -> dict:
    """Every opening twice with colours swapped; A's pair scores and a bootstrap interval,
    in the shape of `mango_match`'s report (fields used by strength.py)."""
    games: list[GtpGame] = []
    with GtpClient(cmd_a, name_a) as ea, GtpClient(cmd_b, name_b) as eb, GtpClient(referee_cmd, "referee") as ref:
        for i, op in enumerate(openings):
            games.append(play_gtp_game(ea, eb, ref, n, komi, op, move_cap, i, True))
            games.append(play_gtp_game(eb, ea, ref, n, komi, op, move_cap, i, False))
    pair_scores = [(games[2 * i].score_for_a() + games[2 * i + 1].score_for_a()) / 2.0 for i in range(len(openings))]
    wins = sum(1 for g in games if g.score_for_a() > 0.75)
    losses = sum(1 for g in games if g.score_for_a() < 0.25)
    ci = bootstrap_mean_interval(pair_scores, seed=seed)
    return {
        "a": name_a, "b": name_b, "board_size": n, "komi": komi, "seed": seed, "driver": "gtp",
        "pairs": len(openings), "games": len(games), "wins_a": wins, "losses_a": losses,
        "draws_a": len(games) - wins - losses,
        "mean_pair_score": float(np.mean(pair_scores)) if pair_scores else 0.0, "ci95": [ci[0], ci[1]],
        "pair_scores": pair_scores, "openings": [list(map(int, op)) for op in openings],
        "games_detail": [{"pair": g.pair, "a_is_black": g.a_is_black, "result": g.result, "score": g.score,
                          "termination": g.termination, "moves": g.moves} for g in games],
    }
