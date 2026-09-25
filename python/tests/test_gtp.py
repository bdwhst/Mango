"""GTP client / external-engine match driver (DESIGN 6.6): vertex conversion, score
parsing, and a real match between two `mango_gtp` random players refereed by a third."""

from __future__ import annotations

import pytest

from mango.gtp import GtpClient, bootstrap_mean_interval, move_to_vertex, parse_final_score, play_gtp_match, vertex_to_move
from tests.test_pipeline import _bin_dir


def test_vertex_conversion_matches_the_engine_convention():
    n = 9
    # Point 0 is the top-left corner (row 0 = rank 9); 'I' is skipped.
    assert move_to_vertex(0, n) == "A9"
    assert move_to_vertex(8, n) == "J9"
    assert move_to_vertex(72, n) == "A1"
    assert move_to_vertex(81, n) == "pass"
    for m in range(n * n + 1):
        assert vertex_to_move(move_to_vertex(m, n), n) == m
    assert vertex_to_move("resign", n) == -1
    assert vertex_to_move(" j1 ", n) == 80
    with pytest.raises(ValueError):
        vertex_to_move("A10", n)
    n = 19
    assert move_to_vertex(18, n) == "T19" and vertex_to_move("T1", n) == 18 * 19 + 18


def test_score_parsing_and_bootstrap():
    assert parse_final_score("B+3.5") == 3.5
    assert parse_final_score("W+2") == -2.0
    assert parse_final_score("0") == 0.0
    with pytest.raises(ValueError):
        parse_final_score("X+1")
    lo, hi = bootstrap_mean_interval([1.0] * 10)
    assert lo == 1.0 and hi == 1.0
    lo, hi = bootstrap_mean_interval([0.0, 1.0] * 10, seed=4)
    assert lo < 0.5 < hi


@pytest.fixture
def gtp_exe():
    b = _bin_dir()
    if b is None:
        pytest.skip("C++ executables not built")
    return b / ("mango_gtp.exe" if b.joinpath("mango_gtp.exe").exists() else "mango_gtp")


def test_gtp_client_talks_to_mango_gtp(gtp_exe):
    with GtpClient([gtp_exe, "--size", "5", "--seed", "3"]) as e:
        assert e.command("protocol_version") == "2"
        assert e.command("boardsize 5") == ""
        assert e.command("play B C3") == ""
        with pytest.raises(Exception):
            e.command("play W C3")  # occupied
        reply = e.command("genmove W")
        assert reply and reply != "C3"
        board = e.command("showboard")
        assert "X" in board and "O" in board


def test_random_vs_random_match_over_gtp(gtp_exe):
    n = 5
    a = [gtp_exe, "--size", n, "--komi", 7.5, "--seed", 11]
    b = [gtp_exe, "--size", n, "--komi", 7.5, "--seed", 12]
    ref = [gtp_exe, "--size", n, "--komi", 7.5]
    openings = [[0, 1], [12, 6], [24, 0]]
    r = play_gtp_match(a, b, ref, n, 7.5, openings, move_cap=50, name_a="ra", name_b="rb", seed=5)
    assert r["pairs"] == 3 and r["games"] == 6 and r["driver"] == "gtp"
    assert r["wins_a"] + r["losses_a"] + r["draws_a"] == 6
    assert 0.0 <= r["ci95"][0] <= r["mean_pair_score"] <= r["ci95"][1] <= 1.0
    for i, g in enumerate(r["games_detail"]):
        assert g["pair"] == i // 2 and g["a_is_black"] == (i % 2 == 0)
        assert g["moves"][:2] == openings[i // 2]
        assert len(g["moves"]) <= 50
        assert g["termination"] in ("two_passes", "move_cap")
        if g["termination"] == "two_passes":
            assert g["moves"][-2:] == [n * n, n * n]
        assert g["result"] == (1 if g["score"] > 0 else (-1 if g["score"] < 0 else 0))
    # Identical seeds on both sides make the second run identical.
    r2 = play_gtp_match(a, b, ref, n, 7.5, openings, move_cap=50, name_a="ra", name_b="rb", seed=5)
    assert [g["moves"] for g in r2["games_detail"]] == [g["moves"] for g in r["games_detail"]]
