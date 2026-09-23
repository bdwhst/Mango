"""MGO2 reader/writer against the C++ fixtures (cpp/tests/test_chunk.cpp writes them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mango.chunk import (EXTRAS_FINAL_OWNERSHIP, EXTRAS_PRUNED_VISITS, EXTRAS_SEARCH_KIND, HEADER_SIZE, Chunk, ChunkHeader,
                         GameRecord, pack_snapshot, parse_chunk, read_chunk, serialize_chunk, serialize_header,
                         unpack_snapshots, write_chunk)

FIXTURES = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "fixtures"


def _load(name: str) -> bytes:
    p = FIXTURES / name
    if not p.exists():
        pytest.skip(f"fixture {name} missing (run mango_tests with MANGO_WRITE_FIXTURES=1)")
    return p.read_bytes()


@pytest.mark.parametrize("name,extras", [("mgo2_fixture.bin", 0),
                                         ("mgo2_fixture_extras.bin", EXTRAS_FINAL_OWNERSHIP | EXTRAS_SEARCH_KIND | EXTRAS_PRUNED_VISITS)])
def test_fixture_parses_and_reserialises_byte_exact(name, extras):
    raw = _load(name)
    chunk = parse_chunk(raw)
    h = chunk.header
    assert h.board_size == 5 and h.komi == 7.5 and h.planes == 17 and h.feature_schema == 1 and h.rules_id == 1
    assert h.num_games == 2 and h.num_positions == 9 and h.simulations_per_move == 4
    assert h.model_id == "0001-deadbeef" and h.config_fingerprint == "0123456789abcdef"
    assert h.chunk_id == 42 and h.run_seed == 7 and h.move_cap == 50 and h.temperature_moves == 4
    assert h.record_extras == extras
    g0, g1 = chunk.games
    assert g0.T == 4 and g1.T == 5
    assert g1.no_resign_game and not g0.no_resign_game
    assert g1.moves.tolist() == [1, 0, 5, 25, 25]  # (0,1), (0,0), (1,0), pass, pass
    cells = unpack_snapshots(g1.snapshots, 5)
    assert cells.shape == (6, 25)
    assert cells[2][0] == 2  # white stone at (0,0) before move 2
    assert cells[3][0] == 0  # captured by move 2
    assert g1.result == 1 and abs(g1.score - 17.5) < 1e-6 and g1.termination == 0
    assert g0.root_value.tolist() == pytest.approx([0.25, 0.15, 0.05, -0.05], abs=1e-6)
    assert g0.visit_actions[0].tolist() == [12, 13] and g0.visit_counts[0].tolist() == [3, 1]
    if extras:
        assert g1.search_kind.tolist() == [1, 0, 1, 1, 1]
        assert g1.final_ownership.shape == (25,) and g1.final_ownership[0] == 1
        assert g1.root_total_visits[0] == 5  # pruned: 3 + 1 <= 5
        assert h.reduced_simulations == 2 and h.full_search_prob == pytest.approx(0.25)
    else:
        assert g1.search_kind is None and g1.final_ownership is None
        assert g1.root_total_visits[0] == 4
    # The Python writer reproduces the C++ bytes exactly.
    assert serialize_chunk(chunk) == raw
    assert serialize_header(h)[:4] == b"MGO2" and len(serialize_header(h)) == HEADER_SIZE


def test_snapshot_packing_round_trip():
    n = 7
    rng = np.random.default_rng(3)
    cells = rng.integers(0, 3, size=n * n).astype(np.int8)
    packed = pack_snapshot(cells, n)
    assert packed.shape == ((n * n + 3) // 4,)
    back = unpack_snapshots(packed[None, :], n)[0]
    assert back.tolist() == cells.tolist()
    # Point 4k lives in bits 0-1 of byte k.
    one = np.zeros(n * n, np.int8)
    one[4] = 2
    assert pack_snapshot(one, n)[1] == 2


def test_malformed_chunks_are_rejected():
    raw = _load("mgo2_fixture.bin")
    with pytest.raises(ValueError):
        parse_chunk(raw[:-3])
    with pytest.raises(ValueError):
        parse_chunk(raw + b"\0")
    with pytest.raises(ValueError):
        parse_chunk(b"XGO2" + raw[4:])
    chunk = parse_chunk(raw)
    chunk.games[0].root_total_visits[0] = 9  # sum 4 != 9 without the pruned bit
    with pytest.raises(ValueError):
        serialize_chunk(chunk)


def test_write_chunk_is_atomic_and_round_trips(tmp_path):
    raw = _load("mgo2_fixture_extras.bin")
    chunk = parse_chunk(raw)
    out = write_chunk(tmp_path / "chunk_000001.mgo", chunk)
    assert out.exists() and not (tmp_path / "chunk_000001.mgo.tmp").exists()
    back = read_chunk(out)
    assert serialize_chunk(back) == raw


def test_python_built_record_matches_layout():
    """A record built in Python serialises with the documented per-game layout."""
    n = 5
    h = ChunkHeader(board_size=n, komi=7.5, model_id="py", move_cap=50)
    g = GameRecord(game_seed=20)  # holdout tag
    g.snapshots = np.zeros((2, 7), np.uint8)
    g.snapshots[1] = pack_snapshot(np.array([1] + [0] * 24, np.int8), n)
    g.moves = np.array([0], np.uint16)
    g.root_value = np.array([0.5], np.float32)
    g.root_max_q = np.array([0.25], np.float32)
    g.root_total_visits = np.array([3], np.uint32)
    g.visit_actions = [np.array([0, 25], np.uint16)]
    g.visit_counts = [np.array([2, 1], np.uint32)]
    g.result = 1
    g.score = 17.5
    raw = serialize_chunk(Chunk(h, [g]))
    assert len(raw) == HEADER_SIZE + 20 + 2 * 7 + 2 + 8 + (6 + 2 * 6)
    body = raw[HEADER_SIZE:]
    assert body[2:4] == b"\x01\x00"  # T = 1
    assert body[4] == 1  # result
    assert body[12:20] == (20).to_bytes(8, "little")  # game_seed
    assert g.is_holdout
    back = parse_chunk(raw)
    assert back.games[0].visit_actions[0].tolist() == [0, 25]
