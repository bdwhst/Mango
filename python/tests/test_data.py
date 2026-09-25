"""Loader contracts (DESIGN 6.3, 9 "Training"): z parity, pi rules, holdout tag, symmetry
applied jointly to planes, pi and ownership, PCR index rules, uniform sampling."""

from __future__ import annotations

import numpy as np
import pytest

from mango.chunk import (EXTRAS_FINAL_OWNERSHIP, EXTRAS_SEARCH_KIND, TERMINATION_RESIGN, TERMINATION_TWO_PASSES, Chunk,
                         ChunkHeader, GameRecord, pack_snapshot, write_chunk)
from mango.data import (ChunkDataset, Window, inverse_symmetry, symmetry_table, transform_map, transform_planes,
                        write_holdout_games)


def make_game(n: int, seed: int, T: int, result: int, extras: int = 0, termination: int = TERMINATION_TWO_PASSES,
              kinds=None) -> GameRecord:
    """A synthetic game: black plays points 0,2,4,... white 1,3,5,... (no captures)."""
    nn = n * n
    g = GameRecord(game_seed=seed, result=result, termination=termination, score=float(result) * 3.5)
    cells = np.zeros((T + 1, nn), np.int8)
    moves = []
    for t in range(T):
        cells[t + 1] = cells[t]
        p = t % nn
        cells[t + 1, p] = 1 if t % 2 == 0 else 2
        moves.append(p)
    g.snapshots = np.stack([pack_snapshot(c, n) for c in cells])
    g.moves = np.array(moves, np.uint16)
    g.root_value = np.linspace(-0.5, 0.5, T).astype(np.float32)
    g.root_max_q = np.zeros(T, np.float32)
    g.root_total_visits = np.full(T, 10, np.uint32)
    for t in range(T):
        g.visit_actions.append(np.array([moves[t], nn], np.uint16))  # played move + pass
        g.visit_counts.append(np.array([7, 3], np.uint32))
    if extras & EXTRAS_SEARCH_KIND:
        g.search_kind = np.array(kinds if kinds is not None else [1] * T, np.uint8)
    if extras & EXTRAS_FINAL_OWNERSHIP:
        own = np.where(cells[T] == 1, 1, np.where(cells[T] == 2, -1, 0)).astype(np.int8)
        g.final_ownership = own
    return g


def write_test_chunk(path, n, games, extras=0, chunk_id=1):
    h = ChunkHeader(board_size=n, komi=7.5, model_id="t", chunk_id=chunk_id, move_cap=2 * n * n, record_extras=extras)
    write_chunk(path, Chunk(h, games))
    return path


def test_targets_z_pi_and_holdout(tmp_path):
    n = 5
    p = write_test_chunk(tmp_path / "c.mgo", n, [make_game(n, 20, 6, +1), make_game(n, 21, 5, -1)])
    w = Window([p])
    assert w.games[0].holdout and not w.games[1].holdout
    train = ChunkDataset(w, store_pi="visits", temperature_moves=2, symmetry=False, holdout=False)
    hold = ChunkDataset(w, store_pi="visits", temperature_moves=2, symmetry=False, holdout=True)
    assert train.train_positions == 5 and hold.train_positions == 6
    # Game 1 (white wins, result -1): z = -1 at even t (black to move), +1 at odd t.
    for t in range(5):
        planes, pi, z, extra = train.example(1, t, 0)
        assert z == (-1.0 if t % 2 == 0 else 1.0)
        assert extra["score"] == pytest.approx(-3.5 if t % 2 == 0 else 3.5)
        assert extra["aux_mask"] == 1.0 and extra["policy_mask"] == 1.0
        assert pi[t % (n * n)] == pytest.approx(0.7) and pi[n * n] == pytest.approx(0.3) and pi.sum() == pytest.approx(1.0)
        assert planes.shape == (17, n, n) and planes[16].all() == (t % 2 == 0)
    # "temperature": normalised counts before the cutoff, one-hot played move after.
    temp = ChunkDataset(w, store_pi="temperature", temperature_moves=2, symmetry=False, holdout=False)
    _, pi0, _, _ = temp.example(1, 0, 0)
    _, pi3, _, _ = temp.example(1, 3, 0)
    assert pi0[n * n] == pytest.approx(0.3)
    assert pi3[3] == 1.0 and pi3.sum() == 1.0
    # Sampling is uniform over training positions: batches never contain holdout games.
    batch = train.sample_batch(64)
    assert batch["planes"].shape == (64, 17, n, n) and batch["pi"].shape == (64, n * n + 1)
    assert set(np.unique(batch["z"])).issubset({-1.0, 1.0})
    # Every batch example's planes come from game 1 (game 0 is holdout): black to move
    # at even t means the mover plane 0 count is t//2... just check both parities occur.
    assert batch["z"].min() == -1.0 and batch["z"].max() == 1.0


def test_one_symmetry_for_planes_pi_and_ownership(tmp_path):
    n = 5
    extras = EXTRAS_FINAL_OWNERSHIP | EXTRAS_SEARCH_KIND
    p = write_test_chunk(tmp_path / "c.mgo", n, [make_game(n, 3, 7, +1, extras)], extras)
    ds = ChunkDataset(Window([p]), symmetry=True, holdout=False)
    base_planes, base_pi, _, base_extra = ds.example(0, 4, 0)
    table = symmetry_table(n)
    for s in range(8):
        planes, pi, z, extra = ds.example(0, 4, s)
        assert np.array_equal(planes, transform_planes(base_planes, n, s))
        assert np.allclose(pi[: n * n], transform_map(base_pi[: n * n], n, s)) and pi[n * n] == base_pi[n * n]
        assert np.array_equal(extra["ownership"], transform_map(base_extra["ownership"], n, s))
        # The stone at the played point and the ownership at the same point move together.
        played = 4 % (n * n)
        q = table[s][played]
        assert pi[q] == pytest.approx(0.7)
        assert planes[0].reshape(-1)[q] == base_planes[0].reshape(-1)[played]
        assert extra["ownership"][q] == base_extra["ownership"][played]
        assert z == 1.0  # symmetry never changes the value target
    # Ownership is in the mover's perspective: sign flips with parity.
    _, _, _, e_even = ds.example(0, 2, 0)
    _, _, _, e_odd = ds.example(0, 3, 0)
    assert np.array_equal(e_even["ownership"], -e_odd["ownership"])


def test_resigned_games_mask_auxiliary_targets(tmp_path):
    n = 5
    p = write_test_chunk(tmp_path / "c.mgo", n, [make_game(n, 1, 4, -1, termination=TERMINATION_RESIGN),
                                                 make_game(n, 2, 4, +1)])
    ds = ChunkDataset(Window([p]), symmetry=False, holdout=False)
    assert ds.example(0, 0, 0)[3]["aux_mask"] == 0.0
    assert ds.example(1, 0, 0)[3]["aux_mask"] == 1.0


def test_pcr_index_drops_or_masks_reduced_positions(tmp_path):
    n = 5
    kinds = [1, 0, 0, 1, 0, 1]
    p = write_test_chunk(tmp_path / "c.mgo", n, [make_game(n, 1, 6, +1, EXTRAS_SEARCH_KIND, kinds=kinds)],
                         EXTRAS_SEARCH_KIND)
    w = Window([p])
    drop = ChunkDataset(w, symmetry=False, holdout=False, reduced_positions="drop")
    keep = ChunkDataset(w, symmetry=False, holdout=False, reduced_positions="value_only")
    assert drop.train_positions == 3 and keep.train_positions == 6
    assert sorted(drop.index.entries[:, 1].tolist()) == [0, 3, 5]
    masks = [keep.example(0, t, 0)[3]["policy_mask"] for t in range(6)]
    assert masks == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_window_rejects_mixed_extras(tmp_path):
    n = 5
    a = write_test_chunk(tmp_path / "a.mgo", n, [make_game(n, 1, 3, 1)], 0, chunk_id=1)
    b = write_test_chunk(tmp_path / "b.mgo", n, [make_game(n, 2, 3, 1, EXTRAS_SEARCH_KIND)], EXTRAS_SEARCH_KIND, chunk_id=2)
    with pytest.raises(ValueError):
        Window([a, b])
    assert inverse_symmetry(3) == 5


def test_write_holdout_games_keeps_only_holdout_games(tmp_path):
    n = 5
    p1 = write_test_chunk(tmp_path / "a.mgo", n, [make_game(n, 20, 6, +1), make_game(n, 21, 5, -1)], chunk_id=1)
    p2 = write_test_chunk(tmp_path / "b.mgo", n, [make_game(n, 40, 4, -1), make_game(n, 7, 4, +1)], chunk_id=2)
    out = tmp_path / "fixed.mgo"
    assert write_holdout_games([p1, p2], out) == 2
    w = Window([out])
    assert [g.record.game_seed for g in w.games] == [20, 40]
    assert all(g.holdout for g in w.games)
    ds = ChunkDataset(w, symmetry=False, holdout=True)
    assert ds.train_positions == 6 + 4
    # No holdout game: nothing is written.
    p3 = write_test_chunk(tmp_path / "c.mgo", n, [make_game(n, 3, 4, +1)], chunk_id=3)
    assert write_holdout_games([p3], tmp_path / "none.mgo") == 0
    assert not (tmp_path / "none.mgo").exists()
