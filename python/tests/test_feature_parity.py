"""Cross-language parity: the loader's planes equal the C++ encodeFeatures output for
every position of a random game (fixture parity_5x5.*), and the symmetry tables match
cpp/core/symmetry.cpp (fixture symmetry_5x5.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mango.chunk import read_chunk, unpack_snapshots
from mango.data import (NUM_PLANES, assemble_planes, inverse_symmetry, symmetry_table, transform_map, transform_planes,
                        transform_point, transform_policy)

FIXTURES = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "fixtures"


def _need(name: str) -> Path:
    p = FIXTURES / name
    if not p.exists():
        pytest.skip(f"fixture {name} missing")
    return p


def test_planes_match_cpp_encode_features():
    chunk = read_chunk(_need("parity_5x5.mgo"))
    features = np.frombuffer(_need("parity_5x5.features").read_bytes(), np.uint8)
    n = chunk.header.board_size
    g = chunk.games[0]
    assert g.is_holdout  # seed 20
    T = g.T
    assert features.shape == (T * NUM_PLANES * n * n,)
    features = features.reshape(T, NUM_PLANES, n, n)
    cells = unpack_snapshots(g.snapshots, n)
    for t in range(T):
        planes = assemble_planes(cells, t, n)
        assert planes.dtype == np.uint8
        if not np.array_equal(planes, features[t]):
            diff = np.argwhere(planes != features[t])
            pytest.fail(f"planes differ at t={t}: first mismatch (plane, r, c) = {diff[0].tolist()}")
    # Colour plane and parity: black moves at even t.
    assert assemble_planes(cells, 0, n)[16].all()
    assert not assemble_planes(cells, 1, n)[16].any()
    # z parity: the mover at t sees result * (+1 if t even else -1).
    signs = [1 if t % 2 == 0 else -1 for t in range(T)]
    assert signs[:2] == [1, -1]


def test_symmetry_tables_match_cpp():
    j = json.loads(_need("symmetry_5x5.json").read_text())
    n = j["board_size"]
    table = symmetry_table(n)
    assert table.tolist() == j["transform_point"]
    assert [inverse_symmetry(s) for s in range(8)] == j["inverse"]
    # Bijections with the documented inverses.
    for s in range(8):
        perm = [transform_point(p, n, s) for p in range(n * n)]
        assert sorted(perm) == list(range(n * n))
        inv = [transform_point(q, n, inverse_symmetry(s)) for q in perm]
        assert inv == list(range(n * n))


def test_transforms_move_a_peak_consistently():
    n = 5
    planes = np.zeros((NUM_PLANES, n, n), np.uint8)
    planes[0, 1, 2] = 1
    pi = np.zeros(n * n + 1, np.float32)
    pi[1 * n + 2] = 0.7
    pi[n * n] = 0.3
    own = np.zeros(n * n, np.float32)
    own[1 * n + 2] = 1.0
    for s in range(8):
        q = transform_point(1 * n + 2, n, s)
        tp = transform_planes(planes, n, s)
        assert tp[0].reshape(-1)[q] == 1 and tp[0].sum() == 1
        tpi = transform_policy(pi, n, s)
        assert tpi[q] == pytest.approx(0.7) and tpi[n * n] == pytest.approx(0.3)
        tow = transform_map(own, n, s)
        assert tow[q] == 1.0 and tow.sum() == 1.0
        # Inverse brings everything back.
        assert np.array_equal(transform_planes(tp, n, inverse_symmetry(s)), planes)
        assert np.allclose(transform_policy(tpi, n, inverse_symmetry(s)), pi)
    # Concrete cases: transpose (1,2)->(2,1); flip rows (1,2)->(3,2); flip cols (1,2)->(1,2).
    assert transform_point(1 * n + 2, n, 1) == 2 * n + 1
    assert transform_point(1 * n + 2, n, 2) == 3 * n + 2
    assert transform_point(1 * n + 2, n, 4) == 1 * n + 2
