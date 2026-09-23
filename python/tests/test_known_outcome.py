"""Known-outcome 5x5 positions: settledness, scores, expansion, and an oracle model
that reads the score off the planes gets 100% sign accuracy."""

from __future__ import annotations

import numpy as np
import torch

from mango.known_outcome import (TEMPLATES_5x5, area_score, is_settled, known_positions, parse_diagram, planes_for,
                                 value_sign_accuracy)


def test_templates_are_settled_with_expected_scores():
    scores = [area_score(parse_diagram(t), 5, 7.5) for t in TEMPLATES_5x5]
    assert scores == [-2.5, 1.5, 17.5, -12.5, -2.5, 1.5]
    assert all(abs(s) >= 1.0 for s in scores)
    for t in TEMPLATES_5x5:
        assert is_settled(parse_diagram(t), 5)
    # Unsettled examples are rejected: a two-point eye, and a group with one eye.
    assert not is_settled(parse_diagram("X X X X X\nX . . X X\nX X X X X\nX X X X X\nX . X X X"), 5)
    assert not is_settled(parse_diagram("X . X X X\nX X X X X\nX X X X X\nX X X X X\nX X X X X"), 5)


def test_expansion_and_mover_sign():
    ps = known_positions(5, 7.5)
    assert len(ps) == len(TEMPLATES_5x5) * 2 * 8 * 2
    signs = {p.mover_sign for p in ps}
    assert signs == {-1.0, 1.0}
    # Colour swap flips the sign of black - white, komi is unchanged.
    a = next(p for p in ps if p.template == 0 and p.symmetry == 0 and not p.swapped and p.black_to_move)
    b = next(p for p in ps if p.template == 0 and p.symmetry == 0 and p.swapped and p.black_to_move)
    assert a.score == -2.5
    assert b.score == -12.5  # 10 - 15 - 7.5
    # Symmetric copies have the same score.
    for p in ps:
        assert p.score == area_score(p.cells, 5, 7.5)
    planes = planes_for(a, 5)
    assert planes.shape == (17, 5, 5) and planes[16].all()
    assert np.array_equal(planes[0], planes[7]) and np.array_equal(planes[8], planes[15])


class Oracle(torch.nn.Module):
    """Value = tanh(area score from the mover's perspective) computed from the planes."""

    def forward(self, x):
        B = x.shape[0]
        values = []
        for i in range(B):
            mover = x[i, 0].reshape(-1).numpy()
            opp = x[i, 8].reshape(-1).numpy()
            black = x[i, 16, 0, 0].item() > 0.5
            cells = np.where(mover > 0, 1 if black else 2, np.where(opp > 0, 2 if black else 1, 0)).astype(np.int8)
            s = area_score(cells, 5, 7.5)
            values.append(np.tanh(s if black else -s))
        return torch.zeros(B, 26), torch.tensor(values, dtype=torch.float32)


def test_oracle_scores_100_percent():
    ps = known_positions(5, 7.5)
    r = value_sign_accuracy(Oracle(), ps, 5, torch.device("cpu"))
    assert r["accuracy"] == 1.0 and r["positions"] == len(ps)
    assert set(r["per_margin"]) == {"1.5", "2.5", "12.5", "16.5", "17.5", "32.5"}
    assert sum(m["positions"] for m in r["per_margin"].values()) == len(ps)
