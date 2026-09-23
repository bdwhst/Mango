"""Known-outcome 5x5 positions for the M3a' value-head check (docs/DESIGN.md section 11).

Each template is a settled position: every group has at least two real eyes, every
empty point is an eye of one colour, so no legal move changes the Tromp-Taylor area
score and both players' best move is to pass. The outcome is therefore the sign of
(black area - white area - komi), from the mover's perspective. Templates are expanded
by the 8 symmetries, both colours to move, and a colour swap (which changes the margin
because komi is one-sided), giving a few hundred positions.

The network input uses a static history (the same stones on all 8 history steps),
which is what a position reached by consecutive passes looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import NUM_PLANES, transform_map

# Rows top to bottom; X black, O white, . empty.
TEMPLATES_5x5 = [
    # 0: white wins by 2.5 (black 15, white 10).
    "X . X O .\n"
    "X X X O O\n"
    ". X X O .\n"
    "X X X O O\n"
    "X . X O .",
    # 1: black wins by 1.5 (black 17, white 8): the smallest two-eyed white group.
    "X O . O .\n"
    "X O O O O\n"
    "X X X X X\n"
    "X . X . X\n"
    "X X X X X",
    # 2: black owns everything (25 vs 0): 17.5.
    "X . X . X\n"
    "X X X X X\n"
    ". X . X .\n"
    "X X X X X\n"
    "X . X . X",
    # 3: white wins by 12.5: black the top two rows (10), white the rest (15).
    "X . X . X\n"
    "X X X X X\n"
    "O O O O O\n"
    "O . O . O\n"
    "O O O O O",
    # 4: white edge group (one chain) with two eyes; white wins by 2.5 (black 15, white 10).
    "O . O O .\n"
    "O O O O O\n"
    "X X X X X\n"
    "X . X . X\n"
    "X X X X X",
    # 5: as 1 with black's eyes elsewhere; black wins by 1.5.
    "X O . O .\n"
    "X O O O O\n"
    "X X X X X\n"
    "X . X X X\n"
    "X X X . X",
]


def parse_diagram(text: str) -> np.ndarray:
    rows = [r.split() for r in text.strip().split("\n")]
    n = len(rows)
    cells = np.zeros(n * n, np.int8)
    for r, row in enumerate(rows):
        assert len(row) == n, "diagram must be square"
        for c, ch in enumerate(row):
            cells[r * n + c] = {".": 0, "X": 1, "O": 2}[ch]
    return cells


def area_score(cells: np.ndarray, n: int, komi: float) -> float:
    """Tromp-Taylor area score black - white - komi (no dead-stone removal)."""
    black = int((cells == 1).sum())
    white = int((cells == 2).sum())
    seen = np.zeros(n * n, bool)
    for p0 in range(n * n):
        if cells[p0] != 0 or seen[p0]:
            continue
        stack, region, touch = [p0], [], set()
        seen[p0] = True
        while stack:
            p = stack.pop()
            region.append(p)
            r, c = divmod(p, n)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    q = rr * n + cc
                    if cells[q] == 0:
                        if not seen[q]:
                            seen[q] = True
                            stack.append(q)
                    else:
                        touch.add(int(cells[q]))
        if touch == {1}:
            black += len(region)
        elif touch == {2}:
            white += len(region)
    return black - white - komi


def is_settled(cells: np.ndarray, n: int) -> bool:
    """Every empty point is a single-point eye of one colour and every group has >= 2 of them."""
    eyes_of_group: dict[int, int] = {}
    # union-find over stones
    parent = list(range(n * n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in range(n * n):
        if cells[p] == 0:
            continue
        r, c = divmod(p, n)
        for dr, dc in ((1, 0), (0, 1)):
            rr, cc = r + dr, c + dc
            if rr < n and cc < n:
                q = rr * n + cc
                if cells[q] == cells[p]:
                    parent[find(p)] = find(q)
    for p in range(n * n):
        if cells[p] != 0:
            continue
        r, c = divmod(p, n)
        colours = set()
        groups = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                q = rr * n + cc
                if cells[q] == 0:
                    return False  # a two-point (or larger) region is not a single eye
                colours.add(int(cells[q]))
                groups.add(find(q))
        if len(colours) != 1 or len(groups) != 1:
            return False
        g = groups.pop()
        eyes_of_group[g] = eyes_of_group.get(g, 0) + 1
    for p in range(n * n):
        if cells[p] != 0 and eyes_of_group.get(find(p), 0) < 2:
            return False
    return True


@dataclass
class KnownPosition:
    cells: np.ndarray  # [n*n] 0/1/2
    black_to_move: bool
    score: float  # black - white - komi
    template: int
    symmetry: int
    swapped: bool

    @property
    def mover_sign(self) -> float:
        return float(np.sign(self.score if self.black_to_move else -self.score))


def known_positions(n: int = 5, komi: float = 7.5, min_margin: float = 1.0) -> list[KnownPosition]:
    out: list[KnownPosition] = []
    for ti, text in enumerate(TEMPLATES_5x5):
        base = parse_diagram(text)
        assert len(base) == n * n
        if not is_settled(base, n):
            raise ValueError(f"template {ti} is not settled")
        for swapped in (False, True):
            cells0 = base.copy()
            if swapped:
                cells0 = np.where(base == 1, 2, np.where(base == 2, 1, 0)).astype(np.int8)
            for sym in range(8):
                cells = transform_map(cells0, n, sym)
                s = area_score(cells, n, komi)
                if abs(s) < min_margin:
                    continue
                for black_to_move in (True, False):
                    out.append(KnownPosition(cells, black_to_move, s, ti, sym, swapped))
    return out


def planes_for(pos: KnownPosition, n: int) -> np.ndarray:
    """Static-history feature planes [17, n, n] float32 for a known position."""
    mover, opp = (1, 2) if pos.black_to_move else (2, 1)
    planes = np.zeros((NUM_PLANES, n * n), np.float32)
    for k in range(8):
        planes[k] = pos.cells == mover
        planes[8 + k] = pos.cells == opp
    if pos.black_to_move:
        planes[16] = 1.0
    return planes.reshape(NUM_PLANES, n, n)


def value_sign_accuracy(model, positions: list[KnownPosition], n: int, device) -> dict:
    """Fraction of positions where sign(v) matches the settled outcome (mover's perspective)."""
    import torch

    model.eval()
    x = torch.from_numpy(np.stack([planes_for(p, n) for p in positions])).to(device)
    with torch.no_grad():
        _, v = model(x)
    v = v.float().cpu().numpy()
    signs = np.array([p.mover_sign for p in positions])
    correct = (np.sign(v) == signs)
    per_template: dict[int, list[bool]] = {}
    for p, ok in zip(positions, correct):
        per_template.setdefault(p.template, []).append(bool(ok))
    margins = np.array([abs(p.score) for p in positions])
    by_margin = {}
    for m in sorted(set(margins.tolist())):
        sel = margins == m
        by_margin[f"{m:.1f}"] = {"positions": int(sel.sum()), "accuracy": float(correct[sel].mean())}
    return {
        "positions": len(positions),
        "accuracy": float(correct.mean()),
        "mean_abs_value": float(np.abs(v).mean()),
        "per_template": {str(k): float(np.mean(v_)) for k, v_ in per_template.items()},
        "per_margin": by_margin,
    }
