"""Training data pipeline (docs/DESIGN.md section 6.3): planes, targets and symmetry
augmentation assembled from MGO2 chunks.

Contracts with the C++ side (pinned by cross-language fixtures):
  * planes: 17 x n x n uint8 - mover stones at t..t-7, opponent stones at t..t-7,
    colour plane (all ones iff black to move); black moves at even t.
  * z_t = result * (+1 if t even else -1)  (mover's perspective)
  * pi_t from the stored root visit counts ("visits": normalised counts;
    "temperature": normalised counts before temperature_moves, one-hot played move after)
  * symmetry: bit 0 transpose, bit 1 flip rows, bit 2 flip columns, applied in that order;
    the same transform is applied to planes, to the board part of pi and to ownership.
  * holdout games: game_seed % 20 == 0 (5%).

M3a' keeps whole chunks in memory (the design's memmap index is an M4 optimisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .chunk import EXTRAS_SEARCH_KIND, TERMINATION_RESIGN, Chunk, GameRecord, read_chunk, unpack_snapshots

NUM_PLANES = 17
HISTORY = 8
NUM_SYMMETRIES = 8
HOLDOUT_MODULUS = 20


# --- symmetries ---------------------------------------------------------------------------

def transform_point(p: int, n: int, sym: int) -> int:
    r, c = divmod(p, n)
    if sym & 1:
        r, c = c, r
    if sym & 2:
        r = n - 1 - r
    if sym & 4:
        c = n - 1 - c
    return r * n + c


_TABLES: dict[int, np.ndarray] = {}


def symmetry_table(n: int) -> np.ndarray:
    """[8, n*n]: table[sym][p] = transformed index of point p."""
    if n not in _TABLES:
        _TABLES[n] = np.array([[transform_point(p, n, s) for p in range(n * n)] for s in range(NUM_SYMMETRIES)], np.int64)
    return _TABLES[n]


def inverse_symmetry(sym: int) -> int:
    return {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 3, 6: 6, 7: 7}[sym]


def transform_map(values: np.ndarray, n: int, sym: int) -> np.ndarray:
    """Board-shaped vector [..., n*n]: out[T(p)] = in[p]."""
    out = np.empty_like(values)
    out[..., symmetry_table(n)[sym]] = values
    return out


def transform_planes(planes: np.ndarray, n: int, sym: int) -> np.ndarray:
    """[C, n, n] -> [C, n, n] with the same point map."""
    flat = planes.reshape(planes.shape[0], n * n)
    return transform_map(flat, n, sym).reshape(planes.shape)


def transform_policy(pi: np.ndarray, n: int, sym: int) -> np.ndarray:
    """[n*n+1] policy: board part transformed, pass unchanged."""
    out = np.empty_like(pi)
    out[: n * n] = transform_map(pi[: n * n], n, sym)
    out[n * n] = pi[n * n]
    return out


# --- assembly -----------------------------------------------------------------------------

def assemble_planes(cells: np.ndarray, t: int, n: int) -> np.ndarray:
    """Feature planes for position t from unpacked snapshots [T+1, n*n] (0/1/2)."""
    black_to_move = t % 2 == 0
    mover, opp = (1, 2) if black_to_move else (2, 1)
    planes = np.zeros((NUM_PLANES, n * n), np.uint8)
    for k in range(HISTORY):
        idx = t - k
        if idx < 0:
            break
        snap = cells[idx]
        planes[k] = snap == mover
        planes[HISTORY + k] = snap == opp
    if black_to_move:
        planes[2 * HISTORY] = 1
    return planes.reshape(NUM_PLANES, n, n)


def policy_from_visits(actions: np.ndarray, counts: np.ndarray, n: int, store_pi: str, t: int, temperature_moves: int,
                       played: int) -> np.ndarray:
    pi = np.zeros(n * n + 1, np.float32)
    if store_pi == "visits" or t < temperature_moves:
        pi[actions.astype(np.int64)] = counts.astype(np.float32)
        s = pi.sum()
        if s > 0:
            pi /= s
    elif store_pi == "temperature":
        pi[int(played)] = 1.0  # the stored played move, never a recomputed argmax
    else:
        raise ValueError(f"unknown store_pi {store_pi!r}")
    return pi


@dataclass
class GameData:
    """One game unpacked for sampling."""

    record: GameRecord
    cells: np.ndarray  # [T+1, n*n] int8
    chunk_id: int
    holdout: bool


class Window:
    """Games of the current window, loaded from chunk files (newest last)."""

    def __init__(self, chunk_paths: list[str | Path]):
        self.chunks: list[Chunk] = [read_chunk(p) for p in chunk_paths]
        if not self.chunks:
            raise ValueError("empty window")
        n = {c.header.board_size for c in self.chunks}
        if len(n) != 1:
            raise ValueError(f"mixed board sizes in the window: {sorted(n)}")
        extras = {c.header.record_extras for c in self.chunks}
        if len(extras) != 1:
            raise ValueError(f"mixed record_extras in the window: {sorted(extras)}")
        self.board_size = n.pop()
        self.record_extras = extras.pop()
        self.games: list[GameData] = []
        for c in self.chunks:
            for g in c.games:
                self.games.append(GameData(g, unpack_snapshots(g.snapshots, self.board_size), c.header.chunk_id, g.is_holdout))

    @property
    def num_games(self) -> int:
        return len(self.games)


class PositionIndex:
    """Index of training positions (game, t) under the sampling rules of DESIGN 6.3."""

    def __init__(self, window: Window, holdout: bool, reduced_positions: str = "drop"):
        self.window = window
        self.n = window.board_size
        entries: list[tuple[int, int]] = []
        for gi, gd in enumerate(window.games):
            if gd.holdout != holdout:
                continue
            g = gd.record
            kinds = g.search_kind if (window.record_extras & EXTRAS_SEARCH_KIND) else None
            for t in range(g.T):
                if kinds is not None and kinds[t] == 0 and reduced_positions == "drop":
                    continue
                entries.append((gi, t))
        self.entries = np.array(entries, np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return int(self.entries.shape[0])


class ChunkDataset:
    """Uniform sampling over training positions with symmetry augmentation."""

    def __init__(self, window: Window, store_pi: str = "visits", temperature_moves: int = 0, symmetry: bool = True,
                 seed: int = 0, holdout: bool = False, reduced_positions: str = "drop"):
        self.window = window
        self.n = window.board_size
        self.store_pi = store_pi
        self.temperature_moves = temperature_moves
        self.symmetry = symmetry
        self.index = PositionIndex(window, holdout=holdout, reduced_positions=reduced_positions)
        self.rng = np.random.default_rng(seed)

    @property
    def train_positions(self) -> int:
        return len(self.index)

    def example(self, gi: int, t: int, sym: int) -> tuple[np.ndarray, np.ndarray, float, dict]:
        gd = self.window.games[gi]
        g = gd.record
        n = self.n
        planes = assemble_planes(gd.cells, t, n)
        pi = policy_from_visits(g.visit_actions[t], g.visit_counts[t], n, self.store_pi, t, self.temperature_moves,
                                int(g.moves[t]))
        sign = 1.0 if t % 2 == 0 else -1.0
        z = float(g.result) * sign
        extra = {
            "score": float(g.score) * sign,
            "aux_mask": 0.0 if g.termination == TERMINATION_RESIGN else 1.0,
            "policy_mask": 1.0 if (g.search_kind is None or g.search_kind[t] != 0) else 0.0,
        }
        if g.final_ownership is not None:
            own = g.final_ownership.astype(np.float32) * sign
            extra["ownership"] = transform_map(own, n, sym) if sym else own
        if sym:
            planes = transform_planes(planes, n, sym)
            pi = transform_policy(pi, n, sym)
        return planes, pi, z, extra

    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        m = len(self.index)
        if m == 0:
            raise ValueError("no training positions")
        picks = self.rng.integers(0, m, size=batch_size)
        syms = self.rng.integers(0, NUM_SYMMETRIES, size=batch_size) if self.symmetry else np.zeros(batch_size, np.int64)
        return self._collate([(int(self.index.entries[i, 0]), int(self.index.entries[i, 1]), int(s)) for i, s in zip(picks, syms)])

    def iterate_all(self, batch_size: int) -> Iterator[dict[str, np.ndarray]]:
        """Every indexed position once, no symmetry (holdout monitor)."""
        for start in range(0, len(self.index), batch_size):
            rows = self.index.entries[start : start + batch_size]
            yield self._collate([(int(gi), int(t), 0) for gi, t in rows])

    def _collate(self, items: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
        n = self.n
        B = len(items)
        planes = np.empty((B, NUM_PLANES, n, n), np.float32)
        pi = np.empty((B, n * n + 1), np.float32)
        z = np.empty(B, np.float32)
        score = np.empty(B, np.float32)
        aux_mask = np.empty(B, np.float32)
        policy_mask = np.empty(B, np.float32)
        ownership = None
        for i, (gi, t, s) in enumerate(items):
            p, q, zz, extra = self.example(gi, t, s)
            planes[i] = p
            pi[i] = q
            z[i] = zz
            score[i] = extra["score"]
            aux_mask[i] = extra["aux_mask"]
            policy_mask[i] = extra["policy_mask"]
            if "ownership" in extra:
                if ownership is None:
                    ownership = np.empty((B, n * n), np.float32)
                ownership[i] = extra["ownership"]
        out = {"planes": planes, "pi": pi, "z": z, "score": score, "aux_mask": aux_mask, "policy_mask": policy_mask}
        if ownership is not None:
            out["ownership"] = ownership
        return out
