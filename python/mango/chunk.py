"""MGO2 game-chunk reader/writer (docs/DESIGN.md section 5.6).

Byte-for-byte compatible with cpp/selfplay/chunk.cpp: every field is packed one by
one, little-endian, at the documented offsets. The committed fixtures
cpp/tests/fixtures/mgo2_fixture*.bin pin the layout on both sides.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"MGO2"
VERSION = 2
HEADER_SIZE = 128

EXTRAS_FINAL_OWNERSHIP = 1
EXTRAS_SEARCH_KIND = 2
EXTRAS_PRUNED_VISITS = 4

TERMINATION_TWO_PASSES = 0
TERMINATION_RESIGN = 1
TERMINATION_MOVE_CAP = 2


@dataclass
class ChunkHeader:
    board_size: int
    komi: float = 7.5
    planes: int = 17
    feature_schema: int = 1
    rules_id: int = 1
    num_games: int = 0
    num_positions: int = 0
    simulations_per_move: int = 0
    model_id: str = ""
    config_fingerprint: str = ""
    chunk_id: int = 0
    run_seed: int = 0
    move_cap: int = 0
    temperature_moves: int = 0
    record_extras: int = 0
    reduced_simulations: int = 0
    full_search_prob: float = 1.0


@dataclass
class GameRecord:
    game_index: int = 0
    result: int = 0
    termination: int = TERMINATION_TWO_PASSES
    no_resign_game: bool = False
    score: float = 0.0
    game_seed: int = 0
    snapshots: np.ndarray = field(default_factory=lambda: np.zeros((1, 0), np.uint8))  # [T+1, P] packed
    moves: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))  # [T], n*n = pass
    search_kind: np.ndarray | None = None  # [T] uint8 or None
    root_value: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    root_max_q: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    root_total_visits: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))  # [T]
    visit_actions: list[np.ndarray] = field(default_factory=list)  # per move: uint16 actions
    visit_counts: list[np.ndarray] = field(default_factory=list)  # per move: uint32 counts
    final_ownership: np.ndarray | None = None  # [n*n] int8 or None

    @property
    def T(self) -> int:
        return int(len(self.moves))

    @property
    def is_holdout(self) -> bool:
        """5% holdout tag, a pure function of the stored seed (DESIGN 5.6)."""
        return self.game_seed % 20 == 0


@dataclass
class Chunk:
    header: ChunkHeader
    games: list[GameRecord]


def packed_snapshot_bytes(n: int) -> int:
    return (n * n + 3) // 4


def unpack_snapshots(packed: np.ndarray, n: int) -> np.ndarray:
    """[T+1, P] packed -> [T+1, n*n] int8 with 0 empty, 1 black, 2 white."""
    nn = n * n
    b = packed.astype(np.uint8)
    out = np.empty((b.shape[0], b.shape[1] * 4), np.int8)
    for k in range(4):
        out[:, k::4] = (b >> (2 * k)) & 3
    return out[:, :nn]


def pack_snapshot(cells: np.ndarray, n: int) -> np.ndarray:
    """[n*n] values 0/1/2 -> packed bytes (point 4k in bits 0-1 of byte k)."""
    P = packed_snapshot_bytes(n)
    padded = np.zeros(P * 4, np.uint8)
    padded[: n * n] = cells.astype(np.uint8)
    out = np.zeros(P, np.uint8)
    for k in range(4):
        out |= padded[k::4] << (2 * k)
    return out


def _cstr(s: str, width: int) -> bytes:
    b = s.encode("ascii")
    if len(b) > width:
        raise ValueError(f"string field too long: {s!r}")
    return b + b"\0" * (width - len(b))


def serialize_header(h: ChunkHeader) -> bytes:
    out = struct.pack(
        "<4sBBBBB3xfIIII32s16sQQHHBxHf20x",
        MAGIC,
        VERSION,
        h.board_size,
        h.planes,
        h.feature_schema,
        h.rules_id,
        h.komi,
        HEADER_SIZE,
        h.num_games,
        h.num_positions,
        h.simulations_per_move,
        _cstr(h.model_id, 32),
        _cstr(h.config_fingerprint, 16),
        h.chunk_id,
        h.run_seed,
        h.move_cap,
        h.temperature_moves,
        h.record_extras,
        h.reduced_simulations,
        h.full_search_prob,
    )
    assert len(out) == HEADER_SIZE
    return out


def validate_game(g: GameRecord, h: ChunkHeader) -> str:
    n = h.board_size
    nn = n * n
    T = g.T
    if g.snapshots.shape != (T + 1, packed_snapshot_bytes(n)):
        return "snapshots shape"
    if len(g.root_value) != T or len(g.root_max_q) != T or len(g.root_total_visits) != T:
        return "root arrays != T"
    if len(g.visit_actions) != T or len(g.visit_counts) != T:
        return "visits != T"
    if T and int(g.moves.max()) > nn:
        return "move out of range"
    pruned = bool(h.record_extras & EXTRAS_PRUNED_VISITS)
    for t in range(T):
        a, c = g.visit_actions[t], g.visit_counts[t]
        if len(a) == 0 or len(a) != len(c):
            return f"move {t}: bad visit entries"
        if int(a.max()) > nn or int(c.min()) == 0:
            return f"move {t}: bad visit values"
        s = int(c.sum())
        if not pruned and s != int(g.root_total_visits[t]):
            return f"move {t}: sum != root_total_visits"
        if pruned and s > int(g.root_total_visits[t]):
            return f"move {t}: pruned sum > root_total_visits"
    want_kind = bool(h.record_extras & EXTRAS_SEARCH_KIND)
    if want_kind != (g.search_kind is not None) or (want_kind and len(g.search_kind) != T):
        return "search_kind presence"
    want_own = bool(h.record_extras & EXTRAS_FINAL_OWNERSHIP)
    if want_own != (g.final_ownership is not None) or (want_own and len(g.final_ownership) != nn):
        return "final_ownership presence"
    if g.result not in (-1, 0, 1):
        return "result"
    return ""


def serialize_game(g: GameRecord, h: ChunkHeader) -> bytes:
    err = validate_game(g, h)
    if err:
        raise ValueError(f"invalid game record: {err}")
    T = g.T
    parts = [
        struct.pack("<HHbBBxfQ", g.game_index, T, g.result, g.termination, 1 if g.no_resign_game else 0, g.score, g.game_seed),
        g.snapshots.astype(np.uint8).tobytes(),
        g.moves.astype("<u2").tobytes(),
    ]
    if h.record_extras & EXTRAS_SEARCH_KIND:
        parts.append(g.search_kind.astype(np.uint8).tobytes())
    rv = np.empty((T, 2), "<f4")
    rv[:, 0] = g.root_value
    rv[:, 1] = g.root_max_q
    parts.append(rv.tobytes())
    for t in range(T):
        a, c = g.visit_actions[t], g.visit_counts[t]
        parts.append(struct.pack("<IH", int(g.root_total_visits[t]), len(a)))
        pairs = np.empty(len(a), dtype=[("a", "<u2"), ("c", "<u4")])
        pairs["a"] = a
        pairs["c"] = c
        parts.append(pairs.tobytes())
    if h.record_extras & EXTRAS_FINAL_OWNERSHIP:
        parts.append(g.final_ownership.astype(np.int8).tobytes())
    return b"".join(parts)


def serialize_chunk(chunk: Chunk) -> bytes:
    h = chunk.header
    h.num_games = len(chunk.games)
    h.num_positions = sum(g.T for g in chunk.games)
    return serialize_header(h) + b"".join(serialize_game(g, h) for g in chunk.games)


def parse_header(buf: bytes | memoryview) -> ChunkHeader:
    if len(buf) < HEADER_SIZE:
        raise ValueError("chunk shorter than its header")
    (magic, version, n, planes, schema, rules, komi, header_size, num_games, num_positions, sims, model_id, fp, chunk_id,
     run_seed, move_cap, temp_moves, extras, reduced, full_prob) = struct.unpack("<4sBBBBB3xfIIII32s16sQQHHBxHf20x",
                                                                                   bytes(buf[:HEADER_SIZE]))
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported chunk version {version}")
    if header_size != HEADER_SIZE:
        raise ValueError("header_size != 128")
    return ChunkHeader(
        board_size=n, komi=komi, planes=planes, feature_schema=schema, rules_id=rules, num_games=num_games,
        num_positions=num_positions, simulations_per_move=sims, model_id=model_id.split(b"\0", 1)[0].decode("ascii"),
        config_fingerprint=fp.split(b"\0", 1)[0].decode("ascii"), chunk_id=chunk_id, run_seed=run_seed,
        move_cap=move_cap, temperature_moves=temp_moves, record_extras=extras, reduced_simulations=reduced,
        full_search_prob=full_prob,
    )


def parse_chunk(buf: bytes) -> Chunk:
    h = parse_header(buf)
    n = h.board_size
    nn = n * n
    P = packed_snapshot_bytes(n)
    mv = memoryview(buf)
    pos = HEADER_SIZE
    games: list[GameRecord] = []
    positions = 0

    def take(nbytes: int) -> memoryview:
        nonlocal pos
        if pos + nbytes > len(mv):
            raise ValueError(f"chunk truncated at byte {pos}")
        out = mv[pos : pos + nbytes]
        pos += nbytes
        return out

    for gi in range(h.num_games):
        game_index, T, result, termination, no_resign, score, game_seed = struct.unpack("<HHbBBxfQ", bytes(take(20)))
        g = GameRecord(game_index=game_index, result=result, termination=termination, no_resign_game=bool(no_resign),
                       score=score, game_seed=game_seed)
        g.snapshots = np.frombuffer(take((T + 1) * P), np.uint8).reshape(T + 1, P).copy()
        g.moves = np.frombuffer(take(2 * T), "<u2").astype(np.uint16)
        if h.record_extras & EXTRAS_SEARCH_KIND:
            g.search_kind = np.frombuffer(take(T), np.uint8).copy()
        rv = np.frombuffer(take(8 * T), "<f4").reshape(T, 2)
        g.root_value = rv[:, 0].astype(np.float32)
        g.root_max_q = rv[:, 1].astype(np.float32)
        totals = np.empty(T, np.uint32)
        for t in range(T):
            total, nnz = struct.unpack("<IH", bytes(take(6)))
            totals[t] = total
            pairs = np.frombuffer(take(6 * nnz), dtype=[("a", "<u2"), ("c", "<u4")])
            g.visit_actions.append(pairs["a"].astype(np.uint16))
            g.visit_counts.append(pairs["c"].astype(np.uint32))
        g.root_total_visits = totals
        if h.record_extras & EXTRAS_FINAL_OWNERSHIP:
            g.final_ownership = np.frombuffer(take(nn), np.int8).copy()
        err = validate_game(g, h)
        if err:
            raise ValueError(f"game {gi}: {err}")
        positions += T
        games.append(g)
    if pos != len(mv):
        raise ValueError("trailing bytes after the last game")
    if positions != h.num_positions:
        raise ValueError("num_positions does not match the records")
    return Chunk(h, games)


def read_chunk(path: str | Path) -> Chunk:
    with open(path, "rb") as f:
        return parse_chunk(f.read())


def read_header(path: str | Path) -> ChunkHeader:
    with open(path, "rb") as f:
        return parse_header(f.read(HEADER_SIZE))


def write_chunk(path: str | Path, chunk: Chunk) -> Path:
    """Atomic publish: <path>.tmp then rename."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(serialize_chunk(chunk))
    os.replace(tmp, path)
    return path
