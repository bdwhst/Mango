"""Human interface (docs/DESIGN.md section 6.7): a tkinter board for playing against a
model, or for playing both colours locally and asking the search for the best move.

The GUI contains no rules and no search. It drives one `mango_gtp --model DIR` process
over GTP (section 5.7) and treats it as the single authority: `play` decides legality
(occupied points, suicide, positional superko), `genmove` is the model's move,
`showboard` is re-read after every change and is what gets drawn, `final_score` is the
Tromp-Taylor score with the model's komi, and `mango-analyze` runs a search from the
current position and returns the top moves (N, Q, P) in the perspective of the side to
move. `undo`, `clear_board` and a restart discard the engine's tree (section 5.4.8).

    python scripts/gui.py --run runs/9x9-r0              # the run's best model
    python scripts/gui.py --model runs/9x9-r0/models/0015-4da2388b --sims 400
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .gtp import LETTERS, GtpClient, GtpError, move_to_vertex, vertex_to_move

MODE_PLAY = "play"          # human vs model
MODE_ANALYSIS = "analysis"  # human plays both colours, the search suggests moves
BLACK, WHITE = "B", "W"


def other(colour: str) -> str:
    return WHITE if colour == BLACK else BLACK


# ----------------------------------------------------------------------------- engine access


class Engine(Protocol):
    def command(self, line: str) -> str: ...
    def close(self) -> None: ...


def find_bin_dir() -> Path | None:
    """Where the C++ executables are: MANGO_BIN_DIR, else the usual build directories."""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.get("MANGO_BIN_DIR")
    candidates = [Path(env)] if env else []
    candidates += [root / "build" / "windows-cuda" / "Release", root / "build" / "windows" / "Release",
                   root / "build" / "macos-mps", root / "build" / "macos", root / "build" / "linux"]
    exe = "mango_gtp" + (".exe" if platform.system() == "Windows" else "")
    for c in candidates:
        if (c / exe).exists():
            return c
    return None


def gtp_exe(bin_dir: str | Path) -> Path:
    return Path(bin_dir) / ("mango_gtp" + (".exe" if platform.system() == "Windows" else ""))


def resolve_model(run: str | Path | None, model: str | Path | None) -> Path:
    """`--model DIR`, or the best model of `--run DIR` (its best.json)."""
    if model:
        return Path(model)
    if not run:
        raise SystemExit("give --run RUN_DIR or --model MODEL_DIR")
    with open(Path(run) / "best.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    return Path(run) / "models" / best["model_id"]


def model_metadata(model_dir: str | Path) -> dict:
    with open(Path(model_dir) / "model.json", "r", encoding="utf-8") as f:
        return json.load(f)


def engine_factory(exe: Path, model_dir: Path, device: str = "auto") -> Callable[[int], Engine]:
    """An engine constructor taking the simulation budget (a launch option of mango_gtp)."""
    def make(sims: int) -> Engine:
        return GtpClient([exe, "--model", model_dir, "--sims", sims, "--device", device], name="mango_gtp")
    return make


# ----------------------------------------------------------------------------- parsing


@dataclass
class MoveStat:
    move: int      # point index row * n + col, or n * n for pass
    visits: int
    q: float       # mean value of the move for the side to move, in [-1, 1]
    prior: float


@dataclass
class Analysis:
    to_move: str
    root_value: float   # v of the root for the side to move (section 5.4.10 value)
    visits: int
    simulations: int
    moves: list[MoveStat] = field(default_factory=list)

    def winrate(self, q: float | None = None) -> float:
        """P(side to move wins) read off a value in [-1, 1]."""
        v = self.root_value if q is None else q
        return (v + 1.0) / 2.0


def parse_showboard(text: str) -> list[list[str]]:
    """The `showboard` reply as rows of '.', 'X', 'O'; row 0 is the top rank."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append(parts[1:])
    n = len(rows)
    if n == 0 or any(len(r) != n for r in rows):
        raise ValueError(f"unrecognised showboard output: {text!r}")
    return rows


def _kv(token: str) -> tuple[str, float]:
    k, _, v = token.partition("=")
    return k, float(v)


def parse_analyze(text: str, n: int, to_move: str) -> Analysis | None:
    """The `mango-analyze` reply: 'root v=.. visits=.. sims=..' then one 'r,c N=.. Q=.. P=..'
    (or 'pass N=..') line per move, best first. None when the engine gave no analysis
    (a player without a search, or a finished game)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("root "):
        return None
    head = dict(_kv(t) for t in lines[0].split()[1:])
    a = Analysis(to_move=to_move, root_value=head["v"], visits=int(head["visits"]), simulations=int(head["sims"]))
    for ln in lines[1:]:
        tok = ln.split()
        if tok[0] == "pass":
            m = n * n
        else:
            r, c = tok[0].split(",")
            m = int(r) * n + int(c)
        stats = dict(_kv(t) for t in tok[1:])
        a.moves.append(MoveStat(move=m, visits=int(stats["N"]), q=stats["Q"], prior=stats["P"]))
    return a


# ----------------------------------------------------------------------------- controller


@dataclass
class GameState:
    n: int
    komi: float
    grid: list[list[str]]
    moves: list[tuple[str, int]] = field(default_factory=list)  # (colour, move)
    to_move: str = BLACK
    over: bool = False
    result: str = ""          # "B+3.5", "W+R", "0" (draw); "" while the game is on
    analysis: Analysis | None = None

    @property
    def last_move(self) -> tuple[str, int] | None:
        return self.moves[-1] if self.moves else None


class GameController:
    """The game as the GUI sees it. Every position change goes through the engine and the
    grid is re-read from `showboard`; the controller only remembers the move list, whose
    turn it is, and how the game ended."""

    def __init__(self, make_engine: Callable[[int], Engine], sims: int, komi: float, move_cap: int | None = None,
                 mode: str = MODE_PLAY, human_colour: str = BLACK, top_k: int = 8):
        self.make_engine = make_engine
        self.sims = sims
        self.komi = komi
        self.mode = mode
        self.human_colour = human_colour
        self.top_k = top_k
        self.engine: Engine = make_engine(sims)
        self.engine.command(f"komi {komi}")
        grid = parse_showboard(self.engine.command("showboard"))
        n = len(grid)
        self.move_cap = move_cap if move_cap else 2 * n * n
        self.state = GameState(n=n, komi=komi, grid=grid)

    # --- game flow -----------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.state.n

    @property
    def model_colour(self) -> str:
        return other(self.human_colour)

    @property
    def model_to_move(self) -> bool:
        return self.mode == MODE_PLAY and not self.state.over and self.state.to_move == self.model_colour

    @property
    def human_to_move(self) -> bool:
        return not self.state.over and not self.model_to_move

    def new_game(self, mode: str | None = None, human_colour: str | None = None, sims: int | None = None) -> None:
        """Clears the board. A changed simulation budget restarts the engine (it is a
        launch option of mango_gtp)."""
        if mode is not None:
            self.mode = mode
        if human_colour is not None:
            self.human_colour = human_colour
        if sims is not None and sims != self.sims:
            self.engine.close()
            self.sims = sims
            self.engine = self.make_engine(sims)
            self.engine.command(f"komi {self.komi}")
        self.engine.command("clear_board")
        self.state = GameState(n=self.n, komi=self.komi, grid=parse_showboard(self.engine.command("showboard")))

    def human_move(self, move: int) -> bool:
        """Plays `move` (point index or n*n for pass) for the side to move. False if it is
        not the human's turn or the engine rejects the move; the position is unchanged."""
        if not self.human_to_move:
            return False
        colour = self.state.to_move
        try:
            self.engine.command(f"play {colour} {move_to_vertex(move, self.n)}")
        except GtpError:
            return False
        self._record(colour, move)
        return True

    def engine_move(self) -> int:
        """Asks the model for its move (play mode, model to move) and plays it."""
        if not self.model_to_move:
            raise RuntimeError("it is not the model's turn")
        colour = self.state.to_move
        reply = self.engine.command(f"genmove {colour}").strip()
        if reply.lower() == "resign":
            self.state.over = True
            self.state.result = f"{other(colour)}+R"
            return -1
        move = vertex_to_move(reply, self.n)
        self._record(colour, move)
        return move

    def pass_move(self) -> bool:
        return self.human_move(self.n * self.n)

    def resign(self) -> None:
        """The side to move resigns (the human, in play mode)."""
        if self.state.over:
            return
        self.state.over = True
        self.state.result = f"{other(self.state.to_move)}+R"
        self.state.analysis = None

    def undo(self) -> int:
        """Retracts moves until it is the human's turn: one move in analysis mode, the
        model's reply and the human's move in play mode. Returns how many were undone.
        The engine discards its tree (section 5.4.8)."""
        st = self.state
        if not st.moves:
            if st.over:  # a resignation before any move
                st.over, st.result = False, ""
            return 0
        count = 1
        if self.mode == MODE_PLAY and st.moves[-1][0] == self.model_colour and len(st.moves) >= 2:
            count = 2
        for _ in range(count):
            self.engine.command("undo")
            colour, _m = st.moves.pop()
            st.to_move = colour
        st.over, st.result, st.analysis = False, "", None
        self._sync_grid()
        return count

    def analyze(self) -> Analysis | None:
        """Searches from the current position for the side to move; None once the game is
        over or when the engine has no search (random player)."""
        if self.state.over:
            self.state.analysis = None
            return None
        text = self.engine.command("mango-analyze")
        a = parse_analyze(text, self.n, self.state.to_move)
        if a is not None:
            a.moves = a.moves[: self.top_k]
        self.state.analysis = a
        return a

    def final_score(self) -> str:
        return self.engine.command("final_score").strip()

    def close(self) -> None:
        self.engine.close()

    # --- internals -----------------------------------------------------------------

    def _record(self, colour: str, move: int) -> None:
        st = self.state
        st.moves.append((colour, move))
        st.to_move = other(colour)
        st.analysis = None
        self._sync_grid()
        n2 = self.n * self.n
        if len(st.moves) >= 2 and st.moves[-1][1] == n2 and st.moves[-2][1] == n2:
            st.over, st.result = True, self.final_score()
        elif len(st.moves) >= self.move_cap:
            st.over, st.result = True, self.final_score()

    def _sync_grid(self) -> None:
        self.state.grid = parse_showboard(self.engine.command("showboard"))


def describe_result(result: str) -> str:
    if result in ("", None):
        return ""
    if result == "0":
        return "Draw"
    side = "Black" if result.startswith("B") else "White"
    if result.endswith("+R"):
        return f"{side} wins by resignation"
    return f"{side} wins by {result[2:]}"


def format_analysis(a: Analysis | None, n: int) -> str:
    if a is None:
        return "(no analysis)"
    side = "Black" if a.to_move == BLACK else "White"
    lines = [f"{side} to move: win {100 * a.winrate():.1f} %  (v = {a.root_value:+.3f}, {a.visits} visits, "
             f"{a.simulations} sims)"]
    for i, m in enumerate(a.moves, 1):
        lines.append(f"{i:2d}. {move_to_vertex(m.move, n):>4}  N={m.visits:<5d} win {100 * a.winrate(m.q):5.1f} %  "
                     f"P={m.prior:.3f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- tkinter view


def _star_points(n: int) -> list[tuple[int, int]]:
    if n < 7:
        return []
    k = 3 if n >= 13 else 2
    pts = [k, n - 1 - k]
    if n % 2 == 1:
        pts.append(n // 2)
    return [(r, c) for r in pts for c in pts]


class App:
    """The window. Engine calls run on a worker thread; only the tk thread touches widgets."""

    BOARD_PX = 640   # at 96 dpi; scaled with the screen's dpi
    MARGIN = 36

    def __init__(self, root, controller: GameController, model_label: str = ""):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.ctrl = controller
        self.busy = False
        self.queue: queue.Queue = queue.Queue()
        root.title(f"Mango {model_label}".strip())
        dpi_scale = max(1.0, root.winfo_fpixels("1i") / 96.0)
        self.board_px = int(self.BOARD_PX * dpi_scale)
        self.margin = int(self.MARGIN * dpi_scale)

        self.canvas = tk.Canvas(root, width=self.board_px, height=self.board_px, bg="#deb887", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

        side = ttk.Frame(root)
        side.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        ttk.Label(side, text=f"Model: {model_label}").grid(row=0, column=0, columnspan=2, sticky="w")

        self.mode_var = tk.StringVar(value=controller.mode)
        self.colour_var = tk.StringVar(value=controller.human_colour)
        self.sims_var = tk.IntVar(value=controller.sims)
        self.auto_var = tk.BooleanVar(value=controller.mode == MODE_ANALYSIS)
        r = 1
        ttk.Label(side, text="Mode").grid(row=r, column=0, sticky="w")
        r += 1
        ttk.Radiobutton(side, text="Play against the model", variable=self.mode_var, value=MODE_PLAY).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Radiobutton(side, text="Analysis: I play both colours", variable=self.mode_var, value=MODE_ANALYSIS).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Label(side, text="My colour").grid(row=r, column=0, sticky="w")
        f = ttk.Frame(side)
        f.grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(f, text="Black", variable=self.colour_var, value=BLACK).pack(side="left")
        ttk.Radiobutton(f, text="White", variable=self.colour_var, value=WHITE).pack(side="left")
        r += 1
        ttk.Label(side, text="Simulations / move (next game)").grid(row=r, column=0, sticky="w")
        ttk.Spinbox(side, from_=1, to=100000, increment=50, textvariable=self.sims_var, width=8).grid(
            row=r, column=1, sticky="w")
        r += 1
        ttk.Checkbutton(side, text="Analyse after every move", variable=self.auto_var).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.btn_new = ttk.Button(side, text="New game", command=self.new_game)
        self.btn_new.grid(row=r, column=0, sticky="ew", pady=(8, 2))
        self.btn_analyze = ttk.Button(side, text="Analyse now", command=self.analyze)
        self.btn_analyze.grid(row=r, column=1, sticky="ew", pady=(8, 2))
        r += 1
        self.btn_pass = ttk.Button(side, text="Pass", command=self.pass_move)
        self.btn_pass.grid(row=r, column=0, sticky="ew", pady=2)
        self.btn_undo = ttk.Button(side, text="Undo", command=self.undo)
        self.btn_undo.grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        self.btn_resign = ttk.Button(side, text="Resign", command=self.resign)
        self.btn_resign.grid(row=r, column=0, sticky="ew", pady=2)
        self.btn_score = ttk.Button(side, text="Score now", command=self.score)
        self.btn_score.grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        self.status = ttk.Label(side, text="", wraplength=300)
        self.status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
        r += 1
        self.info = tk.Text(side, width=44, height=14, font=("Courier New", 10), state="disabled")
        self.info.grid(row=r, column=0, columnspan=2, sticky="nsew")
        r += 1
        ttk.Label(side, text="Moves").grid(row=r, column=0, sticky="w", pady=(8, 0))
        r += 1
        self.moves_text = tk.Text(side, width=44, height=8, font=("Courier New", 10), state="disabled")
        self.moves_text.grid(row=r, column=0, columnspan=2, sticky="nsew")
        self.buttons = [self.btn_new, self.btn_analyze, self.btn_pass, self.btn_undo, self.btn_resign, self.btn_score]

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.redraw()
        self._set_status()
        root.after(50, self._poll)
        # Mode and colour apply to the game in progress; the budget at the next new game.
        self.mode_var.trace_add("write", self._on_mode_or_colour_change)
        self.colour_var.trace_add("write", self._on_mode_or_colour_change)
        if self.ctrl.model_to_move:
            self._run(self.ctrl.engine_move, self._after_model_move)

    def _on_mode_or_colour_change(self, *_args) -> None:
        """Switching to analysis makes the human play both colours from now on (analysis
        on by default); switching to play hands the side to move to the model at once."""
        mode = self.mode_var.get()
        if mode != self.ctrl.mode:
            self.auto_var.set(mode == MODE_ANALYSIS)
        self.ctrl.mode = mode
        self.ctrl.human_colour = self.colour_var.get()
        self.redraw()
        self._set_status()
        if not self.busy:
            self._continue()

    # --- geometry ------------------------------------------------------------------

    def _cell(self) -> float:
        return (self.board_px - 2 * self.margin) / max(1, self.ctrl.n - 1)

    def _xy(self, r: int, c: int) -> tuple[float, float]:
        cell = self._cell()
        return self.margin + c * cell, self.margin + r * cell

    def point_at(self, x: float, y: float) -> tuple[int, int] | None:
        cell = self._cell()
        c = round((x - self.margin) / cell)
        r = round((y - self.margin) / cell)
        n = self.ctrl.n
        if not (0 <= r < n and 0 <= c < n):
            return None
        px, py = self._xy(r, c)
        if abs(px - x) > 0.45 * cell or abs(py - y) > 0.45 * cell:
            return None
        return r, c

    # --- drawing -------------------------------------------------------------------

    def redraw(self) -> None:
        cv, st, n = self.canvas, self.ctrl.state, self.ctrl.n
        cv.delete("all")
        cell = self._cell()
        x0, y0 = self._xy(0, 0)
        x1, y1 = self._xy(n - 1, n - 1)
        for i in range(n):
            x, y = self._xy(i, i)
            cv.create_line(x0, y, x1, y, fill="black")
            cv.create_line(x, y0, x, y1, fill="black")
            cv.create_text(x, y0 - 18, text=LETTERS[i], font=("Helvetica", 10))
            cv.create_text(x0 - 18, y, text=str(n - i), font=("Helvetica", 10))
        for r, c in _star_points(n):
            x, y = self._xy(r, c)
            cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
        rad = 0.46 * cell
        for r in range(n):
            for c in range(n):
                ch = st.grid[r][c]
                if ch not in ("X", "O"):
                    continue
                x, y = self._xy(r, c)
                fill, outline = ("black", "black") if ch == "X" else ("white", "black")
                cv.create_oval(x - rad, y - rad, x + rad, y + rad, fill=fill, outline=outline)
        last = st.last_move
        if last and last[1] < n * n:
            r, c = divmod(last[1], n)
            x, y = self._xy(r, c)
            cv.create_oval(x - rad * 0.35, y - rad * 0.35, x + rad * 0.35, y + rad * 0.35,
                           outline="red" if last[0] == BLACK else "red", width=2)
        if st.analysis is not None and not st.over:
            self._draw_analysis(st.analysis, rad)

    def _draw_analysis(self, a: Analysis, rad: float) -> None:
        """Top moves as discs (best in green) with win % and visits; a pass has no point,
        so when it is the best move (or among the top moves) it is announced in the margin."""
        cv, n = self.canvas, self.ctrl.n
        for i, m in enumerate(a.moves):
            if m.move >= n * n:
                best = i == 0
                colour = "#2e8b57" if best else "#4a7ab5"
                label = "Best move: PASS" if best else f"#{i + 1}: pass"
                label += f"  (win {100 * a.winrate(m.q):.0f} %, {m.visits} visits)"
                x0, y0 = self._xy(0, 0)
                x1, _ = self._xy(0, n - 1)
                cv.create_rectangle(x0 - rad, 4, x1 + rad, self.margin * 0.5, fill=colour, outline="")
                cv.create_text((x0 + x1) / 2, (4 + self.margin * 0.5) / 2, text=label, fill="white",
                               font=("Helvetica", 10, "bold"), tags="pass_hint")
                continue
            r, c = divmod(m.move, n)
            if self.ctrl.state.grid[r][c] != ".":
                continue
            x, y = self._xy(r, c)
            fill = "#2e8b57" if i == 0 else "#6fa8dc"
            cv.create_oval(x - rad, y - rad, x + rad, y + rad, fill=fill, outline="")
            cv.create_text(x, y - rad * 0.3, text=f"{100 * a.winrate(m.q):.0f}", font=("Helvetica", 9, "bold"),
                           fill="white")
            cv.create_text(x, y + rad * 0.35, text=str(m.visits), font=("Helvetica", 8), fill="white")

    def _set_text(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_status(self, extra: str = "") -> None:
        st = self.ctrl.state
        if st.over:
            text = "Game over: " + describe_result(st.result)
        elif self.busy:
            text = "Model thinking..." if self.ctrl.model_to_move else "Analysing..."
        else:
            who = "Black" if st.to_move == BLACK else "White"
            you = " (you)" if self.ctrl.mode == MODE_PLAY else ""
            text = f"{who} to move{you}. Move {len(st.moves) + 1}."
        if extra:
            text += "\n" + extra
        self.status.configure(text=text)
        self._set_text(self.info, format_analysis(st.analysis, self.ctrl.n))
        moves = [f"{i + 1}. {colour} {move_to_vertex(m, self.ctrl.n)}" for i, (colour, m) in enumerate(st.moves)]
        self._set_text(self.moves_text, "  ".join(moves))

    # --- async plumbing --------------------------------------------------------------

    def _run(self, fn: Callable, done: Callable) -> None:
        if self.busy:
            return
        self.busy = True
        for b in self.buttons:
            b.state(["disabled"])
        self._set_status()

        def work():
            try:
                self.queue.put((done, fn(), None))
            except Exception as e:  # shown to the user, the GUI stays up
                self.queue.put((done, None, e))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                done, result, err = self.queue.get_nowait()
                self.busy = False
                for b in self.buttons:
                    b.state(["!disabled"])
                if err is not None:
                    self.redraw()
                    self._set_status(f"Engine error: {err}")
                else:
                    done(result)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # --- actions ---------------------------------------------------------------------

    def _on_click(self, event) -> None:
        p = self.point_at(event.x, event.y)
        if p is None:
            return
        self.play_point(p[0] * self.ctrl.n + p[1])

    def play_point(self, move: int) -> None:
        if self.busy or not self.ctrl.human_to_move:
            return
        self._run(lambda: self.ctrl.human_move(move), self._after_human_move)

    def pass_move(self) -> None:
        self.play_point(self.ctrl.n * self.ctrl.n)

    def _after_human_move(self, ok: bool) -> None:
        self.redraw()
        if not ok:
            self._set_status("Illegal move.")
            return
        self._set_status()
        self._continue()

    def _continue(self) -> None:
        """After any position change: the model replies, or the analysis refreshes."""
        if self.ctrl.model_to_move:
            self._run(self.ctrl.engine_move, self._after_model_move)
        elif self.auto_var.get() and self.ctrl.human_to_move:
            self._run(self.ctrl.analyze, self._after_analysis)

    def _after_model_move(self, _move) -> None:
        self.redraw()
        self._set_status()
        if self.auto_var.get() and self.ctrl.human_to_move:
            self._run(self.ctrl.analyze, self._after_analysis)

    def _after_analysis(self, _a) -> None:
        self.redraw()
        self._set_status()

    def analyze(self) -> None:
        if self.busy or self.ctrl.state.over:
            return
        self._run(self.ctrl.analyze, self._after_analysis)

    def undo(self) -> None:
        if self.busy:
            return
        self._run(self.ctrl.undo, lambda _k: (self.redraw(), self._set_status()))

    def resign(self) -> None:
        if self.busy or self.ctrl.state.over:
            return
        self.ctrl.resign()
        self.redraw()
        self._set_status()

    def score(self) -> None:
        if self.busy:
            return
        self._run(self.ctrl.final_score, lambda s: self._set_status(f"Current score: {describe_result(s)}"))

    def new_game(self) -> None:
        if self.busy:
            return
        mode, colour, sims = self.mode_var.get(), self.colour_var.get(), int(self.sims_var.get())

        def start():
            self.ctrl.new_game(mode=mode, human_colour=colour, sims=sims)

        def started(_r):
            self.redraw()
            self._set_status()
            self._continue()

        self._run(start, started)

    def _on_close(self) -> None:
        try:
            self.ctrl.close()
        finally:
            self.root.destroy()


# ----------------------------------------------------------------------------- start-up chooser


@dataclass
class RunEntry:
    run: Path
    model_dir: Path
    model_id: str
    iteration: int
    board_size: int
    sims: int   # the run's eval_simulations (200 when the config has none)

    @property
    def label(self) -> str:
        return f"{self.run.name}  -  best {self.model_id} (iteration {self.iteration}, {self.board_size}x{self.board_size})"


def run_eval_sims(run: Path, default: int = 200) -> int:
    cfg = run / "config.json"
    if not cfg.exists():
        return default
    with open(cfg, "r", encoding="utf-8") as f:
        return int(json.load(f).get("search", {}).get("eval_simulations") or default)


def list_runs(runs_dir: str | Path) -> list[RunEntry]:
    """Every run under `runs_dir` whose best model exists, newest best.json first."""
    out: list[tuple[float, RunEntry]] = []
    for best in Path(runs_dir).glob("*/best.json"):
        try:
            with open(best, "r", encoding="utf-8") as f:
                b = json.load(f)
            model_dir = best.parent / "models" / b["model_id"]
            meta = model_metadata(model_dir)
        except (OSError, KeyError, ValueError):
            continue
        out.append((best.stat().st_mtime, RunEntry(best.parent, model_dir, b["model_id"], int(b.get("iteration", 0)),
                                                   int(meta["board_size"]), run_eval_sims(best.parent))))
    out.sort(key=lambda t: t[0], reverse=True)
    return [e for _t, e in out]


def choose_model_dialog(runs_dir: Path) -> tuple[Path, int] | None:
    """A small window: pick a run (its best model) or browse to a model directory.
    Returns (model_dir, sims) or None when closed."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    entries = list_runs(runs_dir)
    result: list[tuple[Path, int] | None] = [None]
    root = tk.Tk()
    root.title("Mango: choose a model")
    frame = ttk.Frame(root, padding=12)
    frame.grid(sticky="nsew")
    ttk.Label(frame, text=f"Runs under {runs_dir} (newest first)").grid(row=0, column=0, columnspan=3, sticky="w")
    box = tk.Listbox(frame, width=70, height=max(4, min(12, len(entries) + 1)), exportselection=False)
    for e in entries:
        box.insert("end", e.label)
    if entries:
        box.selection_set(0)
    box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(4, 8))
    ttk.Label(frame, text="Simulations / move").grid(row=2, column=0, sticky="w")
    sims_var = tk.IntVar(value=entries[0].sims if entries else 200)
    ttk.Spinbox(frame, from_=1, to=100000, increment=50, textvariable=sims_var, width=8).grid(row=2, column=1, sticky="w")

    def on_select(_event=None):
        sel = box.curselection()
        if sel:
            sims_var.set(entries[sel[0]].sims)

    box.bind("<<ListboxSelect>>", on_select)

    def ok(_event=None):
        sel = box.curselection()
        if not sel:
            return
        result[0] = (entries[sel[0]].model_dir, int(sims_var.get()))
        root.destroy()

    def browse():
        d = filedialog.askdirectory(title="Model directory (contains model.json)", mustexist=True)
        if d and (Path(d) / "model.json").exists():
            result[0] = (Path(d), int(sims_var.get()))
            root.destroy()

    ttk.Button(frame, text="Play", command=ok).grid(row=3, column=0, sticky="ew", pady=(8, 0))
    ttk.Button(frame, text="Browse model directory...", command=browse).grid(row=3, column=1, sticky="ew", pady=(8, 0))
    ttk.Button(frame, text="Quit", command=root.destroy).grid(row=3, column=2, sticky="ew", pady=(8, 0))
    box.bind("<Double-Button-1>", ok)
    root.bind("<Return>", ok)
    if not entries:
        ttk.Label(frame, text="No run with a best model found; browse to a model directory.").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
    root.mainloop()
    return result[0]


def _fail(message: str) -> int:
    """Errors go to a message box (a double-clicked launcher has no console) and stderr."""
    print(message, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Mango", message)
        root.destroy()
    except Exception:
        pass
    return 2


# ----------------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mango GUI: play against a model, or analyse a game you play yourself.")
    p.add_argument("--run", help="run directory; its best model is used (no --run/--model: a chooser window)")
    p.add_argument("--model", help="model directory (model.pt + model.json)")
    p.add_argument("--bin", help="directory with mango_gtp (default: MANGO_BIN_DIR or the build directories)")
    p.add_argument("--sims", type=int, help="simulations per move (default: the run's eval_simulations, else 200)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--mode", choices=(MODE_PLAY, MODE_ANALYSIS), default=MODE_PLAY)
    p.add_argument("--colour", choices=(BLACK, WHITE), default=BLACK, help="the human's colour in play mode")
    p.add_argument("--top-k", type=int, default=8, help="moves shown by the analysis")
    args = p.parse_args(argv)

    if platform.system() == "Windows":  # crisp rendering on high-dpi screens
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root_dir = Path(__file__).resolve().parents[2]
    bin_dir = Path(args.bin) if args.bin else find_bin_dir()
    if bin_dir is None:
        return _fail("mango_gtp not found: build with scripts/build.ps1 -Torch (or set MANGO_BIN_DIR / --bin)")
    sims = args.sims
    if args.run or args.model:
        model_dir = resolve_model(args.run, args.model)
        if sims is None and args.run:
            sims = run_eval_sims(Path(args.run))
    else:
        chosen = choose_model_dialog(root_dir / "runs")
        if chosen is None:
            return 0
        model_dir, chosen_sims = chosen
        sims = sims if sims is not None else chosen_sims
    sims = int(sims or 200)
    if not (model_dir / "model.json").exists():
        return _fail(f"no model.json in {model_dir}")
    meta = model_metadata(model_dir)

    import tkinter as tk

    ctrl = GameController(engine_factory(gtp_exe(bin_dir), model_dir, args.device), sims=sims,
                          komi=float(meta["komi"]), move_cap=meta.get("move_cap"), mode=args.mode,
                          human_colour=args.colour, top_k=args.top_k)
    root = tk.Tk()
    App(root, ctrl, model_label=f"{meta['model_id']} ({meta['board_size']}x{meta['board_size']}, komi {meta['komi']})")
    root.mainloop()
    return 0


def run() -> int:
    """main() with every error shown in a message box (used by the double-click launchers)."""
    try:
        return main()
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:  # start-up failures: missing model, engine could not start, ...
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(run())
