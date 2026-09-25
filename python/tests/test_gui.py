"""Human interface (DESIGN 6.7): parsing of the engine's replies, the game controller
against a scripted fake engine (turn taking, illegal moves, undo per mode, game end,
analysis refresh), a real game against the random `mango_gtp` player when the binaries
are built, and a smoke test of the tkinter window when a display is available."""

from __future__ import annotations

import pytest

from mango.gtp import GtpClient, GtpError, move_to_vertex, vertex_to_move
from mango.gui import (BLACK, MODE_ANALYSIS, MODE_PLAY, WHITE, Analysis, GameController, describe_result,
                       format_analysis, parse_analyze, parse_showboard)
from tests.test_pipeline import _bin_dir

SHOWBOARD_5 = """
 5 . . . . .
 4 . X . . .
 3 . . O . .
 2 . . . . .
 1 X . . . .
   A B C D E"""


class FakeEngine:
    """A GTP-shaped engine with a capture-free board, enough for the controller's control
    flow: occupied points are illegal, genmove takes the first empty point of a scripted
    list (pass when it is exhausted), mango-analyze returns a fixed report."""

    def __init__(self, n: int = 5, sims: int = 10, script: list[int] | None = None, analyze: str | None = None):
        self.n, self.sims = n, sims
        self.script = list(script or [])
        self.analyze_text = analyze
        self.stones: dict[int, str] = {}
        self.moves: list[tuple[str, int]] = []
        self.komi = 7.5
        self.log: list[str] = []
        self.closed = False

    def command(self, line: str) -> str:
        self.log.append(line)
        tok = line.split()
        cmd = tok[0]
        n2 = self.n * self.n
        if cmd == "komi":
            self.komi = float(tok[1])
            return ""
        if cmd == "clear_board":
            self.stones.clear()
            self.moves.clear()
            return ""
        if cmd == "play":
            colour, m = tok[1], vertex_to_move(tok[2], self.n)
            if m < n2 and m in self.stones:
                raise GtpError("fake: illegal move")
            if m < n2:
                self.stones[m] = colour
            self.moves.append((colour, m))
            return ""
        if cmd == "genmove":
            colour = tok[1]
            m = n2
            while self.script:
                cand = self.script.pop(0)
                if cand == n2 or cand not in self.stones:
                    m = cand
                    break
            if m < n2:
                self.stones[m] = colour
            self.moves.append((colour, m))
            return move_to_vertex(m, self.n)
        if cmd == "undo":
            if not self.moves:
                raise GtpError("fake: cannot undo")
            _c, m = self.moves.pop()
            self.stones.pop(m, None)
            return ""
        if cmd == "showboard":
            rows = []
            for r in range(self.n):
                cells = []
                for c in range(self.n):
                    s = self.stones.get(r * self.n + c)
                    cells.append("." if s is None else ("X" if s == "B" else "O"))
                rows.append(f"{self.n - r:2d} " + " ".join(cells))
            return "\n" + "\n".join(rows) + "\n   " + " ".join("ABCDE"[: self.n])
        if cmd == "final_score":
            b = sum(1 for s in self.stones.values() if s == "B")
            w = sum(1 for s in self.stones.values() if s == "W")
            d = b - w - self.komi
            return "0" if d == 0 else (f"B+{d:g}" if d > 0 else f"W+{-d:g}")
        if cmd == "mango-analyze":
            if self.analyze_text is not None:
                return self.analyze_text
            empties = [m for m in range(n2) if m not in self.stones]
            lines = [f"root v=0.25 visits={self.sims} sims={self.sims}"]
            for i, m in enumerate(empties[:3]):
                lines.append(f"{m // self.n},{m % self.n} N={self.sims - 2 * i} Q={0.3 - 0.1 * i} P={0.2 - 0.05 * i}")
            lines.append("pass N=1 Q=-0.9 P=0.01")
            return "\n".join(lines)
        raise GtpError(f"fake: unknown command {cmd}")

    def close(self) -> None:
        self.closed = True


def controller(mode=MODE_PLAY, human=BLACK, script=None, n=5, sims=10, engines=None, **kw) -> GameController:
    def make(s):
        e = FakeEngine(n=n, sims=s, script=list(script or []))
        if engines is not None:
            engines.append(e)
        return e
    return GameController(make, sims=sims, komi=7.5, mode=mode, human_colour=human, **kw)


# ----------------------------------------------------------------------------- parsing


def test_parse_showboard_rows_are_top_down():
    g = parse_showboard(SHOWBOARD_5)
    assert len(g) == 5 and all(len(r) == 5 for r in g)
    assert g[1][1] == "X" and g[2][2] == "O" and g[4][0] == "X" and g[0][0] == "."
    with pytest.raises(ValueError):
        parse_showboard("nothing here")
    with pytest.raises(ValueError):
        parse_showboard(" 5 . . . . .\n 4 . . .")  # ragged


def test_parse_analyze_orders_and_converts_points():
    text = "root v=0.125 visits=210 sims=200\n2,4 N=120 Q=0.31 P=0.402\npass N=5 Q=-0.7 P=0.01\n0,0 N=3 Q=0.1 P=0.05"
    a = parse_analyze(text, 9, WHITE)
    assert a is not None and a.to_move == WHITE
    assert a.root_value == pytest.approx(0.125) and a.visits == 210 and a.simulations == 200
    assert [m.move for m in a.moves] == [2 * 9 + 4, 81, 0]
    assert a.moves[0].visits == 120 and a.moves[0].q == pytest.approx(0.31) and a.moves[0].prior == pytest.approx(0.402)
    assert a.winrate() == pytest.approx(0.5625) and a.winrate(-1.0) == 0.0
    # No analysis: a player without a search (empty reply) or a finished game.
    assert parse_analyze("", 9, BLACK) is None
    assert parse_analyze("game over", 9, BLACK) is None
    s = format_analysis(a, 9)
    assert "White to move" in s and "E7" in s and "pass" in s and "56.2 %" in s
    assert format_analysis(None, 9) == "(no analysis)"


def test_describe_result():
    assert describe_result("B+3.5") == "Black wins by 3.5"
    assert describe_result("W+R") == "White wins by resignation"
    assert describe_result("0") == "Draw" and describe_result("") == ""


# ----------------------------------------------------------------------------- controller


def test_play_mode_alternates_human_and_model_and_rejects_illegal_moves():
    engines = []
    c = controller(script=[12, 13], engines=engines)
    assert c.n == 5 and c.human_to_move and not c.model_to_move
    assert engines[0].log[:2] == ["komi 7.5", "showboard"]
    assert c.human_move(0)
    assert c.state.grid[0][0] == "X" and c.state.to_move == WHITE and c.model_to_move
    assert not c.human_move(1), "not the human's turn"
    assert c.engine_move() == 12
    assert c.state.grid[2][2] == "O" and c.human_to_move
    assert not c.human_move(12), "occupied: the engine rejects it and nothing changes"
    assert c.state.to_move == BLACK and len(c.state.moves) == 2
    assert c.state.moves == [(BLACK, 0), (WHITE, 12)]
    with pytest.raises(RuntimeError):
        c.engine_move()


def test_model_moves_first_when_the_human_is_white():
    c = controller(human=WHITE, script=[12])
    assert c.model_to_move and not c.human_to_move
    assert c.engine_move() == 12
    assert c.human_to_move and c.state.to_move == WHITE


def test_undo_retracts_a_pair_in_play_mode_and_one_move_in_analysis_mode():
    c = controller(script=[12, 13])
    c.human_move(0)
    c.engine_move()
    assert c.undo() == 2
    assert c.state.moves == [] and c.state.grid[0][0] == "." and c.state.grid[2][2] == "." and c.state.to_move == BLACK
    assert c.undo() == 0
    # After the human's move only (the model has not replied yet): one move.
    c.human_move(3)
    assert c.undo() == 1 and c.state.moves == []
    # Analysis mode: one move at a time, the turn goes back to the retracted colour.
    a = controller(mode=MODE_ANALYSIS)
    a.human_move(0)
    a.human_move(1)
    assert a.state.to_move == BLACK
    assert a.undo() == 1
    assert a.state.moves == [(BLACK, 0)] and a.state.to_move == WHITE and a.state.grid[0][1] == "."
    assert "undo" in a.engine.log


def test_two_passes_end_the_game_and_undo_reopens_it():
    c = controller(mode=MODE_ANALYSIS)
    c.human_move(0)
    assert c.pass_move() and not c.state.over
    assert c.pass_move() and c.state.over
    assert c.state.result == "W+6.5"  # 1 black stone - 7.5 komi on the fake's count
    assert not c.human_move(1), "no moves once the game is over"
    assert c.analyze() is None and c.state.analysis is None
    assert c.undo() == 1 and not c.state.over and c.state.result == "" and c.state.to_move == BLACK
    assert c.human_move(1)


def test_move_cap_ends_the_game():
    c = controller(mode=MODE_ANALYSIS, move_cap=3)
    c.human_move(0)
    c.human_move(1)
    assert not c.state.over
    c.human_move(2)
    assert c.state.over and c.state.result == "W+6.5"  # 2 black, 1 white, komi 7.5 on the fake's count
    assert not c.human_move(3)


def test_resign_is_by_the_side_to_move_and_the_model_may_resign():
    c = controller()
    c.resign()
    assert c.state.over and c.state.result == "W+R"
    c.undo()
    assert not c.state.over and c.state.result == ""
    c.human_move(0)
    c.resign()
    assert c.state.result == "B+R"  # white (the model) was to move
    engines = []
    m = controller(script=[], engines=engines)

    class Resigner(FakeEngine):
        def command(self, line):
            if line.startswith("genmove"):
                return "resign"
            return super().command(line)
    m.engine = Resigner()
    m.human_move(0)
    assert m.engine_move() == -1 and m.state.over and m.state.result == "B+R"


def test_analysis_is_for_the_side_to_move_truncated_to_top_k_and_cleared_by_a_move():
    c = controller(mode=MODE_ANALYSIS, top_k=2)
    c.human_move(0)
    a = c.analyze()
    assert isinstance(a, Analysis) and a.to_move == WHITE and len(a.moves) == 2
    assert a.moves[0].move == 1 and a.moves[0].visits == 10 and c.state.analysis is a
    c.human_move(a.moves[0].move)
    assert c.state.analysis is None
    assert c.state.grid[0][1] == "O"


def test_new_game_restarts_the_engine_only_when_the_budget_changes():
    engines = []
    c = controller(script=[12], engines=engines)
    c.human_move(0)
    c.engine_move()
    c.new_game(mode=MODE_ANALYSIS, human_colour=WHITE)
    assert len(engines) == 1 and "clear_board" in engines[0].log
    assert c.state.moves == [] and c.state.grid[0][0] == "." and c.mode == MODE_ANALYSIS and not c.state.over
    c.new_game(sims=50)
    assert len(engines) == 2 and engines[0].closed and engines[1].sims == 50 and c.sims == 50
    assert engines[1].log[:3] == ["komi 7.5", "clear_board", "showboard"]
    c.close()
    assert engines[1].closed


# ----------------------------------------------------------------------------- real engine


@pytest.fixture
def gtp_exe():
    b = _bin_dir()
    if b is None:
        pytest.skip("C++ executables not built")
    return b / ("mango_gtp.exe" if b.joinpath("mango_gtp.exe").exists() else "mango_gtp")


def test_controller_drives_the_random_mango_gtp_player(gtp_exe):
    def make(sims):
        return GtpClient([gtp_exe, "--size", 5, "--seed", 9], name="random")
    c = GameController(make, sims=1, komi=7.5, mode=MODE_PLAY, human_colour=BLACK)
    try:
        assert c.n == 5
        assert c.human_move(12)  # C3
        m = c.engine_move()
        assert 0 <= m <= 25 and m != 12
        # The drawn grid is the engine's board.
        assert c.state.grid[2][2] == "X"
        if m < 25:
            assert c.state.grid[m // 5][m % 5] == "O"
        assert not c.human_move(12), "occupied"
        assert c.analyze() is None, "the random player has no search"
        assert c.undo() == 2 and c.state.grid[2][2] == "." and c.state.moves == []
        assert c.final_score() == "W+7.5"
        # Two passes end the game with the engine's score.
        c.new_game(mode=MODE_ANALYSIS)
        assert c.pass_move() and c.pass_move() and c.state.over and c.state.result == "W+7.5"
    finally:
        c.close()


# ----------------------------------------------------------------------------- window


def test_window_draws_the_position_and_the_analysis():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        from mango.gui import App
        c = controller(mode=MODE_ANALYSIS, script=[])
        app = App(root, c, model_label="fake")
        root.update()
        app.play_point(12)
        root.update()
        # Drain the worker: the move is played, the analysis follows (auto-analysis on).
        for _ in range(200):
            root.update()
            app._poll()
            if not app.busy and c.state.analysis is not None:
                break
        assert c.state.grid[2][2] == "X" and c.state.analysis is not None
        assert app.point_at(*app._xy(2, 2)) == (2, 2) and app.point_at(-100, -100) is None
        half = app._cell() / 2
        assert app.point_at(app._xy(2, 2)[0] + half, app._xy(2, 2)[1]) is None, "between two points"
        items = app.canvas.find_all()
        assert len(items) > 25  # grid, stone, overlay
        assert "White to move" in app.status.cget("text")
        app.resign()
        assert "Black wins by resignation" in app.status.cget("text")
    finally:
        root.destroy()


def test_a_pass_as_best_move_is_announced_on_the_board():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        from mango.gui import App
        text = "\n".join(["root v=0.9 visits=50 sims=50", "pass N=40 Q=0.92 P=0.6", "0,0 N=6 Q=0.5 P=0.2",
                          "1,1 N=4 Q=0.4 P=0.1"])
        engines = []

        def make(sims):
            e = FakeEngine(n=5, sims=sims, analyze=text)
            engines.append(e)
            return e
        c = GameController(make, sims=50, komi=7.5, mode=MODE_ANALYSIS)
        app = App(root, c, model_label="fake")
        c.analyze()
        app.redraw()
        hints = [app.canvas.itemcget(i, "text") for i in app.canvas.find_withtag("pass_hint")]
        assert hints == ["Best move: PASS  (win 96 %, 40 visits)"]
        # The board moves are still drawn, none of them in the "best" green.
        greens = [i for i in app.canvas.find_all()
                  if app.canvas.type(i) == "oval" and app.canvas.itemcget(i, "fill") == "#2e8b57"]
        assert greens == []
        blues = [i for i in app.canvas.find_all()
                 if app.canvas.type(i) == "oval" and app.canvas.itemcget(i, "fill") == "#6fa8dc"]
        assert len(blues) == 2
    finally:
        root.destroy()


def test_mode_and_colour_switch_apply_to_the_game_in_progress():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")

    def drain(app):
        for _ in range(500):
            root.update()
            app._poll()
            if not app.busy:
                return
        raise AssertionError("worker did not finish")

    try:
        from mango.gui import App
        # Start in play mode (human black, the model would answer white).
        c = controller(mode=MODE_PLAY, script=[12, 13, 14])
        app = App(root, c, model_label="fake")
        root.update()
        # New game first, then the analysis radio: the switch must apply now, not at the next game.
        app.new_game()
        drain(app)
        app.mode_var.set(MODE_ANALYSIS)
        drain(app)
        assert c.mode == MODE_ANALYSIS and app.auto_var.get()
        app.play_point(0)
        drain(app)
        assert c.state.moves == [(BLACK, 0)] and c.state.to_move == WHITE, "no model reply in analysis mode"
        assert c.state.analysis is not None
        # Back to play mode while it is white's turn: the model (white) moves at once.
        app.mode_var.set(MODE_PLAY)
        drain(app)
        assert c.mode == MODE_PLAY and not app.auto_var.get()
        assert c.state.moves == [(BLACK, 0), (WHITE, 12)] and c.human_to_move
        # Taking white mid-game: black is now the model's and it is black's turn.
        app.colour_var.set(WHITE)
        drain(app)
        assert c.human_colour == WHITE and c.state.moves[-1] == (BLACK, 13) and c.state.to_move == WHITE
    finally:
        root.destroy()


# ----------------------------------------------------------------------------- launcher


def test_list_runs_finds_best_models_newest_first(tmp_path):
    import json
    import os
    import time

    from mango.gui import list_runs, run_eval_sims

    def make_run(name, model_id, iteration, sims=None, broken=False):
        r = tmp_path / name
        (r / "models" / model_id).mkdir(parents=True)
        (r / "best.json").write_text(json.dumps({"iteration": iteration, "model_id": model_id}), encoding="utf-8")
        if not broken:
            (r / "models" / model_id / "model.json").write_text(
                json.dumps({"model_id": model_id, "board_size": 9, "komi": 7.5}), encoding="utf-8")
        if sims is not None:
            (r / "config.json").write_text(json.dumps({"search": {"eval_simulations": sims}}), encoding="utf-8")
        return r

    old = make_run("old", "0003-aaaa", 3, sims=50)
    make_run("broken", "0001-bbbb", 1, broken=True)  # best.json points at a model without model.json
    new = make_run("new", "0015-cccc", 15)
    t = time.time()
    os.utime(old / "best.json", (t - 100, t - 100))
    os.utime(new / "best.json", (t, t))
    (tmp_path / "notarun").mkdir()
    runs = list_runs(tmp_path)
    assert [e.run.name for e in runs] == ["new", "old"]
    assert runs[0].model_dir == new / "models" / "0015-cccc" and runs[0].iteration == 15 and runs[0].sims == 200
    assert runs[1].sims == 50 and run_eval_sims(old) == 50 and run_eval_sims(new) == 200
    assert "0015-cccc" in runs[0].label and "iteration 15" in runs[0].label and "9x9" in runs[0].label
    assert list_runs(tmp_path / "missing") == []


def test_launchers_exist_and_point_at_the_gui_script():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cmd = (root / "Mango.cmd").read_bytes()
    assert b"scripts\\gui.py" in cmd and b"pythonw.exe" in cmd and b"\r\n" in cmd  # cmd.exe wants CRLF
    sh = (root / "Mango.command").read_text(encoding="utf-8")
    assert sh.startswith("#!/bin/bash") and "scripts/gui.py" in sh and "\r" not in sh
    launcher = (root / "scripts" / "gui.py").read_text(encoding="utf-8")
    assert "from mango.gui import run" in launcher
