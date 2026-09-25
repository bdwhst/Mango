"""Run configuration: JSON with defaults identical to cpp/core/config.h.

Keys the C++ side does not read (it ignores unknown keys): search.resign_auto,
search.resign_select_scope, search.resign_fpr_target, selfplay.evict_old_chunks,
training.fixed_holdout_iteration, eval.ladder_pairs, eval.ladder_neighbours, eval.gtp_anchor, eval.gtp_workers,
selfplay.processes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, dict[str, Any]] = {
    "board": {"size": 9, "komi": 7.5, "move_cap": -1},
    "search": {
        "simulations": 200,
        "eval_simulations": 200,
        "c_puct": 1.5,
        "fpu": "zero",
        "dirichlet_alpha": -1.0,
        "dirichlet_epsilon": 0.25,
        "temperature_moves": 8,
        "search_symmetry": True,
        "tree_reuse": True,
        "budget_includes_inherited": False,
        "resign_threshold": -1.0,
        "resign_auto": False,  # True: the pipeline selects v_resign per iteration (DESIGN 5.4.7)
        "resign_select_scope": "window",  # games the selection uses: "window" (paper) | "latest" (previous iteration only)
        "resign_fpr_target": 0.05,  # the selection keeps the false-positive rate below this on those games
        "no_resign_fraction": 0.10,
        "nn_cache_size": 0,
        "resolve_terminal_moves": True,  # exact values of game-ending moves in the search (DESIGN 5.4.10, D18)
    },
    "selfplay": {
        "games_per_iteration": 2000,
        "games_in_flight": 128,
        "leaves_per_game": 1,
        "chunk_games": 256,
        "save_sgf": True,
        "evict_old_chunks": True,  # delete chunks that fell out of the window (DESIGN 6.3)
        "processes": 1,  # N >= 2: N mango_selfplay processes per iteration, each with games_in_flight (DESIGN 5.5.1 step 0)
    },
    "training": {
        "res_blocks": 6,
        "filters": 64,
        "batch_size": 256,
        "max_steps_per_iteration": 1000,
        "samples_per_position": 0.25,
        "window_games": 20000,
        "holdout_fraction": 0.05,
        "l2": 1e-4,
        "l2_all_params": False,
        "store_pi": "visits",
        "lr0": 0.01,
        "lr_step1": 30000,
        "lr_step2": 60000,
        "fixed_holdout_iteration": 0,  # >0: freeze that iteration's holdout games as a fixed validation set (DESIGN 6.6)
    },
    "eval": {
        "pairs": 200,
        "gate_threshold": 0.55,
        "opening_moves": 3,
        "ladder_every": 5,
        "gating": True,
        "ladder_pairs": 50,       # pairs per ladder match (DESIGN 6.6)
        "ladder_neighbours": 3,   # nearest ladder entries a new entry plays
        "gtp_anchor": None,       # ladder GTP anchor played by the pipeline; keep None: GNU Go is measured on
                                  # demand by scripts/vs_gnugo.py (DESIGN 6.6), never during training
        "gtp_workers": 4,         # parallel engine trios for a GTP match (scripts/vs_gnugo.py; games sequential per trio)
    },
}


def default_config() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(DEFAULTS)


def merge_config(user: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cfg = default_config()
    for section, values in user.items():
        if section not in cfg:
            raise KeyError(f"unknown config section: {section}")
        for key, value in values.items():
            if key not in cfg[section]:
                raise KeyError(f"unknown config key: {section}.{key}")
            cfg[section][key] = value
    size = cfg["board"]["size"]
    if not (2 <= size <= 19):
        raise ValueError("board.size must be in [2, 19]")
    return cfg


def load_config(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return merge_config(json.load(f))


def effective_move_cap(cfg: dict[str, Any]) -> int:
    mc = cfg["board"]["move_cap"]
    n = cfg["board"]["size"]
    return 2 * n * n if mc < 0 else mc


def effective_dirichlet_alpha(cfg: dict[str, Any]) -> float:
    a = cfg["search"]["dirichlet_alpha"]
    n = cfg["board"]["size"]
    return a if a >= 0 else 0.03 * 361.0 / (n * n)
