"""Run configuration: JSON with defaults identical to cpp/core/config.h."""

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
        "no_resign_fraction": 0.10,
        "nn_cache_size": 0,
    },
    "selfplay": {
        "games_per_iteration": 2000,
        "games_in_flight": 128,
        "leaves_per_game": 1,
        "chunk_games": 256,
        "save_sgf": True,
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
    },
    "eval": {"pairs": 200, "gate_threshold": 0.55, "opening_moves": 3, "ladder_every": 5, "gating": True},
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
