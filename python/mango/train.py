"""Training phase (docs/DESIGN.md section 6.2) and the held-out monitor (6.6).

  loss = mse(v, z) + CE(logits, pi) [policy-target positions] + c * sum ||theta||^2
  SGD momentum 0.9, LR by global step, batch from config; steps per iteration bounded by
  samples_per_position * window_train_positions / batch. Learner checkpoints carry the
  model, optimiser, scaler, global step and RNG states, and are distinct from model
  versions (the self-play models).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import ChunkDataset, Window
from .model import AGZNet


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def learning_rate(step: int, cfg: dict[str, Any]) -> float:
    t = cfg["training"]
    if step < t["lr_step1"]:
        return float(t["lr0"])
    if step < t["lr_step2"]:
        return float(t["lr0"]) * 0.1
    return float(t["lr0"]) * 0.01


def bounded_steps(cfg: dict[str, Any], window_train_positions: int) -> int:
    """steps = min(train_steps, ceil(samples_per_position * window_train_positions / batch))."""
    t = cfg["training"]
    if window_train_positions <= 0:
        return 0
    by_data = math.ceil(float(t["samples_per_position"]) * window_train_positions / int(t["batch_size"]))
    return int(min(int(t["max_steps_per_iteration"]), by_data))


def build_model(cfg: dict[str, Any]) -> AGZNet:
    return AGZNet(int(cfg["board"]["size"]), int(cfg["training"]["res_blocks"]), int(cfg["training"]["filters"]))


def build_optimizer(model: AGZNet, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=float(cfg["training"]["lr0"]), momentum=0.9)


def compute_loss(model: AGZNet, logits: torch.Tensor, value: torch.Tensor, pi: torch.Tensor, z: torch.Tensor,
                 policy_mask: torch.Tensor, l2: float, l2_all_params: bool) -> dict[str, torch.Tensor]:
    value_loss = F.mse_loss(value, z)
    log_probs = F.log_softmax(logits.float(), dim=1)
    ce_per = -(pi * log_probs).sum(dim=1)
    m = policy_mask.sum()
    policy_loss = (ce_per * policy_mask).sum() / m if float(m) > 0 else ce_per.sum() * 0.0
    reg = sum((p.float() ** 2).sum() for p in model.l2_parameters(l2_all_params))
    total = value_loss + policy_loss + l2 * reg
    return {"total": total, "value": value_loss.detach(), "policy": policy_loss.detach(), "l2": (l2 * reg).detach()}


# --- checkpoints --------------------------------------------------------------------------

def save_checkpoint(path: str | Path, model: AGZNet, optimizer: torch.optim.Optimizer, scaler: Any, step: int,
                    cfg: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(step),
        "config_hash": config_hash(cfg),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, model: AGZNet, optimizer: torch.optim.Optimizer | None, scaler: Any,
                    device: torch.device, restore_rng: bool = True) -> int:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
    return int(state["global_step"])


def checkpoint_step(path: str | Path) -> int:
    return int(torch.load(Path(path), map_location="cpu", weights_only=False)["global_step"])


# --- monitor ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(model: AGZNet, ds: ChunkDataset, device: torch.device, batch_size: int = 256,
                     max_positions: int | None = None, seed: int = 0) -> dict[str, float]:
    """Mean value MSE and policy CE over the dataset's indexed positions (no symmetry), or
    over a seeded random subset of `max_positions` of them."""
    model.eval()
    n_pos = 0
    v_sum = 0.0
    p_sum = 0.0
    for batch in ds.iterate_all(batch_size, max_positions=max_positions, seed=seed):
        planes = torch.from_numpy(batch["planes"]).to(device)
        pi = torch.from_numpy(batch["pi"]).to(device)
        z = torch.from_numpy(batch["z"]).to(device)
        logits, value = model(planes)
        v_sum += float(((value - z) ** 2).sum())
        p_sum += float((-(pi * F.log_softmax(logits.float(), dim=1)).sum(dim=1)).sum())
        n_pos += planes.shape[0]
    model.train()
    if n_pos == 0:
        return {"positions": 0, "value_mse": float("nan"), "policy_ce": float("nan")}
    return {"positions": n_pos, "value_mse": v_sum / n_pos, "policy_ce": p_sum / n_pos}


# --- the training phase -------------------------------------------------------------------

def train_until(model: AGZNet, optimizer: torch.optim.Optimizer, scaler: Any, ds: ChunkDataset, cfg: dict[str, Any],
                start_step: int, target_step: int, device: torch.device, ckpt_path: str | Path | None = None,
                ckpt_every: int = 0, log=None) -> dict[str, float]:
    """Trains from global step `start_step` until `target_step` (never "N more steps")."""
    t = cfg["training"]
    batch_size = int(t["batch_size"])
    l2 = float(t["l2"])
    l2_all = bool(t["l2_all_params"])
    use_amp = device.type == "cuda"
    model.train()
    step = start_step
    last: dict[str, float] = {}
    sums = {"total": 0.0, "value": 0.0, "policy": 0.0, "l2": 0.0}
    count = 0
    while step < target_step:
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = ds.sample_batch(batch_size)
        planes = torch.from_numpy(batch["planes"]).to(device, non_blocking=True)
        pi = torch.from_numpy(batch["pi"]).to(device, non_blocking=True)
        z = torch.from_numpy(batch["z"]).to(device, non_blocking=True)
        pmask = torch.from_numpy(batch["policy_mask"]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, value = model(planes)
        losses = compute_loss(model, logits.float(), value.float(), pi, z, pmask, l2, l2_all)
        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()
        step += 1
        for k in sums:
            sums[k] += float(losses[k])
        count += 1
        if ckpt_path is not None and ckpt_every > 0 and step % ckpt_every == 0 and step < target_step:
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, cfg)
        if log is not None and (step % 50 == 0 or step == target_step):
            log(f"step {step}/{target_step} lr {lr:g} loss {float(losses['total']):.4f} "
                f"(v {float(losses['value']):.4f} p {float(losses['policy']):.4f})")
    if count:
        last = {k: v / count for k, v in sums.items()}
    last["steps"] = float(count)
    return last


def make_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def build_window_dataset(chunk_paths: list[str | Path], cfg: dict[str, Any], seed: int, holdout: bool = False,
                         symmetry: bool = True) -> ChunkDataset:
    window = Window(chunk_paths)
    return ChunkDataset(window, store_pi=cfg["training"]["store_pi"],
                        temperature_moves=int(cfg["search"]["temperature_moves"]), symmetry=symmetry, seed=seed,
                        holdout=holdout)
