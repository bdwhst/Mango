"""Trainer contracts (DESIGN 6.2, 9 "Training"): step cap, LR schedule, one-step overfit,
policy-mask normalisation, checkpoint restore reproducing the next step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mango.config import merge_config
from mango.data import ChunkDataset, Window
from mango.train import (bounded_steps, build_model, build_optimizer, checkpoint_step, compute_loss, evaluate_dataset,
                         learning_rate, load_checkpoint, save_checkpoint, train_until)
from tests.test_data import make_game, write_test_chunk

CFG = merge_config({"board": {"size": 5}, "training": {"res_blocks": 1, "filters": 8, "batch_size": 16,
                                                        "max_steps_per_iteration": 100, "samples_per_position": 0.25,
                                                        "lr_step1": 10, "lr_step2": 20}})


def test_step_cap_and_lr_schedule():
    assert bounded_steps(CFG, 0) == 0
    assert bounded_steps(CFG, 64) == 1  # ceil(0.25 * 64 / 16) = 1
    assert bounded_steps(CFG, 6400) == 100  # capped by max_steps_per_iteration
    assert bounded_steps(CFG, 1000) == 16  # ceil(250 / 16)
    assert learning_rate(0, CFG) == 0.01 and learning_rate(9, CFG) == 0.01
    assert learning_rate(10, CFG) == pytest.approx(0.001) and learning_rate(25, CFG) == pytest.approx(0.0001)


def test_policy_loss_averages_over_masked_positions_only():
    torch.manual_seed(0)
    model = build_model(CFG)
    B, na = 4, 26
    logits = torch.randn(B, na)
    value = torch.zeros(B)
    pi = torch.softmax(torch.randn(B, na), 1)
    z = torch.zeros(B)
    full = compute_loss(model, logits, value, pi, z, torch.ones(B), 0.0, False)
    half = compute_loss(model, logits, value, pi, z, torch.tensor([1.0, 1.0, 0.0, 0.0]), 0.0, False)
    ce = -(pi * torch.log_softmax(logits, 1)).sum(1)
    assert float(full["policy"]) == pytest.approx(float(ce.mean()), rel=1e-5)
    assert float(half["policy"]) == pytest.approx(float(ce[:2].mean()), rel=1e-5)
    none = compute_loss(model, logits, value, pi, z, torch.zeros(B), 0.0, False)
    assert float(none["policy"]) == 0.0
    # L2 term is c * sum of squared conv/linear weights (BN excluded).
    reg = compute_loss(model, logits, value, pi, z, torch.ones(B), 1e-4, False)
    expected = 1e-4 * sum((p ** 2).sum() for p in model.l2_parameters(False))
    assert float(reg["l2"]) == pytest.approx(float(expected), rel=1e-5)


def _dataset(tmp_path):
    n = 5
    games = [make_game(n, 1 + i, 12, 1 if i % 2 == 0 else -1) for i in range(8)]
    p = write_test_chunk(tmp_path / "c.mgo", n, games)
    return ChunkDataset(Window([p]), symmetry=False, holdout=False, seed=1)


def test_training_reduces_loss_and_checkpoint_restore_is_exact(tmp_path):
    device = torch.device("cpu")
    ds = _dataset(tmp_path)
    torch.manual_seed(1)
    model = build_model(CFG).to(device)
    opt = build_optimizer(model, CFG)
    before = evaluate_dataset(model, ds, device)
    stats = train_until(model, opt, None, ds, CFG, 0, 30, device)
    after = evaluate_dataset(model, ds, device)
    assert stats["steps"] == 30
    assert after["value_mse"] < before["value_mse"]
    assert after["policy_ce"] < before["policy_ce"]

    # Save at step 30, train 5 more steps -> A. Restore the checkpoint into a fresh
    # model/optimizer and train the same 5 steps -> B. Identical weights.
    ck = save_checkpoint(tmp_path / "ckpt_30.pt", model, opt, None, 30, CFG)
    assert checkpoint_step(ck) == 30
    ds_a = ChunkDataset(ds.window, symmetry=False, holdout=False, seed=5)
    train_until(model, opt, None, ds_a, CFG, 30, 35, device)
    a = {k: v.clone() for k, v in model.state_dict().items()}

    model2 = build_model(CFG).to(device)
    opt2 = build_optimizer(model2, CFG)
    step = load_checkpoint(ck, model2, opt2, None, device)
    assert step == 30
    ds_b = ChunkDataset(ds.window, symmetry=False, holdout=False, seed=5)
    train_until(model2, opt2, None, ds_b, CFG, 30, 35, device)
    for k, v in model2.state_dict().items():
        assert torch.equal(v, a[k]), k
    # "Train until target" never repeats work: already at 35, nothing happens.
    assert train_until(model2, opt2, None, ds_b, CFG, 35, 35, device)["steps"] == 0


def test_holdout_monitor_uses_only_holdout_games(tmp_path):
    n = 5
    games = [make_game(n, 20, 6, 1), make_game(n, 40, 6, -1), make_game(n, 3, 6, 1)]
    p = write_test_chunk(tmp_path / "c.mgo", n, games)
    hold = ChunkDataset(Window([p]), symmetry=False, holdout=True)
    train = ChunkDataset(Window([p]), symmetry=False, holdout=False)
    assert hold.train_positions == 12 and train.train_positions == 6
    model = build_model(CFG)
    r = evaluate_dataset(model, hold, torch.device("cpu"))
    # A bounded evaluation uses a seeded random subset of the index, never its head.
    tr = ChunkDataset(Window([p]), symmetry=False, holdout=False)
    assert evaluate_dataset(model, tr, torch.device("cpu"))["positions"] == 6
    assert evaluate_dataset(model, tr, torch.device("cpu"), max_positions=6)["positions"] == 6
    assert evaluate_dataset(model, tr, torch.device("cpu"), max_positions=3, seed=1)["positions"] == 3
    rows = {seed: [tuple(r) for r in tr.subset_entries(3, seed)] for seed in range(6)}
    assert all(len(v) == 3 and v == sorted(v) for v in rows.values())
    assert len({tuple(v) for v in rows.values()}) > 1  # seed-dependent
    head = [tuple(r) for r in tr.index.entries[:3]]
    assert any(v != head for v in rows.values())  # not simply the oldest positions
    assert rows[1] == [tuple(r) for r in tr.subset_entries(3, 1)]  # reproducible
    assert r["positions"] == 12 and np.isfinite(r["value_mse"]) and np.isfinite(r["policy_ce"])
