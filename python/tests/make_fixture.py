"""Generates the cross-language NN backend fixture used by cpp/tests/test_torch_eval.cpp.

    python python/tests/make_fixture.py

Writes cpp/tests/fixtures/model_5x5_v1/{model.pt, weights.pt, model.json, reference.json}.
The fixture is committed; regenerate only when the model format changes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mango.export import export_model_version  # noqa: E402
from mango.model import AGZNet  # noqa: E402


def main() -> None:
    torch.manual_seed(1234)
    n = 5
    model = AGZNet(n, 2, 16)
    # Populate BN running statistics with random data so eval BN is non-trivial.
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randint(0, 2, (32, 17, n, n)).float())
    model.eval()

    out_root = ROOT / "cpp" / "tests" / "fixtures"
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / "model_5x5_v1"
    if final.exists():
        shutil.rmtree(final)
    d = export_model_version(model, out_root, iteration=0, komi=7.5, move_cap=50, config_fingerprint="fixture")
    d.rename(final)

    # Reference inputs: 8 random binary boards; plane 16 constant per sample.
    g = torch.Generator().manual_seed(99)
    x = torch.randint(0, 2, (8, 17, n, n), generator=g)
    for i in range(8):
        x[i, 16] = i % 2
    with torch.no_grad():
        logits, value = model(x.float())
    ref = {
        "board_size": n,
        "planes": 17,
        "inputs": x.flatten(1).tolist(),
        "logits": logits.tolist(),
        "values": value.tolist(),
    }
    with open(final / "reference.json", "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print("wrote", final)


if __name__ == "__main__":
    main()
