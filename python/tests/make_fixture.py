"""Generates the cross-language NN backend fixtures used by cpp/tests/test_torch_eval.cpp.

    python python/tests/make_fixture.py          # model_5x5_v1_folded from the committed model_5x5_v1
    python python/tests/make_fixture.py --all    # regenerate model_5x5_v1 too (changes its bytes)

model_5x5_v1: {model.pt, weights.pt, model.json, reference.json}, exported with the
BatchNorm layers kept (bn_folded absent, DESIGN 6.4). model_5x5_v1_folded: the same weights
exported with BatchNorm folded into the convolutions, with a copy of the same
reference.json. Both are committed; regenerate only when the model format changes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mango.export import export_model_version, load_eager_model, load_metadata  # noqa: E402
from mango.model import AGZNet  # noqa: E402

FIXTURES = ROOT / "cpp" / "tests" / "fixtures"


def make_folded() -> None:
    """model_5x5_v1_folded: the committed model_5x5_v1 weights, exported with bn_fold=True."""
    src = FIXTURES / "model_5x5_v1"
    dst = FIXTURES / "model_5x5_v1_folded"
    if dst.exists():
        shutil.rmtree(dst)
    meta = load_metadata(src)
    model = load_eager_model(src)
    tmp_root = FIXTURES / ".folded-tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    d = export_model_version(model, tmp_root, iteration=0, komi=meta["komi"], move_cap=meta["move_cap"],
                             config_fingerprint="fixture-folded", bn_fold=True)
    assert d.name == meta["model_id"], (d.name, meta["model_id"])  # same weights, same id
    d.rename(dst)
    shutil.rmtree(tmp_root)
    shutil.copy(src / "reference.json", dst / "reference.json")
    # The folded graph must reproduce the stored reference within the fp32 tolerance.
    ref = json.loads((dst / "reference.json").read_text(encoding="utf-8"))
    scripted = torch.jit.load(str(dst / "model.pt")).eval()
    with torch.no_grad():
        x = torch.tensor(ref["inputs"], dtype=torch.float32).view(-1, ref["planes"], ref["board_size"], ref["board_size"])
        logits, value = scripted(x)
    err = max(float((logits - torch.tensor(ref["logits"])).abs().max()), float((value - torch.tensor(ref["values"])).abs().max()))
    assert err < 1e-4, err
    print("wrote", dst, f"(max abs error vs reference {err:.1e})")


def main() -> None:
    if "--all" not in sys.argv[1:]:
        make_folded()
        return
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
    d = export_model_version(model, out_root, iteration=0, komi=7.5, move_cap=50, config_fingerprint="fixture",
                             bn_fold=False)
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
    make_folded()


if __name__ == "__main__":
    main()
