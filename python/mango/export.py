"""Model versions (docs/DESIGN.md section 6.4).

A model version is an immutable directory `<root>/<model_id>/` holding:
  model.pt     TorchScript module (traced at batch 8, eval mode, fp32, CPU) returning
               (logits [B, N*N+1], value [B])
  weights.pt   eager state dict (for the ladder and re-export)
  model.json   metadata: identity, architecture, and the game contract the model
               was trained under (board size, feature schema, rules, komi, move cap)

model_id = "<iteration:04d>-<first 8 hex of sha256 over the parameter bytes>", so a
re-export of the same weights yields the same id.

BatchNorm folding (DESIGN 6.4, M4a step 3): by default every Conv2d -> BatchNorm2d pair is
fused into one convolution with a bias before tracing, so model.pt has no batch-norm
nodes; model.json records `bn_folded`. weights.pt always keeps the unfolded state dict.
The switch affects new exports only: an existing directory is returned untouched when its
`bn_folded` (absent = false) matches the request and raises ExportVariantMismatch otherwise,
so an A/B export of the two variants of one set of weights uses two different roots:

    python -m mango.export --model-dir runs/R/models/0003-xxxxxxxx --out models_unfolded --no-bn-fold
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .model import NUM_PLANES, AGZNet

MODEL_FORMAT = "torchscript-v1"


class ExportVariantMismatch(ValueError):
    """The model version directory exists with the other `bn_folded` variant."""
FEATURE_SCHEMA = 1
RULES_ID = 1


def weights_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model_id(iteration: int, model: torch.nn.Module) -> str:
    return f"{iteration:04d}-{weights_digest(model)[:8]}"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - best effort
        return ""


def build_metadata(model: AGZNet, model_id: str, komi: float, move_cap: int, config_fingerprint: str = "",
                   bn_folded: bool = False) -> dict[str, Any]:
    return {
        "format": MODEL_FORMAT,
        "model_id": model_id,
        "board_size": model.board_size,
        "planes": model.planes,
        "feature_schema": FEATURE_SCHEMA,
        "rules_id": RULES_ID,
        "komi": float(komi),
        "move_cap": int(move_cap),
        "res_blocks": model.res_blocks,
        "filters": model.filters,
        "export_dtype": "float32",
        "bn_folded": bool(bn_folded),
        "outputs": ["logits", "value"],
        "config_fingerprint": config_fingerprint,
        "torch_version": torch.__version__,
        "git_hash": git_hash(),
        "export_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def export_copy(model: AGZNet) -> AGZNet:
    """A CPU, fp32, eval-mode deep copy. The live (possibly training) model is untouched."""
    return copy.deepcopy(model).cpu().float().eval()


@torch.no_grad()
def fold_batchnorm(model: AGZNet) -> AGZNet:
    """A copy of an eval-mode model with every Conv2d -> BatchNorm2d pair fused into one
    convolution with a bias (W' = W * g/sqrt(var+eps), b' = beta - mean * g/sqrt(var+eps));
    the BN modules become Identity. Forward outputs equal the input model's up to fp32
    rounding (DESIGN 6.4). The input model is untouched."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    m = export_copy(model)
    m.stem_conv = fuse_conv_bn_eval(m.stem_conv, m.stem_bn)
    m.stem_bn = torch.nn.Identity()
    for block in m.tower:
        block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()
    m.policy_conv = fuse_conv_bn_eval(m.policy_conv, m.policy_bn)
    m.policy_bn = torch.nn.Identity()
    m.value_conv = fuse_conv_bn_eval(m.value_conv, m.value_bn)
    m.value_bn = torch.nn.Identity()
    return m


@torch.no_grad()
def trace_model(model: AGZNet) -> torch.jit.ScriptModule:
    """Traces a copy of the model at batch 8 and verifies the trace at batch 1 and 64."""
    model = export_copy(model)
    n = model.board_size
    example = torch.zeros(8, model.planes, n, n)
    traced = torch.jit.trace(model, example)
    for batch in (1, 64):
        x = torch.randint(0, 2, (batch, model.planes, n, n)).float()
        el, ev = model(x)
        tl, tv = traced(x)
        if not torch.allclose(el, tl, atol=1e-5) or not torch.allclose(ev, tv, atol=1e-5):
            raise RuntimeError(f"traced model disagrees with eager model at batch {batch}")
        if tl.shape != (batch, n * n + 1) or tv.shape != (batch,):
            raise RuntimeError("unexpected output shapes from the traced model")
    return traced


def export_model_version(
    model: AGZNet,
    root: str | Path,
    iteration: int,
    komi: float,
    move_cap: int,
    config_fingerprint: str = "",
    model_id: str | None = None,
    bn_fold: bool = True,
) -> Path:
    """Writes `<root>/<model_id>/` atomically and returns its path.

    If the directory already exists (same weights re-exported) it is left untouched,
    provided its `bn_folded` (absent = False) matches `bn_fold`; otherwise
    ExportVariantMismatch is raised rather than handing back the other variant.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if model_id is None:
        model_id = make_model_id(iteration, model)
    final = root / model_id
    if final.exists():
        existing = bool(load_metadata(final).get("bn_folded", False))
        if existing != bool(bn_fold):
            raise ExportVariantMismatch(
                f"{final} exists with bn_folded={existing}, requested bn_fold={bool(bn_fold)}; "
                f"export the other variant into a different root")
        return final
    tmp = root / f".tmp-{model_id}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        exported = export_copy(model)
        traced = trace_model(fold_batchnorm(exported) if bn_fold else exported)
        traced.save(str(tmp / "model.pt"))
        torch.save(exported.state_dict(), tmp / "weights.pt")  # always the unfolded weights
        meta = build_metadata(exported, model_id, komi, move_cap, config_fingerprint, bn_folded=bn_fold)
        with open(tmp / "model.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def load_metadata(model_dir: str | Path) -> dict[str, Any]:
    with open(Path(model_dir) / "model.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_eager_model(model_dir: str | Path) -> AGZNet:
    meta = load_metadata(model_dir)
    model = AGZNet(meta["board_size"], meta["res_blocks"], meta["filters"], meta.get("planes", NUM_PLANES))
    model.load_state_dict(torch.load(Path(model_dir) / "weights.pt", map_location="cpu"))
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    """Re-exports a model version's weights into another root, folded or not (A/B exports)."""
    import argparse

    ap = argparse.ArgumentParser(description="Re-export a Mango model version (DESIGN 6.4)")
    ap.add_argument("--model-dir", required=True, help="an existing models/<model_id>/ directory")
    ap.add_argument("--out", required=True, help="root for the new version (use a different root per variant)")
    ap.add_argument("--no-bn-fold", action="store_true", help="export with the BatchNorm layers kept (today's graph)")
    ap.add_argument("--iteration", type=int, default=None, help="iteration prefix of the new id (default: the source's)")
    args = ap.parse_args(argv)
    meta = load_metadata(args.model_dir)
    model = load_eager_model(args.model_dir)
    iteration = args.iteration if args.iteration is not None else int(meta["model_id"].split("-")[0])
    d = export_model_version(model, args.out, iteration=iteration, komi=meta["komi"], move_cap=meta["move_cap"],
                             config_fingerprint=meta.get("config_fingerprint", ""), bn_fold=not args.no_bn_fold)
    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
