"""Model versions (docs/DESIGN.md section 6.4).

A model version is an immutable directory `<root>/<model_id>/` holding:
  model.pt     TorchScript module (traced at batch 8, eval mode, fp32, CPU) returning
               (logits [B, N*N+1], value [B])
  weights.pt   eager state dict (for the ladder and re-export)
  model.json   metadata: identity, architecture, and the game contract the model
               was trained under (board size, feature schema, rules, komi, move cap)

model_id = "<iteration:04d>-<first 8 hex of sha256 over the parameter bytes>", so a
re-export of the same weights yields the same id.
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


def build_metadata(model: AGZNet, model_id: str, komi: float, move_cap: int, config_fingerprint: str = "") -> dict[str, Any]:
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
) -> Path:
    """Writes `<root>/<model_id>/` atomically and returns its path.

    If the directory already exists (same weights re-exported) it is left untouched.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if model_id is None:
        model_id = make_model_id(iteration, model)
    final = root / model_id
    if final.exists():
        return final
    tmp = root / f".tmp-{model_id}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        exported = export_copy(model)
        traced = trace_model(exported)
        traced.save(str(tmp / "model.pt"))
        torch.save(exported.state_dict(), tmp / "weights.pt")
        meta = build_metadata(exported, model_id, komi, move_cap, config_fingerprint)
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
