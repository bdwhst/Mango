import json
import re

import torch

from mango.export import export_model_version, load_eager_model, load_metadata, make_model_id
from mango.model import AGZNet


def _random_bn_stats(model: AGZNet, n: int) -> None:
    # Populate BN running statistics so eval-mode BN is not the identity.
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.randint(0, 2, (16, 17, n, n)).float())
    model.eval()


def test_export_roundtrip(tmp_path):
    torch.manual_seed(3)
    n = 5
    model = AGZNet(n, 2, 16)
    _random_bn_stats(model, n)
    d = export_model_version(model, tmp_path / "models", iteration=7, komi=7.5, move_cap=50, config_fingerprint="abc")
    assert d.parent == tmp_path / "models"
    assert re.fullmatch(r"0007-[0-9a-f]{8}", d.name)
    meta = load_metadata(d)
    assert meta["model_id"] == d.name
    assert meta["board_size"] == n and meta["res_blocks"] == 2 and meta["filters"] == 16
    assert meta["komi"] == 7.5 and meta["move_cap"] == 50 and meta["feature_schema"] == 1 and meta["rules_id"] == 1
    assert meta["outputs"] == ["logits", "value"]

    scripted = torch.jit.load(str(d / "model.pt")).eval()
    eager = load_eager_model(d)
    with torch.no_grad():
        for batch in (1, 8, 64):
            x = torch.randint(0, 2, (batch, 17, n, n)).float()
            el, ev = eager(x)
            sl, sv = scripted(x)
            assert torch.allclose(el, sl, atol=1e-5)
            assert torch.allclose(ev, sv, atol=1e-5)
            # Softmax is NOT in the graph: rows of logits must not sum to 1.
            assert not torch.allclose(sl.sum(1), torch.ones(batch), atol=1e-3)

    # Same weights -> same id, and re-export is a no-op on an existing directory.
    assert make_model_id(7, model) == d.name
    d2 = export_model_version(model, tmp_path / "models", iteration=7, komi=7.5, move_cap=50)
    assert d2 == d
    # Different iteration or different weights -> different id.
    assert make_model_id(8, model) != d.name
    other = AGZNet(n, 2, 16)
    assert make_model_id(7, other) != d.name
    # No temp directories left behind.
    assert not any(p.name.startswith(".tmp-") for p in (tmp_path / "models").iterdir())


def test_export_does_not_touch_the_live_model(tmp_path):
    torch.manual_seed(5)
    model = AGZNet(5, 1, 8).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    device_before = next(model.parameters()).device
    export_model_version(model, tmp_path, iteration=1, komi=7.5, move_cap=50)
    assert model.training, "export switched the live model to eval mode"
    assert next(model.parameters()).device == device_before
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    # A subsequent training step still updates BN running statistics (train mode).
    stats_before = model.stem_bn.running_mean.clone()
    with torch.no_grad():
        model(torch.randint(0, 2, (4, 17, 5, 5)).float())
    assert not torch.equal(model.stem_bn.running_mean, stats_before)


def test_metadata_is_json_serialisable(tmp_path):
    model = AGZNet(5, 1, 8).eval()
    d = export_model_version(model, tmp_path, iteration=0, komi=7.5, move_cap=50)
    with open(d / "model.json", encoding="utf-8") as f:
        json.load(f)
