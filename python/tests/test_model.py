import torch

from mango.model import AGZNet, count_parameters


def test_shapes_and_ranges():
    torch.manual_seed(0)
    for n, blocks, filters in [(5, 2, 16), (9, 6, 64)]:
        model = AGZNet(n, blocks, filters).eval()
        x = torch.randint(0, 2, (4, 17, n, n)).float()
        logits, value = model(x)
        assert logits.shape == (4, n * n + 1)
        assert value.shape == (4,)
        assert torch.all(value > -1) and torch.all(value < 1)


def test_parameter_count_9x9_default():
    model = AGZNet(9, 6, 64)
    # stem 17*64*9 + 6 blocks * 2 * 64*64*9 + heads; see DESIGN section 6.1 arithmetic
    assert 480_000 < count_parameters(model) < 500_000


def test_batch_invariance_in_eval_mode():
    torch.manual_seed(1)
    model = AGZNet(5, 2, 16).eval()
    x = torch.randint(0, 2, (8, 17, 5, 5)).float()
    with torch.no_grad():
        l8, v8 = model(x)
        for i in range(8):
            l1, v1 = model(x[i : i + 1])
            assert torch.allclose(l8[i], l1[0], atol=1e-5)
            assert torch.allclose(v8[i], v1[0], atol=1e-5)


def test_l2_parameter_selection():
    model = AGZNet(5, 1, 8)
    names = {n for n, p in model.named_parameters()}
    l2 = {id(p) for p in model.l2_parameters()}
    for name, p in model.named_parameters():
        expect = name.endswith(".weight") and "bn" not in name
        assert (id(p) in l2) == expect, name
    assert len(names) > len(l2)
    assert len(list(model.l2_parameters(all_params=True))) == len(names)
