"""AlphaGo Zero network (docs/DESIGN.md section 6.1).

Initial convolutional block + `res_blocks` residual blocks (config counts residual
blocks only), a policy head producing N*N+1 logits (last = pass) and a value head
producing tanh(v) in (-1, 1). `forward` returns (logits, value); the softmax is the
C++ evaluator's job (in fp32), never part of the exported graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLANES = 17


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AGZNet(nn.Module):
    def __init__(self, board_size: int, res_blocks: int, filters: int, planes: int = NUM_PLANES):
        super().__init__()
        if not (2 <= board_size <= 19):
            raise ValueError("board_size must be in [2, 19]")
        self.board_size = board_size
        self.res_blocks = res_blocks
        self.filters = filters
        self.planes = planes
        n2 = board_size * board_size

        self.stem_conv = nn.Conv2d(planes, filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(res_blocks)])

        self.policy_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * n2, n2 + 1)

        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(n2, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        h = self.tower(h)
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        logits = self.policy_fc(p.flatten(1))
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v)).squeeze(1)
        return logits, value

    def l2_parameters(self, all_params: bool = False):
        """Parameters subject to L2 regularisation (DESIGN D9)."""
        for name, p in self.named_parameters():
            if all_params:
                yield p
            elif name.endswith(".weight") and ("bn" not in name):
                yield p


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
