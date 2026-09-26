"""E20：冻结三路表示后的小型条件接口；零初始化保证起点等于 E19。"""

import torch
from torch import nn


class ResidualMap(nn.Module):
    def __init__(self, dim=768, rank=32):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(torch.nn.functional.gelu(self.down(x.float()))).to(x.dtype)


class PatternInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = ResidualMap()
        self.geometry = ResidualMap()

    def forward(self, tokens):
        # 输入是冻结 TCPM 的输出，前面的 BF appearance tokens 逐位保持。
        identity, geometry = tokens[:, -8:-4], tokens[:, -4:]
        return torch.cat([tokens[:, :-8], identity + self.identity(identity),
                          geometry + self.geometry(geometry)], dim=1)


class PatternKVLoRA(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.adapter_down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.adapter_up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, tokens):
        value = self.base(tokens)
        # 只改变 pattern tokens 的 K/V；appearance 的原始投影保持不变。
        delta = self.adapter_up(self.adapter_down(tokens[:, -8:].float())).to(value.dtype)
        return torch.cat([value[:, :-8], value[:, -8:] + delta], dim=1)
