"""Small self-supervised pattern affinity projection."""

import torch
from torch import nn
from torch.nn import functional as F


class PatternAffinityAdapter(nn.Module):
    def __init__(self, extra_dim, use_geometry=True):
        super().__init__()
        self.use_geometry = use_geometry
        self.input_dim = 384 + extra_dim - (4 if not use_geometry else 0)
        self.pattern = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64))
        self.dino = nn.Linear(384, 64, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if not self.use_geometry:
            x = torch.cat((x[..., :384], x[..., 388:]), dim=-1)
        return F.normalize(self.pattern(x) + self.alpha * self.dino(x[..., :384]), dim=-1)
