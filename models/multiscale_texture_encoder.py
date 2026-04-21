from typing import List, Sequence

import torch
import torch.nn as nn


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleTextureEncoder(nn.Module):
    """
    Lightweight 4-scale texture encoder.
    Input:  [B, 3, H, W]
    Output: [T1, T2, T3, T4]
      - T1: [B, C1, H,   W]
      - T2: [B, C2, H/2, W/2]
      - T3: [B, C3, H/4, W/4]
      - T4: [B, C4, H/8, W/8]
    """

    def __init__(self, in_channels: int = 3, stage_channels: Sequence[int] = (64, 128, 256, 256)):
        super().__init__()
        c1, c2, c3, c4 = stage_channels
        self.stage1 = nn.Sequential(_ConvGNAct(in_channels, c1, stride=1), _ConvGNAct(c1, c1, stride=1))
        self.stage2 = nn.Sequential(_ConvGNAct(c1, c2, stride=2), _ConvGNAct(c2, c2, stride=1))
        self.stage3 = nn.Sequential(_ConvGNAct(c2, c3, stride=2), _ConvGNAct(c3, c3, stride=1))
        self.stage4 = nn.Sequential(_ConvGNAct(c3, c4, stride=2), _ConvGNAct(c4, c4, stride=1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        t1 = self.stage1(x)
        t2 = self.stage2(t1)
        t3 = self.stage3(t2)
        t4 = self.stage4(t3)
        return [t1, t2, t3, t4]


class MultiScaleSketchEncoder(MultiScaleTextureEncoder):
    """Sketch counterpart of MultiScaleTextureEncoder."""

