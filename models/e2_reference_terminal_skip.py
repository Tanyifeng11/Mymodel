"""末步参考直连增量：绕过冻结隐藏特征，零点保持f68；非完整ControlNet。"""
import torch
from torch import nn


PARAMETERS = 772


class ReferenceTerminalSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(192, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, reference, mask):
        # reference沿用原pixel-unshuffle亮度/ab，不读目标；mask仅乘一次。
        assert reference.dtype == mask.dtype == torch.float32
        assert reference.shape[1] == 192 and mask.shape[1] == 1
        return self.output(reference)*mask
