"""冻结 RF2 输入保持原协议；文本不会参与方向场计算。"""
from torch import nn

class TriModalField(nn.Module):
    def __init__(self, rf2):
        super().__init__()
        self.rf2 = rf2.eval().requires_grad_(False)

    @property
    def prior(self):
        return self.rf2.prior

    def forward(self, reference, structure, text=None):
        return self.rf2(reference, structure)
