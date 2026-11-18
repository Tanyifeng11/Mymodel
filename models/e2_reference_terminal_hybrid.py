"""保留冻结特征后的4616增量头，叠加772参考直连；仅实验末步使用。"""
from torch import nn
from models.e2_reference_terminal_skip import ReferenceTerminalSkip


PARAMETERS = 5388


class ReferenceTerminalHybrid(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.head.base.eval().requires_grad_(False)
        self.skip = ReferenceTerminalSkip()

    def increment_parameters(self):
        return self.head.increment_parameters()+list(self.skip.parameters())

    def difference(self, feature, mask, phi, reference, base_delta):
        # 原特征增量、参考直连均在同一f68零点；各输出的mask仅乘一次。
        return self.head.from_features(feature,mask,phi)-base_delta+self.skip(reference,mask)
