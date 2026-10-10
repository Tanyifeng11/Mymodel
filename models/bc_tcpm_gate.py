"""边界一致的 TCPM 作用域门控。

训练阶段的 TCPM 使用 garment mask 的内侧区域，推理阶段过去直接使用
未腐蚀的 sketch mask。这个小模块把两条路径统一起来，不改变数据和模型
权重；``legacy`` 模式保留旧行为，便于同一 checkpoint 做配对对照。
"""

import torch
import torch.nn as nn

from garment_mask_utils import build_region_masks


class BoundaryConsistentTCPMGate(nn.Module):
    """将 TCPM 的推理 mask 对齐到训练时的内侧 mask。

    ``consistent`` 返回腐蚀后的服装内部；``soft`` 在边界环带保留一个
    有界的软权重；``legacy`` 原样返回输入 mask。模块没有可学习参数，
    因而不会改变 checkpoint 的加载格式。
    """

    MODES = ("legacy", "consistent", "soft")

    def __init__(self, kernel_size: int = 9, feather: float = 0.15):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.feather = float(feather)

    def forward(self, mask: torch.Tensor, mode: str = "consistent",
                feather: float = None) -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"不支持的 BC-TCPM 模式: {mode}")
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        if mode == "legacy":
            return mask
        original_ndim = mask.ndim
        if original_ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"TCPM mask 需要 [B,1,H,W] 或 [B,H,W]，得到 {tuple(mask.shape)}")
        body, boundary, _ = build_region_masks(mask, self.kernel_size)
        if mode == "consistent":
            output = body
        else:
            amount = self.feather if feather is None else float(feather)
            output = (body + max(0.0, min(1.0, amount)) * boundary).clamp(0.0, 1.0)
        return output[:, 0] if original_ndim == 3 else output
