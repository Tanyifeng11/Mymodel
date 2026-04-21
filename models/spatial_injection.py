from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInjectionAdapter(nn.Module):
    """
    Inject fused multi-scale spatial features into U-Net hidden states via forward hooks.
    Injection rule: hidden_i = hidden_i + alpha_i * proj_i(F_i)
    """

    def __init__(
        self,
        unet: nn.Module,
        fusion_channels: Sequence[int] = (64, 128, 256, 256),
        target_channels: Sequence[int] = (320, 640, 1280, 1280),
        alphas: Sequence[float] = (1.0, 1.0, 0.7, 0.5),
    ):
        super().__init__()
        object.__setattr__(self, "unet", unet)
        self.proj = nn.ModuleList(
            [nn.Conv2d(cin, cout, kernel_size=1) for cin, cout in zip(fusion_channels, target_channels)]
        )
        self.alphas = list(alphas)
        self._fused_features: Optional[List[torch.Tensor]] = None
        self._enabled = False
        self._hooks = []


    def trainable_parameters(self):
        return self.proj.parameters()

    def set_alphas(self, alphas: Sequence[float]):
        self.alphas = list(alphas)

    def set_features(self, fused_features: Optional[List[torch.Tensor]]):
        self._fused_features = fused_features

    def clear_features(self):
        self._fused_features = None

    def enable(self):
        if self._enabled:
            return
        self._enabled = True
        targets = [
            self.unet.down_blocks[0],
            self.unet.down_blocks[1],
            self.unet.down_blocks[2],
            self.unet.mid_block,
        ]
        for idx, module in enumerate(targets):
            self._hooks.append(module.register_forward_hook(self._make_hook(idx)))

    def disable(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._enabled = False

    def _make_hook(self, idx: int):
        def _hook(_module, _inputs, output):
            if self._fused_features is None:
                return output
            if idx >= len(self._fused_features):
                return output

            feat = self._fused_features[idx]
            hidden = output[0] if isinstance(output, tuple) else output
            proj_feat = self.proj[idx](feat)
            if proj_feat.shape[-2:] != hidden.shape[-2:]:
                proj_feat = F.interpolate(
                    proj_feat, size=hidden.shape[-2:], mode="bilinear", align_corners=False
                )
            mixed = hidden + self.alphas[idx] * proj_feat

            if isinstance(output, tuple):
                return (mixed, *output[1:])
            return mixed

        return _hook
