from typing import List, Sequence

import torch
import torch.nn as nn


class MinimalFusionBlock(nn.Module):
    """
    Minimal spatial fusion:
    concat(S_i, T_i) -> 1x1 conv -> GN -> SiLU -> 3x3 conv
    """

    def __init__(self, sketch_channels: int, texture_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(sketch_channels + texture_channels, out_channels, kernel_size=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        # TODO: replace with BF-style bidirectional modulation fusion if needed.

    def forward(self, sketch_feat: torch.Tensor, texture_feat: torch.Tensor) -> torch.Tensor:
        if sketch_feat.shape[-2:] != texture_feat.shape[-2:]:
            texture_feat = torch.nn.functional.interpolate(
                texture_feat, size=sketch_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.net(torch.cat([sketch_feat, texture_feat], dim=1))


class BFMLikeFusionBlock(nn.Module):
    """
    Lightweight BF-like bidirectional modulation fusion.
    S_mod = gamma_t * norm(S) + beta_t
    T_mod = gamma_s * norm(T) + beta_s
    F = Conv([S_mod, T_mod])
    """

    def __init__(self, sketch_channels: int, texture_channels: int, out_channels: int):
        super().__init__()
        self.norm_s = nn.GroupNorm(min(8, sketch_channels), sketch_channels)
        self.norm_t = nn.GroupNorm(min(8, texture_channels), texture_channels)

        self.s_to_gamma_beta = nn.Conv2d(sketch_channels, texture_channels * 2, kernel_size=1)
        self.t_to_gamma_beta = nn.Conv2d(texture_channels, sketch_channels * 2, kernel_size=1)

        self.out = nn.Sequential(
            nn.Conv2d(sketch_channels + texture_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, sketch_feat: torch.Tensor, texture_feat: torch.Tensor) -> torch.Tensor:
        if sketch_feat.shape[-2:] != texture_feat.shape[-2:]:
            texture_feat = torch.nn.functional.interpolate(
                texture_feat, size=sketch_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        gamma_t, beta_t = self.s_to_gamma_beta(sketch_feat).chunk(2, dim=1)
        gamma_s, beta_s = self.t_to_gamma_beta(texture_feat).chunk(2, dim=1)

        s_mod = gamma_s * self.norm_s(sketch_feat) + beta_s
        t_mod = gamma_t * self.norm_t(texture_feat) + beta_t
        return self.out(torch.cat([s_mod, t_mod], dim=1))


class MultiScaleFusion(nn.Module):
    """Fuse [S1..S4] and [T1..T4] into [F1..F4]."""

    def __init__(
        self,
        sketch_channels: Sequence[int],
        texture_channels: Sequence[int],
        out_channels: Sequence[int],
        fusion_type: str = "minimal",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.minimal_blocks = nn.ModuleList(
            [
                MinimalFusionBlock(sc, tc, oc)
                for sc, tc, oc in zip(sketch_channels, texture_channels, out_channels)
            ]
        )
        self.bfm_like_blocks = nn.ModuleList(
            [
                BFMLikeFusionBlock(sc, tc, oc)
                for sc, tc, oc in zip(sketch_channels, texture_channels, out_channels)
            ]
        )

    def set_fusion_type(self, fusion_type: str):
        if fusion_type not in ("minimal", "bfm_like"):
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.fusion_type = fusion_type

    def forward(self, sketch_feats: List[torch.Tensor], texture_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.fusion_type == "bfm_like":
            blocks = self.bfm_like_blocks
        else:
            blocks = self.minimal_blocks
        return [blk(sf, tf) for blk, sf, tf in zip(blocks, sketch_feats, texture_feats)]
