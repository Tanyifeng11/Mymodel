import torch
import torch.nn as nn


class BFTextureConditioner(nn.Module):
    """
    BF-style texture-only conditioner (engineering approximation).

    This module extracts 4-scale texture features from the texture image,
    pools them into a compact representation, and fuses them with CLIP texture
    embeddings to generate context tokens for UNet cross-attention.
    """

    def __init__(
        self,
        clip_embeddings_dim: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        base_channels: int = 32,
        stage_channels=None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        if stage_channels is not None:
            c1, c2, c3, c4 = stage_channels
        else:
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c2),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c3),
            nn.SiLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c4),
            nn.SiLU(),
        )

        fused_dim = clip_embeddings_dim + c1 + c2 + c3 + c4
        self.token_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.SiLU(),
            nn.Linear(fused_dim, num_tokens * cross_attention_dim),
        )
        self.token_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_image_embeds: torch.Tensor, texture_images: torch.Tensor):
        f1 = self.stage1(texture_images)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        pooled = [
            torch.mean(f1, dim=(2, 3)),
            torch.mean(f2, dim=(2, 3)),
            torch.mean(f3, dim=(2, 3)),
            torch.mean(f4, dim=(2, 3)),
        ]
        texture_multi_scale_embed = torch.cat(pooled, dim=1)

        fused = torch.cat([clip_image_embeds, texture_multi_scale_embed], dim=1)
        tokens = self.token_mlp(fused).view(-1, self.num_tokens, self.cross_attention_dim)
        tokens = self.token_norm(tokens)

        feature_shapes = [f1.shape, f2.shape, f3.shape, f4.shape]
        return tokens, feature_shapes
