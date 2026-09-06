import torch
import torch.nn as nn

from models.text_guided_queries import TextGuidedQueries


class BFTextureConditioner(nn.Module):
    def __init__(
        self,
        clip_embeddings_dim: int = 768,
        cross_attention_dim: int = 768,
        num_tokens: int = 16,
        base_channels: int = 32,
        stage_channels=None,
        stage_token_hw=(8, 8),
        texture_mode: str = "patch_resampled",
        text_guidance_dim: int = 0,
        text_guidance_heads: int = 4,
        text_guidance_max_ratio: float = 0.3,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.texture_mode = texture_mode
        self.stage_token_hw = stage_token_hw

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

        self.stage_pool = nn.AdaptiveAvgPool2d(stage_token_hw)

        token_source_dims = [clip_embeddings_dim, c1, c2, c3, c4]
        self.token_source_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, cross_attention_dim),
                nn.SiLU(),
            )
            for dim in token_source_dims
        ])

        self.resampler_queries = nn.Parameter(torch.randn(1, num_tokens, cross_attention_dim) * 0.02)
        self.resampler = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=8,
            batch_first=True,
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(cross_attention_dim, cross_attention_dim * 2),
            nn.SiLU(),
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
        )
        self.token_norm = nn.LayerNorm(cross_attention_dim)
        self.text_guidance = None
        self.text_guidance_enabled = True
        if text_guidance_dim:
            self.configure_text_guidance(text_guidance_dim, text_guidance_heads, text_guidance_max_ratio)

    def configure_text_guidance(self, text_guidance_dim=0, text_guidance_heads=4,
                                text_guidance_max_ratio=0.3):
        self.text_guidance = (
            TextGuidedQueries(self.cross_attention_dim, text_guidance_dim,
                              text_guidance_heads, text_guidance_max_ratio).to(
                device=self.resampler_queries.device, dtype=self.resampler_queries.dtype
            ) if text_guidance_dim else None
        )

    def text_guidance_config(self):
        module = self.text_guidance
        return {
            "text_guidance_dim": module.inner_dim if module is not None else 0,
            "text_guidance_heads": module.num_heads if module is not None else 4,
            "text_guidance_max_ratio": module.max_ratio if module is not None else 0.3,
        }

    def train_resampler_only(self):
        """B/C 共用冻结规则，token MLP、CNN、投影层保持 E5 权重。"""
        self.requires_grad_(False)
        self.resampler_queries.requires_grad_(True)
        self.resampler.requires_grad_(True)
        if self.text_guidance is not None:
            self.text_guidance.requires_grad_(True)

    def _stage_to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.stage_pool(feat)
        return pooled.flatten(2).transpose(1, 2)

    def _build_patch_tokens(self, clip_vision_tokens: torch.Tensor, texture_images: torch.Tensor):
        f1 = self.stage1(texture_images)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        stage_tokens = [
            clip_vision_tokens,
            self._stage_to_tokens(f1),
            self._stage_to_tokens(f2),
            self._stage_to_tokens(f3),
            self._stage_to_tokens(f4),
        ]

        projected = []
        for proj, tokens in zip(self.token_source_proj, stage_tokens):
            projected.append(proj(tokens))
        fused_tokens = torch.cat(projected, dim=1)
        return fused_tokens, [f1.shape, f2.shape, f3.shape, f4.shape]

    def _build_legacy_tokens(self, clip_image_embeds: torch.Tensor, texture_images: torch.Tensor):
        f1 = self.stage1(texture_images)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        pooled = [
            torch.mean(f1, dim=(2, 3), keepdim=False),
            torch.mean(f2, dim=(2, 3), keepdim=False),
            torch.mean(f3, dim=(2, 3), keepdim=False),
            torch.mean(f4, dim=(2, 3), keepdim=False),
        ]

        pooled_clip = clip_image_embeds.unsqueeze(1)
        pooled_cnn = [p.unsqueeze(1) for p in pooled]
        legacy_tokens = [pooled_clip] + pooled_cnn

        projected = []
        for proj, tokens in zip(self.token_source_proj, legacy_tokens):
            projected.append(proj(tokens))
        fused_tokens = torch.cat(projected, dim=1)
        return fused_tokens, [f1.shape, f2.shape, f3.shape, f4.shape]

    def forward(
        self,
        clip_image_embeds: torch.Tensor = None,
        texture_images: torch.Tensor = None,
        clip_vision_tokens: torch.Tensor = None,
        texture_mode: str = None,
        text_embeds: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        apply_text_guidance: bool = True,
    ):
        if texture_images is None:
            raise ValueError("texture_images is required.")

        mode = texture_mode or self.texture_mode

        if mode == "patch_resampled":
            if clip_vision_tokens is None:
                if clip_image_embeds is None:
                    raise ValueError("patch_resampled mode requires clip_vision_tokens or clip_image_embeds.")
                clip_vision_tokens = clip_image_embeds.unsqueeze(1)

            fused_tokens, feature_shapes = self._build_patch_tokens(clip_vision_tokens, texture_images)

        elif mode == "legacy_pooled":
            if clip_image_embeds is None:
                if clip_vision_tokens is None:
                    raise ValueError("legacy_pooled mode requires clip_image_embeds or clip_vision_tokens.")
                clip_image_embeds = clip_vision_tokens.mean(dim=1)

            fused_tokens, feature_shapes = self._build_legacy_tokens(clip_image_embeds, texture_images)

        else:
            raise ValueError(f"Unsupported texture_mode: {mode}")

        bsz = fused_tokens.shape[0]
        query = self.resampler_queries.expand(bsz, -1, -1)
        if self.text_guidance is not None and self.text_guidance_enabled and apply_text_guidance:
            query = self.text_guidance(query, text_embeds, text_mask)
        tokens, _ = self.resampler(query, fused_tokens, fused_tokens, need_weights=False)
        tokens = tokens + self.token_mlp(tokens)
        tokens = self.token_norm(tokens)
        return tokens, feature_shapes
