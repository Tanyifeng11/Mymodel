import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_guided_queries import TextGuidedQueries
from models.text_texture_film import TextTextureFiLM
from models.nexus_texture_adapter import NexusTextureAdapter


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
        film_hidden_dim: int = 0,
        nexus_dim: int = 0,
        nexus_heads: int = 4,
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
        self.film = None
        self.film_enabled = True
        self.configure_film(film_hidden_dim)
        self.nexus = None
        self.nexus_enabled = True
        self.configure_nexus(nexus_dim, nexus_heads)

    def configure_nexus(self, nexus_dim=0, nexus_heads=4):
        if nexus_dim and (self.film is not None or self.text_guidance is not None):
            raise ValueError("E12 不叠加 FiLM 或文本 query 模块")
        self.nexus = (
            NexusTextureAdapter(self.cross_attention_dim, self.stage3[1].num_channels,
                                nexus_dim, nexus_heads).to(
                device=self.resampler_queries.device, dtype=self.resampler_queries.dtype
            ) if nexus_dim else None
        )

    def nexus_config(self):
        return {"nexus_dim": self.nexus.inner_dim if self.nexus is not None else 0,
                "nexus_heads": self.nexus.num_heads if self.nexus is not None else 4}

    def train_nexus_only(self):
        if self.nexus is None:
            raise ValueError("E12 训练需要先创建 Nexus Adapter")
        self.requires_grad_(False)
        self.nexus.requires_grad_(True)

    def configure_film(self, film_hidden_dim=0):
        if film_hidden_dim < 0:
            raise ValueError("film_hidden_dim 不能为负数")
        self.film = (
            TextTextureFiLM(self.cross_attention_dim, self.stage3[1].num_channels, film_hidden_dim).to(
                device=self.resampler_queries.device, dtype=self.resampler_queries.dtype
            ) if film_hidden_dim else None
        )

    def film_config(self):
        return {"film_hidden_dim": self.film.hidden_dim if self.film is not None else 0}

    def train_film_only(self):
        """冻结原 E5 所有纹理参数，仅训练压缩前的文本 FiLM。"""
        if self.film is None:
            raise ValueError("film_only 训练需要先创建 FiLM 模块")
        self.requires_grad_(False)
        self.film.requires_grad_(True)

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

    def train_text_guidance_only(self):
        """E8c：原 E5 包括 query/resampler 全部冻结，只训练新增文本模块。"""
        if self.text_guidance is None:
            raise ValueError("text_only 训练需要先创建文本查询模块")
        self.requires_grad_(False)
        self.text_guidance.requires_grad_(True)

    def _stage_to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = self.stage_pool(feat)
        return pooled.flatten(2).transpose(1, 2)

    def _encode_texture_features(self, texture_images, text_embeds=None, text_mask=None, apply_film=True,
                                 nexus_text_embeds=None, apply_nexus=True):
        f1 = self.stage1(texture_images)
        f2 = self.stage2(f1)
        # 保留 stage3 原始 Sequential 与权重键，在 GN 和 SiLU 之间插入 FiLM。
        if self.film is not None and self.film_enabled and apply_film:
            f3 = self.stage3[1](self.stage3[0](f2))
            f3 = self.stage3[2](self.film(f3, text_embeds, text_mask))
        else:
            f3 = self.stage3(f2)
        if self.nexus is not None and self.nexus_enabled and apply_nexus:
            f3 = self.nexus(f3, text_embeds if nexus_text_embeds is None else nexus_text_embeds)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4

    def _build_patch_tokens(self, clip_vision_tokens: torch.Tensor, texture_images: torch.Tensor,
                            local_detail_grid=None, text_embeds=None, text_mask=None, apply_film=True,
                            nexus_text_embeds=None, apply_nexus=True):
        f1, f2, f3, f4 = self._encode_texture_features(
            texture_images, text_embeds, text_mask, apply_film, nexus_text_embeds, apply_nexus)

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
        if local_detail_grid is not None:
            local = F.adaptive_avg_pool2d(f3, (local_detail_grid, local_detail_grid))
            local = self.token_source_proj[3](local.flatten(2).transpose(1, 2))
            return fused_tokens, [f1.shape, f2.shape, f3.shape, f4.shape], local
        return fused_tokens, [f1.shape, f2.shape, f3.shape, f4.shape]

    def _build_legacy_tokens(self, clip_image_embeds: torch.Tensor, texture_images: torch.Tensor,
                             local_detail_grid=None, text_embeds=None, text_mask=None, apply_film=True,
                             nexus_text_embeds=None, apply_nexus=True):
        f1, f2, f3, f4 = self._encode_texture_features(
            texture_images, text_embeds, text_mask, apply_film, nexus_text_embeds, apply_nexus)

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
        if local_detail_grid is not None:
            local = F.adaptive_avg_pool2d(f3, (local_detail_grid, local_detail_grid))
            local = self.token_source_proj[3](local.flatten(2).transpose(1, 2))
            return fused_tokens, [f1.shape, f2.shape, f3.shape, f4.shape], local
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
        local_detail_source: str = "off",
        local_detail_grid: int = 16,
        apply_film: bool = True,
        nexus_text_embeds: torch.Tensor = None,
        apply_nexus: bool = True,
    ):
        if texture_images is None:
            raise ValueError("texture_images is required.")

        mode = texture_mode or self.texture_mode
        if local_detail_source not in {"off", "resampled", "local"}:
            raise ValueError(f"不支持的 local_detail_source：{local_detail_source}")
        if local_detail_source == "local" and local_detail_grid <= 0:
            raise ValueError("local_detail_grid 必须大于零")
        grid = local_detail_grid if local_detail_source == "local" else None

        if mode == "patch_resampled":
            if clip_vision_tokens is None:
                if clip_image_embeds is None:
                    raise ValueError("patch_resampled mode requires clip_vision_tokens or clip_image_embeds.")
                clip_vision_tokens = clip_image_embeds.unsqueeze(1)

            built = self._build_patch_tokens(clip_vision_tokens, texture_images, grid,
                                             text_embeds, text_mask, apply_film, nexus_text_embeds, apply_nexus)

        elif mode == "legacy_pooled":
            if clip_image_embeds is None:
                if clip_vision_tokens is None:
                    raise ValueError("legacy_pooled mode requires clip_image_embeds or clip_vision_tokens.")
                clip_image_embeds = clip_vision_tokens.mean(dim=1)

            built = self._build_legacy_tokens(clip_image_embeds, texture_images, grid,
                                              text_embeds, text_mask, apply_film, nexus_text_embeds, apply_nexus)

        else:
            raise ValueError(f"Unsupported texture_mode: {mode}")

        fused_tokens, feature_shapes = built[:2]
        bsz = fused_tokens.shape[0]
        query = self.resampler_queries.expand(bsz, -1, -1)
        if self.text_guidance is not None and self.text_guidance_enabled and apply_text_guidance:
            query = self.text_guidance(query, text_embeds, text_mask)
        capture_attention = bool(getattr(self, "probe_capture_resampler_attention", False))
        tokens, resampler_attention = self.resampler(
            query, fused_tokens, fused_tokens, need_weights=capture_attention)
        if capture_attention:
            # Only installed by diagnostic scripts: head-averaged [B, num_queries, num_sources].
            self.last_resampler_attention = resampler_attention.detach()
        tokens = tokens + self.token_mlp(tokens)
        tokens = self.token_norm(tokens)
        if local_detail_source != "off":
            # A 组复用原始 16 token；B 组绕过 8×8 stage_pool 和 resampler。
            local_tokens = tokens if local_detail_source == "resampled" else built[2]
            return tokens, feature_shapes, local_tokens
        return tokens, feature_shapes
