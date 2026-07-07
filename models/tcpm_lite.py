import torch
import torch.nn as nn

class TCPMLite(nn.Module):
    def __init__(self, hidden_dim: int, hidden_ratio: float = 0.25, residual_scale_init: float = 0.0):
        super().__init__()
        mid_dim = max(32, int(hidden_dim * hidden_ratio))
        self.texture_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_stats = {}

    def forward(self, texture_tokens: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        if texture_tokens is None or text_embeds is None:
            return texture_tokens

        texture_importance = texture_tokens.detach().float().abs().mean(dim=1)
        texture_logits = self.texture_proj(texture_importance.to(dtype=texture_tokens.dtype))
        text_pool = text_embeds.float().mean(dim=1).to(dtype=texture_tokens.dtype)
        text_logits = self.text_mlp(text_pool)

        gate = torch.sigmoid(texture_logits + text_logits).unsqueeze(1)
        residual = texture_tokens * gate
        out = texture_tokens + self.residual_scale.to(dtype=texture_tokens.dtype) * residual

        with torch.no_grad():
            self.last_stats = {
                "residual_scale": float(self.residual_scale.detach().float().cpu().item()),
                "gate_mean": float(gate.detach().float().mean().cpu().item()),
                "gate_std": float(gate.detach().float().std(unbiased=False).cpu().item()),
                "residual_norm": float(residual.detach().float().norm(dim=-1).mean().cpu().item()),
            }
        return out
