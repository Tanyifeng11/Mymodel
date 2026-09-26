"""E19-B：从灰度局部像素学习方向/周期表示，固定 A 的 token 映射。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPatternEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 128), nn.SiLU(), nn.Linear(128, 36))

    def forward(self, images):
        images = F.interpolate(images.float(), (128, 128), mode="bilinear", align_corners=False)
        gray = (images * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
        patches = torch.stack([gray[:, y:y + 64, x:x + 64]
                               for y in (0, 64) for x in (0, 64)], 1)
        patches = patches - patches.mean((-2, -1), keepdim=True)
        patches = patches / patches.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        output = self.layers(patches.reshape(-1, 1, 64, 64))
        return F.normalize(output.reshape(len(images), 4, 36), dim=-1)
