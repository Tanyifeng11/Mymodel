"""E19.2-A2：身份与几何没有共享参数；仅身份 CNN 接收身份监督。"""

import torch.nn as nn

from models.explicit_pattern import ExplicitPatternTokens
from models.frequency_pattern import FrequencyBiasedEncoder
from models.learned_pattern import LearnedPatternEncoder


class IdentityGeometryPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = LearnedPatternEncoder()
        self.geometry = FrequencyBiasedEncoder().eval().requires_grad_(False)
        self.mapper = ExplicitPatternTokens().eval().requires_grad_(False)

    def identity_tokens(self, images):
        return self.mapper(self.identity(images))

    def geometry_tokens(self, images):
        return self.mapper(self.geometry(images))

    def forward(self, images):
        # 保留分路输出；A2 不把两路平均为一个向量，也不接生成器。
        return {"identity": self.identity_tokens(images),
                "geometry": self.geometry_tokens(images)}
