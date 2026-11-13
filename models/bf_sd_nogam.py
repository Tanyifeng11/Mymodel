"""无 GAM 的 BFM 通路；启用开关独立于注意力缓存。"""
from models.bf_sd_skip import BFSkipFusion, BFSDNetwork


class NoGAMSkipFusion(BFSkipFusion):
    def __init__(self):
        super().__init__()
        self.enabled = True

    def conditioning_enabled(self, kwargs):
        return self.enabled


class NoGAMNetwork(BFSDNetwork):
    def forward(self, noisy, timestep, text, sketch, reference):
        self.adapter.enabled = True
        self.adapter.condition(sketch, reference)
        value = self.unet(noisy, timestep, encoder_hidden_states=text).sample
        assert self.adapter.injection_counts == [3, 2, 2, 3], self.adapter.injection_counts
        return value
