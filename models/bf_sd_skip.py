"""原样 BF 双向融合 + SD 通道适配；跳跃连接的加法和通道不匹配跳过规则沿用 BF。"""
import torch
from torch import nn
from models.bf_original_fusion import bFMFusion


class BFSkipFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = bFMFusion(1, 3, ngf=144, bottleneck_depth=100, num_resblocks=0)
        self.channels = (320, 640, 1280, 1280)
        self.projections = nn.ModuleList([
            nn.Conv2d(a, b, 1) for a, b in zip((288, 576, 1152, 1152), self.channels)])
        # 从共同 E1 起点平滑接入；首步只有投影有梯度，随后梯度进入原始融合网络。
        for layer in self.projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.features = None
        self.conditional = False
        self.handles = []
        self.injection_counts = [0] * 4

    def condition(self, sketch, reference):
        gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
        self.features = [p(f) for p, f in zip(self.projections, self.fusion(gray, reference))]

    def clear(self):
        self.features = None

    def bind(self, unet):
        assert not self.handles
        def before(_module, args, kwargs):
            self.conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            self.injection_counts = [0] * 4
        self.handles.append(unet.register_forward_pre_hook(before, with_kwargs=True))
        for block_index, block in enumerate(unet.up_blocks):
            level = 3 - block_index
            def inject(_module, args, kwargs, index=level):
                if self.features is None or not self.conditional:
                    return args, kwargs
                skips = kwargs['res_hidden_states_tuple']
                feature = self.features[index]
                updated = []
                for skip in skips:
                    if skip.shape[1] == self.channels[index]:
                        assert skip.shape == feature.shape, (index, skip.shape, feature.shape)
                        skip = skip + feature.to(skip.dtype)
                        self.injection_counts[index] += 1
                    updated.append(skip)
                return args, dict(kwargs, res_hidden_states_tuple=tuple(updated))
            self.handles.append(block.register_forward_pre_hook(inject, with_kwargs=True))

    def unbind(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


class BFSDNetwork(nn.Module):
    def __init__(self, unet, adapter):
        super().__init__()
        self.unet, self.adapter = unet, adapter
        self.adapter.bind(unet)

    def forward(self, noisy, timestep, text, sketch, reference, sketch_states):
        self.adapter.condition(sketch, reference)
        return self.unet(noisy, timestep, encoder_hidden_states=text,
                         cross_attention_kwargs={'sa_hidden_states': sketch_states}).sample

    def trainable_state(self):
        keys = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in keys}

    def load_trainable(self, state):
        own = self.state_dict()
        assert all(name in own and own[name].shape == value.shape for name, value in state.items())
        self.load_state_dict(state, strict=False)
