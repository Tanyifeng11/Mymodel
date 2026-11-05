"""文本引导原 BF 双向交互；保留旧权重键名、四尺度及 CFG 注入规则。"""
import torch
from torch import nn

from models.bf_original_fusion import bFMFusion, affine_transformation
from models.bf_sd_skip import BFSkipFusion, BFSDNetwork


class ResidualTextFiLM(nn.Module):
    def __init__(self, text_dim, channels, hidden_dim=128, rho=.1):
        super().__init__()
        self.rho = rho
        self.head = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 2 * channels))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.last = None

    def forward(self, value, pooled):
        gamma, beta = self.head(pooled).to(value.dtype).chunk(2, -1)
        self.last = (gamma.detach(), beta.detach())
        return value + self.rho * (gamma.tanh()[..., None, None] * value + beta[..., None, None])


class TGTFMFusion(bFMFusion):
    def __init__(self, variant='T3', ngf=144, bottleneck_depth=100, text_dim=768, hidden_dim=128):
        super().__init__(1, 3, ngf=ngf, bottleneck_depth=bottleneck_depth, num_resblocks=0)
        if variant not in ('T0', 'T1', 'T2', 'T3'):
            raise ValueError('第一阶段只允许 T0、T1、T2、T3')
        self.variant = variant
        widths = (ngf * 2, ngf * 4, ngf * 8, ngf * 8)
        branches = {'T0': (), 'T1': ('sketch',), 'T2': ('reference',),
                    'T3': ('sketch', 'reference')}[variant]
        self.text_modulators = nn.ModuleDict()
        for branch in branches:
            # 单分支/双分支消融中的同名新头也采用相同初始化。
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4242 if branch == 'sketch' else 4243)
                self.text_modulators[branch] = nn.ModuleList(
                    ResidualTextFiLM(text_dim, c, hidden_dim) for c in widths)

    def forward(self, sketch, reference, text=None, eos_indices=None):
        if self.variant == 'T0':
            return super().forward(sketch, reference)
        if text is None or eos_indices is None or len(text) != len(sketch):
            raise ValueError('文本调制需要同一前向的 CLIP tokens 及真实 EOS 索引')
        pooled = text[torch.arange(len(text), device=text.device), eos_indices]
        s = self.downconv_channel_unchanged(self.resblocks(self.downconv1(sketch)))
        r = self.G_downconv_channel_unchanged(self.G_resblocks(self.G_downconv1(reference)))
        outputs = []
        for index, level in enumerate(range(2, 6)):
            s = getattr(self, 'downconv%d' % level)(s)
            r = getattr(self, 'G_downconv%d' % level)(r)
            if 'sketch' in self.text_modulators:
                s = self.text_modulators['sketch'][index](s, pooled)
            if 'reference' in self.text_modulators:
                r = self.text_modulators['reference'][index](r, pooled)
            # 原算法先用两路未更新特征生成参数，再分别更新；不得交换次序。
            g_alpha, g_beta = self.get_FiLM_param_(r, str(level), texture=True)
            i_alpha, i_beta = self.get_FiLM_param_(s, str(level))
            s = affine_transformation(s, g_alpha, g_beta)
            r = affine_transformation(r, i_alpha, i_beta)
            outputs.append(s)
        return outputs

    def stats(self):
        result = {}
        for branch, heads in self.text_modulators.items():
            for index, head in enumerate(heads):
                if head.last is not None:
                    gamma, beta = head.last
                    result[f'{branch}{index}_gamma'] = float(gamma.float().abs().mean())
                    result[f'{branch}{index}_beta'] = float(beta.float().abs().mean())
        return result


class TGTFMAdapter(BFSkipFusion):
    def __init__(self, variant='T3', ngf=144, bottleneck_depth=100, text_dim=768,
                 hidden_dim=128, channels=(320, 640, 1280, 1280), eos_token_id=49407):
        nn.Module.__init__(self)
        self.config = dict(variant=variant, ngf=ngf, bottleneck_depth=bottleneck_depth,
                           text_dim=text_dim, hidden_dim=hidden_dim, channels=list(channels),
                           eos_token_id=eos_token_id)
        self.fusion = TGTFMFusion(variant, ngf, bottleneck_depth, text_dim, hidden_dim)
        self.channels = tuple(channels)
        self.projections = nn.ModuleList(nn.Conv2d(c, out, 1) for c, out in zip(
            (ngf * 2, ngf * 4, ngf * 8, ngf * 8), channels))
        for layer in self.projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.handles = []
        self.features = self.context = None
        self.conditional = False
        self.injection_counts = [0] * 4
        self.conditional_calls = self.unconditional_calls = 0

    def condition(self, sketch, reference, text_ids):
        self.clear()
        if text_ids.ndim != 2 or len(text_ids) != len(sketch):
            raise ValueError('文本 ID 与图像 batch 不一致')
        eos = text_ids.eq(self.config['eos_token_id'])
        if not eos.any(1).all():
            raise ValueError('文本缺少 tokenizer 的 EOS')
        # SD1.5 的 padding 与 EOS 可能同 ID，取第一个 EOS，避免把 padding 当内容。
        indices = eos.int().argmax(1)
        gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
        self.context = (gray, reference, indices)

    def bind(self, unet):
        super().bind(unet)
        def update(_module, args, kwargs):
            if not self.conditional:
                self.unconditional_calls += 1
                return
            self.conditional_calls += 1
            if self.context is not None and self.features is None:
                text = kwargs['encoder_hidden_states']
                gray, reference, indices = self.context
                self.features = [p(f) for p, f in zip(self.projections,
                    self.fusion(gray, reference, text, indices))]
        self.handles.append(unet.register_forward_pre_hook(update, with_kwargs=True))

    def clear(self):
        self.features = self.context = None


class TGTFMNetwork(BFSDNetwork):
    def forward(self, noisy, timestep, text, sketch, reference, sketch_states, text_ids):
        self.adapter.condition(sketch, reference, text_ids)
        return self.unet(noisy, timestep, encoder_hidden_states=text,
                         cross_attention_kwargs={'sa_hidden_states': sketch_states}).sample

    def checkpoint(self):
        return dict(format_version=1, fusion_config=self.adapter.config, network=self.trainable_state())

    def load_checked(self, state, legacy=False):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if legacy:
            expected = {name for name in expected if '.text_modulators.' not in name}
        if set(state) != expected:
            raise ValueError('权重键不一致：缺失=%s，多余=%s' %
                             (sorted(expected - set(state)), sorted(set(state) - expected)))
        self.load_trainable(state)
        return dict(legacy=legacy, loaded=len(state), new=sorted(
            name for name, p in self.named_parameters() if p.requires_grad and name not in state))
