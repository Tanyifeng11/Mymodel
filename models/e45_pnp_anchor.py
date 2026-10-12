"""固定权重的下游锚；原 E5/B1 算子与 Sketch 注意力保持原实现。"""
import math
import torch
from torch.nn import functional as F

FEATURE = 'up_blocks.2.resnets.1'
ATTENTION = 'up_blocks.2.attentions.1.transformer_blocks.0.attn1'


def strength(index):
    if index <= 14:
        return .25
    if index <= 24:
        return .125 * (1 + math.cos(math.pi * (index - 14) / 10))
    return 0.


def feature_anchor(student, teacher, weight):
    if weight == 0:
        return student
    assert student.shape == teacher.shape
    return student + weight * (teacher.detach() - student)


def equal_energy(delta, mask):
    if delta.ndim == 3:
        mask = mask.flatten(2).transpose(1, 2)
    masked = delta * mask.to(delta)
    energy = delta.float().square().sum()
    target = masked.float().square().sum()
    amplitude = (target / energy).sqrt() if energy else energy
    applied = delta * amplitude.to(delta)
    actual = applied.float().square().sum()
    return applied, dict(amplitude=float(amplitude), raw_energy=float(energy),
        masked_energy=float(target), applied_energy=float(actual),
        relative_energy_error=float((actual - target).abs() / target) if target else 0.)


def attention_delta(attn, teacher, student, weight):
    """只修正 self 项；学生 V、原 Sketch 项、偏置、残差均不重复应用。"""
    if weight == 0:
        return None
    q, k, v = (student[name] for name in ['q', 'k', 'v'])
    b, n, c = q.shape
    def split(x):
        return x.reshape(b, -1, attn.heads, c // attn.heads).transpose(1, 2)
    qt, kt = teacher['q'], teacher['k']
    assert q.shape == qt.shape and k.shape == kt.shape
    own = F.scaled_dot_product_attention(split(q), split(k), split(v))
    anchored = F.scaled_dot_product_attention(split(qt), split(kt), split(v))
    delta = (weight * (anchored - own)).transpose(1, 2).reshape(b, n, c)
    delta = F.linear(delta, attn.to_out[0].weight, None)
    return attn.to_out[1](delta) / attn.rescale_output_factor


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class Anchor:
    """每次 conditional 外层调用只有一次 Writer、教师和学生各一次 forward。"""
    def __init__(self, runtime, arm, masks, disabled=False, audit_unconditional=False):
        from tools.e44_b1_mask import weights_at
        self.r, self.p = runtime, runtime.processor
        self.arm, self.disabled = arm, disabled
        self.audit_unconditional = audit_unconditional
        self.weights = weights_at(masks, (64, 48), runtime.pipe.device)
        self.steps = []
        self.phase = None
        self.handles = []
        self.teacher_calls = self.student_calls = self.unconditional_calls = 0

    def event(self, name):
        self.events.append(name)

    def injection(self, base, raw, applied):
        if self.phase != 'student':
            return
        self.base, self.raw = base, raw
        self.current['injection_raw_rms'] = rms(raw)
        self.current['injection_applied_rms'] = rms(applied)

    def injection_output(self, module, args, result):
        self.event('b1')
        if self.phase == 'student' and self.p.enabled and self.arm == 'C3' and not self.disabled:
            delta, audit = equal_energy(self.raw, self.p.mask)
            self.current['equal_energy'] = audit
            self.current['injection_applied_rms'] = rms(delta)
            return self.base + delta
        return result

    def feature(self, module, args, result):
        self.event('feature')
        assert torch.is_tensor(result), '真实 ResNet 边界必须为 Tensor'
        if self.phase == 'teacher':
            self.cache['feature'] = result.detach()
        elif self.phase == 'student':
            teacher = self.cache['feature']
            self.current['feature_shape'] = list(result.shape)
            self.current['feature_difference_rms'] = rms(result - teacher)
            weight = self.weight if self.arm in ['C5', 'C7'] else 0.
            output = feature_anchor(result, teacher, weight)
            self.current['feature_correction_rms'] = rms(output - result)
            return output
        return result

    def projection(self, name):
        def capture(module, args, result):
            if self.phase in ['teacher', 'student']:
                self.qkv[name] = result.detach()
        return capture

    def attention(self, module, args, kwargs, result):
        self.event('attention')
        if self.phase == 'teacher':
            self.cache['qkv'] = dict(self.qkv)
        elif self.phase == 'student':
            assert kwargs.get('attention_mask') is None, '本协议仅适用原生成的无 mask self-attention'
            assert torch.is_tensor(result) and result.ndim == 3
            teacher = self.cache['qkv']
            self.current['q_shape'] = list(self.qkv['q'].shape)
            self.current['heads'] = module.heads
            self.current['q_difference_rms'] = rms(self.qkv['q'] - teacher['q'])
            self.current['k_difference_rms'] = rms(self.qkv['k'] - teacher['k'])
            weight = self.weight if self.arm in ['C6', 'C7'] else 0.
            delta = attention_delta(module, teacher, self.qkv, weight)
            self.current['attention_correction_rms'] = rms(delta) if delta is not None else 0.
            if delta is not None:
                return result + delta
        return result

    def install(self):
        from tools.e44_protocol import LAYER
        modules = dict(self.r.pipe.unet.named_modules())
        self.handles.append(modules[LAYER[:-10]].register_forward_hook(self.injection_output))
        self.handles.append(modules[FEATURE].register_forward_hook(self.feature))
        attention = modules[ATTENTION]
        for name in ['q', 'k', 'v']:
            self.handles.append(getattr(attention, 'to_' + name).register_forward_hook(self.projection(name)))
        self.handles.append(attention.register_forward_hook(self.attention, with_kwargs=True))
        self.old_forward = self.r.pipe.unet.forward
        self.p.diagnostic_observer = self.injection
        self.p.spatial_intervention = 'interior' if self.arm == 'C2' and not self.disabled else 'global'
        self.r.active = self.arm != 'C0'
        self.r.pipe.unet.forward = self.forward

    def forward(self, *args, **kwargs):
        from tools.e44_b1_mask import regional_stats
        from tools.e35_freeu_run import digest
        conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
        self.events = []
        if not conditional:
            assert not self.p.enabled
            self.phase = None
            self.unconditional_calls += 1
            output = self.old_forward(*args, **kwargs)
            if self.audit_unconditional:
                baseline = self.old_forward(*args, **kwargs)
                assert torch.equal(output[0], baseline[0]), '同状态无条件预测受锚点影响'
            return output
        latent = args[0] if args else kwargs['sample']
        timestep = args[1] if len(args) > 1 else kwargs['timestep']
        index = len(self.steps)
        enabled = self.p.enabled
        self.weight = strength(index) if enabled and not self.disabled else 0.
        self.current = dict(index=index, timestep=int(timestep), enabled=enabled, weight=self.weight,
            same_state_sha256=digest(latent), injection_raw_rms=0., injection_applied_rms=0.)
        self.cache = {}
        try:
            # 绕过外层 module hook；不重复 Writer，不推进 scheduler，不污染 ε 记录。
            self.p.enabled = False
            self.phase = 'teacher'
            self.qkv = {}
            teacher = self.old_forward(*args, **kwargs)
            self.teacher_calls += 1
            teacher_order = list(self.events)
            self.events = []
            self.p.enabled = enabled
            self.phase = 'student'
            self.qkv = {}
            student = self.old_forward(*args, **kwargs)
            self.student_calls += 1
            assert teacher_order == self.events == ['b1', 'feature', 'attention'], self.events
            assert digest(latent) == self.current['same_state_sha256'], '教师/学生改动了输入 latent'
            if self.arm == 'C4' and self.weight:
                assert isinstance(student, tuple)
                student = (feature_anchor(student[0], teacher[0], self.weight),) + student[1:]
            effect = student[0].float() - teacher[0].float()
            self.current['epsilon_effect'] = regional_stats(effect, self.weights)
            self.current['cfg_effect'] = regional_stats(effect * 7, self.weights)
            self.current['layer_order'] = teacher_order
            self.steps.append(self.current)
            return student
        finally:
            self.p.enabled = enabled
            self.phase = None
            self.cache = {}
            self.qkv = {}
            self.base = self.raw = None

    def close(self):
        self.r.pipe.unet.forward = self.old_forward
        self.p.diagnostic_observer = None
        self.p.spatial_intervention = 'global'
        self.r.active = True
        for handle in self.handles:
            handle.remove()
