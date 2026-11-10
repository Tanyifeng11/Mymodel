"""冻结E2的观测工具；不添加生成参数，不改变旧forward或采样结果。"""
from contextlib import contextmanager
import math

import numpy as np
import torch
import torch.nn.functional as F

STEPS = (0, 10, 25, 40, 49)
PATTERNS = ('v16', 'v32', 'h16', 'h32')
VARIANTS = ('raw', 'shift', 'shuffle') + PATTERNS + tuple(x + '_shift' for x in PATTERNS)


def reference_control(rgb, kind, seed):
    """移动/置换完整RGB三元组；不改变颜色联合直方图。格栅是受控诊断，非真实新面料。"""
    value = np.asarray(rgb, dtype=np.uint8)
    if kind == 'raw':
        return value.copy()
    if kind == 'shift':
        return np.roll(value, (37, 29), axis=(0, 1))
    if kind == 'shuffle':
        rng = np.random.default_rng(seed)
        return value.reshape(-1, 3)[rng.permutation(value.shape[0] * value.shape[1])].reshape(value.shape)
    shifted = kind.endswith('_shift'); name = kind.replace('_shift', '')
    assert name in PATTERNS
    period = int(name[1:]); h, w = value.shape[:2]
    assert h % period == 0 and w % period == 0
    yy, xx = np.indices((h, w)); phase = (xx if name[0] == 'v' else yy).ravel() % period
    pixels = value.reshape(-1, 3)
    intensity = pixels.astype(np.float64) @ np.array([.2989, .5870, .1140])
    order = np.lexsort((np.arange(len(phase)), phase))
    result = np.empty_like(pixels)
    result[order] = pixels[np.argsort(intensity, kind='stable')]
    result = result.reshape(value.shape)
    return np.roll(result, (37, 29), axis=(0, 1)) if shifted else result


def color_signature(rgb):
    # 整数求和及平方和与排列顺序无关，防止把float归约误差误当纹样信号。
    x = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3).astype(np.float64)
    return np.r_[x.mean(0), np.sqrt(np.maximum((x*x).mean(0)-x.mean(0)**2, 0))].astype(np.float32) / 255.


def spectrum_signature(rgb):
    x = np.asarray(rgb, dtype=np.float64) @ np.array([.2989, .5870, .1140])
    h, w = x.shape; x -= x.mean(); power = np.abs(np.fft.fft2(x)) ** 2
    fx, fy = np.fft.fftfreq(w)[None], np.fft.fftfreq(h)[:, None]
    result = []
    for name in PATTERNS:
        p = int(name[1:])
        band = ((np.abs(np.abs(fx)-1/p) < .51/w) & (np.abs(fy) < .51/h)) if name[0] == 'v' else \
               ((np.abs(np.abs(fy)-1/p) < .51/h) & (np.abs(fx) < .51/w))
        result.append(float(power[band].sum() / max(float(power.sum()), 1e-12)))
    return np.asarray(result, dtype=np.float32)


def activation_vectors(value):
    x = value.detach().float()
    assert x.ndim == 4 and len(x) == 1 and torch.isfinite(x).all()
    moments = torch.cat([x.mean((2, 3)), x.std((2, 3), unbiased=False)], 1).flatten()
    indices = torch.linspace(0, x.shape[1]-1, min(32, x.shape[1]), device=x.device).long()
    spatial = F.adaptive_avg_pool2d(x[:, indices], (4, 4)).flatten()
    return dict(moments=moments.cpu().numpy(), spatial=torch.cat([moments, spatial]).cpu().numpy())


class FeatureTap:
    """及时读取，防止旧LeakyReLU(inplace)改写被观察张量；hook不返回替代输出。"""
    def __init__(self, adapter):
        self.adapter, self.handles, self.values = adapter, [], {}

    def __enter__(self):
        f = self.adapter.fusion
        modules = {'ref1': f.G_downconv1, 'ref1b': f.G_downconv_channel_unchanged,
                   'ref2_pure': f.G_downconv2, 'sketch2_before_mod': f.downconv2}
        for level in [3, 4, 5]: modules['ref%d_mixed' % level] = getattr(f, 'G_downconv%d' % level)
        for level in [2, 3, 4, 5]:
            for name in ['alpha', 'beta']:
                modules['film%d_%s' % (level, name)] = getattr(f, 'G_bottleneck_%s_%d' % (name, level))
        for i, projection in enumerate(self.adapter.projections): modules['projection%d' % i] = projection
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(lambda _m, _a, value, n=name:
                self.values.__setitem__(n, activation_vectors(value))))
        def fused(_m, _a, values):
            for i, value in enumerate(values): self.values['fused%d' % (i+2)] = activation_vectors(value)
        self.handles.append(f.register_forward_hook(fused))
        return self

    def __exit__(self, *_):
        for h in self.handles: h.remove()


class TrajectoryTap:
    """只记录原轨迹；额外前向在原采样结束后进行，避免污染原CFG或scheduler。"""
    def __init__(self, pipe, adapter, steps=STEPS):
        self.pipe, self.adapter, self.steps = pipe, adapter, set(steps)
        self.states, self.condition_calls, self.scheduler_calls = {}, 0, 0

    def __enter__(self):
        def observe(_m, args, kwargs, output):
            conditional = 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            if conditional:
                index = self.condition_calls; self.condition_calls += 1
            else:
                index = self.condition_calls-1
            if index not in self.steps: return
            row = self.states.setdefault(index, {})
            kind = 'conditional' if conditional else 'unconditional'
            row[kind] = dict(args=(args[0].detach().clone(), args[1].detach().clone()),
                kwargs=dict(kwargs), epsilon=output[0].detach().clone(), counts=list(self.adapter.injection_counts))
        self.handle = self.pipe.unet.register_forward_hook(observe, with_kwargs=True)
        self.original = self.pipe.scheduler.step
        self.had_step = 'step' in self.pipe.scheduler.__dict__
        self.instance_step = self.pipe.scheduler.__dict__.get('step')
        def scheduler_step(prediction, timestep, sample, *args, **kwargs):
            i = self.scheduler_calls; self.scheduler_calls += 1
            if i in self.steps:
                self.states[i].update(guided=prediction.detach().clone(), sample=sample.detach().clone(),
                                      timestep=int(timestep))
            return self.original(prediction, timestep, sample, *args, **kwargs)
        self.pipe.scheduler.step = scheduler_step
        return self

    def __exit__(self, *_):
        self.handle.remove()
        if self.had_step: self.pipe.scheduler.step = self.instance_step
        else: del self.pipe.scheduler.__dict__['step']


def feature_mix(old, new, gain):
    # 端点保持旧半精度值，避免a+(b-a)的舍入破坏像素回归。
    if gain == 0: return old
    if gain == 1: return new
    return [(a.float()+gain*(b.float()-a.float())).to(a.dtype) for a, b in zip(old, new)]


@contextmanager
def sketch_strength(pipe, scale):
    processors = [(p, p.scale) for p in pipe.unet.attn_processors.values() if hasattr(p, 'scale')]
    try:
        pipe.set_scale(scale)
        yield
    finally:
        for p, value in processors: p.scale = value


def region_change(value, baseline, masks):
    delta, reference = (value.float()-baseline.float()).double(), baseline.double()
    result = {}
    for name, mask in masks.items():
        weight = F.adaptive_avg_pool2d(mask.float(), value.shape[-2:]).double()
        n = float(weight.sum()) * value.shape[1]
        if n == 0:
            result[name] = dict(rms=None, baseline_rms=None, relative_rms=None); continue
        rms = float((delta.square()*weight).sum()/n)**.5
        old = float((reference.square()*weight).sum()/n)**.5
        result[name] = dict(rms=rms, baseline_rms=old, relative_rms=rms/old if old else None)
    return result


def grouped_ridge(x, shifted, labels, groups, fold_map, classes, penalty=1.):
    """训练及测试按服装身份隔离；固定正则，不用测试集选超参。只拟合诊断读出。"""
    x, shifted = np.asarray(x, dtype=np.float64), np.asarray(shifted, dtype=np.float64)
    labels, groups = np.asarray(labels), np.asarray(groups)
    original_prediction, shifted_prediction = np.empty(len(x), int), np.empty(len(x), int)
    for fold in sorted(set(fold_map.values())):
        test = np.asarray([fold_map[g] == fold for g in groups]); train = ~test
        mean, std = x[train].mean(0), x[train].std(0)
        usable = std > 1e-6; std = np.maximum(std, 1e-6)
        def normalize(a):
            z = (a-mean)/std; z[:, ~usable] = 0
            return z/math.sqrt(max(1, int(usable.sum())))
        a = normalize(x[train]); y = np.eye(classes)[labels[train]]; center = y.mean(0)
        dual = np.linalg.solve(a @ a.T + penalty*np.eye(len(a)), y-center)
        weights = a.T @ dual
        original_prediction[test] = (normalize(x[test]) @ weights + center).argmax(1)
        shifted_prediction[test] = (normalize(shifted[test]) @ weights + center).argmax(1)
    return original_prediction, shifted_prediction


def group_accuracy(prediction, labels, groups):
    values = [float(np.mean(np.asarray(prediction)[np.asarray(groups)==g] == np.asarray(labels)[np.asarray(groups)==g]))
              for g in sorted(set(groups))]
    rng = np.random.default_rng(110142); a = np.asarray(values)
    boot = a[rng.integers(len(a), size=(10000, len(a)))].mean(1)
    return dict(groups=len(a), samples=len(labels), accuracy=float(a.mean()),
                descriptive_ci95=np.quantile(boot, [.025, .975]).tolist(),
                inference='exploratory grouped diagnostic;not generator accuracy or confirmatory significance')
