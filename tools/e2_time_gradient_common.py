"""时间/梯度诊断的观测与数学工具；不包含可学习参数或优化器。"""
import copy
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from models.e2_texture_diagnostics import TrajectoryTap

TRAIN_TIMES = (1, 50, 100, 175, 225, 249, 250, 499, 749, 999)
NOISE_SEEDS = (111100, 111101)
SCORE_STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49)
OLD_WEIGHTS = dict(mse=1., gram=1., edge=.05, color=.1, body=.1, rest=.03816244801608925)
_BOOTSTRAP_INDICES = {}


class FullTrace:
    """保存原50步和实际step返回值；额外前向全部放在原采样结束之后。"""
    def __init__(self, pipe, adapter):
        self.pipe, self.tap = pipe, TrajectoryTap(pipe, adapter, range(50))

    def __enter__(self):
        self.tap.__enter__(); self.inner = self.pipe.scheduler.step
        def observe(*args, **kwargs):
            index = self.tap.scheduler_calls
            result = self.inner(*args, **kwargs)
            assert float(kwargs.get('eta', 0.)) == 0., '本协议只诊断原eta0确定性DDIM'
            self.tap.states[index]['next_sample'] = result[0].detach().clone()
            return result
        self.pipe.scheduler.step = observe
        return self.tap

    def __exit__(self, *args):
        self.pipe.scheduler.step = self.inner
        self.tap.__exit__(*args)


def predicted_x0(sample, epsilon, alpha):
    a = torch.as_tensor(alpha, dtype=torch.float32, device=sample.device)
    return (sample.float()-(1-a).sqrt()*epsilon.float())/a.sqrt()


def schedule_record(scheduler, timestep):
    t = int(timestep); previous = t-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
    a = float(scheduler.alphas_cumprod[t])
    ap = float(scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod)
    return dict(t=t,previous_t=previous,alpha=a,previous_alpha=ap,
        epsilon_to_x0=math.sqrt((1-a)/a),
        epsilon_to_next=math.sqrt(max(0.,1-ap))-math.sqrt(ap/a)*math.sqrt(1-a),
        clip_sample=bool(scheduler.config.clip_sample),eta=0.)


def scheduler_counterfactual(scheduler, epsilon, state):
    # 独立副本，避免额外step推进原采样器状态；eta0不抽随机数。
    clone = copy.deepcopy(scheduler)
    return clone.step(epsilon,state['conditional']['args'][1],state['sample'],eta=0.,return_dict=False)[0]


def digest_network(network):
    value=hashlib.sha256()
    for name,tensor in network.state_dict().items():
        value.update(name.encode());value.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def gradient_pair(first, second, weight=None):
    """零/未连接梯度不伪造cos=0；软区域权重在内积中仅使用一次。"""
    if first is None or second is None:
        return dict(first_connected=first is not None,second_connected=second is not None,
            first_norm=None,second_norm=None,dot=None,cosine=None,first_along_second=None)
    a,b=first.detach().double(),second.detach().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    w=1. if weight is None else F.adaptive_avg_pool2d(weight.float(),a.shape[-2:]).double()
    na=float((a.square()*w).sum().sqrt());nb=float((b.square()*w).sum().sqrt())
    dot=float((a*b*w).sum())
    return dict(first_connected=True,second_connected=True,first_norm=na,second_norm=nb,dot=dot,
        cosine=max(-1.,min(1.,dot/(na*nb))) if na and nb else None,
        first_along_second=dot/nb if nb else None)


def gradient_summary(value):
    if value is None:return dict(connected=False,norm=None,finite=True)
    assert torch.isfinite(value).all()
    return dict(connected=True,norm=float(value.detach().double().norm()),finite=True)


def descriptive(values, seed=111142):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],np.float64)
    if not len(a):return dict(n=0,mean=None,ci95=None)
    key=(len(a),seed)
    if key not in _BOOTSTRAP_INDICES:
        _BOOTSTRAP_INDICES[key]=np.random.default_rng(seed).integers(len(a),size=(10000,len(a)))
    boot=a[_BOOTSTRAP_INDICES[key]].mean(1)
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def same_patch_objective(predicted, target, coordinates, descriptor):
    """匹配GT与参考描述子预处理/区域，独立列为诊断项，不加入旧目标。"""
    from models.e2_geometry import extract_patches
    terms=[]
    for size,coords in coordinates.items():
        if not coords:continue
        a=extract_patches((predicted[0].float()+1)*.5,coords,int(size))
        with torch.no_grad():b=descriptor(extract_patches((target[0].float()+1)*.5,coords,int(size))).mean(0)
        terms.append((descriptor(a).mean(0)-b).abs().mean())
    return torch.stack(terms).mean() if terms else predicted.sum()*0
