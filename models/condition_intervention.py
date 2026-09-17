"""固定基线触发时刻的六样本干预；修正 conditional epsilon，随后沿用原 CFG。"""
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from models.condition_response_probe import pair_stats, region_weights


MODES = ('none', 'baseline', 'boundary', 'weaken_texture', 'strengthen_sketch', 'global')


def fixed_trigger_steps(source):
    if source['steps'] != list(range(50)) or source['fractions'] != [0.1, 0.2]:
        raise ValueError('干预需要完整的 50 步、两档扰动基线')
    if [r['step_index'] for r in source['records']] != list(range(50)):
        raise ValueError('基线记录不完整')
    return [r['step_index'] for r in source['records']
            if 8 <= r['step_index'] <= 35 and all(
                s['above_repeat_floor'] and s['cosine'] is not None and s['cosine'] < 0
                for s in r['regions']['boundary']['responses'].values())]


def projected_correction(a, b, weight):
    """u=w*a；移除 b 沿 u 的负投影。软边界分母用 ||w*a||²，不能用 sum(w*a²)。"""
    u = weight.double() * a.double()
    dot = (u * b.double()).sum()
    norm = u.square().sum()
    coefficient = min(0.0, float(dot / norm)) if float(norm) > 0 else 0.0
    return (-coefficient * u).float(), coefficient


@contextmanager
def isolated_forward(processors, prediction):
    states = [(p.scale, {k: v for k, v in vars(p).items() if k.startswith('last_')},
               getattr(p, 'balanced_gate_trace_enabled', None)) for p in processors]
    try:
        for p in processors:
            if hasattr(p, 'balanced_gate_trace_enabled'):
                p.balanced_gate_trace_enabled = False
        devices = [prediction.device.index] if prediction.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        for p, (scale, diagnostics, trace) in zip(processors, states):
            p.scale = scale
            for key in list(vars(p)):
                if key.startswith('last_'):
                    delattr(p, key)
            for key, value in diagnostics.items():
                setattr(p, key, value)
            if trace is not None:
                p.balanced_gate_trace_enabled = trace


class ConditionIntervention:
    def __init__(self, sketch, texture, mask, mode, source_path, output_dir, metadata):
        if mode not in MODES[1:]:
            raise ValueError('未知干预模式')
        source = json.loads(Path(source_path).read_text(encoding='utf-8'))
        self.steps = set(fixed_trigger_steps(source))
        if source['prediction_space'] != 'epsilon_before_cfg':
            raise ValueError('干预只支持 conditional epsilon')
        for key in ('sketch_path', 'texture_path', 'prompt', 'seed', 'gam_ckpt',
                    'guidance_scale', 'texture_scale', 'num_inference_steps', 'mask_info'):
            if source['metadata'][key] != metadata[key]:
                raise ValueError(f'当前条件与触发基线不一致：{key}')
        self.sketch, self.texture = list(sketch), list(texture)
        if not self.sketch or not self.texture or mask is None or mask.shape[:2] != (1, 1):
            raise ValueError('干预需要两路处理器及单样本 mask')
        self.mask, self.mode = mask, mode
        self.source_records = source['records']
        self.kernel = source['region_kernel_input_pixels']
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report = dict(mode=mode, source_probe=str(source_path), metadata=metadata,
                           prediction_space='epsilon_before_cfg', window=[8, 35], fraction=0.2,
                           trigger_steps=sorted(self.steps), records=[])

    @torch.no_grad()
    def apply(self, prediction, forward, step, timestep):
        if prediction.shape[0] != 1:
            raise ValueError('干预仅支持 batch=1')
        source = self.source_records[step]
        if int(timestep) != source['timestep']:
            raise ValueError('采样 timestep 与触发基线不一致')
        if ([float(p.scale) for p in self.sketch] != source['sketch_scales'] or
                [float(p.scale) for p in self.texture] != source['texture_scales']):
            raise ValueError('条件注入强度与触发基线不一致')
        row = dict(step_index=step, timestep=int(timestep), scheduled=step in self.steps,
                   applied=False, correction_rms=0.0)
        result = prediction
        if step in self.steps and self.mode != 'baseline':
            processors = self.sketch + self.texture
            with isolated_forward(processors, prediction):
                if self.mode in ('weaken_texture', 'strengthen_sketch'):
                    chosen = self.texture if self.mode == 'weaken_texture' else self.sketch
                    factor = 0.8 if self.mode == 'weaken_texture' else 1.2
                    for p in chosen:
                        p.scale *= factor
                    result = forward()
                else:
                    repeat = forward()
                    for p in self.sketch:
                        p.scale *= 0.8
                    weak_sketch = forward()
                    for p, scale in zip(self.sketch, source['sketch_scales']):
                        p.scale = scale
                    for p in self.texture:
                        p.scale *= 0.8
                    weak_texture = forward()
                    a = prediction.float() - weak_sketch.float()
                    b = prediction.float() - weak_texture.float()
                    weight = region_weights(self.mask.to(prediction.device), prediction.shape[-2:], self.kernel)[self.mode]
                    stats = pair_stats(a, b, weight)
                    floor = pair_stats(repeat.float() - prediction.float(), repeat.float() - prediction.float(), weight)['a_rms']
                    reliable = stats['a_rms'] is not None and min(stats['a_rms'], stats['b_rms']) > 10 * max(floor or 0, 1e-12)
                    correction, coefficient = projected_correction(a, b, weight)
                    row.update(response=stats, repeat_rms=floor, above_repeat_floor=reliable,
                               coefficient=coefficient)
                    if reliable and coefficient < 0:
                        # 不除以 0.2：只修正实测 20% 纹理响应，不假定完整 guidance 线性。
                        result = (prediction.float() + correction).to(prediction.dtype)
            delta = result.float() - prediction.float()
            if not torch.isfinite(delta).all():
                raise ValueError('干预产生非有限修正')
            row.update(applied=bool(torch.any(delta != 0)), correction_rms=float(delta.square().mean().sqrt()))
        self.report['records'].append(row)
        (self.folder / 'intervention.json').write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return result
