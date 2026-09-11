"""E9 同 latent 前向探针。只观测，不改变用于 scheduler 更新的预测。"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LocalDetailProbe:
    def __init__(self, unet, garment_mask, output_dir, steps):
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.steps = set(steps)
        self.step = -1
        self.garment = garment_mask
        self.adapters = [getattr(p, 'local_detail_adapter') for p in unet.attn_processors.values()
                         if getattr(p, 'local_detail_adapter', None) is not None]
        if len(self.adapters) != 1:
            raise ValueError('传播探针要求恰好一个 E9 旁路')
        self.handle = self.adapters[0].register_forward_hook(self.capture)
        self.captured = None
        self.rows = []

    def capture(self, module, inputs, output):
        if self.step not in self.steps:
            return
        shape = inputs[3] if output.ndim == 3 else output.shape[-2:]
        residual = output.detach().float()
        if residual.ndim == 3:
            residual = residual.transpose(1, 2).reshape(residual.shape[0], -1, *shape)
        mask = F.interpolate(inputs[2].float(), size=shape, mode='nearest') > 0
        garment = F.interpolate(self.garment.float(), size=shape, mode='nearest') > 0
        self.captured = (residual.square().mean(1)[0].cpu().numpy(),
                         mask[0, 0].cpu().numpy(), garment[0, 0].cpu().numpy())

    def compare(self, on_prediction, forward, timestep):
        """on/off 共用当前 latent、文本、参考条件和 timestep；第二次 off 测数值底噪。"""
        if self.step not in self.steps:
            return
        if self.captured is None:
            raise RuntimeError('没有捕获到 E9 残差')
        residual_energy, injection_mask, garment = self.captured
        module = self.adapters[0]
        scale, trace = module.runtime_scale, module.runtime_trace
        saved_stats = module.last_stats
        try:
            module.runtime_scale = 0.0
            module.runtime_trace = None
            # 额外前向不消耗正常生成流程的随机状态。
            devices = [on_prediction.device.index] if on_prediction.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                off = forward()
                repeat_off = forward()
        finally:
            module.runtime_scale, module.runtime_trace = scale, trace
            module.last_stats = saved_stats
        delta = (on_prediction.float()-off.float()).square().mean(1)[0].detach().cpu().numpy()
        floor = (repeat_off.float()-off.float()).square().mean(1)[0].detach().cpu().numpy()
        # 此 E9 层与噪声预测在同一空间分辨率。不同层须重新定义映射。
        if delta.shape != injection_mask.shape:
            raise ValueError('旁路与噪声输出的空间尺寸不同，不能直接比较 mask')
        outside = ~garment
        regions = {'outside_injection': ~injection_mask, 'outside_garment': outside}
        row = {'step_index': self.step, 'timestep': int(timestep), 'runtime_scale': float(scale),
               'injection_area': float(injection_mask.mean()), 'garment_area': float(garment.mean()),
               'injection_outside_garment_pixels': int((injection_mask & outside).sum()),
               'noise_delta_rms': float(np.sqrt(delta.mean())),
               'off_repeat_rms': float(np.sqrt(floor.mean()))}
        for name, region in regions.items():
            row[name+'_pixels'] = int(region.sum())
            for label, energy in [('direct_residual', residual_energy), ('noise_delta', delta), ('off_repeat', floor)]:
                row[label+'_'+name+'_rms'] = float(np.sqrt(energy[region].mean())) if region.any() else None
        np.savez_compressed(self.folder/f'step_{self.step:02d}.npz',
                            injection_mask=injection_mask, garment_mask=garment,
                            direct_residual_rms=np.sqrt(residual_energy),
                            noise_delta_rms=np.sqrt(delta), off_repeat_rms=np.sqrt(floor))
        self.rows.append(row)
        (self.folder/'probe.json').write_text(json.dumps(self.rows, indent=2), encoding='utf-8')

    def close(self):
        self.handle.remove()
