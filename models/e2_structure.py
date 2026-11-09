"""A的增量结构残差：输入草图几何，融合完成后加到四尺度投影；不是原bFM草图编码器。"""
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from models.bf_sd_skip import BFSkipFusion

VERSION = 'A_residual_after_projection_v1_sketch_skeleton_square9_edt32'
MODES = {'U1': (False, False), 'A3': (True, False),
         'A4': (False, True), 'A5': (True, True)}


def structure_maps(sketch):
    """固定条件侧预处理；GT/参考不参与。内部骨架只是候选线，不能称为语义缝线。"""
    from garment_mask_utils import build_sketch_garment_mask, mask_image_to_bool
    from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
    from skimage.morphology import skeletonize
    values = ((sketch.detach().float().cpu() + 1) * 127.5).clamp(0, 255).round().byte()
    result = []
    for item in values:
        image = Image.fromarray(item.permute(1, 2, 0).numpy())
        w, h = image.size
        mask_image, _ = build_sketch_garment_mask(image, w, h, line_threshold=245, close_size=5)
        mask = mask_image_to_bool(mask_image)
        line = 1 - np.asarray(image.convert('L'), dtype=np.float32) / 255.
        outer = mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=1)
        near = binary_dilation(outer, structure=np.ones((9, 9)))
        inner = skeletonize(line > 10 / 255.) & mask & ~near
        distance = distance_transform_edt(mask) - distance_transform_edt(~mask)
        valid = .02 <= float(mask.mean()) <= .95
        result.append(np.stack([line, outer, mask, np.clip(distance / 32., -1, 1), inner,
                                np.full_like(line, float(valid))]).astype(np.float32))
    return torch.from_numpy(np.stack(result)).to(device=sketch.device, dtype=sketch.dtype)


def encoder(channels, width):
    layers = []
    for incoming in [channels, width, width]:
        layers.extend([nn.Conv2d(incoming, width, 3, stride=2, padding=1),
                       nn.GroupNorm(8, width), nn.SiLU()])
    return nn.Sequential(*layers)


class StructureResidual(nn.Module):
    def __init__(self, channels=(320, 640, 1280, 1280), width=32, mode='A5', seed=110909):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.width = mode, width
        self.layered, self.multiscale = MODES[mode]
        # 四种表示具有同一可训练参数量。U1/A4的两组输入都不含语义分层，
        # 固定补零保持输入卷积参数一致；U1不是规范中尚未实现的单分支A1。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.outer = encoder(4, width)
            self.inner = encoder(3, width)
            self.heads = nn.ModuleList([nn.Conv2d(width * 2, c, 1) for c in channels])
            for head in self.heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, maps, sizes):
        scales = [1., .5, .25] if self.multiscale else [1.]
        features = []
        for scale in scales:
            if scale == 1:
                x = maps
            else:
                x = F.interpolate(maps, scale_factor=scale, mode='bilinear', align_corners=False)
                # 对1像素候选线保留覆盖，避免nearest恰好漏掉奇数坐标整条线。
                x = x.clone()
                factor = int(1 / scale)
                x[:, [1, 4]] = F.max_pool2d(maps[:, [1, 4]], factor, factor)
                x[:, 2:3] = F.interpolate(maps[:, 2:3], scale_factor=scale, mode='nearest')
            if self.layered:
                a, b = x[:, [0, 1, 2, 3]], x[:, [0, 4, 2]]
            else:
                # 不分层对照只看同一原线图/前景/距离，不偷用外/内线图。
                a, b = torch.cat([x[:, [0, 2, 3]], torch.zeros_like(x[:, :1])], 1), x[:, [0, 2, 3]]
            features.append(torch.cat([self.outer(a), self.inner(b)], 1))
        valid = maps[:, 5:6, :1, :1]
        result = []
        for head, size in zip(self.heads, sizes):
            fused = sum(F.interpolate(f, size=size, mode='bilinear', align_corners=False)
                        for f in features) / len(features)
            result.append(head(fused) * valid)
        return result


class AStructureAdapter(BFSkipFusion):
    def __init__(self, original, mode='A5', width=32, enabled=True):
        # 不重新初始化bFM/旧投影；这些对象及state_dict前缀原样保留。
        nn.Module.__init__(self)
        self.fusion, self.projections, self.channels = original.fusion, original.projections, original.channels
        self.features, self.conditional = None, False
        self.handles, self.injection_counts = [], [0] * 4
        self.structure = StructureResidual(self.channels, width, mode).to(next(self.projections.parameters()).device)
        self.structure_enabled = enabled
        self.structure_statistics = None
        self.structure_records = []

    def condition(self, sketch, reference):
        super().condition(sketch, reference)
        if not self.structure_enabled:
            return
        maps = structure_maps(sketch)
        deltas = self.structure(maps, [f.shape[-2:] for f in self.features])
        self.features = [f + d.to(f.dtype) for f, d in zip(self.features, deltas)]
        self.structure_statistics = dict(valid=int(maps[:, 5, 0, 0].sum()),
            outer_pixels=float(maps[:, 1].sum()), inner_pixels=float(maps[:, 4].sum()),
            residual_rms=[float(d.detach().float().square().mean().sqrt()) for d in deltas])
        self.structure_records.append(self.structure_statistics)


def attach_structure(network, mode='A5', width=32, enabled=True):
    original = network.adapter
    original.unbind()
    adapter = AStructureAdapter(original, mode, width, enabled)
    network.adapter = adapter
    adapter.bind(network.unet)
    return adapter
