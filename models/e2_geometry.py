"""固定草图几何；历史mask和9×9腐蚀保持原样，不从GT补条件。"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from garment_mask_utils import build_region_masks

VERSION = 'e2_geometry_v1_original_mask_square9_pixel_center_edt'


@dataclass
class GeometryBundle:
    foreground: torch.Tensor
    body: torch.Tensor
    rest: torch.Tensor
    inner_band: torch.Tensor
    background: torch.Tensor
    valid: torch.Tensor
    outer_edge: Optional[torch.Tensor]
    inner_edge: Optional[torch.Tensor]
    signed_distance: Optional[torch.Tensor]
    metadata: dict


def build_geometry(mask, sketch=None, details=True, distance_clip=32.):
    foreground = mask.detach().float().clamp(0, 1)
    body, _, _ = build_region_masks(foreground, kernel_size=9)
    area = foreground.mean((1, 2, 3))
    valid = (area >= .02) & (area <= .95)
    outer = inner = distance = None
    metadata = dict(version=VERSION, mask_area=area.cpu().tolist(), mask_source='provided_legacy_mask',
                    low_area=(area < .02).cpu().tolist(), high_area=(area > .95).cpu().tolist(),
                    details_computed=details, distance_clip=distance_clip,
                    inner_line_representation='thresholded_original_sketch_not_semantic_parts')
    if details:
        from scipy.ndimage import distance_transform_edt
        eroded = -F.max_pool2d(-foreground, 3, 1, 1)
        outer = (foreground - eroded).clamp(0, 1)
        exclusion = F.max_pool2d(outer, 9, 1, 4) > 0
        if sketch is not None:
            gray = ((sketch.detach().float() + 1) * .5 * sketch.new_tensor(
                [.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
            inner = ((gray < 245 / 255.) & (foreground > .5) & ~exclusion).float()
            metadata['has_inner_line'] = inner.flatten(1).any(1).cpu().tolist()
        else:
            metadata['has_inner_line'] = None
        maps = []
        for item in foreground[:, 0].cpu().numpy() > .5:
            # 离散像素中心距离：相邻内/外像素分别为+1/-1，再固定截断归一化。
            signed = distance_transform_edt(item) - distance_transform_edt(~item)
            maps.append(np.clip(signed / distance_clip, -1, 1).astype(np.float32))
        distance = torch.from_numpy(np.stack(maps))[:, None].to(mask.device)
    return GeometryBundle(foreground, body, 1 - body, (foreground - body).clamp(0, 1),
                          1 - foreground, valid, outer, inner, distance, metadata)


class GeometryCache:
    def __init__(self, root):
        self.root = Path(root)

    def key(self, sample_id, mask, sketch, input_hashes):
        values = dict(sample_id=str(sample_id), input_hashes=input_hashes, version=VERSION,
                      shape=list(mask.shape), morphology='legacy_square9', distance_clip=32.,
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for name, tensor in [('mask', mask), ('sketch', sketch)]:
            values[name] = None if tensor is None else hashlib.sha256(
                tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def get(self, sample_id, mask, sketch=None, input_hashes=None):
        key = self.key(sample_id, mask, sketch, input_hashes or {})
        path = self.root / (key + '.pt')
        if path.exists():
            value = torch.load(path, map_location=mask.device, weights_only=False)
            return GeometryBundle(**value), key
        geometry = build_geometry(mask, sketch)
        value = {name: tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor
                 for name, tensor in vars(geometry).items()}
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(path)
        return geometry, key


class InteriorPatchSampler:
    """独立CPU生成器；保存/恢复它即可继续同一采样流，不碰扩散/数据RNG。"""
    def __init__(self, seed=110642):
        self.seed = int(seed)
        self.rng = torch.Generator(device='cpu').manual_seed(self.seed)

    def state_dict(self):
        return dict(seed=self.seed, rng=self.rng.get_state().clone())

    def load_state_dict(self, state):
        if state['seed'] != self.seed:
            raise ValueError('patch采样seed不一致')
        self.rng.set_state(state['rng'].cpu())

    def coordinates(self, mask, size, count=4):
        if min(mask.shape[-2:]) < size:
            return [], 0
        value = (mask.detach().squeeze() > .5).to(torch.int64)
        integral = F.pad(value.cumsum(0).cumsum(1), (1, 0, 1, 0))
        area = (integral[size:, size:] - integral[:-size, size:]
                - integral[size:, :-size] + integral[:-size, :-size])
        valid = (area == size * size).nonzero().cpu()
        number = len(valid)
        if not number:
            return [], 0
        indices = torch.randperm(number, generator=self.rng)[:min(number, count)]
        return [(int(valid[i, 1]), int(valid[i, 0])) for i in indices], number


def extract_patches(image, coordinates, size):
    # 直接张量切片；禁止预测侧转NumPy或乘mask补零。
    return torch.stack([image[:, y:y + size, x:x + size] for x, y in coordinates])
