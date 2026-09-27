"""E22.2：保持周期物理坐标，比较等通道数的三种表示。"""

import math

import torch
import torch.nn.functional as F

from models.harmonic_period import harmonic_period
from models.pattern_canonicalization import fft_orientation
from models.target_pattern_score import gray


@torch.no_grad()
def estimate(images):
    theta, _ = fft_orientation(images)
    period = harmonic_period(images, theta)
    return {"theta": theta.cpu(), "frequency": period["xy"].cpu(),
            "active": period["active"].cpu(), "confidence": period["confidence"].mean(-1).cpu()}


def field(correct, donor, mode, phase=0., size=32):
    # Reference is resized over the entire garment canvas in the frozen E20 protocol.
    # Thus coordinates are full-canvas fractions, not a new garment-bbox warp.
    xy = (torch.arange(size).float()+.5)/size-.5
    y, x = torch.meshgrid(xy, xy, indexing="ij")
    theta = correct["theta"].reshape(())
    u, v = x*theta.cos()+y*theta.sin(), -x*theta.sin()+y*theta.cos()
    frequency = donor["frequency"].flatten()
    active = correct["active"].flatten().float()
    confidence = correct["confidence"].reshape(()).clamp(0, 1)
    const = torch.ones_like(x)
    if mode == "scalar":
        channels = [const*frequency[0]*active[0]/16, const*frequency[1]*active[1]/16,
                    const*confidence, const*0, const*0]
    elif mode == "coordinates":
        channels = [const*frequency[0]*active[0]/16, const*frequency[1]*active[1]/16,
                    const*confidence, 2*x, 2*y]
    else:
        a, b = 2*math.pi*frequency[0]*u+phase, 2*math.pi*frequency[1]*v+phase
        channels = [a.sin()*active[0], a.cos()*active[0], b.sin()*active[1], b.cos()*active[1], const*confidence]
    return torch.stack(channels)[None]


def interior_spectrum(image, mask):
    """固定全画布坐标的 interior 功率谱；去均值并向内羽化，抑制服装轮廓。"""
    mono = gray(F.interpolate(image.float(), (128, 128), mode="area"))[:, None]
    support = F.interpolate(mask.float(), (128, 128), mode="area")
    support = (support >= 1-1e-6).float()
    core = 1-F.max_pool2d(F.pad(1-support, (3,3,3,3), value=1), 7, stride=1)
    window = core*F.avg_pool2d(core, 9, stride=1, padding=4)
    mean = (mono*window).sum((-2,-1), keepdim=True)/window.sum((-2,-1), keepdim=True).clamp_min(1)
    power = torch.fft.fft2(((mono-mean)*window)[:, 0]).abs().square()
    fx = torch.fft.fftfreq(128, device=image.device)*128
    radius = (fx[:,None].square()+fx[None,:].square()).sqrt().round().long()
    radial = power.new_zeros((len(power), 92))
    radial.scatter_add_(1, radius.flatten()[None].expand(len(power),-1), power.flatten(1))
    # Separate normalization prevents the radial summary from overwhelming axis spectra.
    return torch.cat([F.normalize(q, dim=-1) for q in
                      (power.sum(-2)[:,1:33], power.sum(-1)[:,1:33], radial[:,1:33])], -1)/math.sqrt(3)
