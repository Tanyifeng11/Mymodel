"""O4 完整生成读出：固定 interior，低纹理/不明确方向不算成功。"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from eval.eval_utils import estimate_foreground_mask
from eval.metrics import _dilate_binary, _rgb_to_lab
from garment_mask_utils import build_region_masks
from models.target_pattern_score import interior_rectangle
from tools.e21_target_supervision import structure


def axial_distance(a, b):
    return abs((a-b+90) % 180-90)


def orientation(image, roi):
    x, y, w, h = roi
    gray = np.array(image.convert("L"), dtype=np.float64)[y:y+h, x:x+w]/255.
    # Native ROI pixels preserve physical direction; no anisotropic square resize.
    gy, gx = np.gradient(gray)
    xx, yy, xy = np.mean(gx*gx), np.mean(gy*gy), np.mean(gx*gy)
    coherence = math.hypot(xx-yy, 2*xy)/max(xx+yy, 1e-12)
    theta = (math.degrees(math.atan2(2*xy, xx-yy))/2+90) % 180
    window = np.hanning(h)[:,None]*np.hanning(w)[None,:]
    power = abs(np.fft.fft2((gray-gray.mean())*window))**2
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radius = np.hypot(fx,fy)*min(h,w)
    weights = power*((radius>=2)&(radius<=min(h,w)/4))
    phi = np.arctan2(fy,fx)
    z = (weights*np.exp(2j*phi)).sum()/max(weights.sum(),1e-12)
    fft_coherence = float(abs(z))
    fft_theta = (np.degrees(np.angle(z))/2+90) % 180
    bins = np.rint(radius).astype(int)
    radial = np.bincount(bins.ravel(),weights=weights.ravel())
    peak = int(radial.argmax())
    concentration = float(radial[max(2,peak-1):peak+2].sum()/max(radial.sum(),1e-12))
    valid = bool(gray.std()>=.02 and coherence>=.25 and fft_coherence>=.35
                 and concentration>=.20 and axial_distance(theta,fft_theta)<=20)
    return {"theta":float(theta), "fft_theta":float(fft_theta), "coherence":float(coherence),
            "fft_coherence":fft_coherence, "spectral_concentration":concentration,
            "contrast":float(gray.std()), "valid":valid}


def contour(mask):
    t = torch.as_tensor(mask).float()[None,None]
    inner = -F.max_pool2d(-t,3,stride=1,padding=1)
    return (t-inner)[0,0].numpy()>.5


def measure(image, mask_image, sketch, reference, expected, roi):
    mask = to_tensor(mask_image)[None]
    interior, boundary, background = build_region_masks(mask,17)
    regions = {k:v[0,0].numpy()>.99 for k,v in
               (("interior",interior),("boundary",boundary),("background",background))}
    a = np.array(image).astype(float)/255.
    g = to_tensor(image)[None]
    result = structure(g, mask, sketch)
    direction = orientation(image,roi)
    direction["error"] = axial_distance(direction["theta"],expected)
    direction["correct"] = direction["valid"] and direction["error"]<=20
    direction["agreement"] = (1+math.cos(math.radians(2*direction["error"])))/2 if direction["valid"] else 0.
    fg = estimate_foreground_mask(image,image.size)
    pred_edge, target_edge = contour(fg), contour(mask[0,0].numpy()>.5)
    precision = (pred_edge&_dilate_binary(target_edge,5)).sum()/max(pred_edge.sum(),1)
    recall = (target_edge&_dilate_binary(pred_edge,5)).sum()/max(target_edge.sum(),1)
    lab = _rgb_to_lab(np.array(image))
    ref_lab = _rgb_to_lab(np.array(reference))
    color = lab[regions["interior"]].mean(0)
    median = np.median(lab[regions["interior"]],axis=0)
    result.update(direction=direction, contour_f1=float(2*precision*recall/max(precision+recall,1e-12)),
                  leakage=float((a.mean(-1)<.95)[regions["background"]].mean()),
                  background_white_mae=float(abs(a-1)[regions["background"]].mean()),
                  boundary_white_mae=float(abs(a-1)[regions["boundary"]].mean()),
                  interior_lab=color.tolist(), interior_median_lab=median.tolist(),
                  reference_color_delta=float(np.linalg.norm(color-ref_lab.mean((0,1)))))
    return result


def fixed_roi(mask_image):
    return interior_rectangle(build_region_masks(to_tensor(mask_image)[None],17)[0])
