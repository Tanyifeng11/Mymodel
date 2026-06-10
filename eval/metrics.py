"""
Evaluation metrics for multimodal garment generation.
Covers 4 categories for paper-ready ablation tables:

1. Generation Quality: FID, CLIP-I, SSIM, LPIPS
2. Texture Strength:  TSS, TCF, TPF
3. Texture Leakage:   LR, BAS, BCS
4. Structure Preservation: Edge F1, Sketch IoU
"""

import math
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg


# ============================================================================
# Helpers
# ============================================================================

def _open_rgb(path):
    return Image.open(path).convert("RGB")


def _open_mask(path, size):
    if path is None:
        return None
    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return m


def _iter_pixels(img, mask=None):
    px = list(img.getdata())
    if mask is None:
        return px
    m = list(mask.getdata())
    out = [p for p, mm in zip(px, m) if mm > 0]
    return out if out else px


def _pil_to_tensor(pil_img, size=None):
    """Convert PIL to [0,1] tensor [C,H,W]."""
    from torchvision import transforms
    if size:
        pil_img = pil_img.resize(size, Image.BICUBIC)
    return transforms.ToTensor()(pil_img)


def _pil_to_np(pil_img, size=None):
    """Convert PIL to uint8 numpy [H,W,C]."""
    if size:
        pil_img = pil_img.resize(size, Image.BICUBIC)
    return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)


# ============================================================================
# Category 1: Generation Quality
# ============================================================================

# ---- FID (Fréchet Inception Distance) ----

# Lazy-loaded InceptionV3 module for FID computation.
_inception_v3 = None


def _get_inception_v3(device="cuda"):
    """Lazy-load InceptionV3 with the standard FID feature layer (pool3, 2048-d)."""
    global _inception_v3
    if _inception_v3 is not None:
        return _inception_v3.to(device)

    from torchvision.models import inception_v3
    from torchvision.transforms import Normalize

    model = inception_v3(weights="DEFAULT", transform_input=False)
    model.fc = torch.nn.Identity()  # remove classification head
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Standard FID preprocessing
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    _inception_v3 = model
    return model.to(device), normalize


@torch.no_grad()
def extract_inception_features(
    image_paths: List[str],
    batch_size: int = 32,
    device: str = "cuda",
    resize_size: int = 299,
) -> np.ndarray:
    """
    Extract InceptionV3 pool3 features (2048-d) for a list of image paths.
    Returns numpy array of shape [N, 2048].
    """
    from torchvision.transforms.functional import resize as tv_resize

    model, normalize = _get_inception_v3(device)
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = _open_rgb(p).resize((resize_size, resize_size), Image.BICUBIC)
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            t = normalize(t)
            batch_tensors.append(t)
        batch = torch.stack(batch_tensors, dim=0).to(device)
        feat = model(batch).cpu().numpy()
        features.append(feat)

    return np.concatenate(features, axis=0)


def compute_fid(
    gen_features: np.ndarray,
    real_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute FID between generated and real feature sets.

    gen_features: [N_gen, D]
    real_features: [N_real, D]

    FID = ||mu_g - mu_r||^2 + Tr(Sigma_g + Sigma_r - 2*(Sigma_g*Sigma_r)^{1/2})
    """
    mu_g = np.mean(gen_features, axis=0)
    mu_r = np.mean(real_features, axis=0)
    sigma_g = np.cov(gen_features, rowvar=False)
    sigma_r = np.cov(real_features, rowvar=False)

    diff = mu_g - mu_r
    # Numerical stability: add small epsilon to diagonal
    sigma_g = sigma_g + np.eye(sigma_g.shape[0]) * eps
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps

    covmean, _ = linalg.sqrtm(sigma_g @ sigma_r, disp=False)
    # Handle numerical imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # Clip small negative eigenvalues
        covmean = np.maximum(covmean, 0.0)

    fid = float(diff @ diff + np.trace(sigma_g + sigma_r - 2.0 * covmean))
    return max(0.0, fid)


def compute_fid_from_paths(
    gen_paths: List[str],
    real_paths: List[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    """High-level FID: given two lists of image paths, return FID score."""
    gen_feat = extract_inception_features(gen_paths, batch_size=batch_size, device=device)
    real_feat = extract_inception_features(real_paths, batch_size=batch_size, device=device)
    return compute_fid(gen_feat, real_feat)


# ---- CLIP-I (CLIP Image Similarity) ----

_clip_model_cache = None
_clip_processor_cache = None


def _get_clip_model(device="cuda", model_name="openai/clip-vit-large-patch14"):
    """Lazy-load CLIP model for image similarity."""
    global _clip_model_cache, _clip_processor_cache
    if _clip_model_cache is not None:
        return _clip_model_cache.to(device), _clip_processor_cache

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _clip_model_cache = model
    _clip_processor_cache = processor
    return model.to(device), processor


@torch.no_grad()
def extract_clip_image_features(
    image_paths: List[str],
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Extract CLIP image embeddings [N, D] from image paths."""
    model, processor = _get_clip_model(device)
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [_open_rgb(p) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalize
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_clip_i(
    gen_paths: List[str],
    ref_paths: List[str],
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute CLIP-I between generated and reference images.

    Returns:
        clip_i_mean: mean cosine similarity
        clip_i_std:  standard deviation
    """
    gen_feat = extract_clip_image_features(gen_paths, batch_size=batch_size, device=device)
    ref_feat = extract_clip_image_features(ref_paths, batch_size=batch_size, device=device)

    # Pairwise cosine similarity (already L2-normalized, so dot product)
    sims = np.sum(gen_feat * ref_feat, axis=1)  # [N]

    return {
        "clip_i_mean": float(np.mean(sims)),
        "clip_i_std": float(np.std(sims)),
        "clip_i_min": float(np.min(sims)),
        "clip_i_max": float(np.max(sims)),
    }


# ---- SSIM (using skimage or torchmetrics) ----

def compute_ssim(img1: Image.Image, img2: Image.Image, mask: Optional[Image.Image] = None) -> float:
    """
    Compute SSIM between two images, optionally masked.
    Uses skimage if available, otherwise falls back to a simpler proxy.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_func
        arr1 = _pil_to_np(img1)
        arr2 = _pil_to_np(img2.resize(img1.size, Image.BICUBIC))
        if mask is not None:
            mask_np = np.asarray(mask.resize(img1.size, Image.NEAREST).convert("L")) > 127
            arr1 = arr1 * mask_np[..., None]
            arr2 = arr2 * mask_np[..., None]
        return float(ssim_func(arr1, arr2, channel_axis=2, data_range=255))
    except ImportError:
        # Fallback: our simplified ssim_like
        return ssim_like(img1, img2, mask)


# ============================================================================
# Category 2: Texture Strength (纹理强度)
# ============================================================================
# Measures how much the texture reference actually controls the output.
# Higher numbers = stronger texture influence (not necessarily better).

def compute_texture_sensitivity_score(
    gen_image_groups: List[List[str]],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Texture Sensitivity Score (TSS).
    For N different texture images applied to the same sketch,
    generate N images and measure average pairwise dissimilarity.

    gen_image_groups: List of groups, each group is a list of image paths
                      generated with different textures but the same sketch.

    Returns:
        tss_clip:   mean pairwise CLIP cosine distance (higher = more sensitive)
        tss_lab:    mean pairwise LAB color distance
        tss_hsv:    mean pairwise HSV histogram L1
    """
    if len(gen_image_groups) == 0 or len(gen_image_groups[0]) < 2:
        return {"tss_clip": 0.0, "tss_lab": 0.0, "tss_hsv": 0.0}

    model, processor = _get_clip_model(device)
    all_clip_dists = []
    all_lab_dists = []
    all_hsv_dists = []

    for group in gen_image_groups:
        if len(group) < 2:
            continue

        # CLIP features for this group
        images = [_open_rgb(p) for p in group]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Pairwise cosine distance
        sim_matrix = feats @ feats.T
        # Upper triangle (excluding diagonal)
        triu_idx = torch.triu_indices(len(group), len(group), offset=1)
        dists = (1.0 - sim_matrix[triu_idx[0], triu_idx[1]]).cpu().numpy()
        all_clip_dists.extend(dists.tolist())

        # LAB distances
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                lab_dist = _pairwise_lab_distance(images[i], images[j])
                all_lab_dists.append(lab_dist)

                hsv_dist = _pairwise_hsv_hist_l1(images[i], images[j])
                all_hsv_dists.append(hsv_dist)

    return {
        "tss_clip_mean": float(np.mean(all_clip_dists)) if all_clip_dists else 0.0,
        "tss_clip_std": float(np.std(all_clip_dists)) if all_clip_dists else 0.0,
        "tss_lab_mean": float(np.mean(all_lab_dists)) if all_lab_dists else 0.0,
        "tss_lab_std": float(np.std(all_lab_dists)) if all_lab_dists else 0.0,
        "tss_hsv_mean": float(np.mean(all_hsv_dists)) if all_hsv_dists else 0.0,
        "tss_hsv_std": float(np.std(all_hsv_dists)) if all_hsv_dists else 0.0,
    }


def compute_texture_color_fidelity(
    gen_path: str,
    texture_path: str,
    mask_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Texture Color Fidelity (TCF).
    How closely the generated garment's colors match the texture reference.
    Measured within the garment mask only.

    Returns:
        tcf_lab_delta:  LAB mean color distance (lower = better match)
        tcf_hsv_l1:     HSV histogram L1 distance (lower = better match)
        tcf_rgb_l2:     RGB mean L2 distance (lower = better match)
    """
    gen = _open_rgb(gen_path)
    tex = _open_rgb(texture_path).resize(gen.size, Image.BICUBIC)
    mask = _open_mask(mask_path, gen.size) if mask_path else None

    # LAB
    try:
        import cv2
        gen_lab = cv2.cvtColor(_pil_to_np(gen), cv2.COLOR_RGB2LAB).astype(np.float32)
        tex_lab = cv2.cvtColor(_pil_to_np(tex), cv2.COLOR_RGB2LAB).astype(np.float32)
        if mask is not None:
            mask_np = np.asarray(mask, dtype=np.uint8) > 127
            gen_px = gen_lab[mask_np]
            tex_px = tex_lab[mask_np] if mask_np.shape == tex_lab.shape[:2] else tex_lab.reshape(-1, 3)
        else:
            gen_px = gen_lab.reshape(-1, 3)
            tex_px = tex_lab.reshape(-1, 3)
        lab_delta = float(np.linalg.norm(gen_px.mean(axis=0) - tex_px.mean(axis=0)))
    except ImportError:
        lab_delta = 0.0

    # HSV histogram
    hsv_l1 = hist_l1(histogram_rgb(gen, mask=mask), histogram_rgb(tex))

    # RGB mean L2
    g_mean = mean_rgb(gen, mask=mask)
    t_mean = mean_rgb(tex)
    rgb_l2_val = rgb_l2(g_mean, t_mean)

    return {
        "tcf_lab_delta": lab_delta,
        "tcf_hsv_l1": hsv_l1,
        "tcf_rgb_l2": rgb_l2_val,
    }


def compute_texture_pattern_fidelity(
    gen_path: str,
    texture_path: str,
    mask_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Texture Pattern Fidelity (TPF).
    How closely the generated garment's local patterns match the texture.

    Returns:
        tpf_patch_sim:    Patch-level texture similarity (higher = better)
        tpf_gram_l1:      Gram matrix L1 distance at multiple VGG layers (lower = better)
    """
    gen = _open_rgb(gen_path)
    tex = _open_rgb(texture_path).resize(gen.size, Image.BICUBIC)
    mask = _open_mask(mask_path, gen.size) if mask_path else None

    # Patch similarity
    patch_sim = patch_texture_similarity(gen, tex, mask=mask, patch=8)

    # Gram matrix via VGG
    try:
        gram_l1 = _compute_gram_l1(gen, tex, mask=mask)
    except Exception:
        gram_l1 = 0.0

    return {
        "tpf_patch_sim": patch_sim,
        "tpf_gram_l1": gram_l1,
    }


# ============================================================================
# Category 3: Texture Leakage (纹理溢出)
# ============================================================================
# Measures unwanted texture "spill" outside the garment region.
# Lower numbers = less leakage = better.

def compute_texture_leakage(
    gen_path: str,
    mask_path: Optional[str] = None,
    dilate_kernel: int = 13,
) -> Dict[str, float]:
    """
    Compute texture leakage metrics from a generated image and its garment mask.

    Returns:
        leak_colored_frac:     Fraction of background pixels with noticeable color/saturation
        leak_mean_saturation:  Mean saturation in background region
        leak_value_shift:      Mean brightness shift in background (vs. expected white/gray)
        leak_edge_density:     Edge density at garment boundary (high = artifacts)
    """
    try:
        import cv2
    except ImportError:
        return {
            "leak_colored_frac": 0.0,
            "leak_mean_saturation": 0.0,
            "leak_value_shift": 0.0,
            "leak_edge_density": 0.0,
        }

    gen = _open_rgb(gen_path)
    arr = _pil_to_np(gen)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if mask_path is None:
        # Auto-estimate foreground mask
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0
        fg_mask = (sat > 0.10) | (val < 0.88)
        kernel = np.ones((7, 7), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        fg_mask = fg_mask > 127
    else:
        mask_img = Image.open(mask_path).convert("L").resize(gen.size, Image.NEAREST)
        fg_mask = np.asarray(mask_img, dtype=np.uint8) > 127

    # Dilate mask to define boundary/outside
    mask_u8 = fg_mask.astype(np.uint8) * 255
    dilated_u8 = cv2.dilate(mask_u8, np.ones((dilate_kernel, dilate_kernel), np.uint8))
    outside = ~(dilated_u8 > 127)
    boundary = (dilated_u8 > 127) & ~fg_mask

    # ---- Leakage: colored pixels outside ----
    if outside.sum() < 16:
        colored_frac = 0.0
        mean_sat = 0.0
        value_shift = 0.0
    else:
        outside_hsv = hsv[outside]
        sat = outside_hsv[:, 1] / 255.0
        val = outside_hsv[:, 2] / 255.0
        colored_frac = float(((sat > 0.16) & (val < 0.96)).mean())
        mean_sat = float(sat.mean())
        value_shift = float(abs(val.mean() - 0.95))

    # ---- Boundary artifacts: edge density at boundary ----
    if boundary.sum() < 16:
        edge_density = 0.0
    else:
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edge_density = float(np.abs(edges[boundary]).mean())

    return {
        "leak_colored_frac": colored_frac,
        "leak_mean_saturation": mean_sat,
        "leak_value_shift": value_shift,
        "leak_edge_density": edge_density,
    }


# ============================================================================
# Category 4: Structure Preservation (结构保持)
# ============================================================================

def compute_structure_preservation(
    gen_path: str,
    sketch_path: str,
    mask_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Measure how well the generated garment preserves the sketch structure.

    Returns:
        struct_edge_f1:     Edge F1 score between generated and sketch (higher = better)
        struct_iou:         Foreground IoU vs. sketch-derived mask (higher = better)
        struct_edge_l1:     Mean edge L1 distance (lower = better)
    """
    try:
        import cv2
    except ImportError:
        return {"struct_edge_f1": 0.0, "struct_iou": 0.0, "struct_edge_l1": 0.0}

    gen = _open_rgb(gen_path)
    sketch = _open_rgb(sketch_path).resize(gen.size, Image.BICUBIC)

    gen_np = _pil_to_np(gen)
    sketch_np = _pil_to_np(sketch)

    gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sketch_gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sobel edges
    gen_edge = np.abs(cv2.Sobel(gen_gray, cv2.CV_64F, 1, 1, ksize=3))
    sketch_edge = np.abs(cv2.Sobel(sketch_gray, cv2.CV_64F, 1, 1, ksize=3))

    # Normalize to [0, 1] for F1
    gen_edge_n = (gen_edge / (gen_edge.max() + 1e-6)).flatten()
    sketch_edge_n = (sketch_edge / (sketch_edge.max() + 1e-6)).flatten()

    # Binary threshold
    gen_bin = (gen_edge_n > 0.1).astype(np.float32)
    sketch_bin = (sketch_edge_n > 0.3).astype(np.float32)  # sketches have stronger edges

    tp = (gen_bin * sketch_bin).sum()
    fp = (gen_bin * (1 - sketch_bin)).sum()
    fn = ((1 - gen_bin) * sketch_bin).sum()
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    edge_f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # IoU of foreground
    if mask_path:
        mask = np.asarray(Image.open(mask_path).convert("L").resize(gen.size, Image.NEAREST)) > 127
        gen_fg = gen_gray < 240  # approximate foreground
        intersection = (gen_fg & mask).sum()
        union = (gen_fg | mask).sum()
        iou = intersection / max(union, 1.0)
    else:
        iou = 0.0

    # Edge L1
    edge_l1 = float(np.abs(gen_edge - sketch_edge).mean())

    return {
        "struct_edge_f1": float(edge_f1),
        "struct_iou": float(iou),
        "struct_edge_l1": edge_l1,
    }


# ============================================================================
# Internal helpers (unchanged from original, plus additions)
# ============================================================================

def mean_rgb(img, mask=None):
    px = _iter_pixels(img, mask)
    n = max(1, len(px))
    return (
        sum(p[0] for p in px) / n,
        sum(p[1] for p in px) / n,
        sum(p[2] for p in px) / n,
    )


def rgb_l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def histogram_rgb(img, bins=16, mask=None):
    px = _iter_pixels(img, mask)
    hist = [0] * (bins * 3)
    for r, g, b in px:
        hist[min(bins - 1, r * bins // 256)] += 1
        hist[bins + min(bins - 1, g * bins // 256)] += 1
        hist[2 * bins + min(bins - 1, b * bins // 256)] += 1
    s = sum(hist) or 1
    return [h / s for h in hist]


def hist_l1(h1, h2):
    return sum(abs(a - b) for a, b in zip(h1, h2))


def ssim_like(img1, img2, mask=None):
    p1 = _iter_pixels(img1, mask)
    p2 = _iter_pixels(img2, mask)
    n = max(1, min(len(p1), len(p2)))
    mse = sum(
        ((p1[i][0] - p2[i][0]) ** 2 + (p1[i][1] - p2[i][1]) ** 2 + (p1[i][2] - p2[i][2]) ** 2) / 3.0
        for i in range(n)
    ) / n
    return 1.0 / (1.0 + mse / (255.0 * 255.0))


def lpips_like(img1, img2, mask=None):
    p1 = _iter_pixels(img1, mask)
    p2 = _iter_pixels(img2, mask)
    n = max(1, min(len(p1), len(p2)))
    return sum(
        math.sqrt(
            (p1[i][0] - p2[i][0]) ** 2
            + (p1[i][1] - p2[i][1]) ** 2
            + (p1[i][2] - p2[i][2]) ** 2
        )
        / 255.0
        for i in range(n)
    ) / n


def patch_texture_similarity(gen_img, tex_img, mask=None, patch=8):
    if tex_img.size != gen_img.size:
        tex_img = tex_img.resize(gen_img.size, Image.BICUBIC)
    p1 = _iter_pixels(gen_img, mask)
    p2 = _iter_pixels(tex_img, mask)
    n = max(1, min(len(p1), len(p2)))
    step = max(1, patch * patch)
    sims = []
    for i in range(0, n, step):
        a = p1[i : i + step]
        b = p2[i : i + step]
        if not a or not b:
            continue
        ma = tuple(sum(x[c] for x in a) / len(a) for c in range(3))
        mb = tuple(sum(x[c] for x in b) / len(b) for c in range(3))
        sims.append(1.0 / (1.0 + rgb_l2(ma, mb)))
    return sum(sims) / max(1, len(sims))


def _pairwise_lab_distance(img1, img2):
    """LAB color distance proxy between two PIL images."""
    try:
        import cv2
        arr1 = _pil_to_np(img1)
        arr2 = _pil_to_np(img2.resize(img1.size, Image.BICUBIC))
        lab1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2LAB).astype(np.float32)
        return float(np.linalg.norm(lab1.mean(axis=(0, 1)) - lab2.mean(axis=(0, 1))))
    except ImportError:
        m1 = mean_rgb(img1)
        m2 = mean_rgb(img2)
        return rgb_l2(m1, m2)


def _pairwise_hsv_hist_l1(img1, img2):
    """HSV histogram L1 distance between two PIL images."""
    return hist_l1(histogram_rgb(img1), histogram_rgb(img2))


# Lazy VGG for Gram-based metrics
_vgg_gram = None


def _get_vgg_gram(device="cuda"):
    global _vgg_gram
    if _vgg_gram is not None:
        return _vgg_gram
    from torchvision.models import vgg19, VGG19_Weights
    from torchvision.transforms import Normalize

    feats = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    for p in feats.parameters():
        p.requires_grad = False
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _vgg_gram = (feats.to(device), norm)
    return _vgg_gram


def _compute_gram_l1(img1_pil, img2_pil, mask=None):
    """Compute Gram matrix L1 distance using VGG19 relu3_1 and relu4_1."""
    vgg, norm = _get_vgg_gram("cuda" if torch.cuda.is_available() else "cpu")
    device = next(vgg.parameters()).device

    t1 = _pil_to_tensor(img1_pil).unsqueeze(0).to(device)
    t2 = _pil_to_tensor(img2_pil).unsqueeze(0).to(device)
    t1 = norm(t1)
    t2 = norm(t2)

    if mask is not None:
        m = _pil_to_tensor(mask).unsqueeze(0).to(device)
        m = F.interpolate(m, size=t1.shape[-2:], mode="nearest")
        t1 = t1 * m
        t2 = t2 * m

    def gram(x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        return (x @ x.transpose(1, 2)) / (c * h * w + 1e-6)

    # relu3_1 = layer 18, relu4_1 = layer 27
    loss = 0.0
    for layer_idx in [17, 26]:
        f1 = vgg[:layer_idx + 1](t1)
        f2 = vgg[:layer_idx + 1](t2)
        loss += float(F.l1_loss(gram(f1), gram(f2)).item())

    return loss / 2.0


# ============================================================================
# Original evaluate_pair (preserved for backward compatibility)
# ============================================================================

def evaluate_pair(gen_path, target_path=None, texture_path=None, mask_path=None):
    gen = _open_rgb(gen_path)
    mask = _open_mask(mask_path, gen.size) if mask_path else None
    out = {}

    g_mean = mean_rgb(gen, mask=mask)
    out["gen_mean_r"], out["gen_mean_g"], out["gen_mean_b"] = g_mean

    if target_path:
        tgt = _open_rgb(target_path).resize(gen.size)
        out["lpips_like"] = lpips_like(gen, tgt, mask=mask)
        out["ssim_like"] = ssim_like(gen, tgt, mask=mask)
        out["hist_l1_target"] = hist_l1(histogram_rgb(gen, mask=mask), histogram_rgb(tgt, mask=mask))
        out["mean_rgb_l2_target"] = rgb_l2(g_mean, mean_rgb(tgt, mask=mask))

    if texture_path:
        tex = _open_rgb(texture_path).resize(gen.size)
        out["hist_l1_texture"] = hist_l1(histogram_rgb(gen, mask=mask), histogram_rgb(tex, mask=mask))
        out["patch_texture_similarity"] = patch_texture_similarity(gen, tex, mask=mask, patch=8)
    return out


# ============================================================================
# Full evaluation: run ALL metrics on a generated image
# ============================================================================

def evaluate_full(
    gen_path: str,
    target_path: Optional[str] = None,
    texture_path: Optional[str] = None,
    sketch_path: Optional[str] = None,
    mask_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run the complete evaluation suite on a single generated image.
    Returns a flat dict of all metrics for one sample.
    """
    result = {}

    # Quality (if target available)
    if target_path:
        result["ssim"] = compute_ssim(_open_rgb(gen_path), _open_rgb(target_path),
                                       _open_mask(mask_path, _open_rgb(gen_path).size) if mask_path else None)

    # Texture strength
    if texture_path:
        tcf = compute_texture_color_fidelity(gen_path, texture_path, mask_path)
        tpf = compute_texture_pattern_fidelity(gen_path, texture_path, mask_path)
        result.update(tcf)
        result.update(tpf)

    # Texture leakage
    leak = compute_texture_leakage(gen_path, mask_path)
    result.update(leak)

    # Structure preservation
    if sketch_path:
        struct = compute_structure_preservation(gen_path, sketch_path, mask_path)
        result.update(struct)

    # Also run original evaluate_pair for backward compat
    orig = evaluate_pair(gen_path, target_path, texture_path, mask_path)
    result.update(orig)

    return result
