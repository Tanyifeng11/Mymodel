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
import warnings
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from eval.eval_utils import (
    estimate_foreground_mask,
    existing_file,
    prepare_evaluation_masks,
    safe_open_rgb,
)


# ============================================================================
# Helpers
# ============================================================================

def _open_rgb(path):
    image = safe_open_rgb(path)
    if image is None:
        raise FileNotFoundError(f"image missing or unreadable: {path}")
    return image


def _open_mask(path, size):
    if not existing_file(path):
        return None
    try:
        return Image.open(path).convert("L").resize(size, Image.NEAREST)
    except Exception:
        return None


def _iter_pixels(img, mask=None):
    px = list(img.getdata())
    if mask is None:
        return px
    m = list(mask.getdata())
    return [p for p, mm in zip(px, m) if mm > 0]


def _nan_result(keys, reason):
    warnings.warn(reason, RuntimeWarning)
    result = {key: float("nan") for key in keys}
    result["metric_warnings"] = [reason]
    return result


def _valid_pixels(mask, min_valid_pixels):
    return mask is not None and int(mask.sum()) >= int(min_valid_pixels)


def _rgb_to_lab(arr):
    try:
        from skimage.color import rgb2lab

        return rgb2lab(arr.astype(np.float32) / 255.0).astype(np.float32)
    except ImportError:
        try:
            import cv2

            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] *= 100.0 / 255.0
            lab[..., 1:] -= 128.0
            return lab
        except ImportError:
            rgb = arr.astype(np.float32) / 255.0
            rgb = np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                ((rgb + 0.055) / 1.055) ** 2.4,
            )
            xyz = rgb @ np.array(
                [
                    [0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041],
                ],
                dtype=np.float32,
            ).T
            xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
            delta = 6.0 / 29.0
            f_xyz = np.where(
                xyz > delta**3,
                np.cbrt(xyz),
                xyz / (3.0 * delta * delta) + 4.0 / 29.0,
            )
            return np.stack(
                [
                    116.0 * f_xyz[..., 1] - 16.0,
                    500.0 * (f_xyz[..., 0] - f_xyz[..., 1]),
                    200.0 * (f_xyz[..., 1] - f_xyz[..., 2]),
                ],
                axis=-1,
            ).astype(np.float32)


def _gradient_magnitude(gray):
    gray = gray.astype(np.float32) / 255.0
    grad_y, grad_x = np.gradient(gray)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _binary_edges(gray, low_threshold=0.08, high_threshold=0.16):
    try:
        import cv2

        return cv2.Canny(
            gray.astype(np.uint8),
            int(low_threshold * 1000),
            int(high_threshold * 1000),
        ) > 0
    except ImportError:
        magnitude = _gradient_magnitude(gray)
        adaptive = max(
            low_threshold,
            float(np.percentile(magnitude, 80)) if magnitude.size else low_threshold,
        )
        return magnitude >= adaptive


def _dilate_binary(mask, kernel_size=5):
    try:
        import cv2

        return cv2.dilate(
            mask.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ) > 0
    except ImportError:
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        return np.asarray(image.filter(ImageFilter.MaxFilter(kernel_size))) > 127


def _hsv_histogram(arr, mask=None, bins=(24, 16, 16)):
    hsv = np.asarray(Image.fromarray(arr, mode="RGB").convert("HSV"), dtype=np.uint8)
    pixels = hsv[mask] if mask is not None else hsv.reshape(-1, 3)
    if pixels.size == 0:
        return None
    parts = []
    for channel, channel_bins in enumerate(bins):
        hist, _ = np.histogram(
            pixels[:, channel],
            bins=channel_bins,
            range=(0, 256),
        )
        total = hist.sum()
        if total <= 0:
            return None
        parts.append(hist.astype(np.float64) / total / len(bins))
    return np.concatenate(parts)


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
        if isinstance(_inception_v3, tuple):
            model, normalize = _inception_v3
            return model.to(device), normalize
        _inception_v3 = None

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

    _inception_v3 = (model, normalize)
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
    from scipy import linalg

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
    """Compute FID with torchmetrics first and a torchvision fallback."""
    gen_paths = [path for path in gen_paths if existing_file(path)]
    real_paths = [path for path in real_paths if existing_file(path)]
    if len(gen_paths) < 2 or len(real_paths) < 2:
        raise ValueError(
            f"FID needs at least 2 generated and 2 real images; "
            f"got generated={len(gen_paths)}, real={len(real_paths)}"
        )
    device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        for paths, real in ((real_paths, True), (gen_paths, False)):
            for index in range(0, len(paths), batch_size):
                batch = []
                for path in paths[index : index + batch_size]:
                    image = _open_rgb(path).resize((299, 299), Image.BICUBIC)
                    tensor = torch.from_numpy(
                        np.asarray(image, dtype=np.float32) / 255.0
                    ).permute(2, 0, 1)
                    batch.append(tensor)
                metric.update(torch.stack(batch).to(device), real=real)
        return float(metric.compute().detach().cpu().item())
    except Exception as torchmetrics_error:
        warnings.warn(
            f"torchmetrics FID unavailable, using torchvision fallback: "
            f"{torchmetrics_error}",
            RuntimeWarning,
        )

    gen_feat = extract_inception_features(
        gen_paths, batch_size=batch_size, device=device
    )
    real_feat = extract_inception_features(
        real_paths, batch_size=batch_size, device=device
    )
    return compute_fid(gen_feat, real_feat)


# ---- CLIP-I (CLIP Image Similarity) ----

_clip_model_cache = {}


def _get_clip_model(device="cuda", model_name="openai/clip-vit-large-patch14"):
    """Lazy-load CLIP model for image similarity."""
    global _clip_model_cache
    device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    cache_key = str(model_name)
    if cache_key in _clip_model_cache:
        model, processor = _clip_model_cache[cache_key]
        return model.to(device), processor

    try:
        from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name)
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            processor = CLIPImageProcessor.from_pretrained(model_name)
    except Exception as clip_model_error:
        try:
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
            )

            try:
                model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            except Exception:
                model = CLIPVisionModelWithProjection.from_pretrained(
                    model_name, subfolder="models/image_encoder"
                )
            try:
                processor = CLIPImageProcessor.from_pretrained(model_name)
            except Exception:
                processor = CLIPImageProcessor()
            warnings.warn(
                f"CLIPModel unavailable at {model_name}; using "
                f"CLIPVisionModelWithProjection: {clip_model_error}",
                RuntimeWarning,
            )
        except Exception as vision_error:
            raise RuntimeError(
                f"failed to load CLIP model from {model_name}; "
                f"CLIPModel error={clip_model_error}; vision error={vision_error}"
            ) from vision_error
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _clip_model_cache[cache_key] = (model, processor)
    return model.to(device), processor


@torch.no_grad()
def extract_clip_image_features(
    image_paths: List[str],
    batch_size: int = 16,
    device: str = "cuda",
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """Extract CLIP image embeddings [N, D] from image paths."""
    image_paths = [path for path in image_paths if existing_file(path)]
    if not image_paths:
        raise ValueError("no valid images for CLIP-I")
    device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model, processor = _get_clip_model(device, model_name=model_name)
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [_open_rgb(p) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        if hasattr(model, "get_image_features"):
            feat = model.get_image_features(**inputs)
        else:
            feat = model(**inputs).image_embeds
        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalize
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_clip_i(
    gen_paths: List[str],
    ref_paths: List[str],
    batch_size: int = 16,
    device: str = "cuda",
    model_name: str = "openai/clip-vit-large-patch14",
) -> Dict[str, float]:
    """
    Compute CLIP-I between generated and reference images.

    Returns:
        clip_i_mean: mean cosine similarity
        clip_i_std:  standard deviation
    """
    sims = compute_clip_i_values(
        gen_paths,
        ref_paths,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
    )

    return {
        "clip_i_mean": float(np.mean(sims)),
        "clip_i_std": float(np.std(sims)),
        "clip_i_min": float(np.min(sims)),
        "clip_i_max": float(np.max(sims)),
    }


def compute_clip_i_values(
    gen_paths: List[str],
    ref_paths: List[str],
    batch_size: int = 16,
    device: str = "cuda",
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    if len(gen_paths) != len(ref_paths):
        raise ValueError(
            f"CLIP-I needs paired paths; got generated={len(gen_paths)}, "
            f"reference={len(ref_paths)}"
        )
    valid_pairs = [
        (gen_path, ref_path)
        for gen_path, ref_path in zip(gen_paths, ref_paths)
        if existing_file(gen_path) and existing_file(ref_path)
    ]
    if not valid_pairs:
        raise ValueError("no valid generated/reference pairs for CLIP-I")
    valid_gen, valid_ref = zip(*valid_pairs)
    gen_feat = extract_clip_image_features(
        list(valid_gen),
        batch_size=batch_size,
        device=device,
        model_name=model_name,
    )
    ref_feat = extract_clip_image_features(
        list(valid_ref),
        batch_size=batch_size,
        device=device,
        model_name=model_name,
    )
    return np.sum(gen_feat * ref_feat, axis=1)


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
        return {
            "tss_clip_mean": float("nan"),
            "tss_clip_std": float("nan"),
            "tss_lab_mean": float("nan"),
            "tss_lab_std": float("nan"),
            "tss_hsv_mean": float("nan"),
            "tss_hsv_std": float("nan"),
        }

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
        "tss_clip_mean": float(np.mean(all_clip_dists)) if all_clip_dists else float("nan"),
        "tss_clip_std": float(np.std(all_clip_dists)) if all_clip_dists else float("nan"),
        "tss_lab_mean": float(np.mean(all_lab_dists)) if all_lab_dists else float("nan"),
        "tss_lab_std": float(np.std(all_lab_dists)) if all_lab_dists else float("nan"),
        "tss_hsv_mean": float(np.mean(all_hsv_dists)) if all_hsv_dists else float("nan"),
        "tss_hsv_std": float(np.std(all_hsv_dists)) if all_hsv_dists else float("nan"),
    }


def compute_texture_color_fidelity(
    gen_path: str,
    texture_path: str,
    mask_path: Optional[str] = None,
    sketch_path: Optional[str] = None,
    target_path: Optional[str] = None,
    min_valid_pixels: int = 50,
    mask_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Texture Color Fidelity (TCF).
    How closely the generated garment's colors match the texture reference.
    Measured within the garment mask only.

    Returns:
        tcf_lab_delta:  LAB mean color distance (lower = better match)
        tcf_hsv_l1:     HSV histogram L1 distance (lower = better match)
        tcf_rgb_l2:     RGB mean L2 distance (lower = better match)
    """
    keys = [
        "tcf_lab_delta",
        "tcf_hsv_l1",
        "tcf_rgb_l2",
        "gen_lab_mean_L",
        "gen_lab_mean_a",
        "gen_lab_mean_b",
        "tex_lab_mean_L",
        "tex_lab_mean_a",
        "tex_lab_mean_b",
        "garment_mask_pixels",
    ]
    gen = safe_open_rgb(gen_path)
    tex = safe_open_rgb(texture_path)
    if gen is None:
        return _nan_result(keys, f"TCF skipped: generated image missing: {gen_path}")
    if tex is None:
        return _nan_result(keys, f"TCF skipped: texture image missing: {texture_path}")

    tex = tex.resize(gen.size, Image.BICUBIC)
    if mask_bundle is None:
        mask_bundle = prepare_evaluation_masks(
            gen.size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=gen_path,
        )
    garment_mask = mask_bundle.get("garment")
    garment_pixels = int(garment_mask.sum()) if garment_mask is not None else 0
    if not _valid_pixels(garment_mask, min_valid_pixels):
        result = _nan_result(
            keys,
            f"TCF skipped: garment mask has {garment_pixels} pixels, "
            f"minimum is {min_valid_pixels}",
        )
        result["garment_mask_pixels"] = garment_pixels
        return result

    gen_arr = _pil_to_np(gen)
    tex_arr = _pil_to_np(tex)
    try:
        gen_lab = _rgb_to_lab(gen_arr)
        tex_lab = _rgb_to_lab(tex_arr)
    except Exception as exc:
        result = _nan_result(keys, f"TCF LAB conversion failed: {exc}")
        result["garment_mask_pixels"] = garment_pixels
        return result

    gen_lab_mean = gen_lab[garment_mask].mean(axis=0)
    tex_lab_mean = tex_lab.reshape(-1, 3).mean(axis=0)
    lab_delta = float(np.linalg.norm(gen_lab_mean - tex_lab_mean))

    gen_hsv_hist = _hsv_histogram(gen_arr, garment_mask)
    tex_hsv_hist = _hsv_histogram(tex_arr)
    hsv_l1 = (
        float(np.abs(gen_hsv_hist - tex_hsv_hist).sum())
        if gen_hsv_hist is not None and tex_hsv_hist is not None
        else float("nan")
    )
    gen_rgb_mean = gen_arr[garment_mask].astype(np.float32).mean(axis=0)
    tex_rgb_mean = tex_arr.reshape(-1, 3).astype(np.float32).mean(axis=0)
    rgb_l2_val = float(np.linalg.norm(gen_rgb_mean - tex_rgb_mean))

    return {
        "tcf_lab_delta": lab_delta,
        "tcf_hsv_l1": hsv_l1,
        "tcf_rgb_l2": rgb_l2_val,
        "gen_lab_mean_L": float(gen_lab_mean[0]),
        "gen_lab_mean_a": float(gen_lab_mean[1]),
        "gen_lab_mean_b": float(gen_lab_mean[2]),
        "tex_lab_mean_L": float(tex_lab_mean[0]),
        "tex_lab_mean_a": float(tex_lab_mean[1]),
        "tex_lab_mean_b": float(tex_lab_mean[2]),
        "garment_mask_pixels": garment_pixels,
        "metric_warnings": [],
    }


def compute_texture_pattern_fidelity(
    gen_path: str,
    texture_path: str,
    mask_path: Optional[str] = None,
    sketch_path: Optional[str] = None,
    target_path: Optional[str] = None,
    min_valid_pixels: int = 50,
    mask_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Texture Pattern Fidelity (TPF).
    How closely the generated garment's local patterns match the texture.

    Returns:
        tpf_patch_sim:    Patch-level texture similarity (higher = better)
        tpf_gram_l1:      Gram matrix L1 distance at multiple VGG layers (lower = better)
    """
    keys = ["tpf_patch_sim", "tpf_gram_l1"]
    gen = safe_open_rgb(gen_path)
    tex = safe_open_rgb(texture_path)
    if gen is None:
        return _nan_result(keys, f"TPF skipped: generated image missing: {gen_path}")
    if tex is None:
        return _nan_result(keys, f"TPF skipped: texture image missing: {texture_path}")
    tex = tex.resize(gen.size, Image.BICUBIC)
    if mask_bundle is None:
        mask_bundle = prepare_evaluation_masks(
            gen.size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=gen_path,
        )
    garment_mask = mask_bundle.get("garment")
    garment_pixels = int(garment_mask.sum()) if garment_mask is not None else 0
    if not _valid_pixels(garment_mask, min_valid_pixels):
        return _nan_result(
            keys,
            f"TPF skipped: garment mask has {garment_pixels} pixels, "
            f"minimum is {min_valid_pixels}",
        )
    mask = Image.fromarray(garment_mask.astype(np.uint8) * 255, mode="L")

    # Patch similarity
    patch_sim = patch_texture_similarity(gen, tex, mask=mask, patch=8)

    # Gram matrix via VGG
    try:
        gram_l1 = _compute_gram_l1(gen, tex, mask=mask)
        metric_warnings = []
    except Exception as exc:
        gram_l1 = float("nan")
        metric_warnings = [f"TPF Gram skipped: {exc}"]
        warnings.warn(metric_warnings[0], RuntimeWarning)

    return {
        "tpf_patch_sim": patch_sim,
        "tpf_gram_l1": gram_l1,
        "metric_warnings": metric_warnings,
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
    target_path: Optional[str] = None,
    sketch_path: Optional[str] = None,
    min_valid_pixels: int = 50,
    mask_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute texture leakage metrics from a generated image and its garment mask.

    Returns:
        leak_colored_frac:     Fraction of background pixels with noticeable color/saturation
        leak_mean_saturation:  Mean saturation in background region
        leak_value_shift:      Mean brightness shift in background (vs. expected white/gray)
        leak_edge_density:     Edge density at garment boundary (high = artifacts)
    """
    keys = [
        "leak_colored_frac",
        "leak_mean_saturation",
        "leak_value_shift",
        "leak_edge_density",
        "outside_mask_pixels",
        "boundary_mask_pixels",
    ]
    gen = safe_open_rgb(gen_path)
    if gen is None:
        return _nan_result(keys, f"leakage skipped: generated image missing: {gen_path}")
    arr = _pil_to_np(gen)
    hsv = np.asarray(gen.convert("HSV"), dtype=np.float32)
    if mask_bundle is None:
        mask_bundle = prepare_evaluation_masks(
            gen.size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=gen_path,
            kernel_size=dilate_kernel,
        )
    outside = mask_bundle.get("outside")
    boundary = mask_bundle.get("boundary")
    outside_pixels = int(outside.sum()) if outside is not None else 0
    boundary_pixels = int(boundary.sum()) if boundary is not None else 0
    metric_warnings = []

    if outside_pixels < min_valid_pixels:
        result = _nan_result(
            keys,
            f"leakage skipped: outside mask has {outside_pixels} pixels, "
            f"minimum is {min_valid_pixels}",
        )
        result["outside_mask_pixels"] = outside_pixels
        result["boundary_mask_pixels"] = boundary_pixels
        return result

    sat = hsv[..., 1] / 255.0
    colored = sat > 0.16
    target = safe_open_rgb(target_path)
    target_arr = None
    if target is not None:
        target_arr = _pil_to_np(target, size=gen.size)
        target_hsv = np.asarray(
            Image.fromarray(target_arr, mode="RGB").convert("HSV"),
            dtype=np.float32,
        )
        color_delta = np.linalg.norm(
            arr.astype(np.float32) - target_arr.astype(np.float32), axis=2
        ) / math.sqrt(3.0 * 255.0 * 255.0)
        colored = colored | (color_delta > 0.08)
        value_shift = float(
            np.abs(hsv[..., 2] - target_hsv[..., 2])[outside].mean() / 255.0
        )
    else:
        value_shift = float("nan")
        metric_warnings.append(
            f"leak_value_shift skipped: target image missing: {target_path}"
        )

    colored_frac = float(colored[outside].mean())
    mean_sat = float(sat[outside].mean())

    if boundary_pixels < min_valid_pixels:
        edge_density = float("nan")
        metric_warnings.append(
            f"leak_edge_density skipped: boundary mask has {boundary_pixels} pixels"
        )
    else:
        gray = np.asarray(gen.convert("L"), dtype=np.float32)
        edges = _binary_edges(gray)
        if target_arr is not None:
            target_gray = np.asarray(
                Image.fromarray(target_arr, mode="RGB").convert("L"),
                dtype=np.float32,
            )
            target_edges = _binary_edges(target_gray)
            edge_density = float((edges != target_edges)[boundary].mean())
        else:
            edge_density = float(edges[boundary].mean())

    for warning in metric_warnings:
        warnings.warn(warning, RuntimeWarning)

    return {
        "leak_colored_frac": colored_frac,
        "leak_mean_saturation": mean_sat,
        "leak_value_shift": value_shift,
        "leak_edge_density": edge_density,
        "outside_mask_pixels": outside_pixels,
        "boundary_mask_pixels": boundary_pixels,
        "metric_warnings": metric_warnings,
    }


# ============================================================================
# Category 4: Structure Preservation (结构保持)
# ============================================================================

def compute_structure_preservation(
    gen_path: str,
    sketch_path: str,
    mask_path: Optional[str] = None,
    target_path: Optional[str] = None,
    min_valid_pixels: int = 50,
    mask_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Measure how well the generated garment preserves the sketch structure.

    Returns:
        struct_edge_f1:     Edge F1 score between generated and sketch (higher = better)
        struct_iou:         Foreground IoU vs. sketch-derived mask (higher = better)
        struct_edge_l1:     Mean edge L1 distance (lower = better)
    """
    keys = [
        "struct_edge_f1",
        "struct_edge_precision",
        "struct_edge_recall",
        "struct_iou",
        "struct_edge_l1",
        "edge_f1",
        "edge_precision",
        "edge_recall",
        "sketch_iou",
        "edge_l1",
        "sketch_edge_pixels",
        "gen_edge_pixels",
    ]
    gen = safe_open_rgb(gen_path)
    sketch = safe_open_rgb(sketch_path)
    if gen is None:
        return _nan_result(keys, f"structure skipped: generated image missing: {gen_path}")
    if sketch is None:
        return _nan_result(keys, f"structure skipped: sketch image missing: {sketch_path}")
    sketch = sketch.resize(gen.size, Image.BICUBIC)

    gen_gray = np.asarray(gen.convert("L"), dtype=np.float32)
    sketch_gray = np.asarray(sketch.convert("L"), dtype=np.float32)

    gen_edge = _binary_edges(gen_gray, low_threshold=0.08, high_threshold=0.16)
    sketch_edge = _binary_edges(
        sketch_gray, low_threshold=0.04, high_threshold=0.12
    )
    gen_edge_pixels = int(gen_edge.sum())
    sketch_edge_pixels = int(sketch_edge.sum())
    if (
        gen_edge_pixels < min_valid_pixels
        or sketch_edge_pixels < min_valid_pixels
    ):
        result = _nan_result(
            keys,
            f"structure skipped: edge pixels generated={gen_edge_pixels}, "
            f"sketch={sketch_edge_pixels}, minimum={min_valid_pixels}",
        )
        result["gen_edge_pixels"] = gen_edge_pixels
        result["sketch_edge_pixels"] = sketch_edge_pixels
        return result

    sketch_dilated = _dilate_binary(sketch_edge, kernel_size=5)
    gen_dilated = _dilate_binary(gen_edge, kernel_size=5)
    precision = float((gen_edge & sketch_dilated).sum() / gen_edge_pixels)
    recall = float((sketch_edge & gen_dilated).sum() / sketch_edge_pixels)
    edge_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    if mask_bundle is None:
        mask_bundle = prepare_evaluation_masks(
            gen.size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=gen_path,
        )
    sketch_mask = mask_bundle.get("garment")
    gen_mask = estimate_foreground_mask(gen, gen.size)
    if (
        not _valid_pixels(sketch_mask, min_valid_pixels)
        or not _valid_pixels(gen_mask, min_valid_pixels)
    ):
        iou = float("nan")
        metric_warnings = ["struct_iou skipped: invalid generated or sketch mask"]
    else:
        intersection = int((gen_mask & sketch_mask).sum())
        union = int((gen_mask | sketch_mask).sum())
        iou = float(intersection / union) if union >= min_valid_pixels else float("nan")
        metric_warnings = []

    gen_magnitude = _gradient_magnitude(gen_gray)
    sketch_magnitude = _gradient_magnitude(sketch_gray)
    edge_l1 = float(np.abs(gen_magnitude - sketch_magnitude).mean())

    return {
        "struct_edge_f1": float(edge_f1),
        "struct_edge_precision": precision,
        "struct_edge_recall": recall,
        "struct_iou": float(iou),
        "struct_edge_l1": edge_l1,
        "edge_f1": float(edge_f1),
        "edge_precision": precision,
        "edge_recall": recall,
        "sketch_iou": float(iou),
        "edge_l1": edge_l1,
        "sketch_edge_pixels": sketch_edge_pixels,
        "gen_edge_pixels": gen_edge_pixels,
        "metric_warnings": metric_warnings,
    }


# ============================================================================
# Internal helpers (unchanged from original, plus additions)
# ============================================================================

def mean_rgb(img, mask=None):
    px = _iter_pixels(img, mask)
    if not px:
        return (float("nan"), float("nan"), float("nan"))
    n = len(px)
    return (
        sum(p[0] for p in px) / n,
        sum(p[1] for p in px) / n,
        sum(p[2] for p in px) / n,
    )


def rgb_l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def histogram_rgb(img, bins=16, mask=None):
    px = _iter_pixels(img, mask)
    if not px:
        return [float("nan")] * (bins * 3)
    hist = [0] * (bins * 3)
    for r, g, b in px:
        hist[min(bins - 1, r * bins // 256)] += 1
        hist[bins + min(bins - 1, g * bins // 256)] += 1
        hist[2 * bins + min(bins - 1, b * bins // 256)] += 1
    s = sum(hist) or 1
    return [h / s for h in hist]


def hist_l1(h1, h2):
    values = [abs(a - b) for a, b in zip(h1, h2)]
    return float(sum(values)) if values and all(math.isfinite(v) for v in values) else float("nan")


def ssim_like(img1, img2, mask=None):
    p1 = _iter_pixels(img1, mask)
    p2 = _iter_pixels(img2, mask)
    n = min(len(p1), len(p2))
    if n == 0:
        return float("nan")
    mse = sum(
        ((p1[i][0] - p2[i][0]) ** 2 + (p1[i][1] - p2[i][1]) ** 2 + (p1[i][2] - p2[i][2]) ** 2) / 3.0
        for i in range(n)
    ) / n
    return 1.0 / (1.0 + mse / (255.0 * 255.0))


def lpips_like(img1, img2, mask=None):
    p1 = _iter_pixels(img1, mask)
    p2 = _iter_pixels(img2, mask)
    n = min(len(p1), len(p2))
    if n == 0:
        return float("nan")
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
    n = min(len(p1), len(p2))
    if n == 0:
        return float("nan")
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
    return sum(sims) / len(sims) if sims else float("nan")


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

    if existing_file(target_path):
        tgt = _open_rgb(target_path).resize(gen.size)
        out["lpips_like"] = lpips_like(gen, tgt, mask=mask)
        out["ssim_like"] = ssim_like(gen, tgt, mask=mask)
        out["hist_l1_target"] = hist_l1(histogram_rgb(gen, mask=mask), histogram_rgb(tgt, mask=mask))
        out["mean_rgb_l2_target"] = rgb_l2(g_mean, mean_rgb(tgt, mask=mask))

    if existing_file(texture_path):
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
    compute_leakage: bool = True,
    compute_structure: bool = True,
    min_valid_pixels: int = 50,
    mask_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete evaluation suite on a single generated image.
    Returns a flat dict of all metrics for one sample.
    """
    result = {}
    metric_warnings = []
    gen = safe_open_rgb(gen_path)
    if gen is None:
        return _nan_result(
            [
                "ssim",
                "tcf_lab_delta",
                "tcf_hsv_l1",
                "tcf_rgb_l2",
                "tpf_patch_sim",
                "tpf_gram_l1",
                "leak_colored_frac",
                "leak_mean_saturation",
                "leak_value_shift",
                "leak_edge_density",
                "struct_edge_f1",
                "struct_iou",
                "struct_edge_l1",
            ],
            f"all metrics skipped: generated image missing: {gen_path}",
        )

    if mask_bundle is None:
        mask_bundle = prepare_evaluation_masks(
            gen.size,
            mask_path=mask_path,
            sketch_path=sketch_path,
            target_path=target_path,
            gen_path=gen_path,
        )
    result.update(mask_bundle.get("stats", {}))
    metric_warnings.extend(mask_bundle.get("warnings", []))

    if existing_file(target_path):
        try:
            target = _open_rgb(target_path)
            mask = mask_bundle.get("garment")
            mask_image = (
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                if _valid_pixels(mask, min_valid_pixels)
                else None
            )
            result["ssim"] = compute_ssim(gen, target, mask_image)
        except Exception as exc:
            result["ssim"] = float("nan")
            metric_warnings.append(f"SSIM skipped: {exc}")
    else:
        result["ssim"] = float("nan")
        metric_warnings.append(f"SSIM skipped: target image missing: {target_path}")

    tcf = compute_texture_color_fidelity(
        gen_path,
        texture_path,
        mask_path=mask_path,
        sketch_path=sketch_path,
        target_path=target_path,
        min_valid_pixels=min_valid_pixels,
        mask_bundle=mask_bundle,
    )
    tpf = compute_texture_pattern_fidelity(
        gen_path,
        texture_path,
        mask_path=mask_path,
        sketch_path=sketch_path,
        target_path=target_path,
        min_valid_pixels=min_valid_pixels,
        mask_bundle=mask_bundle,
    )
    metric_warnings.extend(tcf.pop("metric_warnings", []))
    metric_warnings.extend(tpf.pop("metric_warnings", []))
    result.update(tcf)
    result.update(tpf)

    if compute_leakage:
        leak = compute_texture_leakage(
            gen_path,
            mask_path=mask_path,
            target_path=target_path,
            sketch_path=sketch_path,
            min_valid_pixels=min_valid_pixels,
            mask_bundle=mask_bundle,
        )
        metric_warnings.extend(leak.pop("metric_warnings", []))
        result.update(leak)
    else:
        result.update(
            {
                key: float("nan")
                for key in (
                    "leak_colored_frac",
                    "leak_mean_saturation",
                    "leak_value_shift",
                    "leak_edge_density",
                )
            }
        )

    if compute_structure:
        struct = compute_structure_preservation(
            gen_path,
            sketch_path,
            mask_path=mask_path,
            target_path=target_path,
            min_valid_pixels=min_valid_pixels,
            mask_bundle=mask_bundle,
        )
        metric_warnings.extend(struct.pop("metric_warnings", []))
        result.update(struct)
    else:
        result.update(
            {
                key: float("nan")
                for key in (
                    "struct_edge_f1",
                    "struct_edge_precision",
                    "struct_edge_recall",
                    "struct_iou",
                    "struct_edge_l1",
                    "edge_f1",
                    "edge_precision",
                    "edge_recall",
                    "sketch_iou",
                    "edge_l1",
                )
            }
        )

    try:
        result.update(evaluate_pair(gen_path, target_path, texture_path, mask_path))
    except Exception as exc:
        metric_warnings.append(f"legacy pair metrics skipped: {exc}")
    result["metric_warnings"] = sorted(set(metric_warnings))
    return result
