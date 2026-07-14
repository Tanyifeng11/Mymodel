"""Distribution metrics used by the low-cost diagnostics."""

from typing import Dict, List, Optional

import numpy as np
import torch

from eval.metrics import compute_fid_from_paths, extract_inception_features


def resolve_device(device: str) -> str:
    return device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"


def compute_kid_from_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subsets: int = 50,
    subset_size: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Unbiased polynomial-kernel MMD, reported as KID mean/std."""
    n = min(int(subset_size), len(real_features), len(fake_features))
    if n < 2:
        raise ValueError("KID needs at least 2 real and 2 generated images")
    rng = np.random.default_rng(seed)
    dim = real_features.shape[1]
    values = []
    for _ in range(int(subsets)):
        x = real_features[rng.choice(len(real_features), n, replace=False)]
        y = fake_features[rng.choice(len(fake_features), n, replace=False)]
        k_xx = (x @ x.T / dim + 1.0) ** 3
        k_yy = (y @ y.T / dim + 1.0) ** 3
        k_xy = (x @ y.T / dim + 1.0) ** 3
        value = (
            (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
            - 2.0 * k_xy.mean()
        )
        values.append(float(value))
    return {
        "kid_mean": float(np.mean(values)),
        "kid_std": float(np.std(values)),
        "kid_subsets": int(subsets),
        "kid_subset_size": int(n),
    }


def compute_clean_fid(
    real_dir: str,
    fake_dir: str,
    device: str = "cuda",
    num_workers: int = 0,
) -> Dict[str, Optional[float]]:
    try:
        from cleanfid import fid

        value = fid.compute_fid(
            real_dir,
            fake_dir,
            device=resolve_device(device),
            num_workers=int(num_workers),
        )
        return {"clean_fid": float(value), "clean_fid_error": None}
    except Exception as error:
        return {"clean_fid": None, "clean_fid_error": str(error)}


def compute_distribution_metrics(
    real_paths: List[str],
    fake_paths: List[str],
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 42,
    kid_subsets: int = 50,
    kid_subset_size: int = 100,
    clean_fid: bool = False,
    real_dir: Optional[str] = None,
    fake_dir: Optional[str] = None,
) -> Dict[str, object]:
    device = resolve_device(device)
    legacy_fid, backend = compute_fid_from_paths(
        fake_paths,
        real_paths,
        batch_size=batch_size,
        device=device,
        return_backend=True,
    )
    real_features = extract_inception_features(real_paths, batch_size, device)
    fake_features = extract_inception_features(fake_paths, batch_size, device)
    result = {
        "num_real": len(real_paths),
        "num_generated": len(fake_paths),
        "legacy_fid": float(legacy_fid),
        "legacy_fid_backend": backend,
        **compute_kid_from_features(
            real_features,
            fake_features,
            subsets=kid_subsets,
            subset_size=kid_subset_size,
            seed=seed,
        ),
    }
    if clean_fid:
        if not real_dir or not fake_dir:
            result.update({"clean_fid": None, "clean_fid_error": "real_dir/fake_dir missing"})
        else:
            result.update(compute_clean_fid(real_dir, fake_dir, device=device))
    return result
