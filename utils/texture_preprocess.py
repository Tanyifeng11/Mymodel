import random
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms


def _crop_tile_texture(
    image: Image.Image,
    out_size: Tuple[int, int],
    crop_scale_min: float = 0.4,
    crop_scale_max: float = 0.9,
) -> Image.Image:
    w, h = image.size
    min_side = min(w, h)
    scale = random.uniform(crop_scale_min, crop_scale_max)
    crop_size = max(32, int(min_side * scale))
    crop_size = min(crop_size, w, h)
    left = random.randint(0, max(0, w - crop_size))
    top = random.randint(0, max(0, h - crop_size))
    crop = image.crop((left, top, left + crop_size, top + crop_size))

    target_w, target_h = out_size
    tile_size = max(target_w, target_h)
    tile = crop.resize((max(1, tile_size // 2), max(1, tile_size // 2)), Image.BICUBIC)
    canvas = Image.new("RGB", (tile_size, tile_size))
    for yy in range(0, tile_size, tile.size[1]):
        for xx in range(0, tile_size, tile.size[0]):
            canvas.paste(tile, (xx, yy))
    return canvas.resize((target_w, target_h), Image.BICUBIC)


def preprocess_texture_image(
    image: Image.Image,
    width: int,
    height: int,
    mode: str = "crop_tile",
    crop_scale_min: float = 0.4,
    crop_scale_max: float = 0.9,
) -> torch.Tensor:
    """
    Shared texture preprocessing utility.
    Returns normalized tensor in [-1, 1] with shape [3, H, W].
    """
    mode = "plain_resize" if mode == "plain" else mode
    if mode not in ("plain_resize", "crop_tile"):
        raise ValueError(f"Unsupported texture_preprocess_mode: {mode}")

    if mode == "crop_tile":
        proc = _crop_tile_texture(
            image,
            out_size=(width, height),
            crop_scale_min=crop_scale_min,
            crop_scale_max=crop_scale_max,
        )
    else:
        proc = image.resize((width, height), Image.BICUBIC)

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return to_tensor(proc)
