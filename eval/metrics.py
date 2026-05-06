import math


def _open_rgb(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _open_mask(path, size):
    from PIL import Image

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
    # lightweight proxy for LPIPS in dependency-limited environments.
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
    from PIL import Image

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
