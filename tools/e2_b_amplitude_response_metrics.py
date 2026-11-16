"""冻结 B 的幅度配对诊断：实际 PNG 审计与固定目标基底上的响应。"""
import cv2
import numpy as np

from tools.e2_b_last_step_metrics import highpass, precise_l


MATRIX = np.array([[.4124564, .3575761, .1804375],
                   [.2126729, .7151522, .0721750],
                   [.0193339, .1191920, .9503041]])
WHITE = np.array([.95047, 1., 1.08883])
INVERSE_MATRIX = np.linalg.inv(MATRIX)
SLOPE_EPSILON = 1e-6


def to_lab(pixels):
    """与已有双精度 sRGB→Lab 口径一致；L 的范围为 0–100。"""
    rgb = np.asarray(pixels, np.float64) / 255
    linear = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (linear @ MATRIX.T) / WHITE
    f = np.where(xyz > .008856, np.maximum(xyz, .008856) ** (1 / 3),
                 7.787037 * xyz + 16 / 116)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), -1)


def from_lab(lab):
    """返回未裁剪的 0–1 RGB，用于先判断色域再量化。"""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack((fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200), -1)
    xyz = np.where(f > 6 / 29, f ** 3, (f - 16 / 116) / 7.787037) * WHITE
    linear = xyz @ INVERSE_MATRIX.T
    return np.where(linear > .0031308, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055,
                    12.92 * linear)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


def _cosine(a, b):
    return _ratio(np.sum(a * b), np.linalg.norm(a) * np.linalg.norm(b))


def transform_reference(rgb_uint8, factor):
    """只请求改变 L 高频；色域限制和 PNG 量化后的串扰均单独记录。"""
    original = np.asarray(rgb_uint8)
    assert original.dtype == np.uint8 and original.ndim == 3 and original.shape[-1] == 3
    factor = float(factor)
    source = to_lab(original)
    detail = highpass(source[..., 0])
    detail = detail - detail.mean()
    delta = (factor - 1) * detail
    requested = source.copy()
    requested[..., 0] += delta
    lower = np.ones(source.shape[:2], np.float64)
    requested_outside = np.zeros(source.shape[:2], bool)
    clipped_fraction = 0.
    if factor == 1.:
        # 基线必须逐字节相同，不能经过 Lab/RGB 往返后再声称没有干预。
        result = original.copy()
    else:
        valid = lambda rgb: ((rgb >= -1e-10) & (rgb <= 1 + 1e-10)).all(-1)
        requested_outside = ~valid(from_lab(requested))
        lower[requested_outside] = 0.
        upper = np.ones_like(lower)
        for _ in range(12):
            middle = (lower + upper) / 2
            trial = source.copy()
            trial[..., 0] += middle * delta
            ok = valid(from_lab(trial))
            lower = np.where(ok, middle, lower)
            upper = np.where(ok, upper, middle)
        limited = source.copy()
        limited[..., 0] += lower * delta
        rgb = from_lab(limited)
        clipped_fraction = float(((rgb < -1e-10) | (rgb > 1 + 1e-10)).mean())
        result = np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)

    actual = to_lab(result)
    before, after, wanted = [highpass(v[..., 0]) for v in (source, actual, requested)]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    original_rms = rms(before)
    light_change = actual[..., 0] - source[..., 0]
    low = cv2.GaussianBlur(light_change, (0, 0), 24)
    ab_change = np.abs(actual[..., 1:] - source[..., 1:])
    saturation = lambda v: float(((v == 0) | (v == 255)).any(-1).mean())
    audit = dict(
        requested_factor=factor,
        requested_hp_rms_ratio=_ratio(rms(wanted), original_rms),
        effective_hp_rms_ratio=_ratio(rms(after), original_rms),
        effective_projection_gain=_ratio(np.sum(after * before), np.sum(before ** 2)),
        original_hp_rms=original_rms, modified_hp_rms=rms(after),
        hp_correlation=_cosine(after, before),
        gamut_requested_outside_fraction=float(requested_outside.mean()),
        gamut_restricted_fraction=float((lower < 1.).mean()),
        gamut_scale_min=float(lower.min()), gamut_scale_mean=float(lower.mean()),
        clipped_rgb_channel_fraction=clipped_fraction,
        original_saturated_pixel_fraction=saturation(original),
        modified_saturated_pixel_fraction=saturation(result),
        original_mean_l=float(source[..., 0].mean()), modified_mean_l=float(actual[..., 0].mean()),
        mean_l_change=float(light_change.mean()), low_l_mae=float(np.abs(low).mean()),
        low_l_max=float(np.abs(low).max()), ab_mae=float(ab_change.mean()),
        ab_max=float(ab_change.max()), requested_delta_l_rms=rms(delta),
        realized_delta_l_rms=rms(light_change), low_frequency_preserved_exactly=bool(factor == 1.))
    return result, audit


def projection_features(output, positive, positive_base, mask):
    """以 α=1 的 VAE 目标为固定基底；正交量为能量比，各自目标增益理想恒为 1。"""
    inside = cv2.distanceTransform(np.asarray(mask, np.uint8), cv2.DIST_L2, 5) >= 16
    assert inside.any()
    h, p, base = [highpass(precise_l(v))[inside] for v in (output, positive, positive_base)]
    base_energy, target_energy = float(np.mean(base ** 2)), float(np.mean(p ** 2))
    x = _ratio(np.mean(p * base), base_energy)
    y = _ratio(np.mean(h * base), base_energy)
    g = _ratio(np.mean(h * p), target_energy)
    own_orthogonal = _ratio(np.mean((h - g * p) ** 2), target_energy) if g is not None else None
    fixed_orthogonal = _ratio(np.mean((h - y * base) ** 2), base_energy) if y is not None else None
    return dict(fixed_target_projection=x, fixed_output_projection=y, own_target_gain=g,
                own_orthogonal_energy=own_orthogonal,
                fixed_basis_orthogonal_residual_ratio=fixed_orthogonal,
                target_base_correlation=_cosine(p, base), own_correlation=_cosine(h, p),
                effective_target_rms_ratio=_ratio(np.sqrt(target_energy), np.sqrt(base_energy)),
                fixed_basis_energy=base_energy, own_target_energy=target_energy,
                output_energy=float(np.mean(h ** 2)), interior_pixels=int(inside.sum()))


def response_summary(arms):
    """斜率分母来自实际 VAE 目标，绝不使用名义 α 差替代。"""
    segments = {}
    for name, lo, hi in (('low', 'a075', 'a100'), ('high', 'a100', 'a125'),
                         ('center', 'a075', 'a125')):
        a, b = arms[lo], arms[hi]
        def difference(key):
            return float(b[key] - a[key]) if a[key] is not None and b[key] is not None else None
        dx, dy = difference('fixed_target_projection'), difference('fixed_output_projection')
        identified = dx is not None and abs(dx) > SLOPE_EPSILON
        segments[name] = dict(low_arm=lo, high_arm=hi, delta_x=dx, delta_y=dy,
                              identifiable=identified,
                              slope=float(dy / dx) if identified and dy is not None else None)
    def monotonic(key):
        values = [segments[name][key] for name in ('low', 'high')]
        return all(v > SLOPE_EPSILON for v in values) if all(v is not None for v in values) else None
    return dict(low_slope=segments['low']['slope'], high_slope=segments['high']['slope'],
                center_slope=segments['center']['slope'], segments=segments,
                target_monotonic=monotonic('delta_x'), output_monotonic=monotonic('delta_y'),
                slope_denominator_epsilon=SLOPE_EPSILON,
                monotonic_definition='两个相邻区间均严格增加，容差 1e-6；平坦响应不算通过')
