"""验证字节基线、可达变幅、色域边界和已知线性响应。"""
import numpy as np

from tools.e2_b_amplitude_response_metrics import (
    from_lab, projection_features, response_summary, transform_reference,
)


def lab_pattern(amplitude=8., chroma=(0., 0.), quantize=True):
    _, x = np.indices((128, 128))
    value = np.empty((128, 128, 3), np.float64)
    value[..., 0] = 55 + amplitude * np.cos(2 * np.pi * x / 16)
    value[..., 1:] = chroma
    rgb = from_lab(value) * 255
    return np.rint(rgb).astype(np.uint8) if quantize else rgb


def test_factor_one_is_byte_exact_even_at_gamut_boundaries():
    picture = np.random.default_rng(7).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    picture[0, :3] = [[0, 0, 0], [255, 255, 255], [255, 0, 255]]
    actual, audit = transform_reference(picture, 1.)
    assert np.array_equal(actual, picture) and not np.shares_memory(actual, picture)
    assert audit['ab_mae'] == audit['mean_l_change'] == audit['low_l_mae'] == 0
    assert audit['gamut_restricted_fraction'] == 0
    assert abs(audit['effective_projection_gain'] - 1) < 1e-12


def test_unsaturated_reference_changes_texture_without_chroma_shift():
    picture = lab_pattern(amplitude=12., chroma=(12., -8.))
    low, lo = transform_reference(picture, .75)
    high, hi = transform_reference(picture, 1.25)
    assert lo['effective_hp_rms_ratio'] < 1 < hi['effective_hp_rms_ratio']
    assert not np.array_equal(low, high)
    for audit in (lo, hi):
        assert audit['gamut_restricted_fraction'] == 0
        assert abs(audit['effective_hp_rms_ratio'] - audit['requested_hp_rms_ratio']) < .015
        assert audit['hp_correlation'] > .995 and audit['ab_mae'] < .2
        assert abs(audit['mean_l_change']) < .1
        assert audit['low_frequency_preserved_exactly'] is False


def test_gamut_limit_preserves_stripe_order_instead_of_rgb_clipping():
    _, x = np.indices((128, 128))
    stripe = np.where(x % 16 < 8, 240, 32).astype(np.uint8)
    picture = np.repeat(stripe[..., None], 3, axis=2)
    actual, audit = transform_reference(picture, 2.)
    assert audit['gamut_restricted_fraction'] > 0
    assert audit['effective_hp_rms_ratio'] < audit['requested_hp_rms_ratio']
    assert audit['clipped_rgb_channel_fraction'] == 0
    assert audit['hp_correlation'] > .99 and audit['ab_mae'] < .01
    assert actual[stripe == 240, 0].mean() > actual[stripe == 32, 0].mean()


def test_known_point_four_response_has_point_four_slopes_and_constant_own_gain():
    mask = np.zeros((128, 128), np.uint8)
    mask[4:-4, 4:-4] = 1
    base = lab_pattern(amplitude=8., quantize=False)
    arms = {}
    # 连续 RGB 避开 PNG 量化：已知 L 正弦的响应斜率必须为 0.4。
    for label, factor in (('a075', .75), ('a100', 1.), ('a125', 1.25)):
        target = lab_pattern(amplitude=8. * factor, quantize=False)
        output = lab_pattern(amplitude=8. * factor * .4, quantize=False)
        arms[label] = projection_features(output, target, base, mask)
        assert abs(arms[label]['own_target_gain'] - .4) < 1e-10
        assert arms[label]['own_orthogonal_energy'] < 1e-15
        assert arms[label]['fixed_basis_orthogonal_residual_ratio'] < 1e-15
    summary = response_summary(arms)
    for key in ('low_slope', 'high_slope', 'center_slope'):
        assert abs(summary[key] - .4) < 1e-10
    assert summary['target_monotonic'] and summary['output_monotonic']


def test_identical_reachable_targets_do_not_produce_a_spurious_slope():
    arms = {label: dict(fixed_target_projection=1., fixed_output_projection=value)
            for label, value in (('a075', .3), ('a100', .4), ('a125', .5))}
    summary = response_summary(arms)
    assert summary['low_slope'] is summary['high_slope'] is summary['center_slope'] is None
    assert summary['target_monotonic'] is False
    assert summary['segments']['center']['delta_x'] == 0
    assert abs(summary['segments']['center']['delta_y'] - .2) < 1e-12
