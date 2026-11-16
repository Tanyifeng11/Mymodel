import numpy as np

from tools.e2_b_s0_cast_trace import FIELDS, cast_comparison, half_ulp_distance


def test_half_grid_distance_merges_signed_zero_and_handles_sign_and_binade():
    values = np.array([-1., -0., 0., 1.], dtype=np.float16)
    upward = np.nextafter(values, np.float16(np.inf))
    assert np.array_equal(half_ulp_distance(values, upward), np.ones(4, dtype=np.int32))
    assert half_ulp_distance(np.array([-0.], np.float16), np.array([0.], np.float16))[0] == 0
    tiny = np.nextafter(np.float16(0), np.float16(1))
    assert half_ulp_distance(np.array([-tiny], np.float16), np.array([tiny], np.float16))[0] == 2


def test_cast_chord_decomposition_does_not_treat_erased_chord_as_gradient():
    def sample(value):
        pre = np.array([value], dtype=np.float32)
        native = pre.astype(np.float16)
        return {field: pre if field in ('precast', 'delta') else native for field in FIELDS}
    # 两个不同FP32输入都舍入到1，观察弦消失，残差弦恰好补偿它。
    result = cast_comparison(sample(1.0001), sample(.9999), sample(1.), .125)
    assert result['fields']['precast']['chord']['nonzero'] == 1
    assert result['fields']['native']['chord']['nonzero'] == 0
    assert result['precast_nonzero_native_zero'] == 1
    assert result['decomposition_max_abs'] == 0
    assert result['rounding_chord_to_precast_norm'] == 1
    assert result['native_precast_chord_cosine'] is None
