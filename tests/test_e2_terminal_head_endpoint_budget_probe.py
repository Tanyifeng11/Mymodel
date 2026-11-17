import torch

from tools.e2_b_endpoint_losses import luminance
from tools.e2_terminal_head_endpoint_budget_probe import amplitude_parts, configurations, pixel_comparison, PROTECTION


def test_nine_part_decomposition_exactly_preserves_original_amplitude():
    torch.manual_seed(127)
    positive = torch.rand(1, 3, 96, 96)*.5+.25
    rgb = positive*.8+.1
    weight = torch.ones(1, 1, 96, 96)
    coords = [(x, y) for x, y in ((0, 0), (8, 0), (16, 0), (24, 0), (0, 8), (8, 8), (16, 8), (24, 8))]
    expected, _ = luminance(rgb, positive, weight, coords)
    actual = amplitude_parts(rgb, positive, weight, coords)
    assert actual['mean'] == float(expected['amplitude']) and len(actual['parts']) == 9
    assert actual['global_weight'] == 1/9
    assert abs(actual['continuous']['identity_residual']) < 1e-6
    assert abs(actual['continuous']['g']-actual['continuous']['A']*actual['continuous']['c']) < 1e-6


def test_counterfactuals_keep_actual_C1_and_raw_update_sign_and_norm():
    current, raw = torch.tensor([-.3, .4]), torch.tensor([2., 1.])
    values = configurations(current, raw)
    assert list(values) == ['zero', 'C1', 'twice_C1', 'norm_raw']
    assert torch.equal(values['C1'], current) and torch.equal(values['twice_C1'], 2*current)
    assert torch.allclose(values['norm_raw'].norm(), current.norm())
    assert torch.dot(values['norm_raw'], raw) < 0


def test_subset_pixel_comparison_never_claims_complete_group_pass():
    original = {name: .1 for name in PROTECTION}
    current = dict(original, old_projection=.1001)
    result = pixel_comparison(current, original)
    assert result['per_image_residue_cap_failed'] and not result['original_group_gate_assessed']
    assert 'passed' not in result
