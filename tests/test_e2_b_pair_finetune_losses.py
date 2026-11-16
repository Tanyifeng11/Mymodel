"""检查幅度偏置、实际目标标定、分支梯度和逐项位移保护。"""
import pytest
import torch

from models.e2_lab_anchor import lab_to_rgb
from tools.e2_b_pair_finetune_losses import (
    ARMS, pair_coefficients, pair_terms, project_displacement, projection,
)


def picture(amplitude):
    x = torch.arange(96, dtype=torch.float32)[None, :].expand(96, -1)
    detail = torch.cos(2 * torch.pi * x / 16)
    light = 55 + amplitude * detail
    return lab_to_rgb(torch.stack((light, torch.zeros_like(light), torch.zeros_like(light)))[None])


def weight():
    value = torch.zeros(1, 1, 96, 96)
    value[..., 24:72, 24:72] = 1
    return value


def targets():
    return {arm: picture(amplitude) for arm, amplitude in zip(ARMS, (6., 8., 10.))}


def test_exact_target_has_unit_gain_and_zero_pair_absolute_and_orthogonal_errors():
    torch.set_num_threads(2)
    positive = targets()
    result = pair_terms(positive, positive, weight())
    assert float(result['pair']) == 0
    assert float(result['absolute']) == 0
    assert abs(float(result['details']['slope']) - 1) < 1e-7
    for arm in ARMS:
        features = result['details']['arms'][arm]
        assert features['g'].dtype == torch.float32
        assert abs(float(features['g']) - 1) < 1e-7
        assert float(features['orth']) < 1e-12


def test_constant_projection_offset_escapes_pair_loss_but_not_absolute_alignment():
    positive = targets()
    # 每档都比目标少2个L幅度单位：响应差分正确，但基准gain仅0.75。
    shifted = {arm: picture(amplitude) for arm, amplitude in zip(ARMS, (4., 6., 8.))}
    result = pair_terms(shifted, positive, weight())
    assert float(result['pair']) < 1e-10
    assert abs(float(result['absolute']) - .0625) < 1e-6
    assert abs(float(result['details']['slope']) - 1) < 1e-5


def test_target_span_uses_actual_target_and_rejects_identical_targets():
    positive = {arm: picture(amplitude) for arm, amplitude in zip(ARMS, (7.6, 8., 8.4))}
    output = {arm: picture(amplitude) for arm, amplitude in zip(ARMS, (3.8, 4., 4.2))}
    result = pair_terms(output, positive, weight())
    assert abs(float(result['details']['target_span']) - .1) < 1e-6
    assert abs(float(result['details']['slope']) - .5) < 1e-5
    identical = dict.fromkeys(ARMS, picture(8.))
    with pytest.raises(AssertionError, match='跨度'):
        pair_terms(output, identical, weight())


def test_branch_coefficients_match_joint_image_gradients_and_have_opposite_signs():
    positive = targets()
    low = torch.tensor(4., requires_grad=True)
    high = torch.tensor(6., requires_grad=True)
    outputs = {'a075': picture(low), 'a100': picture(5.), 'a125': picture(high)}
    result = pair_terms(outputs, positive, weight())
    gradients = torch.autograd.grad(result['pair'], (low, high))
    values = result['details']['arms']
    coefficients = pair_coefficients(values['a075']['y'], values['a125']['y'],
                                     values['a075']['x'], values['a125']['x'])
    assert not coefficients['low'].requires_grad
    assert float(coefficients['low']) > 0 and float(coefficients['high']) < 0
    branch_low = projection(picture(low), positive['a075'], positive['a100'], weight())['y']
    branch_high = projection(picture(high), positive['a125'], positive['a100'], weight())['y']
    recovered = (torch.autograd.grad(branch_low * coefficients['low'], low)[0],
                 torch.autograd.grad(branch_high * coefficients['high'], high)[0])
    for expected, actual in zip(gradients, recovered):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_mask_excludes_distant_background_and_zero_energy_is_rejected():
    target = picture(8.)
    output = picture(4.)
    before = projection(output, target, target, weight())
    # 高通有8像素邻域；改动留在与衣内统计区域相隔至少16像素处。
    changed = output.clone()
    changed[..., :8, :] = 0
    changed[..., -8:, :] = 1
    after = projection(changed, target, target, weight())
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    constant = torch.full_like(target, .5)
    with pytest.raises(AssertionError, match='能量'):
        projection(output, constant, constant, weight())


def test_displacement_projection_respects_each_distinct_margin():
    raw = torch.tensor([1., -.1])
    gradients = torch.eye(2)
    margins = torch.tensor([.01, .18])
    delta, report = project_displacement(raw, gradients, margins)
    assert report['feasible']
    dots = (gradients @ delta) / raw.norm()
    assert bool((dots >= margins - 1e-6).all())
    assert abs(float(dots[1]) - .18) < 1e-6
    assert float(delta.norm()) > 0


def test_incompatible_positive_margins_reject_nonzero_update():
    raw = torch.tensor([1., .5])
    gradients = torch.tensor([[1., 0.], [-1., 0.]])
    delta, report = project_displacement(raw, gradients, torch.tensor([.01, .18]), max_sweeps=32)
    assert not report['feasible']
    assert torch.equal(delta, torch.zeros_like(raw))


def test_double_gram_prevents_false_rejection_from_nearly_opposing_constraints():
    """双精度dual不能修复已经在float32矩阵乘法中丢失的Gram精度。"""
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(1001, generator=generator)
    first = first / first.norm()
    orthogonal = torch.randn(1001, generator=generator)
    orthogonal = orthogonal - first * (orthogonal @ first)
    orthogonal = orthogonal / orthogonal.norm()
    gradients = torch.stack((first, -first + .1 * orthogonal))
    rows = gradients / gradients.norm(dim=1)[:, None]
    raw = -(rows[0] + rows[1])
    raw = raw / raw.norm()
    margins = torch.tensor([.18, .18])

    # 两项均活跃，可直接解旧float32 Gram的精确dual，排除迭代不足的解释。
    old_gram = (rows @ rows.T).double()
    old_offset = (rows @ (raw / raw.norm())).double() - margins.double()
    old_dual = torch.linalg.solve(old_gram, -old_offset)
    assert bool((old_dual > 0).all())
    assert float((old_offset + old_gram @ old_dual).abs().max()) < 1e-12
    old_delta64 = raw.double() + raw.norm().double() * (rows.double().T @ old_dual)
    old_actual_slack = rows.double() @ old_delta64 / raw.norm().double() - margins.double()
    assert float(old_actual_slack.min()) < -1e-6

    delta, report = project_displacement(raw, gradients, margins)
    assert report['feasible'] and delta.dtype == raw.dtype
    actual = rows.double() @ delta.double() / raw.norm().double() - margins.double()
    assert bool((actual >= -1e-6).all())
    assert report['actual_slack_min'] == float(actual.min())
    assert report['projected_norm'] == float(delta.norm())
    assert report['actual_violations'] == report['gram_violations'] == 0
    assert report['iterations'] <= 4096 and report['internal_break']
    assert report['maximum_reconstruction_slack_difference'] < 1e-6
