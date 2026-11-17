from types import SimpleNamespace

import torch

from tools.e2_reference_terminal_fp32_gradients import task_values
from tools.e2_state_b_endpoint import Objective


def test_explicit_rgb_preserves_original_seven_loss_values_and_gradients():
    # 直接对比原Objective真实损失实现，避免新RGB支路悄悄改写任务语义。
    torch.manual_seed(17)
    theta = torch.tensor(.01, requires_grad=True)
    positive = .2+.6*torch.rand(1, 3, 64, 64)
    rgb = (positive+theta).clamp(0, 1)
    z = torch.zeros(1, 4, 8, 8)+theta
    base_z = torch.ones_like(z)
    base_rgb = positive*.98
    pair = dict(case_id='test', patch_coordinates=[(0, 0)])
    entry = dict(pair=pair, condition='one')
    objective = SimpleNamespace(records={('test', 'one'): (
        torch.ones_like(z)*.1, positive, positive*.95,
        dict(mask=torch.ones(1, 1, 8, 8)), torch.ones(1, 1, 64, 64), None)})
    surrogate = lambda value: (value-positive).square().mean()
    objective.metric = lambda unused: SimpleNamespace(surrogate=surrogate)
    objective.rgb = lambda value: rgb if value is z else base_rgb
    old, _ = Objective.score_latents(objective, entry, z, base_z, residue=surrogate)
    explicit = task_values(objective, entry, z, rgb, base_rgb)
    assert tuple(old) == tuple(explicit)
    for name in old:
        assert torch.equal(old[name], explicit[name]), name
        reference = torch.autograd.grad(old[name], theta, retain_graph=True)[0]
        actual = torch.autograd.grad(explicit[name], theta, retain_graph=True)[0]
        assert torch.equal(reference, actual), name
