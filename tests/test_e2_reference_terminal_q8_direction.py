import numpy as np
import torch

from tools.e2_reference_terminal_q8_direction import q8_ste, select_controls, central


def test_q8_forward_is_png_rounding_and_backward_is_explicit_proxy():
    value = torch.linspace(0, 1, 4097).half().float().requires_grad_()
    output = q8_ste(value)
    png = np.rint(value.detach().numpy()*255).astype(np.uint8)
    expected = torch.from_numpy(png).float()/255
    assert torch.equal(output.detach(), expected)
    derivative = torch.autograd.grad(output.sum(), value)[0]
    real_round_derivative = torch.autograd.grad(((value*255).round()/255).sum(), value)[0]
    assert torch.equal(derivative, torch.ones_like(value))
    assert torch.count_nonzero(real_round_derivative) == 0


def test_central_comparison_uses_ray_factor_not_parameter_norm():
    values = {h: 3+2*h+7*h*h for h in (-1., -.5, 0., .5, 1.)}
    assert central(values, .5) == central(values, 1.) == 2


def test_selection_is_seven_seen_failures_plus_four_pastel_alpha1():
    def row(index, fail, pastel):
        return dict(case=str(index), sample_condition='r', group='pastel' if pastel else 'neutral',
            arm='a100', f68_stations=dict(official_PNG=dict(old_projection=.1)),
            G1_observation=dict(stations=dict(official_PNG=dict(old_projection=.101 if fail else .099))))
    rows = [row(i, True, False) for i in range(7)]+[row(i, False, True) for i in range(7, 11)]
    rows.append(dict(row(11, False, True), arm='a125'))
    chosen = select_controls(rows)
    assert [v['case'] for v in chosen] == [str(i) for i in range(11)]
    assert sum(len(v['reasons']) for v in chosen) == 11
