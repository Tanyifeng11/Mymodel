"""小张量核验G先于独立q，缩放还原及非有限诊断不会伪造通过。"""
from types import SimpleNamespace

import torch

from tools.e2_b_endpoint_losses import TASKS
from tools.e2_state_b_backward_repro import compare_tensor, gradients_with_head, vector_summary, VARIANTS


class ToyObjective:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
        self.events = []

    def rollout(self, entry):
        self.events.append('rollout')
        return self.parameter * 2, torch.zeros(2)

    def metric(self, entry):
        return SimpleNamespace(surrogate=None)

    def score_latents(self, entry, predicted, base, residue=None):
        self.events.append('head_leaf' if predicted.is_leaf else 'head_generator')
        return {task: (i + 1) * predicted.square().sum() for i, task in enumerate(TASKS)}, {}


def test_g_and_q_have_correct_scale_and_independent_head_graph():
    obj = ToyObjective()
    hooks = []
    obj.parameter.register_hook(lambda value: hooks.append('parameter_backward'))
    result = gradients_with_head(obj, {}, [obj.parameter], before_backward=lambda _: obj.events.append('pixels'))
    assert obj.events == ['rollout', 'head_generator', 'pixels', 'head_leaf']
    assert hooks == ['parameter_backward'] * 7
    for i in range(7):
        torch.testing.assert_close(result['gradient'][i], (i + 1) * 8 * obj.parameter.detach())
        torch.testing.assert_close(result['q'][i], (i + 1) * 4 * obj.parameter.detach())
    assert result['values'] == result['head_values']
    assert obj.parameter.grad is None
    scaled = gradients_with_head(obj, {}, [obj.parameter], scale=1024.)
    assert torch.equal(scaled['gradient'], result['gradient'])
    assert torch.equal(scaled['q'], result['q'])


def test_comparisons_keep_zero_and_nonfinite_explicit():
    x = torch.tensor([1., -2.])
    assert compare_tensor(x, x)['equal']
    assert abs(compare_tensor(-x, x)['cosine'] + 1) < 1e-12
    assert compare_tensor(x, torch.zeros_like(x))['finite']
    bad = torch.tensor([float('inf'), 0.])
    assert compare_tensor(bad, x)['max_abs'] is None
    report = vector_summary(bad, {'C0': x.double()})
    assert not report['finite'] and report['nonfinite'] == 1 and report['directional']['C0'] is None


def test_diagnostic_variants_change_one_factor_against_clean():
    assert [v['name'] for v in VARIANTS] == ['clean0', 'clean1', 'pixels_before', 'threads2', 'loss_scale1024']
    assert {k: v for k, v in VARIANTS[0].items() if k != 'name'} == {k: v for k, v in VARIANTS[1].items() if k != 'name'}
    for config in VARIANTS[2:]:
        assert sum(config[key] != VARIANTS[0][key] for key in ('threads', 'pixels_before', 'loss_scale')) == 1
