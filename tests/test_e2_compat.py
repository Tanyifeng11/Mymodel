from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.e2_compat import (compare_snapshots, compute_losses, cpu_snapshot,
                             e2_objective, load_trainable_strict, observe_prediction)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.5))
        self.adapter = SimpleNamespace(features=[])

    def forward(self, noisy, timestep):
        result = noisy * self.scale
        self.adapter.features = [result]
        return result

    def load_trainable(self, state):
        self.load_state_dict(state)


class VAE(nn.Module):
    config = SimpleNamespace(scaling_factor=.18215)

    def decode(self, value):
        return SimpleNamespace(sample=value * 2)


def fixture():
    net = Network()
    pipe = SimpleNamespace(vae=VAE())

    def losses(pipe, network, batch, generator, style, audit_mode=False):
        noise = torch.randn((1, 1, 4, 4), generator=generator)
        t = torch.randint(0, 1000, (1,), generator=generator)
        epsilon = network(noise, t)
        decoded = pipe.vae.decode(epsilon / pipe.vae.config.scaling_factor).sample
        return dict(base=decoded.square().mean(), body=decoded.abs().mean(), rest=decoded.mean()), dict(t=int(t[0]))

    return pipe, net, SimpleNamespace(losses=losses)


def test_missing_extra_and_wrong_shape_old_keys_are_rejected():
    network = Network()
    load_trainable_strict(network, {'scale': torch.tensor(.2)})
    for state in [{}, {'scale': torch.tensor(.2), 'extra': torch.tensor(0.)}, {'scale': torch.zeros(2)}]:
        with pytest.raises(ValueError, match='checkpoint'):
            load_trainable_strict(network, state)


def test_all_off_preserves_loss_forward_rng_and_does_not_call_fft():
    pipe, network, core = fixture()
    rng = torch.Generator().manual_seed(42)
    initial = rng.get_state().clone()
    with observe_prediction(pipe, network) as captured:
        raw, token = core.losses(pipe, network, {}, rng, None)
    before = cpu_snapshot(raw, token, captured, e2_objective(raw), rng)

    def forbidden(*args, **kwargs):
        raise AssertionError('关闭FFT仍消耗采样或调用新分支')

    rng.set_state(initial)
    loss, raw, token, captured, stats = compute_losses(pipe, network, {}, rng, None, core,
                                                     fft=forbidden, fft_weight=0)
    after = cpu_snapshot(raw, token, captured, loss, rng)
    assert max(compare_snapshots(before, after).values()) == 0
    assert stats == {}
    gradient = torch.autograd.grad(loss, [captured['epsilon'], captured['vae_input'], network.scale])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradient)
    assert 'decode' not in pipe.vae.__dict__
    assert not network._forward_hooks


def test_observation_restores_methods_even_when_old_loss_raises():
    pipe, network, _ = fixture()
    with pytest.raises(RuntimeError):
        with observe_prediction(pipe, network):
            raise RuntimeError('旧前向异常')
    assert 'decode' not in pipe.vae.__dict__
    assert not network._forward_hooks
