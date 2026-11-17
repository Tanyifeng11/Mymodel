import torch
from torch import nn
from torch.nn import functional as F

from models.e2_state_texture_b import StateTextureB
from models.e2_reference_terminal_head import ReferenceTerminalHead, reference_statistics, LOG_RMS_MEAN, LOG_RMS_STD
from tools.e2_b_endpoint_losses import set_parameters
from tools.e2_b_material_losses import highpass
from tools.e2_terminal_head_preflight import fixed_perturbation
from tools.e2_reference_terminal_preflight import terminal_replay, scheduler_configuration


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(64, 4, 3, padding=1)
    def forward(self, zt, x0, reference, mask, structure, timestep, color=None):
        return self.output(F.silu(zt))*mask


def inputs():
    torch.manual_seed(93)
    feature = torch.randn(1, 64, 4, 4)
    y = torch.rand(1, 1, 32, 32)
    reference = torch.cat((F.pixel_unshuffle(y*2-1, 8), torch.randn(1, 128, 4, 4)), 1)
    mask = torch.tensor([0., .25, .5, 1.]).view(1, 1, 1, 4).expand(1, 1, 4, 4)
    return feature, dict(x0=None, reference=reference, mask=mask, structure=None, timestep=torch.tensor([1]))


def test_reference_only_first64_per_sample_and_fixed_calibration():
    _, values = inputs()
    reference = values['reference'].repeat(2, 1, 1, 1)
    reference[1, :64] *= .25
    statistics = reference_statistics(reference)
    y = (F.pixel_shuffle(reference[:, :64], 8)+1)/2
    rms = highpass(y).square().mean((1, 2, 3), keepdim=True).sqrt()
    expected = (((rms+1e-6).log()-LOG_RMS_MEAN)/LOG_RMS_STD).tanh()
    assert statistics['phi'].shape == (2, 1, 1, 1)
    assert torch.equal(statistics['rms'], rms) and torch.equal(statistics['phi'], expected)
    reference[:, 64:] += 1000
    assert torch.equal(reference_statistics(reference)['phi'], expected)
    assert statistics['phi'][0] != statistics['phi'][1] and not statistics['phi'].requires_grad


def test_only_4616_new_parameters_and_ordered_zero_buffers():
    head = ReferenceTerminalHead(StateTextureB(color_channels=2, reference_channels=192))
    params = head.increment_parameters()
    assert sum(p.numel() for p in params) == 4616
    assert [id(p) for p in params] == [id(head.delta_weight0), id(head.delta_bias0), id(head.delta_weight1), id(head.delta_bias1)]
    assert sum(p.numel() for p in head.base.parameters()) == 274372
    assert all(not p.requires_grad for p in head.base.parameters())
    assert all(p.requires_grad and not torch.count_nonzero(p) for p in params)
    assert float(head.log_rms_mean) == LOG_RMS_MEAN and float(head.log_rms_std) == LOG_RMS_STD


def test_terminal_gating_zero_parity_and_nonzero_fused_conv_single_mask():
    feature, values = inputs(); head = ReferenceTerminalHead(ToyBase())
    native = head.base(feature, **values)
    head.active_calls = 24
    assert torch.equal(head(feature, **values), native)
    set_parameters(head.increment_parameters(), fixed_perturbation(4616))
    head.reset_capture()
    assert torch.equal(head(feature, **values), native)
    head.active_calls = 24
    assert torch.equal(head(feature, **dict(values, timestep=torch.tensor([21]))), native)
    head.active_calls = 24; full = head(feature, **values); record = head.last_record
    phi = record['statistics']['phi'].reshape(())
    expected = F.conv2d(F.silu(feature), head.base.output.weight+(head.delta_weight0+phi*head.delta_weight1),
        head.base.output.bias+(head.delta_bias0+phi*head.delta_bias1), padding=1)*values['mask']
    assert torch.equal(full, expected) and not torch.equal(full, native)
    assert torch.equal(full, head.from_features(record['feature'], record['mask'], record['statistics']['phi']))
    assert not torch.equal(full, expected*values['mask'])
    parts = torch.autograd.grad(full.square().sum(), head.increment_parameters())
    assert torch.allclose(parts[2], parts[0]*phi, rtol=1e-6, atol=1e-6)
    assert torch.allclose(parts[3], parts[1]*phi, rtol=1e-6, atol=1e-6)
    assert all(p.grad is None and not p.requires_grad for p in head.base.parameters())


def test_cache_adapter_uses_bound_reference_raw_epsilon_and_original_scheduler():
    feature, values = inputs(); head = ReferenceTerminalHead(ToyBase())
    set_parameters(head.increment_parameters(), fixed_perturbation(4616))
    class Scheduler:
        timesteps = torch.tensor([1])
        alphas_cumprod = torch.tensor([.99915, .9983])
        def step(self, epsilon, t, zt, return_dict):
            return ((zt-epsilon*.01).to(zt.dtype),)
    scheduler = Scheduler(); zt = torch.randn(1, 4, 4, 4).half(); raw = torch.randn_like(zt)
    record = dict(epsilon_kind='raw_unmodified_CFG', index=49, t=1, zt=zt, raw_epsilon=raw,
        alpha=float(scheduler.alphas_cumprod[1]), feature=F.silu(feature), mask=values['mask'],
        target_record=(None, None, None, dict(reference=values['reference'])))
    actual, base, delta, corrected = terminal_replay(head, record, scheduler, 'cpu')
    phi = head.statistics(values['reference'])['phi']
    assert torch.equal(delta, head.from_features(record['feature'], record['mask'], phi))
    alpha = scheduler.alphas_cumprod[1]
    expected_epsilon = (raw.float()-(alpha/(1-alpha)).sqrt()*delta).half()
    assert torch.equal(corrected, expected_epsilon)
    assert torch.equal(actual, scheduler.step(expected_epsilon, 1, zt, False)[0])
    assert torch.equal(base, scheduler.step(raw, 1, zt, False)[0].float())


def test_scheduler_default_name_order_only_is_ignored_and_raw_difference_kept():
    actual = dict(clip_sample=False, steps_offset=1, _use_default_values=['beta_end', 'beta_start'])
    reference = dict(clip_sample=False, steps_offset=1, _use_default_values=['beta_start', 'beta_end'])
    result = scheduler_configuration(actual, reference)
    assert result['normalized_equal'] and set(result['raw_diff']) == {'_use_default_values'}
    assert result['actual'] == actual and result['reference'] == reference
    assert actual['_use_default_values'] == ['beta_end', 'beta_start']


def test_scheduler_math_or_key_changes_are_not_normalized_away():
    reference = dict(clip_sample=False, steps_offset=1, _use_default_values=['beta_end', 'beta_start'])
    assert not scheduler_configuration(dict(reference, clip_sample=True), reference)['normalized_equal']
    assert not scheduler_configuration(dict(reference, steps_offset=0), reference)['normalized_equal']
    assert not scheduler_configuration(dict(reference, extra=None), reference)['normalized_equal']
    assert not scheduler_configuration(dict(reference, _use_default_values=['beta_start']), reference)['normalized_equal']
