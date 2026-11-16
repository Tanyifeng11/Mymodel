import torch
from torch import nn
from torch.nn import functional as F

from models.e2_state_texture_b import StateTextureB
from models.e2_terminal_output_head import TerminalOutputHead
from tools.e2_terminal_head_preflight import fixed_perturbation, terminal_replay
from tools.e2_b_endpoint_losses import set_parameters


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(64, 4, 3, padding=1)
    def forward(self, zt, x0, reference, mask, structure, timestep, color=None):
        return self.output(F.silu(zt))*mask


def inputs():
    torch.manual_seed(91)
    feature = torch.randn(1, 64, 4, 4)
    mask = torch.tensor([0., .25, .5, 1.]).view(1, 1, 1, 4).expand(1, 1, 4, 4)
    return feature, dict(x0=None, reference=None, mask=mask, structure=None, timestep=torch.tensor([1]))


def test_only_2308_new_parameters_trainable_and_real_base_frozen():
    head = TerminalOutputHead(StateTextureB(color_channels=2, reference_channels=192))
    assert sum(p.numel() for p in head.increment_parameters()) == 2308
    assert sum(p.numel() for p in head.base.parameters()) == 274372
    assert all(not p.requires_grad for p in head.base.parameters())
    assert all(p.requires_grad for p in head.increment_parameters())
    assert all(torch.count_nonzero(p) == 0 for p in head.increment_parameters())


def test_zero_increment_matches_base_and_only_call24_t1_is_gated():
    feature, values = inputs()
    head = TerminalOutputHead(ToyBase())
    native = head.base(feature, **values)
    head.active_calls = 24
    assert torch.equal(head(feature, **values), native)
    assert torch.equal(head.last_record['feature'], F.silu(feature))
    set_parameters(head.increment_parameters(), fixed_perturbation())
    head.reset_capture()
    assert torch.equal(head(feature, **values), native)  # 早到t1不能启用。
    head.active_calls = 24
    assert torch.equal(head(feature, **dict(values, timestep=torch.tensor([21]))), native)
    head.active_calls = 24
    changed = head(feature, **values)
    assert not torch.equal(changed, native)
    assert torch.equal(changed, head.from_features(head.last_record['feature'], values['mask']))


def test_nonzero_full_and_cache_share_exact_conv_and_single_mask():
    feature, values = inputs()
    head = TerminalOutputHead(ToyBase())
    set_parameters(head.increment_parameters(), fixed_perturbation())
    head.active_calls = 24
    full = head(feature, **values)
    record = head.last_record
    cached = head.from_features(record['feature'], record['mask'])
    expected = F.conv2d(F.silu(feature), head.base.output.weight+head.delta_weight,
        head.base.output.bias+head.delta_bias, padding=1)*values['mask']
    assert torch.equal(full, cached) and torch.equal(cached, expected)
    assert not torch.equal(cached, expected*values['mask'])
    gradients = torch.autograd.grad(cached.square().sum(), head.increment_parameters())
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradients)
    assert all(p.grad is None and not p.requires_grad for p in head.base.parameters())


def test_replay_uses_raw_epsilon_and_real_scheduler_once_for_corrected_state():
    feature, values = inputs()
    head = TerminalOutputHead(ToyBase())
    set_parameters(head.increment_parameters(), fixed_perturbation())
    class Scheduler:
        timesteps = torch.tensor([1])
        alphas_cumprod = torch.tensor([.99915, .9983])
        def step(self, epsilon, t, zt, return_dict):
            return ((zt-epsilon*.01).to(zt.dtype),)
    scheduler = Scheduler()
    zt = torch.randn(1, 4, 4, 4).half()
    raw = torch.randn_like(zt)
    record = dict(epsilon_kind='raw_unmodified_CFG', index=49, t=1, zt=zt, raw_epsilon=raw,
        alpha=float(scheduler.alphas_cumprod[1]), feature=F.silu(feature), mask=values['mask'])
    actual, base, delta, corrected = terminal_replay(head, record, scheduler, 'cpu')
    alpha = scheduler.alphas_cumprod[1]
    expected_epsilon = (raw.float()-(alpha/(1-alpha)).sqrt()*delta).half()
    assert torch.equal(corrected, expected_epsilon)
    assert torch.equal(actual, scheduler.step(expected_epsilon, 1, zt, False)[0])
    assert torch.equal(base, scheduler.step(raw, 1, zt, False)[0].float())
