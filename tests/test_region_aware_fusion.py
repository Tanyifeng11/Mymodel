from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.bf_original_fusion import bFMFusion
from models.region_aware_fusion import RegionFusionAdapter, RegionFusionNetwork


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


class SkipBlock(nn.Module):
    def forward(self, hidden, res_hidden_states_tuple):
        return sum(value.mean((1, 2, 3)) for value in res_hidden_states_tuple)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList(SkipBlock() for _ in range(4))

    def forward(self, sample, timestep, encoder_hidden_states=None, cross_attention_kwargs=None):
        outputs = []
        for block, channels, side, count in zip(self.up_blocks, (16, 16, 8, 4), (2, 4, 8, 16), (3, 2, 2, 3)):
            skips = tuple(sample.new_zeros(len(sample), channels, side, side) for _ in range(count))
            outputs.append(block(sample, res_hidden_states_tuple=skips))
        return SimpleNamespace(sample=torch.stack(outputs).sum(0))


def make_network(variant='region'):
    return RegionFusionNetwork(TinyUNet(), RegionFusionAdapter(
        variant=variant, ngf=2, bottleneck_depth=3, channels=(4, 8, 16, 16)))


def inputs(batch=2):
    return (torch.randn(batch, 3, 128, 128), torch.randn(batch, 3, 128, 128),
            torch.ones(batch, 1, 128, 128))


def test_original_bf_baseline_is_exact():
    adapter = make_network('bf').adapter
    original = bFMFusion(1, 3, ngf=2, bottleneck_depth=3, num_resblocks=0)
    original.load_state_dict(adapter.fusion.state_dict())
    sketch, reference, _ = inputs()
    gray = (sketch * sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    for layer in adapter.projections:
        nn.init.normal_(layer.weight, std=.1)
    adapter.condition(sketch, reference)
    expected = [p(f) for p, f in zip(adapter.projections, original(gray, reference))]
    assert all(torch.equal(a, b) for a, b in zip(adapter.features, expected))


def test_region_and_reference_dropout_preserve_sketch_path():
    fusion = make_network().adapter.fusion
    sketch, reference, mask = inputs()
    sources = fusion.encode(sketch[:, :1], reference)
    for region, keep in [(torch.zeros_like(mask), torch.ones(2)), (mask, torch.zeros(2))]:
        output = fusion.fuse(sources, fusion.regions(region), torch.tensor([10, 900]), keep)
        assert all(torch.equal(value, source[0]) for value, source in zip(output, sources))
    keep = torch.tensor([0., 1.])
    output = fusion.fuse(sources, fusion.regions(mask), 100, keep)
    assert all(torch.equal(value[0], source[0][0]) for value, source in zip(output, sources))
    assert any(not torch.equal(value[1], source[0][1]) for value, source in zip(output, sources))


def test_reference_cannot_change_features_outside_region():
    fusion = make_network().adapter.fusion
    sketch, reference, mask = inputs()
    mask.zero_()
    mask[..., 32:96, 32:96] = 1
    region = fusion.regions(mask)
    a = fusion.fuse(fusion.encode(sketch[:, :1], reference), region, 100, torch.ones(2))
    b = fusion.fuse(fusion.encode(sketch[:, :1], reference.flip(-1)), region, 100, torch.ones(2))
    changed_inside = False
    for first, second in zip(a, b):
        outside = F.interpolate(region, first.shape[-2:], mode='area').eq(0).expand_as(first)
        assert torch.equal(first[outside], second[outside])
        changed_inside |= not torch.equal(first, second)
    assert changed_inside


@pytest.mark.parametrize('variant', ['bf', 'concat', 'region'])
def test_zero_projection_and_gradient_chain(variant):
    network = make_network(variant)
    sketch, reference, mask = inputs()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01)
    for step in range(3):
        optimizer.zero_grad()
        output = network(torch.zeros(2, 1), torch.tensor([100, 800]), None,
                         sketch, reference, {}, mask)
        if step == 0:
            assert torch.count_nonzero(output) == 0
        assert network.adapter.injection_counts == [3, 2, 2, 3]
        (output - 1).square().mean().backward()
        assert network.adapter.projections[0].weight.grad.abs().sum() > 0
        if step == 2:
            fusion = network.adapter.fusion
            parameters = [fusion.downconv1[0].weight, fusion.G_downconv1[0].weight]
            if variant == 'region':
                parameters += [fusion.bottleneck_alpha_2[-1].weight,
                    fusion.G_bottleneck_alpha_2[-1].weight,
                    fusion.region_gates[0].weight, fusion.time_gates[0][-1].weight]
            for parameter in parameters:
                assert torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
        optimizer.step()
        network.adapter.clear()


def test_live_timesteps_use_cached_encoding_and_skip_unconditional(monkeypatch):
    network = make_network()
    adapter = network.adapter
    sketch, reference, mask = inputs()
    for projection in adapter.projections:
        nn.init.normal_(projection.weight, std=.1)
    for gate in adapter.fusion.time_gates:
        nn.init.zeros_(gate[0].weight)
        nn.init.zeros_(gate[0].bias)
        gate[0].weight.data[0, 0] = 1
        gate[-1].weight.data[:, 0] = 2
    calls = []
    original = adapter.fusion.encode
    def encode(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(adapter.fusion, 'encode', encode)
    adapter.condition(sketch, reference, mask)
    source = torch.zeros(2, 1)
    low = network.unet(source, timestep=0, cross_attention_kwargs={'sa_hidden_states': {}}).sample
    high = network.unet(source, torch.tensor([999, 600]), cross_attention_kwargs={'sa_hidden_states': {}}).sample
    assert not torch.equal(low, high)
    assert len(calls) == 1
    assert torch.count_nonzero(network.unet(source, 100).sample) == 0
    adapter.enabled = False
    assert torch.count_nonzero(network.unet(source, 100, cross_attention_kwargs={'sa_hidden_states': {}}).sample) == 0
    adapter.unbind()
    assert not adapter.handles and adapter.context is None
    assert torch.count_nonzero(network.unet(source, 100, cross_attention_kwargs={'sa_hidden_states': {}}).sample) == 0


def test_checkpoint_roundtrip_and_architecture_validation(tmp_path):
    network = make_network()
    for layer in network.adapter.projections:
        nn.init.normal_(layer.weight, std=.1)
    sketch, reference, mask = inputs()
    expected = network(torch.zeros(2, 1), 100, None, sketch, reference, {}, mask)
    path = tmp_path / 'fusion.pt'
    torch.save(network.checkpoint(), path)
    state = torch.load(path, weights_only=False)
    restored = RegionFusionNetwork(TinyUNet(), RegionFusionAdapter(**state['fusion_config']))
    restored.load_checkpoint(state)
    actual = restored(torch.zeros(2, 1), 100, None, sketch, reference, {}, mask)
    assert torch.equal(expected, actual)
    with pytest.raises(ValueError, match='配置'):
        make_network('bf').load_checkpoint(state)
    state['network'].pop(next(iter(state['network'])))
    with pytest.raises(ValueError, match='不完整'):
        restored.load_checkpoint(state)


def test_bfloat16_training_has_finite_gradients():
    network = make_network()
    for layer in network.adapter.projections:
        nn.init.normal_(layer.weight, std=.1)
    sketch, reference, mask = inputs()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = network(torch.zeros(2, 1), torch.tensor([100, 900]), None,
                         sketch, reference, {}, mask)
        loss = (output.float() - 1).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert network.adapter.fusion.time_gates[0][-1].weight.grad.abs().sum() > 0


def test_half_precision_cached_inference_is_finite():
    network = make_network().half().eval()
    sketch, reference, mask = inputs()
    with torch.inference_mode():
        network.adapter.condition(sketch.half(), reference.half(), mask)
        output = network.unet(torch.zeros(2, 1).half(), 981,
                              cross_attention_kwargs={'sa_hidden_states': {}}).sample
    assert torch.isfinite(output).all()


def test_shared_sketch_mask_disables_blank_and_keeps_closed_outline():
    from tools.train_region_fusion import sketch_mask
    blank = torch.ones(3, 128, 128)
    assert torch.count_nonzero(sketch_mask(blank)) == 0
    outlined = blank.clone()
    outlined[:, 24:104, 24:27] = -1
    outlined[:, 24:104, 101:104] = -1
    outlined[:, 24:27, 24:104] = -1
    outlined[:, 101:104, 24:104] = -1
    mask = sketch_mask(outlined)
    assert mask[0, 64, 64] == 1 and mask[0, 0, 0] == 0


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5, 6])
def test_training_objective_matches_historical_full_t3(monkeypatch, seed):
    import sys
    from tools import bf_t3_frozen_core as core
    from tools.train_region_fusion import compute_losses
    from garment_mask_utils import build_region_masks
    def distance(a, b, mask=None):
        return ((a - b).abs() * (mask if mask is not None else 1)).mean()
    functions = SimpleNamespace(masked_edge_l1=distance,
        texture_color_stat_loss=lambda a, b, mask: distance(a, b, mask),
        build_region_masks=build_region_masks, masked_l1_loss=distance)
    monkeypatch.setitem(sys.modules, 'train_GAM_texture_joint', functions)
    monkeypatch.setattr(core, 'sketch_states', lambda *args: {})
    class VAE:
        config = SimpleNamespace(scaling_factor=.18)
        def encode(self, value):
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=value, std=torch.ones_like(value) * .1))
        def decode(self, value):
            return SimpleNamespace(sample=value)
    pipe = SimpleNamespace(vae=VAE(), reference_unet=SimpleNamespace(dtype=torch.float32),
        text_encoder=lambda ids: (ids.float()[..., None],),
        tokenizer=lambda *args, **kwargs: SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long)))
    def network(noisy, timestep, text, sketch, reference, sa, **kwargs):
        return noisy * .1 + reference * .01
    batch = dict(vae_cloth=torch.rand(1, 3, 16, 16), vae_sketch=torch.rand(1, 3, 16, 16),
        texture_image=torch.rand(1, 3, 16, 16), garment_mask=torch.ones(1, 1, 16, 16),
        fusion_mask=torch.ones(1, 1, 16, 16), input_ids=torch.ones(1, 77, dtype=torch.long),
        dataset_index=torch.tensor([7]))
    before, old_token = core.losses(pipe, network, batch, torch.Generator().manual_seed(seed), distance)
    after, new_token = compute_losses(pipe, network, batch, torch.Generator().manual_seed(seed), distance, core)
    assert torch.equal(before['base'] + .1 * before['body'] + .03816244801608925 * before['rest'], after['total'])
    assert old_token['noise_sha256'] == new_token['noise_sha256']
    assert old_token['t'] == new_token['timestep'] and old_token['drop'] == new_token['drop']
