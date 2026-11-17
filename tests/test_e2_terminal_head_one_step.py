import torch

from models.e2_state_texture_b import StateTextureB
from models.e2_terminal_output_head import TerminalOutputHead
from tools.e2_b_endpoint_losses import flatten, set_parameters
from tools.e2_terminal_head_one_step import combine_gradients, one_update, effective_difference, device_tree


def config():
    return dict(lr=3e-5, weight_decay=.01, gradient_clip=1., scale=.5,
        displacement_cap=.007854645, projection_sweeps=4096, default_margin=.01,
        protected_amplitude_margin=.18, protected_condition=['validation_1641', 'r2692_native'],
        pair_gradient_norm_fraction=.25, extra_absolute_gradient_norm_fraction=.25)


def test_combination_keeps_original_means_norm_matching_and_margin_rows():
    shards = []
    for index, case in enumerate(('validation_621', 'validation_1641', 'validation_870', 'validation_1526')):
        shards.append(dict(case=case, gradients=torch.full((1, 7, 3), float(index+1)),
            absolute_gradient=torch.tensor([1., index+1., 3.]), pair_gradient=torch.tensor([2., 3., index+1.]),
            items=[dict(case=case, condition='r2692_native')]))
    spec = dict(training=config(), replay_weights=[1., 2., 3., 4., 5., 6., 7.])
    directions, constraints, margins, _ = combine_gradients(shards, spec)
    merged = torch.cat([s['gradients'] for s in shards]).mean(0).mul(torch.arange(1., 8.)[:, None]).sum(0)
    old_abs = torch.stack([shards[i]['absolute_gradient'] for i in (0, 3)]).mean(0)
    new_abs = torch.stack([shards[i]['absolute_gradient'] for i in (1, 2)]).mean(0)
    pair = torch.stack([s['pair_gradient'] for s in shards]).mean(0)
    unit = lambda x: x*merged.norm()/x.norm()
    assert torch.allclose(directions['C0'], merged+unit(old_abs))
    assert torch.allclose(directions['C1'], merged+unit(old_abs)+.25*unit(pair)+.25*unit(new_abs))
    assert constraints.shape == (28, 3)
    assert torch.isclose(margins[8], torch.tensor(.18)) and torch.count_nonzero(margins != .01) == 1


def test_independent_adamw_single_step_is_shrunk_and_not_norm_inflated():
    directions = torch.tensor([1., 2.])
    constraints = torch.eye(2)
    parameters = [torch.nn.Parameter(torch.zeros(2))]
    report = one_update(parameters, directions, constraints, torch.full((2,), .01), config())
    assert report['optimizer_proposals'] == report['parameter_updates'] == 1
    assert report['projection']['feasible'] and report['effective_projection_sweeps'] == 8192
    assert report['configured_projection_sweeps'] == 4096 and report['minimum_cosine'] > 0
    assert report['actual_norm'] <= report['raw_norm']*.500001
    assert torch.all(flatten(parameters) < 0)
    independent = [torch.nn.Parameter(torch.zeros(2))]
    second = one_update(independent, directions, constraints, torch.full((2,), .01), config())
    assert torch.equal(flatten(parameters), flatten(independent)) and second['actual_descent_sha256'] == report['actual_descent_sha256']


def test_effective_merged_weight_change_keeps_fp32_addition_rounding_and_base_frozen():
    head = TerminalOutputHead(StateTextureB(color_channels=2, reference_channels=192))
    with torch.no_grad():
        head.base.output.weight.fill_(.125)
        head.base.output.bias.fill_(.125)
    original = {key: tensor.clone() for key, tensor in head.base.state_dict().items()}
    increment = torch.linspace(-1e-7, 1e-7, 2308)
    set_parameters(head.increment_parameters(), increment)
    expected = torch.cat([(head.base.output.weight+head.delta_weight).detach().flatten(),
        (head.base.output.bias+head.delta_bias).detach().flatten()]).double()
    expected -= torch.cat([head.base.output.weight.detach().flatten(), head.base.output.bias.detach().flatten()]).double()
    assert torch.equal(effective_difference(head), expected)
    assert not torch.equal(expected, increment.double())
    assert all(torch.equal(value, head.base.state_dict()[key]) for key, value in original.items())
    assert all(not p.requires_grad for p in head.base.parameters())


def test_cached_target_transfer_preserves_dtype_values_and_stops_gradient():
    tensor = torch.tensor([1.234567], dtype=torch.float32, requires_grad=True)
    saved = (tensor, {'weight': torch.tensor([.25], dtype=torch.float32)}, None)
    restored = device_tree(saved, 'cpu')
    assert torch.equal(restored[0], tensor) and restored[0].dtype == tensor.dtype
    assert not restored[0].requires_grad and torch.equal(restored[1]['weight'], saved[1]['weight'])
