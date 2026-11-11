"""验证时间窗边界、完整batch分母及与旧FFT共有patch流。"""
import torch

from models.e2_local_fft import LocalFFTLoss


def test_window_reduces_over_all_slots_and_keeps_legacy_patch_stream():
    rng = torch.Generator().manual_seed(173)
    predicted = torch.rand((6, 3, 96, 96), generator=rng).requires_grad_()
    reference = torch.rand(predicted.shape, generator=rng)
    mask = torch.zeros((6, 1, 96, 96))
    mask[:, :, 8:-8, 8:-8] = 1
    times = torch.tensor([99, 100, 249, 250, 100, 225])
    keep = torch.tensor([True, True, True, True, False, True])
    reference[5].zero_()
    ids = torch.arange(6)
    original = LocalFFTLoss()
    candidate = LocalFFTLoss(t_min=100)
    a, old_stats = original(predicted, reference, mask, times, ids, keep)
    b, new_stats = candidate(predicted, reference, mask, times, ids, keep)
    assert (old_stats['active_n'], old_stats['valid_n']) == (4, 3)
    assert (new_stats['active_n'], new_stats['valid_n']) == (3, 2)
    assert new_stats['samples'][0]['reason'] == 'timestep'
    for index in [1, 2, 5]:
        assert new_stats['samples'][index] == old_stats['samples'][index]
    assert torch.equal(original.sampling_state()['rng'], candidate.sampling_state()['rng'])
    old_gradient, = torch.autograd.grad(a, predicted, retain_graph=True)
    new_gradient, = torch.autograd.grad(b, predicted)
    assert torch.equal(old_gradient[1:3], new_gradient[1:3])
    assert old_gradient[0].norm() > 0
    assert torch.count_nonzero(new_gradient[[0, 3, 4, 5]]) == 0
