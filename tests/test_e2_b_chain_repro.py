"""固定q的重复VJP、缩放及真实FP32→FP16反向cast。"""
import torch

from tools.e2_b_chain_repro import GROUPS, repeated_vjp


def test_repeated_and_rebuilt_vjp_keep_q_fixed_and_parameters_untouched():
    parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
    original = parameter.detach().clone()
    q = torch.tensor([.5, 2.])
    results, _ = repeated_vjp(parameter.square(), [parameter], q)
    rebuilt, _ = repeated_vjp(parameter.square(), [parameter], q, repeats=1, scale=1024.)
    expected = 2 * original * q
    assert all(torch.equal(row, expected) for row in results + rebuilt)
    assert torch.equal(parameter, original) and parameter.grad is None
    assert torch.equal(q, torch.tensor([.5, 2.]))


def test_terminal_cast_records_underflow_and_scale_recovery():
    parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
    q = torch.tensor([1e-9, 1e-4])
    small, native_small = repeated_vjp(parameter.half().float(), [parameter], q, terminal_hook=True)
    large, native_large = repeated_vjp(parameter.half().float(), [parameter], q,
        repeats=1, scale=1024., terminal_hook=True)
    assert small[0][0] == 0 and large[0][0] != 0
    assert native_small[0]['zero_from_nonzero_seed'] == 1
    assert native_large[0]['zero_from_nonzero_seed'] == 0
    assert all(row['matches_expected_cast'] for row in native_small + native_large)


def test_scaled_nonfinite_gradient_remains_explicit():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    rows, terminal = repeated_vjp(parameter.half().float(), [parameter], torch.tensor([100.]),
        repeats=1, scale=1024., terminal_hook=True)
    assert not torch.isfinite(rows[0]).all()
    assert not terminal[0]['native_summary']['finite']


def test_protocol_is_twelve_vjps_and_math_runs_last():
    assert len(GROUPS) * 3 == 12
    assert [g['scale'] for g in GROUPS] == [1., 1., 1024., 1.]
    assert [g['math_only'] for g in GROUPS] == [False, False, False, True]
    assert GROUPS[1]['start'] == 24 and GROUPS[1]['hops'] == 1
