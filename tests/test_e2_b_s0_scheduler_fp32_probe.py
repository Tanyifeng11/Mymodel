from types import SimpleNamespace

import torch

from tools.e2_b_s0_scheduler_fp32_probe import PrecisionStep, cast_output


def test_only_active_scheduler_inputs_are_float32_and_outputs_return_half():
    calls = []
    def scheduler(epsilon, timestep, sample, return_dict):
        calls.append((epsilon.dtype, sample.dtype))
        return sample+epsilon/3, sample-epsilon/7
    wrapper = PrecisionStep(scheduler)
    z, eps = torch.tensor([1.], dtype=torch.float16), torch.tensor([.001], dtype=torch.float16)
    for index in range(50):
        result = wrapper(eps, 981-index*20, z, return_dict=False)
        if index < 25:
            expected = scheduler(eps, 981-index*20, z, return_dict=False)
            calls.pop()
            assert all(torch.equal(a, b) for a, b in zip(result, expected))
        assert all(value.dtype == torch.float16 for value in result)
    assert calls[:25] == [(torch.float16, torch.float16)]*25
    assert calls[25:] == [(torch.float32, torch.float32)]*25
    assert wrapper.count == 50 and len(wrapper.prefix) == len(wrapper.active) == 25


def test_both_ddim_outputs_preserve_return_structure_and_native_dtype():
    original = SimpleNamespace(prev_sample=torch.tensor([.123456]), pred_original_sample=torch.tensor([.777777]))
    casted = cast_output(original, torch.float16)
    assert type(casted) is SimpleNamespace
    assert torch.equal(casted.prev_sample, original.prev_sample.half())
    assert torch.equal(casted.pred_original_sample, original.pred_original_sample.half())
    assert original.prev_sample.dtype == original.pred_original_sample.dtype == torch.float32
    pair = cast_output((original.prev_sample, original.pred_original_sample), torch.float16)
    assert isinstance(pair, tuple) and len(pair) == 2 and all(v.dtype == torch.float16 for v in pair)
