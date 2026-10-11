"""核验旧E5未保存的BF辅助头是否影响生成；不改权重或生成结果。"""
import gc
import hashlib
import torch
from models.bf_texture_module import BFTextureConditioner
from tools.e43_protocol import E5, EXPECTED_E5, OUT, sha, write


def digest(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    assert sha(E5) == EXPECTED_E5
    torch.set_num_threads(4)
    checkpoint = torch.load(E5, map_location='cpu', weights_only=False)
    state = checkpoint['bf_texture_conditioner']
    del checkpoint
    gc.collect()
    channels = tuple(int(state['stage%d.0.weight' % i].shape[0]) for i in range(1, 5))
    clip_dim = int(state['token_source_proj.0.0.weight'].shape[0])
    token_n, dim = state['resampler_queries'].shape[1:]
    models = []
    missing_list = []
    for seed in [42, 123]:
        torch.manual_seed(seed)
        m = BFTextureConditioner(clip_embeddings_dim=clip_dim, cross_attention_dim=dim,
            num_tokens=token_n, stage_channels=channels).eval().requires_grad_(False)
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) == {
            'pattern_head.0.weight', 'pattern_head.0.bias',
            'pattern_head.2.weight', 'pattern_head.2.bias'}
        assert not m.pattern_loss_enabled
        models.append(m)
        missing_list.append(missing)
    states = [m.state_dict() for m in models]
    different = [k for k in states[0] if not torch.equal(states[0][k], states[1][k])]
    assert set(different) == set(missing_list[0])
    active = [{k: v for k, v in s.items() if not k.startswith('pattern_head.')} for s in states]
    assert digest(active[0]) == digest(active[1])
    # 辅助头若被调用，立即失败；分别核验正参考与零参考BF输出。
    def forbid(*args):
        raise AssertionError('生成调用了pattern_head')
    hooks = [m.pattern_head.register_forward_pre_hook(forbid) for m in models]
    torch.manual_seed(43042)
    clip = torch.randn(1, 5, clip_dim)
    texture = torch.randn(1, 3, 64, 64)
    checks = []
    with torch.inference_mode():
        for zero in [False, True]:
            kw = dict(clip_vision_tokens=torch.zeros_like(clip) if zero else clip,
                texture_images=torch.zeros_like(texture) if zero else texture)
            a, b = [m(**kw)[0] for m in models]
            assert torch.equal(a, b)
            checks.append(dict(zero_inputs=zero, tokens_identical=True))
    for h in hooks:
        h.remove()
    result = dict(passed=True,checkpoint_sha256=EXPECTED_E5,missing_keys=missing_list[0],
        cross_initialization_different_tensors=different,active_state_sha256=digest(active[0]),
        pattern_head_disabled=True,pattern_head_called=False,positive_and_zero_checks=checks,
        scope='CPU reconstruction from identical original checkpoint; old per-job whole-module hashes differ only in uncheckpointed disabled auxiliary head by loader construction',
        outputs_regenerated=0,generation_training_steps=0)
    write(OUT/'audit/bf_missing_head_audit.json', result)
    print('E43 BF AUDIT', result, flush=True)


if __name__ == '__main__':
    main()
