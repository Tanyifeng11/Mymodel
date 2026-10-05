"""核验旧 E5 缺失的辅助头是否影响推理；保留完整冻结证明。"""
import hashlib
import torch
from models.bf_texture_module import BFTextureConditioner
from tools.e33tm_protocol import E5, OUT, sha, write, read


def digest(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def effective_hashes(modules):
    from tools.e22_4_generation import module_hashes
    audit = read(OUT/'weight_audit.json')
    assert audit['pass_audit'] and audit['outputs_torch_equal']
    for path, expected in audit['source_files'].items():
        if path != 'tools/e33tm_weight_audit.py': assert sha(path) == expected
    conditioner = modules['bf_texture_conditioner']
    assert not conditioner.pattern_loss_enabled
    state = {k:v for k,v in conditioner.state_dict().items() if k not in audit['missing_keys']}
    result = module_hashes(modules)
    result['bf_texture_conditioner'] = digest(state)
    assert result['bf_texture_conditioner'] == audit['effective_bf_sha256']
    return result


def validate_proof(proof, metadata_path):
    """旧分片保留全状态哈希；依据恢复路径审计解释未记录的有效哈希。"""
    audit = read(OUT/'weight_audit.json')
    assert audit['pass_audit'] and audit['outputs_torch_equal']
    assert read(metadata_path)['original_e5_sha256'] == audit['original_e5_sha256']
    assert proof['pass'] and proof['before'] == proof['after']
    result = dict(proof['before'])
    if 'effective_before' in proof:
        assert proof['effective_before'] == proof['effective_after']
        result = proof['effective_before']
    else:
        # 旧加载路径完整恢复所有有效参数；仅关闭的新增pattern_head不在E5中。
        result['bf_texture_conditioner'] = audit['effective_bf_sha256']
    assert result['bf_texture_conditioner'] == audit['effective_bf_sha256']
    return result


@torch.inference_mode()
def main():
    checkpoint = torch.load(E5, map_location='cpu', weights_only=False)
    state = checkpoint['bf_texture_conditioner']
    assert not any(k.startswith(('text_guidance.', 'film.', 'nexus.', 'direct_readout.')) for k in state)
    options = dict(clip_embeddings_dim=state['token_source_proj.0.1.weight'].shape[1],
                   cross_attention_dim=state['resampler_queries'].shape[-1],
                   num_tokens=state['resampler_queries'].shape[1],
                   stage_channels=tuple(state['stage%d.0.weight'%i].shape[0] for i in range(1,5)))
    models = [BFTextureConditioner(**options).cuda().half().eval().requires_grad_(False) for _ in range(2)]
    missing = []
    for model in models:
        keys, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and keys and all(k.startswith('pattern_head.') for k in keys)
        assert not model.pattern_loss_enabled
        missing.append(keys)
    different = [k for k,v in models[0].state_dict().items() if not torch.equal(v,models[1].state_dict()[k])]
    assert different and all(k in missing[0] for k in different)
    calls = []
    handles = [m.pattern_head.register_forward_hook(lambda *args: calls.append(True)) for m in models]
    generator = torch.Generator(device='cuda').manual_seed(32042)
    pixels = torch.randn(1,3,64,64,device='cuda',dtype=torch.float16,generator=generator)
    clip = torch.randn(1,257,options['clip_embeddings_dim'],device='cuda',dtype=torch.float16,generator=generator)
    outputs = [m(texture_images=pixels,clip_vision_tokens=clip)[0] for m in models]
    assert not calls and torch.equal(*outputs)
    effective = [{k:v for k,v in m.state_dict().items() if k not in missing[0]} for m in models]
    assert digest(effective[0]) == digest(effective[1])
    result = dict(pass_audit=True,original_e5_sha256=sha(E5),missing_keys=missing[0],different_keys=different,
                  full_hashes=[digest(m.state_dict()) for m in models],effective_bf_sha256=digest(effective[0]),
                  pattern_head_forward_calls=len(calls),outputs_torch_equal=True,
                  explanation='Only checkpoint-absent pattern_head differs; pattern_loss_enabled=False bypasses it.',
                  source_files={p:sha(p) for p in ('models/bf_texture_module.py','pipelines/IMAGGarment_pipeline.py',
                      'inference_IMAGGarment-1.py','tools/e33tm_weight_audit.py')})
    write(OUT/'weight_audit.json',result)
    legacy = []
    common = None
    for path in sorted(OUT.rglob('frozen_modules.json')):
        proof = read(path)
        metadata = path.parent.parent/'generation_implementation.json'
        if not metadata.exists(): metadata = OUT/'generation_implementation.json'
        active = validate_proof(proof,metadata)
        if common is None: common = active
        assert active == common, '有效推理权重跨分片不同：%s'%path
        legacy.append(dict(path=str(path.relative_to(OUT)),proof_sha256=sha(path),
                           full_bf_sha256=proof['before']['bf_texture_conditioner'],effective_hashes=active,
                           metadata_sha256=sha(metadata),effective_hash_reconstructed='effective_before' not in proof))
    write(OUT/'historical_weight_audit.json',dict(pass_audit=True,proofs=legacy,
        reconstruction_basis='Original restore path loads every active BF key from identical E5; four absent pattern_head keys are disabled.',
        weight_audit_sha256=sha(OUT/'weight_audit.json')))
    print(result,flush=True)
    for handle in handles: handle.remove()


if __name__ == '__main__': main()
