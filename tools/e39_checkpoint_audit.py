"""核验旧 E5 未加载的辅助头；只在 CPU 构造模块，不改动推理权重或结果。"""
import hashlib
import torch
from models.bf_texture_module import BFTextureConditioner
from tools.e39_protocol import E5, OUT, EXPECTED_E5, sha, write


def run():
    assert sha(E5) == EXPECTED_E5
    state = torch.load(E5, map_location='cpu')['bf_texture_conditioner']
    channels = tuple(int(state['stage%d.0.weight' % i].shape[0]) for i in range(1, 5))
    records = []
    for seed in [39051, 39052]:
        torch.manual_seed(seed)
        module = BFTextureConditioner(clip_embeddings_dim=1024, cross_attention_dim=768,
            num_tokens=16, stage_channels=channels, texture_mode='patch_resampled')
        missing, unexpected = module.load_state_dict(state, strict=False)
        assert set(missing) == {'pattern_head.0.weight', 'pattern_head.0.bias',
                                'pattern_head.2.weight', 'pattern_head.2.bias'}
        assert not unexpected and not module.pattern_loss_enabled
        hashes = {k:hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()
                  for k,v in module.state_dict().items()}
        records.append(dict(seed=seed, missing=list(missing), unexpected=list(unexpected),
            pattern_loss_enabled=module.pattern_loss_enabled, tensor_sha256=hashes))
    changed = [k for k in records[0]['tensor_sha256']
               if records[0]['tensor_sha256'][k] != records[1]['tensor_sha256'][k]]
    assert set(changed) == set(records[0]['missing'])
    result = dict(checkpoint_sha256=sha(E5), cpu_only=True, records=records,
        different_across_initializations=changed, loaded_tensors_identical=True,
        inactive_pattern_head_only=True,
        forward_guard='pattern_head(tokens) only runs when pattern_loss_enabled is True; E5 default is False')
    write(OUT/'audit/checkpoint_compatibility.json', result)
    print('E39 CHECKPOINT AUDIT COMPLETE: inactive pattern_head only', flush=True)


if __name__ == '__main__':
    run()
