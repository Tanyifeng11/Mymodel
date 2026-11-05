"""用实际日志、权重和图片核验四组工程筛选，不依赖评价分数。"""
import csv
import math
import torch

from tools.tg_tfm_experiment import ROOT, START, ARMS, read, write, sha


def audit():
    torch.set_num_threads(2)
    protocol = read(ROOT / 'protocol.json')
    rows = read(ROOT / 'dev_evaluation.json')
    assert len(rows) == 64 and len({r['id'] for r in rows}) == 32
    assert sha(START) == protocol['screen_initialization_sha256']
    original = torch.load(START, map_location='cpu', weights_only=False)['network']
    base_keys = set(original)
    completes = {arm: read(ROOT / 'screen' / arm / 'complete.json') for arm in ARMS}
    configs = {arm: read(ROOT / 'screen' / arm / 'config.json') for arm in ARMS}
    first = completes['T0']
    results, latents = {}, {}
    for row in rows:
        for key in ('sketch', 'reference', 'gt'):
            assert sha(row[key]) == row[key + '_sha256']
    for arm in ARMS:
        folder = ROOT / 'screen' / arm
        complete = completes[arm]
        assert complete['updates'] == 1000 and complete['run'] == first['run']
        assert complete['stream_hashes'] == first['stream_hashes']
        assert configs[arm]['source_sha256'] == configs['T0']['source_sha256']
        logs = list(csv.DictReader((folder / 'training_log.csv').open(encoding='utf-8-sig')))
        assert [int(row['step']) for row in logs] == list(range(1, 1001))
        assert all(math.isfinite(float(v)) for row in logs for v in row.values())
        checkpoint = folder / 'step01000.pt'
        assert sha(checkpoint) == complete['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['step'] == 1000 and state['run'] == complete['run']
        assert state['fusion_config'] == configs[arm]['fusion_config']
        parameters = state['network']
        added = set(parameters) - base_keys
        assert base_keys <= set(parameters)
        assert all('.text_modulators.' in key for key in added)
        assert bool(added) == (arm != 'T0')
        assert all(torch.isfinite(value).all().item() for value in parameters.values())
        assert sum(value.numel() for value in parameters.values()) == complete['trainable_parameters']
        loaded = read(folder / 'legacy_load.json')
        assert loaded['legacy'] and loaded['loaded'] == len(base_keys)
        assert set(loaded['new']) == added
        count = 0
        for row in rows:
            path = folder / 'outputs' / row['condition'] / (row['id'] + '.png')
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256']
            assert meta['checkpoint_sha256'] == complete['checkpoint_sha256']
            assert meta['input_sketch_sha256'] == row['sketch_sha256']
            assert meta['input_reference_sha256'] == row['reference_sha256']
            assert meta['caption'] == row['caption'] and meta['seed'] == row['seed']
            assert meta['cfg_calls'] == [50, 50] and meta['cfg_policy'] == 'original_conditional_only'
            assert meta['GT_used_in_inference'] is False
            assert meta['native_size'] == [384, 512] and meta['cfg'] == 7. and meta['steps'] == 50
            key = row['id']
            if key in latents:
                assert latents[key] == meta['initial_latent_sha256']
            latents[key] = meta['initial_latent_sha256']
            count += 1
        assert count == 64
        results[arm] = dict(updates=1000, continuous_finite_logs=True, finite_checkpoint=True,
            checkpoint_sha256=complete['checkpoint_sha256'], images=count,
            expected_trainable_namespace=True, added_parameter_tensors=len(added),
            cfg_calls=[50, 50], gt_used_in_inference=False)
        del state, parameters
    write(ROOT / 'audit/four_arm_integrity.json', dict(passed=True, arms=results,
        same_shared_initialization=True, same_training_run=True, same_random_stream=True,
        same_initial_latents=True, all_input_output_hashes_valid=True,
        freezing_evidence='checkpoint namespace matches original trainable tensors plus new heads; frozen modules disabled in hashed loader source',
        limitation='frozen backbone tensors were not independently snapshotted at every update'))
    print('FOUR_ARM_INTEGRITY_PASSED', flush=True)


if __name__ == '__main__':
    audit()
