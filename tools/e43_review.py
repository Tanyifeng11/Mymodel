"""本地核对下载结果、共同前缀、固定分母、bootstrap和阶段判定。"""
import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.e43_protocol import OUT, read, write, sha


def review(root):
    out = Path(root) / OUT
    folder = out / 'local_review'
    config = read(folder/'locked_protocol.json')
    splits = read(folder/'split_sources.json')
    pre = read(folder/'preflight.json')
    decision = read(folder/'pilot/decision.json')
    assert read(out/'download_verification.json')['sha256_verified']
    assert pre['passed'] and pre['input_hashes_unchanged'] and pre['frozen']['parameters_unchanged']
    assert pre['frozen']['E5_sha256'] == config['E5_sha256']
    bf_audit = read(folder/'audit/bf_missing_head_audit.json')
    assert bf_audit['passed'] and bf_audit['pattern_head_disabled'] and not bf_audit['pattern_head_called']
    for name, rows in splits.items():
        assert len(rows) == dict(pilot=24, train=128, validation=48, holdout=128)[name]
        assert len({r['group_id'] for r in rows}) == len(rows)
        for other, values in splits.items():
            if name != other:
                assert not {r['group_id'] for r in rows} & {r['group_id'] for r in values}
    for check in pre['checks']:
        assert check['original']['trajectory_hashes'] == check['wrapped']['trajectory_hashes']
        assert len(check['original']['trajectory_hashes']) == 50
        assert sha(folder/'precheck'/(check['id']+'_original.png')) == sha(folder/'precheck'/(check['id']+'_alpha1.png'))
        for test in check['tests']:
            assert test['pass_identical'] and test['trajectory_hashes'] == check['wrapped']['trajectory_hashes'][test['start']:]
    whole_bf_hashes = [pre['frozen']['frozen_modules']['bf_texture_conditioner']]
    for path in (folder/'pilot').glob('runtime_shard*.json'):
        runtime = read(path)
        assert runtime['input_hashes_unchanged'] and runtime['frozen']['parameters_unchanged']
        for name, value in runtime['frozen']['frozen_modules'].items():
            if name != 'bf_texture_conditioner':
                assert value == pre['frozen']['frozen_modules'][name]
        whole_bf_hashes.append(runtime['frozen']['frozen_modules']['bf_texture_conditioner'])
    records, invalid, timing = [], [], []
    png_n = 0
    for i, row in enumerate(splits['pilot']):
        sample = folder/'pilot/samples'/row['id']/'seed42'
        record = read(sample/'complete.json')
        feature = read(sample/'features.json')
        assert record['id'] == row['id'] and record['group_id'] == row['group_id'] and record['seed'] == 42
        assert len(feature['values']) == 14 and np.isfinite(feature['values']).all() and feature['feature_steps'] == [1, 5, 10]
        arms = {v['arm']: v for v in record['outputs']}
        base = arms['alpha1']
        assert len(arms) == (7 if i < 4 else 3)
        with np.load(sample/'common_prefix.npz') as prefix:
            assert hashlib.sha256(prefix['initial'].tobytes()).hexdigest() == base['initial_hash']
            assert hashlib.sha256(prefix['step10'].tobytes()).hexdigest() == base['prefix_latent_hash'] == base['trajectory_hashes'][9]
        assert base['unet_calls'] == 106 and base['prefix_unet_calls'] == 26
        for name, value in arms.items():
            path = sample/(name+'.png')
            assert sha(path) == value['output_png_sha256'] and Image.open(path).size == (384, 512)
            png_n += 1
            timing.append(value)
            if name == 'alpha1':
                continue
            start = value['start_step'] - 1
            assert len(value['trajectory_hashes']) == 50-start
            assert value['branch_start_hash'] == (base['prefix_latent_hash'] if start == 10 else base['initial_hash'])
            assert value['unet_calls'] == (50-start)*(2 if value['effective_alpha'] == 1 else 4)
            if not feature['mask_valid']:
                assert value['effective_alpha'] == 1 and value['trajectory_hashes'] == base['trajectory_hashes'][start:]
                assert value['output_png_sha256'] == base['output_png_sha256']
        if not feature['mask_valid']:
            invalid.append(record['id'])
        records.append(record)
    assert png_n == 88 and len(records) == 24 and len(invalid) == 3
    keys = ['LPIPS_cloth', 'HP_gray', 'LAB', 'IoU', 'EdgeF1', 'Leak']
    for kind in ['comparisons', 'controls']:
        rows = records if kind == 'comparisons' else records[:4]
        for arm, comparison in decision[kind].items():
            for key in keys:
                values = []
                for row in rows:
                    a = {v['arm']: v['metrics'] for v in row['outputs']}
                    if a[arm][key] is not None and a['alpha1'][key] is not None:
                        values.append(a[arm][key]-a['alpha1'][key])
                x = np.asarray(values)
                rng = np.random.default_rng(43042)
                ci = np.quantile(x[rng.integers(len(x), size=(2000, len(x)))].mean(1), [.025, .975])
                expected = comparison['metrics'][key]
                assert len(x) == expected['groups'] and math.isclose(x.mean(), expected['mean'], abs_tol=1e-12)
                assert np.allclose(ci, expected['ci95'], atol=1e-12)
    safe_n = any_n = 0
    directions, oracle, failures = Counter(), [], {'low': [], 'high': []}
    for row in records:
        a = {v['arm']: v['metrics'] for v in row['outputs']}
        base = a['alpha1']
        if base['LPIPS_cloth'] is None:
            continue
        gains, safe_positive = [], False
        for arm in ['low', 'high']:
            v = a[arm]
            gain = base['LPIPS_cloth']-v['LPIPS_cloth']
            gains.append(gain)
            safe = v['IoU']-base['IoU'] >= -.005 and v['EdgeF1']-base['EdgeF1'] >= -.01 and v['Leak']-base['Leak'] <= .002
            safe_positive |= safe and gain > 0
            directions[arm] += bool(safe and gain > 0 and v['HP_gray'] < base['HP_gray'])
            failures[arm].append(dict(id=row['id'], delta_lpips=-gain, delta_iou=v['IoU']-base['IoU'], delta_hp=v['HP_gray']-base['HP_gray']))
        safe_n += safe_positive
        any_n += max(gains) > 0
        oracle.append(max(0, max(gains)))
    assert (len(oracle), safe_n, any_n) == (decision['valid_n'], decision['safe_positive_gain_n'], decision['any_positive_gain_n'])
    assert directions['low'] == 10 and directions['high'] == 0 and decision['reasons'] == ['no_safe_noncolor_gain_in_both_directions']
    assert math.isclose(np.mean(oracle), decision['oracle_upper_bound']['mean'], abs_tol=1e-12)
    resources = dict(pilot_unet_calls=sum(v['unet_calls'] for v in timing),
        pilot_sum_recorded_generation_seconds=sum(v['seconds'] for v in timing),
        pilot_peak_allocated_GiB=max(v['peak_memory_bytes'] for v in timing)/2**30,
        pilot_gpu_job_seconds=301+243, precheck_gpu_job_seconds=146, total_GPU_card_hours=(301+243+146)/3600,
        per_arm={arm: dict(n=sum(v['arm'] == arm for v in timing),
            mean_seconds=float(np.mean([v['seconds'] for v in timing if v['arm'] == arm])),
            total_unet_calls=sum(v['unet_calls'] for v in timing if v['arm'] == arm)) for arm in {v['arm'] for v in timing}},
        deployment_note='原E5每图100次UNet；早期探测后回退106次，非1后缀186次。当前为分支采集计时，未实测独立控制器部署速度。')
    result = dict(passed=True, generated_png_n=png_n, precheck_png_n=4, sources=24, valid_sources=21,
        invalid_mask_ids=invalid, source_splits_disjoint=True, all_generated_png_sha256_verified=True,
        prefix_npz_hashes_verified=True, precheck_50step_and_png_identity_verified=True,
        active_frozen_modules_consistent=True, bf_whole_module_hashes=whole_bf_hashes, bf_auxiliary_head_audit=bf_audit,
        raw_input_content_verified_on_server_not_locally_recomputed=True,
        local_metric_scope='复算逐图指标的来源汇总、bootstrap和门槛；本地未重新运行LPIPS模型',
        safe_noncolor_counts=dict(directions), resources=resources,
        worst_cases={arm: dict(lpips=sorted(v, key=lambda x:x['delta_lpips'], reverse=True)[:3],
            iou=sorted(v, key=lambda x:x['delta_iou'])[:3]) for arm, v in failures.items()},
        no_later_generation_or_policy_fitted=not (folder/'policy.pkl').exists())
    write(out/'local_verification.json', result)
    def panel(ids, arms, name):
        sheet = Image.new('RGB', (144*(len(arms)+1), 220*len(ids)), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, sid in enumerate(ids):
            i = next(i for i, r in enumerate(splits['pilot']) if r['id'] == sid)
            source = Image.open(folder/'pilot/contact_sheets'/('fixed_%03d.jpg' % (i//6)))
            sheet.paste(source.crop((288, (i%6)*220, 428, (i%6)*220+187)), (0, j*220))
            draw.text((0, j*220+190), 'GT '+sid, fill='black')
            for k, arm in enumerate(arms, 1):
                image = Image.open(folder/'pilot/samples'/sid/'seed42'/(arm+'.png'))
                image.thumbnail((140, 187))
                sheet.paste(image, (k*144, j*220))
                draw.text((k*144, j*220+190), arm, fill='black')
                draw.text((k*144, j*220+204), sid, fill='black')
        sheet.save(folder/'pilot/contact_sheets'/name)
    panel([r['id'] for r in splits['pilot'][:4]], ['alpha1', 'low', 'full_low', 'unmasked_low', 'high', 'full_high', 'unmasked_high'], 'local_controls.jpg')
    panel(['210013766', '195833180', '162607053'], ['alpha1', 'low', 'high'], 'local_failures.jpg')
    print('E43 VERIFIED', png_n, 'PNGs;', len(oracle), 'measurable sources;', dict(directions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    review(parser.parse_args().root)
