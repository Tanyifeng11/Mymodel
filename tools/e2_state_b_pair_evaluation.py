"""固定一次 B 候选更新后的 S0 评测；评测候选不等于接受或推广候选。"""
import argparse
import copy
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools.e2_b_amplitude_response_metrics import projection_features, response_summary
from tools.e2_b_last_step_metrics import score_pixels
from tools.e2_b_endpoint_losses import TASKS
from tools.e2_fft_experiment import read, write, sha
from tools.e2_state_b_amplitude_response import CHECKPOINT_SHA, ARMS, pixels, change_audit


SCORES = ('l_amplitude', 'l_amplitude_error', 'l_correlation', 'l_target_gain',
          'l_target_gain_error', 'l_hp_mae', 'palette_swd', 'spatial_ab_mae_goal',
          'spatial_ab_mae_positive', 'low_l', 'low_l_positive', 'old_projection',
          'old_projection_channel_abs_sum', 'old_l_projection', 'edge', 'l_edge')
PROTECTION = ('l_amplitude_error', 'palette_swd', 'spatial_ab_mae_goal', 'low_l',
              'old_projection', 'old_l_projection', 'edge', 'l_edge', 'l_correlation')


def linked_copy(source, destination):
    """各评测作业使用独立素材目录，不并发改写训练资产。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    digest = sha(source)
    assert sha(destination) == digest, str(destination)
    return digest


def metric_means(rows):
    values = {key: [r['scores'][key] for r in rows if r['scores'][key] is not None]
              for key in SCORES}
    return dict(count=len(rows), metrics={k: float(np.mean(v)) if v else None for k, v in values.items()},
                amplitude_pass=sum(r['scores']['amplitude_pass'] for r in rows),
                l_quality_pass=sum(r['scores']['l_quality_pass'] for r in rows))


def aggregate(rows, split):
    groups = {}
    for group in ['all'] + sorted({r['group'] for r in rows}):
        selected = rows if group == 'all' else [r for r in rows if r['group'] == group]
        groups[group] = ({a: metric_means([r for r in selected if r['arm'] == a]) for a in ARMS}
                         if split == 'replay' else metric_means(selected))
    result = dict(groups=groups)
    if split == 'replay':
        pairs = []
        for case, condition in sorted({(r['case'], r['condition']) for r in rows}):
            selected = [r for r in rows if (r['case'], r['condition']) == (case, condition)]
            assert len(selected) == 3
            pairs.append(dict(case=case, condition=condition, group=selected[0]['group'],
                              response=response_summary({r['arm']: r['projection'] for r in selected})))
        result['responses'] = pairs
    return result


def strict_replay_protection(rows, baselines):
    """确定性像素保护只给出诊断，不改变候选、不触发重新训练或选点。"""
    per_image, groups = [], []
    for row in rows:
        prior = baselines[row['case'], row['condition'], row['arm']]['scores']
        current = row['scores']
        delta = {k: float(current[k] - prior[k]) if current[k] is not None and prior[k] is not None
                 else None for k in PROTECTION}
        residue = prior['old_projection']
        failed = residue is not None and current['old_projection'] is not None and (
            current['old_projection'] > residue + 1e-5 + 1e-4 * abs(residue))
        per_image.append(dict(case=row['case'], condition=row['condition'], arm=row['arm'],
                              delta=delta, initial_residue_cap_failed=bool(failed)))
    for group in sorted({r['group'] for r in rows}):
        for arm in ARMS:
            selected = [r for r in rows if r['group'] == group and r['arm'] == arm]
            changes = {}
            failures = []
            for key in PROTECTION:
                values = [(r['scores'][key], baselines[r['case'], r['condition'], arm]['scores'][key])
                          for r in selected]
                values = [(a, b) for a, b in values if a is not None and b is not None]
                if not values:
                    changes[key] = None
                    continue
                delta = float(np.mean([a - b for a, b in values]))
                changes[key] = delta
                tolerance = 1e-8 if key == 'old_projection' else 1e-5
                if (-delta if key == 'l_correlation' else delta) > tolerance:
                    failures.append(key)
            groups.append(dict(group=group, arm=arm, delta=changes, failures=failures))
    return dict(reference='f68 same condition and amplitude', per_image=per_image, groups=groups,
                passed=not any(r['initial_residue_cap_failed'] for r in per_image)
                and not any(r['failures'] for r in groups),
                purpose='只读严格像素回归诊断；不代表训练候选已被接受',
                group_tolerance=1e-5, group_old_projection_tolerance=1e-8,
                per_image_residue_tolerance='1e-5 + 1e-4 * abs(f68)')


def training_baselines(args):
    values, hashes = {}, {}
    for shard in range(4):
        path = args.out / 'gradients' / ('shard' + str(shard) + '.json')
        report = read(path)
        assert report['protocol_sha256'] == sha(args.out / 'protocol.json')
        assert report['frozen_unchanged'] and report['training_updates'] == 0
        assert report['checkpoint_sha256'] == CHECKPOINT_SHA
        hashes[path.name] = sha(path)
        for row in report['values']:
            assert set(row['tasks']) == set(TASKS)
            values[row['case'], row['sample_condition']] = row
    assert len(values) == 48
    return values, hashes


def training_comparison(current, baseline):
    delta = {key: current[key] - baseline[key] for key in TASKS}
    tolerance = {key: 1e-5 + 1e-4 * abs(baseline[key]) for key in TASKS}
    return dict(delta=delta, tolerance=tolerance,
                failures=[key for key in TASKS if delta[key] > tolerance[key]])


def replay_inputs(args, spec):
    source = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    assert sha(source / 'result.json') == spec['amplitude_source_result_sha256']
    old = read(source / 'generated.json')
    assert old['frozen_unchanged'] and old['training_updates'] == 0
    assert old['checkpoint_sha256'] == CHECKPOINT_SHA
    manifest = read(source / 'manifest.json')
    baselines = {(r['case'], r['condition'], r['arm']): r for r in old['rows']}
    assert len(baselines) == len(spec['inputs']) == 48
    pairs = {p['case_id']: p for p in spec['train']}
    entries = []
    for item in spec['inputs']:
        case, condition, arm = item['case'], item['condition'], item['arm']
        baseline = baselines[case, condition, arm]
        assert item['sample_condition'] == baseline['sample_condition']
        assert item['reference_sha256'] == baseline['reference_sha256']
        folder = args.out / 'cases' / case
        names = ['mask.png', 'train_anchor.png', condition + '_positive.png']
        names += [item['sample_condition'] + '_' + suffix + '.png' for suffix in ('ref', 'goal', 'positive')]
        assets = {}
        for name in names:
            relative = 'cases/' + case + '/' + name
            digest = sha(folder / name)
            assert digest == manifest[relative]
            if relative in spec['assets']:
                assert digest == spec['assets'][relative]
            assets[name] = digest
        assert pairs[case]['seed'] == baseline['seed']
        entries.append(dict(item, pair=pairs[case], folder=folder, assets=assets,
                            expected_digest=baseline['noise_digest'], baseline=baseline,
                            source_image=source / 'images' / arm / (case + '_' + condition + '.png')))
    return entries, baselines, old


def holdout_inputs(args, spec, destination):
    source = args.project / 'output_eval/e2_state_b_material_20261113'
    assert sha(source / 'protocol.json') == spec['material_source_protocol_sha256']
    historical = read(source / 'protocol.json')
    originals = {p['case_id']: p for p in historical['holdout']}
    assert len(spec['holdout']) == len(originals) == 11
    train_roots = {p['identity_group'] for p in spec['train']}
    protected = {p[k] for p in spec['holdout'] for k in ('identity_group', 'donor_identity_group')}
    assert len(protected) == 22 and not train_roots & protected
    runtime = copy.copy(args)
    runtime.out = destination / 'eval_assets'
    entries = []
    for pair in spec['holdout']:
        case = pair['case_id']
        prior = originals[case]
        for key in ('identity_group', 'donor_identity_group', 'seed', 'sketch', 'stratum'):
            assert pair[key] == prior[key]
        src, folder = source / 'cases' / case, runtime.out / 'cases' / case
        generation = read(src / 'generation.json')
        assert sha(args.data / pair['sketch']) == pair['source_hashes']['target']['sketch']
        common = {'mask.png': linked_copy(src / 'mask.png', folder / 'mask.png')}
        assert common['mask.png'] == pair['mask_sha256']
        for condition in ('real_a', 'real_b'):
            name = condition + '_ref.png'
            common[name] = linked_copy(src / name, folder / name)
            assert common[name] == pair['reference_hashes'][condition]
        for seed_index in range(2):
            seeded = dict(pair, seed=pair['seed'] + 100000 * seed_index)
            for condition in ('real_a', 'real_b'):
                key = condition + '_s' + str(seed_index)
                meta = next(r for r in generation['rows'] if r['key'] == key)
                assert meta['seed'] == seeded['seed']
                names = [key + '_goal.png', key + '_positive.png', key + '_m0.png',
                         'real_a_s' + str(seed_index) + '_m0.png']
                assets = dict(common)
                for name in sorted(set(names)):
                    assets[name] = linked_copy(src / name, folder / name)
                entries.append(dict(case=case, condition=condition, group=pair['stratum'],
                    pair=seeded, folder=folder, seed_index=seed_index, assets=assets,
                    expected_digest=meta['digest'], source_generation_sha256=sha(src / 'generation.json')))
    assert len(entries) == 44
    return runtime, entries


def score_entry(entry, picture, split):
    folder, condition = entry['folder'], entry['condition']
    mask = np.asarray(Image.open(folder / 'mask.png')) > 0
    if split == 'replay':
        prefix, anchor_name = entry['sample_condition'], 'train_anchor.png'
        positive_base = pixels(folder / (condition + '_positive.png'))
    else:
        prefix = condition + '_s' + str(entry['seed_index'])
        anchor_name = 'real_a_s' + str(entry['seed_index']) + '_m0.png'
        positive_base = None
    positive, goal, anchor = [pixels(folder / name) for name in
                              (prefix + '_positive.png', prefix + '_goal.png', anchor_name)]
    result = dict(scores=score_pixels(picture, positive, goal, anchor, mask,
                                      entry['pair']['patch_coordinates']), assets=entry['assets'])
    if split == 'replay':
        result['projection'] = projection_features(picture, positive, positive_base, mask)
        result['output_change_from_f68'] = change_audit(picture, pixels(entry['source_image']), mask)
    else:
        m0 = pixels(folder / (prefix + '_m0.png'))
        result['outside_mae_to_m0'] = float(np.abs(picture.astype(np.float32) - m0)[~mask].mean() / 255)
        result['source_generation_sha256'] = entry['source_generation_sha256']
    return result


@torch.no_grad()
def generate(args):
    spec = read(args.out / 'protocol.json')
    assert spec['sampler'] == dict(steps=50, cfg=7., sketch_scale=.6, active=[25, 50])
    destination = args.out / 'evaluation' / args.arm / args.split
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / 'result.json').exists(), '评测已完成，不能覆盖结果'
    source_amplitude = args.project / 'output_eval/e2_state_b_amplitude_response_20261116'
    assert sha(source_amplitude / 'generated.json') == spec['amplitude_source_generated_sha256']
    old = read(source_amplitude / 'generated.json')
    assert old['checkpoint_sha256'] == CHECKPOINT_SHA and old['frozen_unchanged']
    if args.split == 'replay':
        entries, baselines, old = replay_inputs(args, spec)
        original_tasks, gradient_hashes = training_baselines(args)
        runtime = copy.copy(args)
    else:
        runtime, entries = holdout_inputs(args, spec, destination)
        baselines = None
    checkpoint = (args.project / 'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
                  if args.arm == 'f68' else args.out / 'training' / (args.arm + '.pt'))
    checkpoint_sha = sha(checkpoint)
    if args.arm == 'f68':
        assert checkpoint_sha == CHECKPOINT_SHA
    reuse = args.arm == 'f68' and args.split == 'replay'
    checkpoint_flags = {}
    checkpoint_report = {}
    if reuse:
        b_hash, net_hash = old['b_hash'], old['net_hash']
        environment = old['environment']
    else:
        from models.e2_state_texture_b import StateTextureB
        from tools import e2_state_b_balanced as balanced
        from tools import e2_state_b_endpoint as endpoint
        from tools import e2_state_b_pilot as pilot
        from tools.e2_state_b_pair_finetune import Objective
        from tools.e2_oracle_texture import weights_digest
        from tools.e2_fft_experiment import environment as environment_fn
        runtime.action = 'train'  # 固定历史确定性后端；本文件不创建优化器。
        core, pipe, net = pilot.load_core(runtime)
        torch.set_num_threads(min(2, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
        assert pipe.scheduler.__class__.__name__ == 'DDIMScheduler'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        checkpoint_report = payload.get('report', {})
        if args.arm != 'f68':
            assert checkpoint_report['protocol_sha256'] == sha(args.out / 'protocol.json')
        checkpoint_flags = {k: checkpoint_report[k] for k in ('parameter_updates', 'adopted', 'acceptance',
                            'projection', 'minimum_cosine', 'arm') if k in checkpoint_report}
        b = StateTextureB(color_channels=2, reference_channels=192).to(args.device)
        b.load_state_dict(payload['b'])
        del payload
        b.eval().requires_grad_(False)
        net.eval().requires_grad_(False)
        pipe.vae.half()
        b_hash, net_hash = weights_digest(b), weights_digest(net)
        assert net_hash == old['net_hash']
        environment = environment_fn()
    rows, started = [], time.monotonic()
    objective, objective_case, initialized_cases = None, None, set()
    for entry in entries:
        case, condition = entry['case'], entry['condition']
        suffix = entry['arm'] if args.split == 'replay' else 's' + str(entry['seed_index'])
        target = destination / 'images' / (case + '_' + condition + '_' + suffix + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        if reuse:
            assert sha(entry['source_image']) == entry['baseline']['png_sha256']
            linked_copy(entry['source_image'], target)
            picture = pixels(target)
            digest = entry['expected_digest']
            calls = entry['baseline']['b_calls']
            assert calls == 25
        else:
            if args.split == 'replay' and objective_case != case:
                assert case not in initialized_cases, '回放按身份连续排列，目标只初始化一次'
                items = [r for r in spec['inputs'] if r['case'] == case]
                assert len(items) == 12
                objective = Objective(runtime, pipe, None, b, entry['pair'], items)
                objective_case = case
                initialized_cases.add(case)
            recorded = []
            def record(module, positional, keywords, output):
                recorded.append(int(keywords['timestep']))
            sample_condition = entry['sample_condition'] if args.split == 'replay' else condition
            with b.register_forward_hook(record, with_kwargs=True):
                if args.split == 'replay':
                    image, digest, tap, latent = endpoint.final_sample(
                        runtime, core, pipe, net, entry['pair'], sample_condition, b)
                else:
                    image, digest, tap = balanced.sample(runtime, core, pipe, net, entry['pair'],
                                                        sample_condition, b, mode='raw')
            assert tap.index == 50 and recorded == pipe.scheduler.timesteps[25:].tolist()
            calls = len(recorded)
            assert calls == 25 and digest == entry['expected_digest']
            image.save(target)
            picture = np.asarray(image)
        row = dict(case=case, condition=condition, group=entry['group'], model_arm=args.arm,
                   seed=entry['pair']['seed'], noise_digest=digest, b_calls=calls,
                   png_sha256=sha(target), image=target.relative_to(args.out).as_posix(),
                   **score_entry(entry, picture, args.split))
        if args.split == 'replay':
            row.update(arm=entry['arm'], factor=entry['factor'], sample_condition=entry['sample_condition'])
            original = original_tasks[case, entry['sample_condition']]
            assert original['noise_digest'] == digest
            if reuse:
                row['training_tasks'] = dict(original['tasks'])
                for key in SCORES:
                    actual, expected = row['scores'][key], entry['baseline']['scores'][key]
                    assert (actual is None and expected is None) or abs(actual - expected) < 1e-8, key
            else:
                # 最终真实采样latent及同一次轨迹未加末步B的base；不需要反传或predictor。
                state = tap.states[-1]
                assert state['index'] == 49 and state['t'] == int(pipe.scheduler.timesteps[-1])
                base = pipe.scheduler.step(state['epsilon'].to(args.device), pipe.scheduler.timesteps[-1],
                                           state['zt'].to(args.device), return_dict=False)[0]
                task_entry = dict(pair=entry['pair'], condition=entry['sample_condition'])
                tasks, _ = objective.score_latents(task_entry, latent.float(), base.float(),
                                                   residue=objective.metric(task_entry).surrogate)
                row['training_tasks'] = endpoint.numbers(tasks)
            row['training_task_comparison'] = training_comparison(row['training_tasks'], original['tasks'])
        else:
            row['seed_index'] = entry['seed_index']
        rows.append(row)
        write(destination / 'partial.json', dict(rows=rows, complete=False))
        print('PAIR_EVALUATED', args.arm, args.split, len(rows), case, condition, suffix, flush=True)
    assert len(rows) == (48 if args.split == 'replay' else 44)
    if not reuse:
        assert weights_digest(b) == b_hash and weights_digest(net) == net_hash
    assert sha(checkpoint) == checkpoint_sha
    result = dict(arm=args.arm, split=args.split, rows=rows, **aggregate(rows, args.split),
        protocol_sha256=sha(args.out / 'protocol.json'), source_sha256=sha(Path(__file__)),
        checkpoint_sha256=checkpoint_sha, checkpoint_flags=checkpoint_flags, checkpoint_report=checkpoint_report,
        frozen_unchanged=True, b_hash=b_hash, net_hash=net_hash, environment=environment,
        evaluation_optimizer_updates=0, candidate_acceptance_decided_here=False,
        baseline_reused=reuse, baseline_source=str(source_amplitude) if reuse else None,
        heldout_used_for_training=False, heldout_used_for_evaluation=args.split == 'holdout',
        seconds=time.monotonic() - started,
        scope=('4个已见身份的三档响应；不是独立泛化验证' if args.split == 'replay' else
               '11个训练隔离但重复使用身份、两参考两seed；只评估alpha=1，不验证斜率泛化'))
    if baselines is not None:
        result['strict_replay_protection'] = strict_replay_protection(rows, baselines)
        failures = [dict(case=r['case'], condition=r['condition'], arm=r['arm'],
                         failures=r['training_task_comparison']['failures'])
                    for r in rows if r['training_task_comparison']['failures']]
        result['training_task_protection'] = dict(passed=not failures, failures=failures, comparisons=48 * len(TASKS),
            tolerance='1e-5+1e-4*abs(initial)', baseline_gradient_json_sha256=gradient_hashes,
            purpose='实际完整S0端点七项task验证；不自动接受或采用候选')
    write(destination / 'result.json', result)
    print('PAIR_EVALUATION_COMPLETE', args.arm, args.split, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arm', choices=('f68', 'C0', 'C1'), required=True)
    parser.add_argument('--split', choices=('replay', 'holdout'), required=True)
    args = parser.parse_args()
    args.out, args.project = args.out.resolve(), args.project.resolve()
    args.data = args.project.parent / 'datasets/BF'
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    generate(args)


if __name__ == '__main__':
    main()
