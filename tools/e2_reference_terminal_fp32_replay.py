"""固定FP32支路C0/C1各一步候选的48条件缓存末步回放；不续训、不读取holdout。"""
import argparse
import copy
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_reference_terminal_fp32_propose as proposal
from tools import e2_state_b_pair_evaluation as evaluation
from tools import e2_state_b_pair_summary as summary
from tools.e2_b_scaled_replay_review import replay_eligibility
from tools.e2_state_b_amplitude_response import read, write, sha
from tools.e2_state_b_backward_repro import tensor_sha
from tools.e2_oracle_texture import weights_digest


SOURCE = Path(__file__).resolve().parents[1]
EXTRA = ('tools/e2_reference_terminal_fp32_replay.py',
         'scripts/e2_reference_terminal_fp32_replay.sbatch')


def inputs(args):
    previous_args = argparse.Namespace(**vars(args)); previous_args.out = args.proposals
    spec, _, _ = proposal.inputs(previous_args)
    previous = proposal.verify(previous_args)
    assert all(sha(args.proposals/n) == h for n, h in read(args.proposals/'manifest.json').items())
    proposals = read(args.proposals/'proposals.json')
    assert proposals['numerical_qualification'] and not proposals['stop_before_replay'] and proposals['source_final_verification']
    return spec, previous, proposals


@torch.no_grad()
def run(args):
    from tools import e2_state_b_pilot as pilot
    from tools.e2_fft_experiment import environment
    started = time.monotonic()
    spec, previous, proposals = inputs(args)
    source = proposal.fresh.direction.delta.post.q8.r16.source
    fresh, branch = proposal.fresh, proposal.fresh.direction
    runtime = copy.copy(args); runtime.action = 'train'; runtime.out = args.source_run
    entries, _, _ = evaluation.replay_inputs(runtime, spec)
    baseline_file = args.source_run/'evaluation/f68/replay/result.json'
    baseline = read(baseline_file)
    baseline_rows = {(r['case'], r['condition'], r['arm']): r for r in baseline['rows']}
    baseline_tasks = {}
    for index in range(4):
        data = torch.load(args.gradients/('gradients/shard%d.pt' % index), map_location='cpu', weights_only=False)
        for v in data['values']:
            baseline_tasks[v['case'], v['sample_condition']] = v
    assert len(baseline_tasks) == 48
    assert not args.out.exists(), '不覆盖FP32支路48条件回放'
    args.out.mkdir(parents=True)
    files = [args.proposals/n for n in read(args.proposals/'manifest.json')]+[args.proposals/'manifest.json', baseline_file]
    protocol = dict(kind='fixed one-step candidates,new anchored FP32 output branch,cached frozen S0 prefixes',
        original_S0_code_modified=False, original_sampler=spec['sampler'],
        scope='48 seen conditions per arm;not new full-S0 execution or holdout',
        parameter_point='saved C0/C1 independent original-step proposals,not updated during evaluation',
        task_baselines='fresh native f68 continuous RGB baselines from qualified new G;never old R32 task values',
        pixel_baseline='original unchanged f68 PNGs and original per-image/group gates',
        new_S0=0, UNet_sampling_calls=0, reverse_passes=0, optimizer_steps=0, holdout_read=0,
        planned_cache_observations=96, planned_zero_PNG_and_seven_task_controls=48,
        original_acceptance=spec['acceptance'], adopted_checkpoints=[],
        input_sha256={str(p.resolve()): sha(p) for p in files},
        source_sha256=dict(previous['source_sha256'], **{n: sha(SOURCE/n) for n in EXTRA}),
        execution_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
    write(args.out/'protocol.json', protocol)
    result = dict(status='incomplete', passed=False, models={}, completed_cache_observations=0,
        zero_PNG_and_seven_task_controls=0, new_S0=0, UNet_sampling_calls=0, reverse_passes=0,
        optimizer_steps=0, holdout_read=0, adopted_checkpoints=[], formal_adoption_eligibility=False)
    tensors = dict(points={})
    error = None
    try:
        _, pipe, net = pilot.load_core(runtime)
        net.eval().requires_grad_(False); pipe.vae.float().eval().requires_grad_(False)
        heads = {}
        for arm in ('C0', 'C1'):
            payload = torch.load(args.proposals/(arm+'.pt'), map_location='cpu', weights_only=False)
            assert sha(args.proposals/(arm+'.pt')) == proposals['arms'][arm]['checkpoint_sha256']
            assert payload['report']['protocol_sha256'] == sha(args.proposals/'protocol.json')
            head = source.load_head(args, args.device)
            for name in source.INCREMENTS:
                getattr(head, name).copy_(payload['increments'][name])
            head.eval().requires_grad_(False)
            assert tensor_sha(fresh.legacy.flatten(head.increment_parameters())) == payload['report']['head_sha256']
            heads[arm] = head
        hashes = dict(net=weights_digest(net), vae=weights_digest(pipe.vae),
            bases={a: weights_digest(h.base) for a, h in heads.items()},
            heads={a: tensor_sha(fresh.legacy.flatten(h.increment_parameters())) for a, h in heads.items()})
        result.update(environment=environment(), frozen_hashes=hashes)
        rows = {arm: [] for arm in heads}
        overlay = read(args.source_run/'one_step_overlay.json')
        for index, pair in enumerate(spec['train']):
            cache, _ = source.load_cache(args, index, overlay)
            objective = source.objective_from_cache(runtime, pipe, heads['C0'], pair, cache)
            meta = read(args.basis/('cache/shard%d.json' % index))
            audit = source.preflight.scheduler_configuration(pipe.scheduler.config, meta['scheduler']['config'])
            assert audit['normalized_equal']
            assert hashes['net'] == cache['net_hash'] and hashes['vae'] == meta['vae_hash_fp32']
            assert all(v == cache['b_hash'] for v in hashes['bases'].values())
            pipe.scheduler.set_timesteps(50, device=args.device)
            for entry in [r for r in entries if r['case'] == pair['case_id']]:
                case, condition = entry['case'], entry['sample_condition']; key = case+'/'+condition
                record = cache['records'][condition]; original = baseline_tasks[case, condition]
                base_z = record['final'].to(args.device)
                pipe.vae.half()
                try:
                    native = branch.delta.post.q8.r16.native_rgb_half(pipe.vae, base_z).float()
                finally:
                    pipe.vae.float()
                assert tensor_sha(native) == original['native_RGB_sha256']
                assert tensor_sha(torch.from_numpy(branch.delta.quantize(native))) == original['pixel_array_sha256']
                reference = branch.delta.decode_raw32(pipe.vae, base_z.float())
                no_b = pipe.scheduler.step(record['raw_epsilon'].to(args.device), pipe.scheduler.timesteps[-1],
                    record['zt'].to(args.device), return_dict=False)[0].float()
                no_b_rgb = objective.rgb(no_b).detach()
                zero_tasks = fresh.task_values(objective, dict(pair=pair, condition=condition), base_z.float(), native, no_b_rgb)
                assert {n: float(v) for n, v in zero_tasks.items()} == original['tasks']
                result['zero_PNG_and_seven_task_controls'] += 1
                feature, mask, base_b = [record[n].to(args.device) for n in ('feature', 'mask', 'base_delta')]
                phi = heads['C0'].statistics(record['target_record'][3]['reference'].to(args.device))['phi']
                assert float(phi) == original['phi']
                kappa = branch.delta.post.coefficient(torch.tensor(record['alpha'], device=args.device, dtype=torch.float32),
                    pipe.scheduler.final_alpha_cumprod.to(args.device).float())
                tensors['points'][key] = dict(base_latent=base_z.cpu(), kappa=float(kappa), phi=float(phi), candidates={})
                for arm, head in heads.items():
                    current_b = head.from_features(feature, mask, phi)
                    difference = current_b-base_b
                    z = branch.continuous_latent(base_z, kappa, current_b, base_b)
                    rgb = branch.delta.anchored_rgb(native, reference, branch.delta.decode_raw32(pipe.vae, z))
                    tasks = fresh.task_values(objective, dict(pair=pair, condition=condition), z, rgb, no_b_rgb)
                    numbers = {n: float(v) for n, v in tasks.items()}
                    image = branch.delta.quantize(rgb)
                    target = args.out/arm/(case+'_'+condition+'.png'); target.parent.mkdir(exist_ok=True)
                    Image.fromarray(image).save(target)
                    comparison = evaluation.training_comparison(numbers, original['tasks'])
                    row = dict(case=case, condition=entry['condition'], sample_condition=condition,
                        group=entry['group'], arm=entry['arm'], factor=entry['factor'], model_arm=arm,
                        image=target.relative_to(args.out).as_posix(), png_sha256=sha(target),
                        pixel_array_sha256=tensor_sha(torch.from_numpy(image)), rgb_float32_sha256=tensor_sha(rgb),
                        latent_continuous_sha256=tensor_sha(z), training_tasks=numbers,
                        training_task_comparison=comparison, cached_frozen_prefix=True, new_S0_calls=0,
                        **evaluation.score_entry(entry, image, 'replay'))
                    rows[arm].append(row)
                    saved = dict(latent=z.cpu(), delta_B_difference=difference.cpu())
                    item = next(v for v in spec['inputs'] if (v['case'], v['sample_condition']) == (case, condition))
                    if source.preflight.old.key(item) in fresh.SENTINELS:
                        saved['rgb'] = rgb.cpu()
                    tensors['points'][key]['candidates'][arm] = saved
                    result['completed_cache_observations'] += 1
                print('FP32_CANDIDATE_REPLAY', len(tensors['points']), key, flush=True)
        assert result['completed_cache_observations'] == 96 and result['zero_PNG_and_seven_task_controls'] == 48
        for arm, candidate in rows.items():
            failures = [dict(case=r['case'], sample_condition=r['sample_condition'], failures=r['training_task_comparison']['failures'])
                        for r in candidate if r['training_task_comparison']['failures']]
            report = dict(arm=arm, split='replay', rows=candidate, **evaluation.aggregate(candidate, 'replay'),
                training_task_protection=dict(passed=not failures, failures=failures, conditions=48, tasks=336),
                strict_replay_protection=evaluation.strict_replay_protection(candidate, baseline_rows))
            report['original_replay_checks'] = replay_eligibility(proposals['arms'][arm], report, baseline)
            report['pastel_quality_pass'] = {a: sum(r['scores']['l_quality_pass'] for r in candidate if r['group'] == 'pastel' and r['arm'] == a)
                                             for a in ('a075', 'a100', 'a125')}
            result['models'][arm] = report
        # 这里只计算原创新度；尚无新结构完整S0控制或holdout，不授予采用资格。
        reports = dict(f68=dict(replay=baseline), **{a: dict(replay=r) for a, r in result['models'].items()})
        result['innovation'] = summary.innovation(reports, dict(C1=dict(eligible=False)))
        assert weights_digest(net) == hashes['net'] and weights_digest(pipe.vae) == hashes['vae']
        assert all(weights_digest(h.base) == hashes['bases'][a] and tensor_sha(fresh.legacy.flatten(h.increment_parameters())) == hashes['heads'][a]
                   for a, h in heads.items())
        result.update(status='complete', passed=True, frozen_unchanged=True,
            meaning_of_passed='execution/zero controls only;inspect original_replay_checks and pastel quality,not adoption')
    except Exception as caught:
        error = caught
        result.update(status='failed', passed=False, error_type=type(caught).__name__, message=str(caught))
    try:
        inputs(args)
        assert all(sha(Path(n)) == h for n, h in protocol['input_sha256'].items())
        assert all(sha(SOURCE/n) == h for n, h in protocol['source_sha256'].items())
        result['source_final_verification'] = True
    except Exception as caught:
        error = error or caught
        result.update(status='failed', passed=False, source_final_verification=False, source_error=str(caught))
    result['seconds'] = time.monotonic()-started
    torch.save(tensors, args.out/'diagnostic.pt')
    result['tensor_file_sha256'] = sha(args.out/'diagnostic.pt')
    write(args.out/'result.json', result)
    names = ['protocol.json', 'diagnostic.pt', 'result.json']+[p.relative_to(args.out).as_posix() for p in sorted(args.out.glob('*/*.png'))]
    write(args.out/'manifest.json', {n: sha(args.out/n) for n in names})
    with zipfile.ZipFile(args.out/'review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for n in names+['manifest.json']:
            archive.write(args.out/n, n)
    print('FP32_REPLAY_REVIEW', result['passed'], sha(args.out/'review.zip'), flush=True)
    if error is not None:
        raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/share/home/u2515283058/Mymodel'))
    args = parser.parse_args(); args.prior = args.prior.resolve(); args.project = args.project.resolve()
    auto = args.prior/'autonomous_repair_20261116'
    for name, value in dict(fixed='scaled_projection_retry', preflight='terminal_head_preflight',
        basis='reference_terminal_preflight_v2', source_run='reference_terminal_one_step',
        global_source='reference_terminal_global_amplitude', r16='reference_terminal_r16_diagnostic',
        q8='reference_terminal_q8_direction_diagnostic', post='reference_terminal_post_latent_probe',
        fp32='reference_terminal_fp32_delta_probe', direction='reference_terminal_fp32_direction',
        gradients='reference_terminal_fp32_gradients', proposals='reference_terminal_fp32_proposals',
        out='reference_terminal_fp32_replay').items():
        setattr(args, name, auto/value)
    args.data = args.project.parent/'datasets/BF'
    args.checkpoint = args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.device = 'cuda:0'; torch.set_num_threads(2); cv2.setNumThreads(2)
    run(args)
