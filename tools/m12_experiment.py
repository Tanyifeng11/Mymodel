"""M12 零训练实验的分阶段执行、核验和 Slurm 调度。"""
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from tools.m12_protocol import *

GPU_SH = 'submit/m12_gpu.sh'
CPU_SH = 'submit/m12_cpu.sh'
SOURCE_FILES = ['models/m12_fixed_dual_path.py', 'tools/m12_protocol.py', 'tools/m12_runtime.py',
    'tools/m12_experiment.py', 'tools/m54_runtime.py', 'models/m54_candidates.py',
    'models/reference_attention_variants.py', 'tools/m54_eval.py', 'tools/e44_runtime.py',
    'tools/m54_input_candidates.py', 'tools/m54_prepare.py', 'tools/e48_holdout.py', GPU_SH, CPU_SH]


def native_generate(runtime, row, path):
    import torch
    runtime.effects_enabled = False
    latent_steps = []
    original = runtime.pipe.scheduler.step

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        latent_steps.append(result[0].detach().cpu())
        return result

    runtime.pipe.scheduler.step = capture
    try:
        runtime.generate(row, path, seed=42)
    finally:
        runtime.pipe.scheduler.step = original
    trace = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    trace['step_latents'] = torch.stack(latent_steps)
    torch.save(trace, path.with_suffix('.pt'))
    metadata = read(path.with_suffix('.json'))
    metadata['diagnostics_sha256'] = sha(path.with_suffix('.pt'))
    write(path.with_suffix('.json'), metadata)


def compare_trace(left, right):
    import torch
    first = torch.load(left.with_suffix('.pt'), map_location='cpu', weights_only=False)
    second = torch.load(right.with_suffix('.pt'), map_location='cpu', weights_only=False)
    fields = {key: dict(exact=bool(torch.equal(first[key], second[key])),
        maximum_absolute_difference=float((first[key].float()-second[key].float()).abs().max()))
        for key in protocol()['replay']['fields'] if key != 'PNG'}
    result = dict(fields=fields, png_exact=sha(left) == sha(right))
    result['passed'] = result['png_exact'] and all(value['exact'] for value in fields.values())
    assert result['passed'], result
    return result


def audit():
    import torch
    from PIL import Image
    from tools.e44_runtime import sketch_masks
    from tools.m54_runtime import make_runtime as native_runtime, release
    from tools.m12_runtime import make_runtime
    protocol()
    assert read(RUN/'audit/input_review_approval.json')['passed']
    valid, invalid = [], []
    for row in rows('pilot'):
        _, _, info = sketch_masks(Image.open(row['sketch']).convert('RGB').resize((384, 512)), 'cpu')
        (valid if info['valid'] else invalid).append(row)
    examples = valid[:2] + invalid[:1]
    assert len(valid) >= 2
    results = []
    for arm in BASES:
        runtime = native_runtime(arm, str(RUN/'audit/native_runtime'/arm))
        try:
            for row in examples:
                path = image('audit/native', arm, row['id'])
                native_generate(runtime, row, path)
        finally:
            release(runtime)
        runtime = make_runtime(arm, *COARSE[arm], 'audit/fusion_runtime/'+arm)
        try:
            for row in examples:
                path = image('audit/fusion', arm, row['id'])
                runtime.generate(row, path, seed=42)
                results.append(dict(arm=arm, id=row['id'], **compare_trace(path, image('audit/native', arm, row['id']))))
            write(RUN/'audit'/('projection_sources_'+arm+'.json'), dict(hook=LAYER,
                source_hashes=runtime.source_hashes, source_gain=float(runtime.processor.b1.gain),
                independently_stored_projections=runtime.processor.b1.projection.weight.data_ptr() != runtime.processor.m10.projection.weight.data_ptr(),
                source_checkpoint_sha256=B1_SHA, output_projection_source='frozen E5 UNet native to_out[0] weight; no bias in residual',
                all_parameters_frozen=all(not parameter.requires_grad for parameter in runtime.processor.parameters())))
        finally:
            release(runtime)
    runtime = make_runtime('OFF', 1., 1., 'audit/off_runtime')
    try:
        runtime.active = False
        path = image('audit/off', 'OFF', examples[0]['id'])
        runtime.generate(examples[0], path, seed=42)
        off_check = compare_trace(path, image('audit/native', 'E5', examples[0]['id']))
    finally:
        release(runtime)
    write(RUN/'audit/round0_complete.json', dict(passed=True, replays=results, reference_off=off_check,
        exact_step_latents=True, exact_conditional_unconditional_epsilon=True, exact_png=True,
        invalid_gate_examples=len(invalid[:1]), scientific_updates=0, protocol_sha256=sha(RUN/'protocol.json'), commit=commit()))
    print('M12_ROUND0_BIT_EXACT_PASS', len(results), flush=True)


def reuse_arm(source_stage, source_arm, target_stage, target_arm, source_root=RUN):
    source_root = Path(source_root)
    source = source_root/source_stage/'images'/source_arm
    copied = []
    for row in rows(target_stage):
        source_image = source/(row['id']+'.png')
        assert source_image.exists(), source_image
        for path in source.glob(row['id']+'*'):
            if path.suffix == '.pt' and source_root != RUN:
                continue
            destination = image(target_stage, target_arm, row['id']).parent/path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                try:
                    os.link(path, destination)
                except OSError:
                    shutil.copy2(path, destination)
            assert sha(path) == sha(destination)
            copied.append(dict(path=str(destination.relative_to(RUN)), sha256=sha(path)))
    write(RUN/target_stage/(target_arm+'_complete.json'), dict(n=len(rows(target_stage)), reused=True,
        source_root=str(source_root), source_stage=source_stage, source_arm=source_arm,
        split_sha256=sha(RUN/'splits'/('dev32.json' if target_stage == 'dev32' else 'pilot.json')),
        input_and_inference_checks='native replay audit; frozen source hashes; same identity and exact input hashes; initial latent/schedule verified during evaluation',
        files=copied, commit=commit()))
    folder = RUN/target_stage/target_arm
    folder.mkdir(parents=True,exist_ok=True)
    original_config = source_root/source_stage/source_arm/'config.json'
    if original_config.exists():
        shutil.copy2(original_config,folder/'config.json')
    else:
        write(folder/'config.json',dict(arm=target_arm,reused=True,source_root=str(source_root),
            source_stage=source_stage,source_arm=source_arm,source_checkpoint=read(RUN/'checkpoint_manifest.json').get(source_arm),
            regional_native_branch_rms='historical M9/M10_GAIN comparator; unavailable, not used for dual-path energy conclusions',
            protocol_sha256=sha(RUN/'protocol.json')))
    for filename in ['checkpoint_manifest.json','input_manifest.json','seed_and_sampler.json']:
        shutil.copy2(RUN/filename,folder/filename)


def prepare_stage(stage):
    protocol()
    if (RUN/stage/'stage_config.json').exists():
        return
    planned = arms(stage)
    assert all(0 <= value <= 1 for pair in planned.values() for value in pair) or stage == 'energy'
    freeze(RUN/stage/'stage_config.json', dict(stage=stage, arms=planned,
        protocol_sha256=sha(RUN/'protocol.json'), frozen_before_stage_outputs=True,
        split_sha256=sha(RUN/'splits'/('dev32.json' if stage == 'dev32' else 'calibration.json' if stage == 'calibration' else 'pilot.json'))))
    if stage == 'refine':
        for arm in BASES:
            reuse_arm('pilot', arm, stage, arm)
        for arm, pair in planned.items():
            if arm in COARSE and COARSE[arm] == pair:
                reuse_arm('pilot', arm, stage, arm)
    if stage == 'energy':
        for arm in BASES:
            reuse_arm('pilot', arm, stage, arm)
        choice = read(RUN/'selection/candidate.json')
        reuse_arm('refine', choice['selected'], stage, choice['selected'])
    if stage == 'dev32':
        for arm, root, source_stage in [('M9', ORIGINAL, 'dev32_u2400'),
                ('M10_GAIN', ORIGINAL/'m10_gain_r1', 'dev32')]:
            reuse_arm(source_stage, arm, stage, arm, root)


def generate(stage, arm):
    from tools.m12_runtime import make_runtime
    from tools.m54_runtime import release
    protocol()
    assert read(RUN/'audit/round0_complete.json')['passed']
    pair = arms(stage)[arm]
    assert not (RUN/stage/(arm+'_complete.json')).exists()
    runtime = make_runtime(arm, *pair, stage+'/runtime/'+arm)
    try:
        for row in rows(stage):
            path = image(stage, arm, row['id'])
            assert not path.exists(), 'PRESERVE_FAILED_ATTEMPT_BEFORE_RETRY'
            metadata, _, _ = runtime.generate(row, path, seed=42)
            print('M12_RGB', stage, arm, row['id'], round(metadata['seconds'], 3), flush=True)
        freeze(RUN/stage/arm/'config.json', dict(arm=arm, alpha_b=pair[0], alpha_m=pair[1],
            protocol_sha256=sha(RUN/'protocol.json'), hook_b1=LAYER, hook_m10=LAYER,
            same_state_parallel=True, source_hashes=runtime.source_hashes, scientific_updates=0))
        for filename in ['checkpoint_manifest.json', 'input_manifest.json', 'seed_and_sampler.json']:
            shutil.copy2(RUN/filename, RUN/stage/arm/filename)
        write(RUN/stage/(arm+'_complete.json'), dict(n=len(rows(stage)), scientific_updates=0,
            alpha_b=pair[0], alpha_m=pair[1], protocol_sha256=sha(RUN/'protocol.json'), commit=commit()))
    finally:
        release(runtime)


def verify(stage, full=False):
    import torch
    from tools.e44_runtime import digest
    condition_count = 0
    rms_count = 0
    rms_issues = []
    planned = arms(stage)
    for arm in planned:
        assert read(RUN/stage/(arm+'_complete.json'))['n'] == len(rows(stage))
        all_records = []
        for row in rows(stage):
            path = image(stage, arm, row['id'])
            metadata = read(path.with_suffix('.json'))
            assert sha(path) == metadata['png_sha256'] and metadata['seed'] == 42
            assert len(metadata['steps']) == 50 and not metadata['gt_used_in_inference']
            assert sha(path.with_suffix('.pt')) == metadata['diagnostics_sha256']
            if full:
                trace = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
                assert digest(trace['initial_latent']) == metadata['initial_sha256']
                assert digest(trace['final_latent']) == metadata['final_sha256']
                for key in ['conditional_epsilon', 'unconditional_epsilon', 'step_latents']:
                    assert len(trace[key]) == 50 and torch.isfinite(trace[key]).all()
            with path.with_name(path.stem+'_rms.csv').open(encoding='utf-8') as stream:
                residuals = list(csv.DictReader(stream))
            assert len(residuals) == 400
            for record in residuals:
                values = [float(record[key]) for key in ['b1_rms', 'm10_rms', 'fusion_rms']]
                assert all(math.isfinite(value) for value in values)
                if record['branch'] == 'unconditional' or record['enabled'] == 'False':
                    assert values[2] == 0
                bound = abs(float(record['alpha_b']))*values[0] + abs(float(record['alpha_m']))*values[1]
                assert values[2] <= bound * 1.01 + 1e-7, ('RMS_TRIANGLE_BOUND', arm, row['id'])
                if min(values[:2]) > 0 and max(values[:2])/min(values[:2]) > 100:
                    rms_issues.append(dict(id=row['id'], step=record['step'], region=record['region'],
                        b1_m10_ratio=max(values[:2])/min(values[:2]), classification='diagnostic scale imbalance; no threshold-based identity exclusion'))
            all_records.extend(residuals)
            condition_count += 1
            rms_count += len(residuals)
        write_csv(RUN/stage/arm/'residual_rms_by_step.csv', all_records)
    write(RUN/'audit'/(stage+('_full50' if full else '')+'_integrity.json'), dict(passed=True,
        conditions=condition_count, residual_region_rows=rms_count, source_imbalance_diagnostics=rms_issues,
        all_outputs_finite=True, unconditional_exact_zero=True, triangle_bound_passed=True,
        raw_full50_reloaded=bool(full), scientific_updates=0, commit=commit()))
    return condition_count


def panels(stage, evaluated_arms):
    from PIL import Image, ImageDraw
    for arm in evaluated_arms:
        group = rows(stage)
        canvas = Image.new('RGB', (6*192, math.ceil(len(group)/6)*280), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(group):
            horizontal, vertical = (index % 6)*192, (index // 6)*280
            canvas.paste(Image.open(image(stage, arm, row['id'])).convert('RGB').resize((192,256)), (horizontal,vertical+22))
            draw.text((horizontal+2,vertical+2), row['id'], fill='black')
        folder = RUN/stage/arm
        folder.mkdir(parents=True,exist_ok=True)
        canvas.save(folder/'output_grid.png')


def evaluate(stage):
    import numpy as np
    from tools import m54_eval as evaluator
    from tools.e44_metrics import paired
    from tools.e35_freeu_protocol import bootstrap
    verify(stage)
    evaluator.OUT = RUN
    evaluator.image_path = image
    evaluator.split_path = lambda unused: RUN/'splits'/('dev32.json' if stage == 'dev32' else 'pilot.json')
    evaluator.evaluate(stage)
    records = read(RUN/stage/'per_image.json')
    means = read(RUN/stage/'means.json')
    evaluated_arms = list(dict.fromkeys(record['variant'] for record in records))
    paired_results = {arm: {baseline: {key: paired(records,arm,baseline,key) for key in CORE}
        for baseline in evaluated_arms if baseline in BASES+['M9','M10_GAIN'] and baseline != arm} for arm in evaluated_arms}
    write(RUN/stage/'paired_all_comparators.json', paired_results)
    for arm in evaluated_arms:
        selected = [record for record in records if record['variant'] == arm]
        write_csv(RUN/stage/arm/'per_image_metrics.csv', selected)
        write(RUN/stage/arm/'paired_bootstrap.json', paired_results[arm])
        base = {record['id']:record for record in records if record['variant']=='E5'}
        tails = [dict(id=record['id'], delta_f1=record['contour_f1']-base[record['id']]['contour_f1'])
            for record in selected if record['contour_f1']-base[record['id']]['contour_f1'] < -.15]
        write_csv(RUN/stage/arm/'tail_cases.csv', tails)
    if stage == 'pilot':
        interactions = {}
        for combined, single_b, single_m in [('C55','C50','C05'),('C11','B1','M10')]:
            indexed = {(record['variant'],record['id']):record for record in records}
            interactions[combined] = {key:bootstrap([indexed[combined,row['id']][key]-indexed[single_b,row['id']][key]
                -indexed[single_m,row['id']][key]+indexed['E5',row['id']][key] for row in rows(stage)]) for key in CORE}
        write(RUN/stage/'interactions.json', interactions)
    if stage == 'dev32':
        candidate = read(RUN/'selection/dev_frozen.json')['selected']
        primary = paired_results[candidate]['M10_GAIN']
        checks = dict(lpips=primary['lpips_gt']['ci95'][1]<0,
            f1=primary['contour_f1']['ci95'][0]>=-.020, iou=primary['iou']['ci95'][0]>=-.010,
            leak=primary['leak']['ci95'][1]<=.005)
        write(RUN/stage/'primary_comparison.json', dict(comparator='M10_GAIN', metrics=primary,
            checks=checks, passed=all(checks.values()), no_claim_of_universal_best_single_branch=True))
        legacy = read(ORIGINAL/'dev32_u2400/per_image.json')
        legacy += read(ORIGINAL/'m10_gain_r1/dev32/per_image.json')
        old = {(record['variant'],record['id']):record for record in legacy}
        native_checks = []
        for record in records:
            if record['variant'] in BASES+['M9','M10_GAIN']:
                previous = old[record['variant'],record['id']]
                for key in CORE:
                    assert np.isclose(record[key],previous[key],rtol=0,atol=1e-12), ('LEGACY_METRIC_MISMATCH',record['variant'],record['id'],key)
                native_checks.append(dict(arm=record['variant'],id=record['id'],all_four_metrics_exact=True))
        write(RUN/'audit/dev32_legacy_metrics.json', native_checks)
    if stage == 'energy':
        candidate = read(RUN/'selection/candidate.json')['selected']
        comparison = {}
        for control in arms(stage):
            comparison[control] = dict(paired_candidate_minus_control={key:paired(records,candidate,control,key) for key in CORE},
                control_dominates_candidate=dominates(means[control],means[candidate]))
        energy = {}
        for arm in [candidate]+list(arms(stage)):
            source = RUN/('refine' if arm==candidate else 'energy')/arm/'residual_rms_by_step.csv'
            with source.open(encoding='utf-8') as stream:
                energy[arm] = sum(float(record['fusion_energy']) for record in csv.DictReader(stream)
                    if record['branch']=='conditional' and record['region']=='all')
        for control in arms(stage):
            comparison[control]['achieved_energy_ratio_to_combination'] = energy[control]/energy[candidate] if energy[candidate]>0 else None
        write(RUN/stage/'energy_interpretation.json',dict(comparisons=comparison,total_actual_energies=energy,
            content_complementarity_not_established_if_control_dominates=any(value['control_dominates_candidate'] for value in comparison.values()),
            limitation='calibration-derived fixed constants can change their own trajectories; report achieved energy rather than claiming exact matching'))
    panels(stage,evaluated_arms)
    print('M12_EVALUATION_COMPLETE',stage,means,flush=True)


def energy_constants():
    verify('calibration')
    choice = read(RUN/'selection/candidate.json')
    totals = {}
    for arm in arms('calibration'):
        with (RUN/'calibration'/arm/'residual_rms_by_step.csv').open(encoding='utf-8') as stream:
            records = [record for record in csv.DictReader(stream) if record['branch']=='conditional' and record['region']=='all']
        totals[arm] = {key:sum(float(record[key]) for record in records) for key in ['b1_energy','m10_energy','fusion_energy']}
    combo = totals[choice['selected']]['fusion_energy']
    constants = {}
    planned = {}
    for baseline, key in [('B1','b1_energy'),('M10','m10_energy')]:
        scale = (combo/totals[baseline][key])**.5 if totals[baseline][key]>0 else None
        acceptable = scale is not None and .5 <= scale <= 2.
        constants[baseline] = dict(scale=scale, acceptable=acceptable, native_energy=totals[baseline][key], target_combo_energy=combo)
        if acceptable:
            planned['EM_'+baseline] = [scale,0.] if baseline=='B1' else [0.,scale]
    freeze(RUN/'selection/energy_constants.json', dict(constants=constants, arms=planned,
        calibration_split_sha256=sha(RUN/'splits/calibration.json'), frozen_before_pilot_energy_controls=True,
        interpretation='single fixed calibration-derived constants; actual achieved energies reported separately'))
    print('M12_ENERGY_CONSTANTS',constants,flush=True)


def freeze_dev():
    choice = read(RUN/'selection/candidate.json')
    assert choice['selected']
    value = dict(**choice, checkpoint_manifest_sha256=sha(RUN/'checkpoint_manifest.json'),
        input_manifest_sha256=sha(RUN/'input_manifest.json'), metric_version_sha256=sha('tools/m54_eval.py'),
        sampler=protocol()['inference'], bootstrap=protocol()['bootstrap'], valid_mask='native unchanged',
        hook_b1=LAYER, hook_m10=LAYER, energy_constants_sha256=sha(RUN/'selection/energy_constants.json'),
        frozen_before_any_dev_candidate_output=True, frozen_utc=utc(), commit=commit())
    if (RUN/'selection/dev_frozen.json').exists():
        assert read(RUN/'selection/dev_frozen.json')['selected'] == choice['selected']
        return
    assert not list((RUN/'dev32').glob('images/*/*.png'))
    freeze(RUN/'selection/dev_frozen.json',value)
    print('M12_DEV_CANDIDATE_FROZEN',choice['selected'],choice['alphas'],sha(RUN/'selection/dev_frozen.json'),flush=True)


def finalize():
    from tools.m54_finalize import package
    protocol()
    stages = ['pilot']
    coarse = read(RUN/'selection/coarse.json')
    if coarse['selected']:
        stages.append('refine')
    selected = read(RUN/'selection/candidate.json')['selected'] if (RUN/'selection/candidate.json').exists() else None
    if selected:
        stages += ['calibration','dev32']
        if arms('energy'):
            stages.append('energy')
    checks = {stage:verify(stage,True) for stage in stages}
    manifests = list(RUN.rglob('frozen_*.json'))
    assert manifests and all(read(path)['unchanged'] for path in manifests)
    for stage in ['pilot','calibration','dev32']:
        for row in rows(stage):
            for key in ['gt','sketch','reference']:
                source = Path(row[key])
                destination = RUN/'inputs'/row['id']/(key+source.suffix)
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source,destination)
                assert sha(source)==sha(destination)
    dev_go = bool(selected and read(RUN/'dev32/gates.json')[selected]['passed'])
    primary_go = bool(selected and read(RUN/'dev32/primary_comparison.json')['passed'])
    write(RUN/'audit/final_complete.json',dict(passed=True, conditions_verified=checks,
        scientific_updates=0, selected=selected, dev_six_gates_passed=dev_go, primary_comparison_passed=primary_go,
        outcome='DEV_GO_REQUIRES_ENERGY_INTERPRETATION' if dev_go else 'COMPLETE_NO_DEV_GO' if selected else 'STOP_NO_ELIGIBLE_COMBINATION',
        frozen_source_records=len(manifests), independent_human_review='NOT_RUN',
        dev32_reused_development_only=True, code_commit=commit(), utc=utc()))
    files = {str(path.relative_to(RUN)):path for path in RUN.rglob('*') if path.is_file()
        and path.suffix in ['.json','.csv','.png','.jpg','.log','.err','.psv'] and '.tar.gz' not in path.name}
    package(RUN/'m12_review.tar.gz',files)
    print('M12_SERVER_FINAL_PASS',dev_go,primary_go,flush=True)


def accounting():
    from tools.m54_finalize import package
    jobs = read(RUN/'resources/jobs.json')
    git_jobs = RUN/'resources/git_sync_job_ids.csv'
    if git_jobs.exists():
        known = {job['id'] for job in jobs}
        for line in git_jobs.read_text().splitlines():
            if line.strip().isdigit() and int(line.strip()) not in known:
                identity = int(line.strip())
                jobs.append(dict(name='GIT_SYNC_'+str(identity),id=identity,action='git_pull',
                    auxiliary=True,gpu=False,stage=None,arm=None))
                known.add(identity)
        write(RUN/'resources/jobs.json',jobs)
    raw = subprocess.check_output(['sacct','-X','-P','-j',','.join(str(job['id']) for job in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    indexed = {int(record['JobIDRaw']):record for record in csv.DictReader(raw.splitlines(),delimiter='|') if record['JobIDRaw'].isdigit()}
    assert set(indexed)=={job['id'] for job in jobs}
    ledger=[]
    for job in jobs:
        record=indexed[job['id']]
        assert record['State'].split()[0] in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED']
        if job['name']=='FINAL':
            assert record['State']=='COMPLETED' and record['ExitCode']=='0:0'
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',record['AllocTRES'])
        gpu_count=int(match.group(1)) if match else sum(map(int,re.findall(r'gres/gpu:[^=,]+=(\d+)',record['AllocTRES'])))
        seconds=int(record['ElapsedRaw'])
        ledger.append(dict(job_id=job['id'], name=job['name'], state=record['State'],exit_code=record['ExitCode'],
            allocated_gpus=gpu_count, allocated_cpus=int(record['AllocCPUS']),seconds=seconds,
            gpu_hours=gpu_count*seconds/3600,cpu_hours=int(record['AllocCPUS'])*seconds/3600))
    write_csv(RUN/'runtime_resource_ledger.csv',ledger)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    write(RUN/'resources/accounting.json',dict(records=ledger,gpu_hours=sum(record['gpu_hours'] for record in ledger),
        cpu_hours=sum(record['cpu_hours'] for record in ledger),includes_failed_allocations=True,scientific_updates=0))
    package(RUN/'m12_ledger.tar.gz',{name:RUN/name for name in ['runtime_resource_ledger.csv','resources/jobs.json','resources/accounting.psv','resources/accounting.json']})
    print('M12_ACCOUNTING_COMPLETE',len(ledger),flush=True)


def submit(name, action, gpu=True, stage=None, arm=None):
    jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
    if any(job['name']==name for job in jobs):
        return
    command=['sbatch','--parsable','--job-name=M12_'+name,GPU_SH if gpu else CPU_SH,action]
    if stage:
        command += ['--stage',stage]
    if arm:
        command += ['--arm',arm]
    output=subprocess.run(command,capture_output=True,text=True)
    assert output.returncode==0,output.stderr
    job=dict(name=name,id=int(output.stdout.strip().split(';')[0]),action=action,gpu=gpu,stage=stage,arm=arm,utc=utc(),commit=commit())
    jobs.append(job)
    write(RUN/'resources/jobs.json',jobs)
    print('M12_SUBMITTED',job,flush=True)


def advance():
    import fcntl
    (RUN/'resources').mkdir(parents=True,exist_ok=True)
    with (RUN/'resources/dispatch.lock').open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
        if not jobs:
            submit('PREPARE','prepare',False)
            return
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(job['id']) for job in jobs),'--format=JobIDRaw,State,ExitCode'],text=True)
        states={int(line.split('|')[0]):line.split('|')[1:] for line in raw.splitlines() if line.split('|')[0].isdigit()}
        print('M12_STATES',[(job['name'],job['id'],states.get(job['id'])) for job in jobs],flush=True)
        if any(states.get(job['id'],['PENDING'])[0].split()[0] in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']
                for job in jobs if not job.get('superseded') and not job.get('auxiliary')):
            print('M12_ENGINEERING_REPAIR_REQUIRED',flush=True)
            return
        def completed(name):
            return any(job['name']==name and states.get(job['id'],['',''])[:2]==['COMPLETED','0:0'] for job in jobs)
        if not completed('PREPARE'):
            return
        if not (RUN/'audit/input_review_approval.json').exists():
            print('M12_ASSISTANT_INPUT_REVIEW_PENDING',flush=True)
            return
        submit('ROUND0','audit')
        if not completed('ROUND0'):
            return
        assert read(RUN/'audit/round0_complete.json')['passed']
        for stage in ['pilot','refine','calibration','energy','dev32']:
            if stage=='refine':
                if not select_candidates('pilot')['selected']:
                    break
            if stage=='calibration':
                if not select_candidates('refine')['selected']:
                    break
            if stage=='energy':
                energy_constants()
                if not arms(stage):
                    continue
            if stage=='dev32':
                freeze_dev()
            prepare_stage(stage)
            for arm in arms(stage):
                if not (RUN/stage/(arm+'_complete.json')).exists():
                    submit(stage.upper()+'_'+arm,'generate',True,stage,arm)
            stage_jobs=[stage.upper()+'_'+arm for arm in arms(stage) if any(job['name']==stage.upper()+'_'+arm for job in jobs)]
            if any(not (RUN/stage/(arm+'_complete.json')).exists() for arm in arms(stage)) or any(not completed(name) for name in stage_jobs):
                return
            if stage=='calibration':
                continue
            submit(stage.upper()+'_EVAL','evaluate',True,stage)
            if not completed(stage.upper()+'_EVAL'):
                return
        submit('FINAL','finalize',False)
        if completed('FINAL'):
            accounting()
            print('M12_SERVER_COMPLETE_LOCAL_REVIEW_PENDING',flush=True)


def retry_failed(job_id):
    jobs = read(RUN/'resources/jobs.json')
    failed = next(job for job in jobs if job['id']==job_id)
    assert not failed.get('superseded')
    state = subprocess.check_output(['sacct','-X','-n','-P','-j',str(job_id),'--format=State,ExitCode'],text=True).strip().split('|')
    assert state[0].split()[0] in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']
    archive = RUN/'failures'/str(job_id)
    archive.mkdir(parents=True,exist_ok=False)
    paths = []
    if failed['action']=='audit':
        paths = [path for path in (RUN/'audit').iterdir() if path.name in ['native','fusion','off','native_runtime','fusion_runtime','off_runtime','round0_complete.json']
            or path.name.startswith('projection_sources_')]
    elif failed['action']=='generate':
        stage, arm = failed['stage'], failed['arm']
        paths = [RUN/stage/'images'/arm,RUN/stage/'runtime'/arm,RUN/stage/arm,RUN/stage/(arm+'_complete.json')]
    elif failed['action']=='evaluate':
        paths = [path for path in (RUN/failed['stage']).iterdir() if path.is_file()
            and path.suffix in ['.json','.csv'] and path.name!='stage_config.json'
            and (not path.name.endswith('_complete.json') or path.name=='evaluation_complete.json')]
    preserved = []
    for path in paths:
        if not path.exists():
            continue
        target = archive/'artifacts'/path.relative_to(RUN)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),str(target))
        preserved.append(str(target.relative_to(RUN)))
    original_name = failed['name']
    write(archive/'attempt.json',dict(original_job=failed,slurm_state=state,preserved_artifacts=preserved,
        log_files_retained=True,repair_commit=commit(),scientific_updates=0,utc=utc()))
    failed.update(name=original_name+'_FAILED_'+str(job_id),logical_name=original_name,superseded=True)
    write(RUN/'resources/jobs.json',jobs)
    submit(original_name,failed['action'],failed['gpu'],failed.get('stage'),failed.get('arm'))


def revise_inputs():
    assert not (RUN/'audit/input_review_approval.json').exists()
    assert not (RUN/'audit/native').exists() and not (RUN/'pilot/images').exists()
    revision = protocol()['input_revision']
    assert revision == 1
    archive = RUN/('input_preflight_r'+str(revision))
    archive.mkdir(parents=True,exist_ok=False)
    for name in ['splits','audit','protocol.json','checkpoint_manifest.json','input_manifest.json','seed_and_sampler.json',
            'm12_input_review.tar.gz','m12_input_review.tar.gz.manifest.json','m12_input_review.tar.gz.sha256']:
        path = RUN/name
        if path.exists():
            shutil.move(str(path),str(archive/name))
    write(archive/'rejection.json',dict(rejected=True,reviewer='assistant',
        reason='actual pre-output R1 review found briefcase accepted by caption substring top handle; explicitly exclude briefcase',
        pilot_accessory_ids=[],calibration_accessory_ids=['149505935'],
        source_protocol_sha256=sha(archive/'protocol.json'),scientific_outputs_seen=False,
        no_metric_mask_threshold_changes=True,utc=utc(),commit=commit()))
    prepare()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','approve_inputs','advance','audit','generate','evaluate','finalize','accounting','retry_failed','revise_inputs'])
    parser.add_argument('--stage',choices=['pilot','refine','calibration','energy','dev32'],default='pilot')
    parser.add_argument('--arm')
    parser.add_argument('--failed-job',type=int)
    options=parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        write(RUN/'resources/code_entries'/(os.environ['SLURM_JOB_ID']+'.json'),dict(commit=commit(),arguments=vars(options),
            source_files={filename:sha(filename) for filename in SOURCE_FILES if Path(filename).exists()}))
    if options.action=='retry_failed':
        retry_failed(options.failed_job)
    elif options.action=='generate':
        generate(options.stage,options.arm)
    elif options.action=='evaluate':
        evaluate(options.stage)
    else:
        globals()[options.action]()


if __name__=='__main__':
    main()
