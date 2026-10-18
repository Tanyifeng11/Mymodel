"""M12 的冻结输入、强度网格及身份配对决策规则。"""
import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tools.m54_protocol import OUT as ORIGINAL, E5, E5_SHA, B1, B1_SHA, LAYER, CONFIG as M54_CONFIG
from tools.m54_protocol import read, write, sha, commit, split_path as original_split, freeze

RUN = Path('/share/home/u2515283058/Mymodel/output_eval/m12_fixed_dual_path_20261018')
COARSE = dict(E5=[0., 0.], B1=[1., 0.], M10=[0., 1.], C05=[0., .5], C50=[.5, 0.],
    C55=[.5, .5], C15=[1., .5], C51=[.5, 1.], C11=[1., 1.])
BASES = ['E5', 'B1', 'M10']
CORE = ['lpips_gt', 'contour_f1', 'iou', 'leak']
SOURCE_CHECKPOINTS = dict(E5=E5, B1=B1, M10=B1,
    M9=ORIGINAL/'training/M9/checkpoints/step2400.pt',
    M10_GAIN=ORIGINAL/'m10_gain_r1/training/checkpoints/step2400.pt')


def utc():
    return datetime.now(timezone.utc).isoformat()


def rows(stage):
    return read(RUN/'splits'/('dev32.json' if stage == 'dev32' else 'calibration.json' if stage == 'calibration' else 'pilot.json'))


def image(stage, arm, identity):
    return RUN/stage/'images'/arm/(identity+'.png')


def arms(stage):
    if stage == 'pilot':
        return COARSE
    if stage == 'refine':
        return read(RUN/'selection/coarse.json')['refinement_grid']
    if stage == 'calibration':
        choice = read(RUN/'selection/candidate.json')
        return dict(B1=[1., 0.], M10=[0., 1.], **{choice['selected']: choice['alphas']})
    if stage == 'energy':
        return read(RUN/'selection/energy_constants.json')['arms']
    choice = read(RUN/'selection/dev_frozen.json')
    return dict(E5=[0., 0.], B1=[1., 0.], M10=[0., 1.], **{choice['selected']: choice['alphas']})


def protocol():
    value = read(RUN/'protocol.json')
    assert value['scientific_updates'] == 0 and value['coarse_grid'] == COARSE
    for name, expected in value['splits_sha256'].items():
        assert sha(RUN/'splits'/(name+'.json')) == expected
    return value


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('id\n', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def prepare():
    if (RUN/'protocol.json').exists():
        protocol()
        return
    from tools.e48_holdout import exposure_sources
    from tools import m54_prepare
    from tools.m54_finalize import package
    eligible_path = Path('output_eval/e48_b1_structure_20261014/holdout_audit/eligible.json')
    eligible = read(eligible_path)
    exposed, registry = exposure_sources({row['id'] for row in eligible})
    registry = [entry for entry in registry if RUN.resolve() not in Path(entry['path']).resolve().parents]
    exposed = {identity for entry in registry for identity in entry['validation_ids']}
    held = {row['id'] for name in ['train1024', 'dev32'] for row in read(original_split(name))}
    old_out = m54_prepare.OUT
    m54_prepare.OUT = RUN
    try:
        pilot = m54_prepare.balanced(eligible, 24, set(exposed) | held)
    finally:
        m54_prepare.OUT = old_out
    blocked = exposed | held | {row['id'] for row in pilot}
    calibration = sorted([row for row in eligible if row['id'] not in blocked],
        key=lambda row: hashlib.sha256(('M12_CALIBRATION_12042_'+row['id']).encode()).hexdigest())[:8]
    assert len(calibration) == 8
    cohorts = dict(pilot=pilot, calibration=calibration, dev32=read(original_split('dev32')))
    identities = [{row['id'] for row in group} for group in cohorts.values()]
    assert all(not left & right for index, left in enumerate(identities) for right in identities[index+1:])
    signatures = read('output_eval/e48_b1_structure_20261014/holdout_audit/validation_signatures.json')
    hashes = {}
    for stage, group in cohorts.items():
        freeze(RUN/'splits'/(stage+'.json'), group)
        for row in group:
            for key in ['gt', 'sketch', 'reference']:
                value = sha(row[key])
                if stage != 'dev32':
                    assert value == signatures[row[key]]['file_sha256']
                hashes[row[key]] = value
    checkpoint_manifest = {}
    for name, path in SOURCE_CHECKPOINTS.items():
        available = path.exists()
        if available:
            checkpoint_hash = sha(path)
        else:
            assert name == 'M9'
            checkpoint_hash = read(path.with_suffix('.json'))['sha256']
            cleanup = read(ORIGINAL/'audit/cleanup_20261017.json')
            assert cleanup['complete'] and any(entry['path']==str(path) and entry.get('deleted')
                and entry['sha256']==checkpoint_hash for entry in cleanup['entries'])
        checkpoint_manifest[name] = dict(path=str(path),sha256=checkpoint_hash,available=available,
            hash_evidence='recomputed current file' if available else 'verified pre-cleanup SHA256 and retained sidecar; historical outputs only',
            origin='frozen B1 checkpoint plus frozen E5 output projection' if name=='M10' else 'frozen source')
    assert checkpoint_manifest['E5']['sha256'] == E5_SHA and checkpoint_manifest['B1']['sha256'] == B1_SHA
    model_manifest = read(ORIGINAL/'manifests/model_file_hashes.json')
    assert all(sha(path) == expected for path, expected in model_manifest.items())
    write(RUN/'checkpoint_manifest.json', checkpoint_manifest)
    write(RUN/'input_manifest.json', hashes)
    write(RUN/'audit/exposure_registry.json', registry)
    write(RUN/'audit/input_cohorts.json', dict(new_pilot=True, pilot_ids=sorted(identities[0]),
        calibration_ids=sorted(identities[1]), dev32_ids=sorted(identities[2]),
        historical_exposed_count=len(exposed), eligible_count=len(eligible),
        audit_source=str(eligible_path), audit_source_sha256=sha(eligible_path),
        historical_exact_pixel_phash_audit=True, refreshed_exposure_exclusion=True,
        writer_pretraining_overlap='UNKNOWN', independent_human_review='NOT_RUN',
        model_outputs_used_for_selection=False))
    value = dict(experiment='M12-P1', scientific_updates=0, authorization='开始实施',
        source_plan_sha256='9b53c0e0b7c9dfc59e8ea49d0a532347b6b8d894839d6d731907ef7a72af415a',
        coarse_grid=COARSE, arm_aliases=dict(C00='E5', C10='B1', C01='M10'), hook_b1=LAYER, hook_m10=LAYER,
        fusion='same unmodified E5 hidden; independent frozen projections; one add-back',
        branch_off='native valid mask and conditional-only CFG; all identities remain scored',
        source_checkpoint_manifest_sha256=sha(RUN/'checkpoint_manifest.json'),
        inference=dict(size=[384, 512], steps=50, cfg=7., seed=42, sampler='DDIM',
            sketch=.6, texture=1., ipa=1., prompt='original caption unchanged', reference_rng='native frozen Writer'),
        replay=dict(criterion='bit exact', fields=['initial_latent', 'final_latent', 'conditional_epsilon',
            'unconditional_epsilon', 'step_latents', 'PNG']),
        pilot=dict(n=24, reused=False, lpips_mean_lt=0., f1_mean_ge=-.05, extreme_count_lt=3,
            pareto='reject if either native B1 or native M10 weakly dominates all four core means and strictly improves at least one',
            ranking='tail count; F1 CI safety and lower bound; IoU CI safety and lower bound; LPIPS; arm',
            refinement='one grid; interior center +/- .25; boundary1 => .5,.75,1; all values frozen before local outputs'),
        residual_audit=dict(source='post projection and rescale, before add-back',
            regions='all; fixed Sketch foreground >=.5 on32x24; fixed19px boundary >0; complement foreground',
            diagnostic_inactive_paths='native residuals measured at the same state when valid, zero coefficient never changes output',
            checks='finite; unconditional and invalid gate exactly zero; triangle bound with float16 rounding tolerance1percent'),
        energy_match=dict(calibration_n=8, disjoint_pilot_dev=True, bounds=[.5, 2.],
            rule='one fixed sqrt(total combo energy / total native single-path energy) per branch on calibration trajectories',
            interpretation='report achieved energy on new trajectories; no test-time or step-wise rescaling'),
        bootstrap=dict(unit='identity', draws=10000, seed=32042), six_gates=M54_CONFIG['safety'],
        primary_comparator='M10_GAIN', primary_metric='LPIPS paired CI upper<0',
        primary_noninferiority=dict(contour_f1_lower=-.020, iou_lower=-.010, leak_upper=.005),
        comparison_arms=['B1', 'M10', 'M10_GAIN', 'M9'], dev32='one frozen candidate; reused development cohort',
        independent_human_review='NOT_RUN', strict_unseen_generalization=False,
        splits_sha256={name: sha(RUN/'splits'/(name+'.json')) for name in cohorts},
        frozen_before_outputs=True, frozen_utc=utc(), commit=commit())
    freeze(RUN/'protocol.json', value)
    write(RUN/'seed_and_sampler.json', value['inference'])
    m54_prepare.OUT = RUN
    try:
        for name in ['pilot', 'calibration']:
            m54_prepare.panels(cohorts[name], name)
    finally:
        m54_prepare.OUT = old_out
    files = {str(path.relative_to(RUN)): path for path in (RUN/'audit/input_review').rglob('*.png')}
    files.update({str(path.relative_to(RUN)): path for path in (RUN/'splits').glob('*.json')})
    files['audit/input_cohorts.json'] = RUN/'audit/input_cohorts.json'
    package(RUN/'m12_input_review.tar.gz', files)
    print('M12_PREPARED_NEW_PILOT24_CALIBRATION8', sha(RUN/'protocol.json'), flush=True)


def approve_inputs():
    protocol()
    freeze(RUN/'audit/input_review_approval.json', dict(passed=True, reviewer='assistant',
        actual_reviewed_ids={stage: [row['id'] for row in rows(stage)] for stage in ['pilot', 'calibration']},
        split_hashes=protocol()['splits_sha256'], model_outputs_seen=False,
        scope='input correspondence and readable garments; caption pattern tags remain exploratory',
        independent_human_review='NOT_RUN'))


def dominates(left, right):
    directions = dict(lpips_gt=-1, contour_f1=1, iou=1, leak=-1)
    differences = [directions[key] * (left[key] - right[key]) for key in CORE]
    return all(value >= 0 for value in differences) and any(value > 0 for value in differences)


def select_candidates(stage):
    means, gates = read(RUN/stage/'means.json'), read(RUN/stage/'gates.json')
    eligible, rejected = [], {}
    for arm, pair in arms(stage).items():
        if not all(value > 0 for value in pair):
            continue
        gate = gates[arm]
        differences = gate['paired_E5']
        reasons = []
        if differences['lpips_gt']['mean'] >= 0:
            reasons.append('mean_delta_lpips>=0')
        if differences['contour_f1']['mean'] < -.05:
            reasons.append('mean_delta_f1<-.05')
        if len(gate['extreme_events']) >= 3:
            reasons.append('extreme_events>=3')
        for baseline in ['B1', 'M10']:
            if dominates(means[baseline], means[arm]):
                reasons.append('dominated_by_'+baseline)
        if reasons:
            rejected[arm] = reasons
        else:
            eligible.append(arm)

    def ranking(arm):
        gate = gates[arm]
        f1_lower = gate['paired_E5']['contour_f1']['ci95'][0]
        iou_lower = gate['paired_E5']['iou']['ci95'][0]
        return len(gate['extreme_events']), f1_lower < -.020, -f1_lower, iou_lower < -.010, -iou_lower, means[arm]['lpips_gt'], arm

    ordered = sorted(eligible, key=ranking)
    selected = ordered[0] if ordered else None
    result = dict(selected=selected, alphas=arms(stage)[selected] if selected else None,
        eligible=ordered, rejected=rejected, stage=stage, ranking=protocol()['pilot']['ranking'],
        protocol_sha256=sha(RUN/'protocol.json'), new_training_updates=0)
    if stage == 'pilot':
        grid = {}
        if selected:
            values = [[.5, .75, 1.] if value == 1 else [value-.25, value, value+.25] for value in result['alphas']]
            for alpha_b in values[0]:
                for alpha_m in values[1]:
                    match = next((arm for arm, pair in COARSE.items() if pair == [alpha_b, alpha_m]), None)
                    grid[match or ('R%03d_%03d' % (round(alpha_b*100), round(alpha_m*100)))] = [alpha_b, alpha_m]
        result['refinement_grid'] = grid
        result['grid_frozen_before_refinement_outputs'] = True
        freeze(RUN/'selection/coarse.json', result)
    else:
        result['local_eligible_count'] = len(ordered)
        result['grid_evaluated_count'] = len(arms(stage))
        freeze(RUN/'selection/candidate.json', result)
    print('M12_SELECTION', stage, selected, result['alphas'], 'eligible', ordered, flush=True)
    return result
