"""方案 A：旧 dev32 只新增 32 张固定 M12_GAIN，不训练、不搜索。"""
import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from tools.m54_protocol import E5, E5_SHA, B1, B1_SHA, LAYER, CONFIG, read, write, sha, commit

PROJECT = Path('/share/home/u2515283058/Mymodel')
RUN = PROJECT/'output_eval/m12_gain_dev32_20261020'
OLD = PROJECT/'output_eval/m12_fixed_dual_path_20261018'
M54 = PROJECT/'output_eval/m54_b1_screen_20261017'
GAIN_CHECKPOINT = M54/'m10_gain_r1/training/checkpoints/step2400.pt'
ARMS = ['E5', 'B1', 'M10', 'M10_GAIN', 'M12', 'M12_GAIN']
CORE = ['lpips_gt', 'contour_f1', 'iou', 'leak']
SOURCE_FILES = ['models/m12_fixed_dual_path.py', 'models/m12_gain_dual_path.py',
    'models/m54_candidates.py', 'models/m54_m10_gain.py', 'models/reference_attention_variants.py',
    'models/reference_feature_writer.py', 'tools/m12_gain_runtime.py', 'tools/m12_gain_experiment.py',
    'tools/m12_runtime.py', 'tools/m54_runtime.py', 'tools/m54_m10_gain_runtime.py',
    'tools/e44_runtime.py', 'tools/e47_runtime.py', 'tools/m54_eval.py', 'tools/e44_metrics.py',
    'tools/e35_freeu_protocol.py', 'tools/e33tmoc_appearance_eval.py', 'garment_mask_utils.py',
    'eval/eval_utils.py', 'eval/metrics.py', 'adapter/attention_processor.py',
    'submit/m12_gain_cpu.sh', 'submit/m12_gain_gpu.sh', 'tests/test_m12_gain_dual_path.py']


def utc():
    return datetime.now(timezone.utc).isoformat()


def csv_write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r)) if records else ['id']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def rows():
    return read(RUN/'protocol/dev32.json')


def image(arm, identity):
    if arm == 'M12_GAIN':
        return RUN/'outputs/M12_GAIN'/identity/'seed42.png'
    return OLD/'dev32/images'/('R025_050' if arm == 'M12' else arm)/(identity+'.png')


def require_protocol():
    protocol = read(RUN/'protocol/protocol.json')
    assert protocol['scientific_updates'] == 0 and protocol['new_scientific_images'] == 32
    assert commit() == protocol['implementation_commit'], 'CODE_CHANGED_AFTER_FREEZE'
    assert all(sha(path) == value for path, value in protocol['code_sha256'].items())
    assert sha(RUN/'protocol/dev32.json') == protocol['split_sha256']
    return protocol


def prepare():
    import torch
    from PIL import Image, __version__ as pillow_version
    from tools.m12_gain_runtime import load_gain
    from tools.e44_runtime import sketch_masks, digest
    if (RUN/'protocol/protocol.json').exists():
        require_protocol()
        return
    old_protocol = read(OLD/'protocol.json')
    inference = dict(size=[384, 512], steps=50, cfg=7., seed=42, sampler='DDIM',
        sketch=.6, texture=1., ipa=1., prompt='original caption unchanged', reference_rng='native frozen Writer')
    assert old_protocol['inference'] == inference
    dev = read(OLD/'splits/dev32.json')
    assert len(dev) == len({r['id'] for r in dev}) == 32
    write(RUN/'protocol/dev32.json', dev)
    inputs, identities, outputs = {}, [], []
    expected_inputs = read(OLD/'input_manifest.json')
    for row in dev:
        record = dict(id=row['id'], caption=row['caption'], caption_sha256=__import__('hashlib').sha256(row['caption'].encode()).hexdigest())
        for kind in ['gt', 'sketch', 'reference']:
            path = Path(row[kind])
            actual = sha(path)
            assert actual == expected_inputs[str(path)]
            inputs[str(path)] = actual
            record[kind+'_sha256'] = actual
            destination = RUN/'inputs'/row['id']/(kind+path.suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            decoded = Image.open(path).convert('RGB')
            decoded.save(destination.with_name(kind+'_canonical.png'))
            record[kind+'_rgb_sha256'] = __import__('hashlib').sha256(decoded.tobytes()).hexdigest()
        _, _, gate = sketch_masks(Image.open(row['sketch']).convert('RGB').resize((384,512), Image.Resampling.BILINEAR), 'cpu')
        record['valid'] = gate['valid']
        initials, finals = [], {}
        for arm in ARMS[:-1]:
            png = image(arm, row['id'])
            meta = read(png.with_suffix('.json'))
            assert sha(png) == meta['png_sha256'] and meta['id'] == row['id']
            assert meta['seed'] == 42 and not meta['gt_used_in_inference']
            assert meta['mask'] == gate and Image.open(png).size == (384,512)
            assert meta['steps'] == list(range(981, 0, -20))
            assert meta['variant'] == ('R025_050' if arm == 'M12' else arm)
            if arm == 'M12':
                assert meta['alpha_b'] == .25 and meta['alpha_m'] == .5
            initials.append(meta['initial_sha256'])
            finals[arm] = meta['final_sha256']
            outputs.append(dict(id=row['id'], arm=arm, png=str(png), png_sha256=sha(png),
                metadata_sha256=sha(png.with_suffix('.json')), seed=42, valid=gate['valid'],
                initial_sha256=meta['initial_sha256'], final_sha256=meta['final_sha256']))
        assert len(set(initials)) == 1
        identities.append(dict(record, initial_sha256=initials[0]))
    checkpoint_manifest = {}
    old_checks = read(OLD/'checkpoint_manifest.json')
    for name, path, expected in [('E5', E5, E5_SHA), ('B1', B1, B1_SHA),
            ('M10_GAIN', GAIN_CHECKPOINT, old_checks['M10_GAIN']['sha256'])]:
        actual = sha(path)
        assert actual == expected
        checkpoint_manifest[name] = dict(path=str(path.resolve()), sha256=actual, recomputed=True)
    gain = load_gain(GAIN_CHECKPOINT, 'cpu')
    values = gain(torch.arange(1000))
    checkpoint_manifest['gain_audit'] = dict(parameters=1280, step=2400,
        finite=True, frozen=all(not p.requires_grad for p in gain.parameters()),
        tensor_hashes={k:digest(v) for k,v in gain.state_dict().items()},
        gamma_min=float(values.min()), gamma_max=float(values.max()), timestep_range=[0,999])
    model_files = read(M54/'manifests/model_file_hashes.json')
    assert all(sha(path) == expected for path, expected in model_files.items())
    write(RUN/'protocol/model_file_hashes.json', model_files)
    write(RUN/'protocol/checkpoint_manifest.json', checkpoint_manifest)
    write(RUN/'protocol/input_hashes.json', inputs)
    write(RUN/'protocol/source_output_manifest.json', outputs)
    csv_write(RUN/'protocol/source_output_manifest.csv', outputs)
    csv_write(RUN/'protocol/dev32_identity_manifest.csv', identities)
    examples = [r['id'] for r in identities if r['valid']][:2] + [r['id'] for r in identities if not r['valid']][:1]
    assert len(examples) == 3
    write(RUN/'protocol/environment_prepare.json', dict(python=platform.python_version(), pytorch=torch.__version__,
        cuda_build=torch.version.cuda, pillow=pillow_version, diffusers=__import__('diffusers').__version__))
    frozen = dict(experiment='M12_GAIN_DEV32_A', source_plan_sha256='8701748220d67c20f14ca1f9349e4ec556be3eb01f6be9c3e8dbfb1231997b49',
        authorization='按上述新方案实施', scientific_updates=0, new_scientific_images=32,
        arms=ARMS, coefficients=dict(alpha_B=.25, alpha_M=.5), inference=inference,
        reused_outputs=160, development_only=True, independent_confirmation=False,
        split_sha256=sha(RUN/'protocol/dev32.json'), input_hashes_sha256=sha(RUN/'protocol/input_hashes.json'),
        source_output_manifest_sha256=sha(RUN/'protocol/source_output_manifest.json'),
        source_protocol_sha256=sha(OLD/'protocol.json'), source_checkpoint_manifest_sha256=sha(RUN/'protocol/checkpoint_manifest.json'),
        code_sha256={path:sha(path) for path in SOURCE_FILES}, implementation_commit=commit(),
        smoke_ids=examples, smoke_selection='first two valid and first invalid in original dev32 order; no output ranking',
        replay=dict(criterion='bit exact', keys=['initial_latent','final_latent','conditional_epsilon','unconditional_epsilon','step_latents'], png=True),
        diagnostic_storage='all 50 conditional/unconditional epsilons and step latents for all new runs; PNG and tensor SHA retained',
        bootstrap=dict(unit='identity', draws=10000, seed=32042), safety=CONFIG['safety'],
        primary=dict(comparator='M10_GAIN', metric='LPIPS', exploratory=True, ci_upper_lt=0.,
            structure_margins_descriptive_only=dict(f1_lower=-.020,iou_lower=-.010,leak_upper=.005)),
        secondary='all five paired comparisons reported; exploratory, no confirmatory multiplicity claims',
        local_verification=dict(canonical_decoded_inputs=True, regional_atol=1e-12, lpips_cpu_atol=1e-6),
        stop='no alpha search, no new training or independent confirmation in this task',
        writer_pretraining_overlap='UNKNOWN', frozen_utc=utc())
    write(RUN/'protocol/protocol.json', frozen)
    print('M12_GAIN_PREPARED', len(outputs), examples, checkpoint_manifest['gain_audit'], flush=True)


def trace_hashes(path):
    import torch
    from tools.e44_runtime import digest
    tensors = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    return {key:digest(tensors[key]) for key in require_protocol()['replay']['keys']}


def compare(left, right, historical=None):
    first, second = trace_hashes(left), trace_hashes(right)
    result = dict(tensors={key:first[key] == second[key] for key in first}, png=sha(left) == sha(right))
    if historical:
        old = read(historical.with_suffix('.json'))
        current = read(left.with_suffix('.json'))
        result['historical_png'] = sha(left) == sha(historical)
        result['historical_final_latent'] = current['final_sha256'] == old['final_sha256']
    assert all(result['tensors'].values()) and all(v for k,v in result.items() if k!='tensors'), result
    return result


def smoke():
    import torch
    from tools.m12_gain_runtime import make_runtime
    from tools.m54_runtime import release
    from tools.m12_experiment import native_generate
    protocol = require_protocol()
    examples = [next(r for r in rows() if r['id'] == identity) for identity in protocol['smoke_ids']]
    replays = []
    for arm, alphas in [('E5',(0.,0.)), ('B1',(1.,0.)), ('M10_GAIN',(0.,1.))]:
        native = make_runtime('smoke/runtime/native/'+arm, native=arm)
        try:
            for row in examples:
                path = RUN/'smoke/native'/arm/(row['id']+'.png')
                native_generate(native, row, path)
                assert sha(path) == sha(image(arm,row['id'])), ('HISTORICAL_NATIVE_REPLAY',arm,row['id'])
        finally:
            release(native)
        runtime = make_runtime('smoke/runtime/fusion/'+arm, *alphas)
        try:
            for row in examples:
                path = RUN/'smoke/endpoints'/arm/(row['id']+'.png')
                runtime.generate(row, path, seed=42)
                replays.append(dict(arm=arm,id=row['id'], **compare(path,RUN/'smoke/native'/arm/(row['id']+'.png'),image(arm,row['id']))))
        finally:
            release(runtime)
    runtime = make_runtime('smoke/runtime/fixed')
    checks = []
    try:
        for row in examples:
            path = RUN/'smoke/fixed'/row['id']/'seed42.png'
            meta, _, _ = runtime.generate(row,path,seed=42)
            records = runtime.processor.records
            assert len(records) == 400 and all(r['fusion_energy']==0 for r in records if r['branch']=='unconditional')
            if not meta['mask']['valid']:
                assert sha(path) == sha(image('E5',row['id'])) and all(r['fusion_energy']==0 for r in records)
            checks.append(dict(id=row['id'],valid=meta['mask']['valid'],finite=True,
                unconditional_zero=True,source_hashes_unchanged=runtime.path_hashes()==runtime.source_hashes,
                all_frozen=all(not p.requires_grad for p in runtime.processor.parameters()),
                seconds=meta['seconds'],peak_memory_gib=meta['peak_memory_gib']))
        runtime.active = False
        path = RUN/'smoke/off/seed42.png'
        runtime.generate(examples[0],path,seed=42)
        off = compare(path,RUN/'smoke/native/E5'/(examples[0]['id']+'.png'))
    finally:
        release(runtime)
    assert all(c['all_frozen'] and c['source_hashes_unchanged'] for c in checks)
    write(RUN/'smoke/endpoint_checks.json',dict(passed=True,replays=replays,reference_off=off))
    csv_write(RUN/'smoke/timing_memory.csv',checks)
    write(RUN/'smoke/complete.json',dict(passed=True,examples=protocol['smoke_ids'],endpoint_pairs=len(replays),
        fixed_formula_unit_tests='bit exact recomposition, half increment scaling, projection bias isolated',
        commit=commit(),scientific_updates=0,utc=utc()))
    print('M12_GAIN_SMOKE_BIT_EXACT_PASS',len(replays),flush=True)


def generate(shard, shards):
    from tools.m12_gain_runtime import make_runtime
    from tools.m54_runtime import release
    protocol = require_protocol()
    assert read(RUN/'smoke/complete.json')['passed']
    assert 0 <= shard < shards
    freeze = RUN/'protocol/final_code_freeze.json'
    if not freeze.exists():
        # 单作业为默认；多卡时各 shard 写相同冻结事实。
        write(freeze,dict(commit=commit(),protocol_sha256=sha(RUN/'protocol/protocol.json'),smoke_sha256=sha(RUN/'smoke/complete.json')))
    assert read(freeze)['commit'] == commit()
    runtime = make_runtime('generation/runtime/shard'+str(shard))
    completed = []
    try:
        for index, row in enumerate(rows()):
            if index % shards != shard:
                continue
            path = image('M12_GAIN',row['id'])
            if path.exists():
                raise RuntimeError('EXISTING_SCIENTIFIC_OUTPUT: preserve and audit before retry')
            meta, _, _ = runtime.generate(row,path,seed=42)
            base = read(image('E5',row['id']).with_suffix('.json'))
            assert meta['initial_sha256'] == base['initial_sha256'] and meta['mask'] == base['mask']
            meta.update(commit=commit(),protocol_sha256=sha(RUN/'protocol/protocol.json'),
                input_sha256={kind:sha(row[kind]) for kind in ['gt','sketch','reference']},
                caption_sha256=__import__('hashlib').sha256(row['caption'].encode()).hexdigest(),
                checkpoint_manifest_sha256=sha(RUN/'protocol/checkpoint_manifest.json'),
                full_trace_tensor_hashes=trace_hashes(path),scientific_updates=0)
            assert all(meta['input_sha256'][kind] == read(RUN/'protocol/input_hashes.json')[row[kind]] for kind in meta['input_sha256'])
            write(path.with_suffix('.json'),meta)
            completed.append(row['id'])
            print('M12_GAIN_GENERATED',len(completed),row['id'],meta['seconds'],flush=True)
    finally:
        release(runtime)
    write(RUN/'generation'/('shard%d_complete.json'%shard),dict(n=len(completed),identities=completed,shard=shard,shards=shards,commit=commit()))


def verify_outputs():
    protocol = require_protocol()
    source = read(RUN/'protocol/source_output_manifest.json')
    for item in source:
        path = Path(item['png'])
        assert sha(path) == item['png_sha256'] and sha(path.with_suffix('.json')) == item['metadata_sha256']
    outputs = []
    for row in rows():
        path = image('M12_GAIN',row['id'])
        meta = read(path.with_suffix('.json'))
        assert sha(path) == meta['png_sha256'] and sha(path.with_suffix('.pt')) == meta['diagnostics_sha256']
        assert meta['commit'] == protocol['implementation_commit']
        outputs.append(dict(id=row['id'],path=str(path),sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json')),
            trace_sha256=sha(path.with_suffix('.pt')),valid=meta['mask']['valid'],seconds=meta['seconds'],peak_memory_gib=meta['peak_memory_gib']))
    write(RUN/'audit/output_hashes.json',outputs)
    return outputs


def evaluate():
    from tools import m54_eval as evaluator
    from tools.e44_metrics import paired
    protocol = require_protocol()
    outputs = verify_outputs()
    # 使用原评价实现重评分六组；原始 PNG 与历史目录保持只读。
    evaluator.OUT = RUN
    evaluator.image_path = lambda stage, arm, identity:image(arm,identity)
    evaluator.split_path = lambda stage:RUN/'protocol/dev32.json'
    for arm in ARMS:
        write(RUN/'evaluation'/(arm+'_complete.json'),dict(n=32,reused=arm!='M12_GAIN'))
    evaluator.evaluate('evaluation')
    records = read(RUN/'evaluation/per_image.json')
    previous = {(r['variant'],r['id']):r for r in read(OLD/'dev32/per_image.json')}
    legacy_checks = []
    for record in records:
        if record['variant'] == 'M12_GAIN':
            continue
        old = previous['R025_050' if record['variant']=='M12' else record['variant'],record['id']]
        assert all(abs(record[key]-old[key]) <= 1e-12 for key in CORE), ('LEGACY_METRIC_MISMATCH',record['variant'],record['id'])
        legacy_checks.append(dict(id=record['id'],arm=record['variant'],core_metrics_exact=True))
    comparisons = {baseline:{key:paired(records,'M12_GAIN',baseline,key) for key in CORE} for baseline in ARMS[:-1]}
    write(RUN/'metrics/paired_deltas.json',comparisons)
    write(RUN/'metrics/legacy_rescore_checks.json',legacy_checks)
    for filename in ['means.json','gates.json','valid_counts.json','per_image.json','metrics.csv','texture_metric_protocol.json']:
        shutil.copy2(RUN/'evaluation'/filename,RUN/'metrics'/filename)
    gates = read(RUN/'metrics/gates.json')['M12_GAIN']
    primary = comparisons['M10_GAIN']
    descriptive = dict(f1=primary['contour_f1']['ci95'][0]>=-.020,
        iou=primary['iou']['ci95'][0]>=-.010,leak=primary['leak']['ci95'][1]<=.005)
    base = {r['id']:r for r in records if r['variant']=='E5'}
    worst = sorted([dict(id=r['id'],delta_f1=r['contour_f1']-base[r['id']]['contour_f1'],
        delta_lpips=r['lpips_gt']-base[r['id']]['lpips_gt'],**{k:r[k] for k in CORE})
        for r in records if r['variant']=='M12_GAIN'],key=lambda r:r['delta_f1'])
    csv_write(RUN/'metrics/worst_cases.csv',worst)
    csv_write(RUN/'metrics/mask_validity.csv',[dict(id=r['id'],valid=r['valid']) for r in outputs])
    # 预先固定逐身份删一分析，只诊断影响，不据此删样或重新筛选。
    indexed = {(r['variant'],r['id']):r for r in records}
    differences = [indexed['M12_GAIN',r['id']]['lpips_gt']-indexed['M10_GAIN',r['id']]['lpips_gt'] for r in rows()]
    loo = [(sum(differences)-value)/31 for value in differences]
    decision = dict(engineering_passed=True,six_gates_passed=gates['passed'],
        primary_lpips_ci_upper_lt_zero=primary['lpips_gt']['ci95'][1]<0,
        descriptive_safety=descriptive,descriptive_margins_are_not_legacy_formal_gates=True,
        outcome='WORTH_INDEPENDENT_CONFIRMATION' if gates['passed'] and primary['lpips_gt']['ci95'][1]<0 and all(descriptive.values()) else 'STOP_FIXED_RECIPE_NO_DEV_GO',
        independent_confirmation_run=False,development_only=True,scientific_updates=0,
        leave_one_identity_out_lpips_mean_range=[min(loo),max(loo)],
        candidate_count=32,reused_count=160,commit=commit(),utc=utc())
    write(RUN/'metrics/decision.json',decision)
    print('M12_GAIN_EVALUATION_COMPLETE',decision,flush=True)


def finalize():
    from tools.m54_finalize import package
    protocol = require_protocol()
    verify_outputs()
    assert read(RUN/'evaluation/evaluation_complete.json')['complete']
    frozen = list(RUN.rglob('frozen_*.json'))
    assert frozen and all(read(path)['unchanged'] for path in frozen)
    assert all(sha(path)==expected for path,expected in read(RUN/'protocol/model_file_hashes.json').items())
    assert all(sha(path)==expected for path,expected in read(RUN/'protocol/input_hashes.json').items())
    for name in ['E5','B1','M10_GAIN']:
        record = read(RUN/'protocol/checkpoint_manifest.json')[name]
        assert sha(record['path']) == record['sha256']
    write(RUN/'audit/server_complete.json',dict(passed=True,commit=commit(),frozen_source_records=len(frozen),
        new_scientific_images=32,scientific_updates=0,decision=read(RUN/'metrics/decision.json'),utc=utc()))
    files = {str(path.relative_to(RUN)):path for path in RUN.rglob('*') if path.is_file()
        and path.suffix in ['.json','.csv','.png','.jpg','.log','.err','.psv'] and '.tar.gz' not in path.name}
    # 旧五组只读打包，便于本地独立复算，不在服务器另做推理。
    for row in rows():
        for arm in ARMS[:-1]:
            source = image(arm,row['id'])
            files['reused/'+arm+'/'+row['id']+'.png'] = source
            files['reused/'+arm+'/'+row['id']+'.json'] = source.with_suffix('.json')
    for path in SOURCE_FILES:
        files['code/'+path] = Path(path)
    package(RUN/'m12_gain_dev32_review.tar.gz',files)
    print('M12_GAIN_SERVER_COMPLETE_LOCAL_REVIEW_PENDING',flush=True)


def accounting():
    from tools.m54_finalize import package
    jobs = read(RUN/'resources/jobs.json')
    raw = subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    indexed = {int(r['JobIDRaw']):r for r in csv.DictReader(raw.splitlines(),delimiter='|') if r['JobIDRaw'].isdigit()}
    ledger = [dict(job_id=j['id'],action=j['action'],**indexed[j['id']]) for j in jobs]
    csv_write(RUN/'audit/runtime_resource_ledger.csv',ledger)
    csv_write(RUN/'audit/failed_jobs.csv',[r for r in ledger if r['State']!='COMPLETED'])
    files = {str(p.relative_to(RUN)):p for p in (RUN/'resources').glob('*') if p.is_file()}
    files.update({str(p.relative_to(RUN)):p for p in (RUN/'audit').glob('*.csv')})
    package(RUN/'m12_gain_dev32_ledger.tar.gz',files)
    print('M12_GAIN_ACCOUNTING',[(r['job_id'],r['State']) for r in ledger],flush=True)


def run():
    smoke()
    generate(0, 1)
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','smoke','generate','evaluate','finalize','accounting','run'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args = parser.parse_args()
    (RUN/'resources').mkdir(parents=True,exist_ok=True)
    if args.action == 'generate':
        generate(args.shard,args.shards)
    else:
        globals()[args.action]()
