"""E45 真 GPU 回放、固定八臂采样与归档；从不写 E44 结果。"""
import argparse
import platform
import re
import subprocess
import tarfile
from pathlib import Path
import torch
from PIL import Image
from models.e45_pnp_anchor import Anchor
from tools.e45_protocol import (OUT, OLD, CKPT, CKPT_SHA, ARMS, CONFIG, SIZE,
    read, write, sha, commit, protocol, image_path, historical)


def runtime(namespace):
    from tools import e44_runtime as module
    module.OUT = OUT / namespace / 'runtime'
    r = module.Runtime('B1')
    r.load(CKPT)
    r.processor.eval().requires_grad_(False)
    r.writer.unet.eval().requires_grad_(False)
    r.pipe.unet.eval().requires_grad_(False)
    return r


def run_case(r, row, arm, stage, disabled=False):
    from tools.e44_runtime import sketch_masks
    from tools.e44_b1_mask import region_masks
    path = image_path(arm, row['id'], stage)
    trace = path.with_name(path.stem + '_anchor.json')
    if trace.exists():
        saved = read(trace)
        assert saved['arm'] == arm and saved['disabled'] == disabled
        assert sha(path) == saved['generation']['png_sha256']
        assert sha(path.with_suffix('.pt')) == saved['generation']['diagnostics_sha256']
        return saved
    sketch = Image.open(row['sketch']).convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
    raw, _, info = sketch_masks(sketch, 'cpu')
    controller = Anchor(r, arm, region_masks(raw), disabled, audit_unconditional=stage.startswith('s0'))
    writer_before = r.writer.calls
    controller.install()
    try:
        record, _, _ = r.generate(row, path)
    finally:
        controller.close()
    assert len(controller.steps) == 50
    assert controller.teacher_calls == controller.student_calls == controller.unconditional_calls == 50
    assert r.writer.calls - writer_before == (50 if info['valid'] and arm != 'C0' else 0)
    baseline = read(historical('C1', row['id']).with_suffix('.json'))
    assert record['initial_sha256'] == baseline['initial_sha256']
    if arm in ['C0','C1','C2'] or disabled:
        source_arm = 'C1' if disabled else arm
        old = read(historical(source_arm, row['id']).with_suffix('.json'))
        assert record['png_sha256'] == old['png_sha256'], (arm, row['id'], 'RGB 回放失败')
        assert record['final_sha256'] == old['final_sha256'], (arm, row['id'], 'latent 回放失败')
    if disabled:
        old = torch.load(historical('C1', row['id']).with_suffix('.pt'), map_location='cpu', weights_only=False)
        new = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        assert torch.equal(new['unconditional_epsilon'], old['unconditional_epsilon'])
        assert torch.equal(new['conditional_epsilon'], old['conditional_epsilon'])
    if not info['valid']:
        assert record['png_sha256'] == read(historical('C0', row['id']).with_suffix('.json'))['png_sha256']
        assert all(x['weight'] == 0 for x in controller.steps)
    energy = [x['equal_energy']['relative_energy_error'] for x in controller.steps if 'equal_energy' in x]
    if energy:
        assert max(energy) < CONFIG['equal_energy_max_relative_error'], 'half 精度下的逐步等能量误差超过预锁定0.3%'
    saved = dict(arm=arm, disabled=disabled, generation=record, steps=controller.steps,
        teacher_calls=controller.teacher_calls, student_calls=controller.student_calls,
        unconditional_calls=controller.unconditional_calls, writer_calls=r.writer.calls-writer_before,
        same_state_unconditional_audit=controller.audit_unconditional,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, training_updates=0,
        exact_historical_replay=arm in ['C0','C1','C2'] or disabled, source_commit=commit())
    write(trace, saved)
    print('E45 RGB', stage, arm, row['id'], 'seconds', round(record['seconds'], 2), flush=True)
    return saved


def frozen_processor(r):
    from tools.e22_4_generation import module_hashes
    return module_hashes({'b1_processor':r.processor})


def audit():
    rows, frozen = protocol()
    smoke = [next(r for r in rows if r['id']==x) for x in frozen['smoke_ids']]
    import diffusers
    write(OUT / 'manifests/environment.json', dict(commit=commit(), python=platform.python_version(),
        torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), protocol_sha256=sha(OUT/'protocol.json')))
    r = runtime('s0')
    before = frozen_processor(r)
    results = []
    try:
        for row in smoke:
            results.append(run_case(r, row, 'C1', 's0/replay'))
        for row in smoke[:2]:
            results.append(run_case(r, row, 'C0', 's0/replay'))
            results.append(run_case(r, row, 'C2', 's0/replay'))
            for arm in ARMS[2:]:
                results.append(run_case(r, row, arm, 's0/zero', disabled=True))
        probes = [run_case(r, row, 'C5', 's0/feature_probe') for row in smoke[:2]]
        trace = [step for value in results[:8] for step in value['steps'] if step['index'] in [0,15]]
        feature_nonzero = all(step['feature_difference_rms'] > 1e-7 for step in trace)
        attention_nonzero = all(step['q_difference_rms'] > 1e-7 and step['k_difference_rms'] > 1e-7 for step in trace)
        assert feature_nonzero, '下游 Feature 未证实存在差异；禁止进入科学比较'
        for value in probes:
            old = read(historical('C1', value['generation']['id']).with_suffix('.json'))
            assert value['generation']['final_sha256'] != old['final_sha256'], '非零 Feature 无终图 latent 响应'
        # 额外同状态无条件核验：teacher/anchors 不改变原无条件算子。
        # 零强度完整轨迹已逐 ε 严格核验，非零轨迹只对同输入作比较。
        assert before == frozen_processor(r)
        write(OUT/'s0/audit.json', dict(pass_s0=True, feature_nonzero=feature_nonzero,
            attention_applicable=attention_nonzero, actual_layer_order=['b1','feature','attention'],
            global_exact_n=8, e5_exact_n=2, interior_exact_n=2, zero_full_ddim_cases=12,
            frozen_b1_before=before, frozen_b1_after=frozen_processor(r),
            attention_status='APPLICABLE' if attention_nonzero else 'NOT_APPLICABLE',
            trace=trace, feature_probe_seconds=[v['generation']['seconds'] for v in probes],
            commit=commit(), protocol_sha256=sha(OUT/'protocol.json')))
    finally:
        r.close()


def generate(arm, stage):
    rows, frozen = protocol()
    audit_result = read(OUT/'s0/audit.json')
    assert audit_result['pass_s0']
    if arm in ['C6','C7']:
        assert audit_result['attention_applicable'], 'Q/K 锚点不适用'
    if stage == 's1':
        rows = [next(r for r in rows if r['id']==x) for x in frozen['smoke_ids']]
    else:
        assert read(OUT/'s1/smoke_complete.json')['pass_s1']
    r = runtime(stage+'/'+arm)
    before = frozen_processor(r)
    try:
        records = [run_case(r, row, arm, stage) for row in rows]
        assert before == frozen_processor(r)
        assert sha(CKPT) == CKPT_SHA
    finally:
        r.close()
    write(OUT/stage/(arm+'_complete.json'), dict(arm=arm, n=len(records), ids=[r['generation']['id'] for r in records],
        protocol_sha256=sha(OUT/'protocol.json'), frozen_b1=before, commit=commit()))


def smoke():
    _, frozen = protocol()
    arms = ARMS if read(OUT/'s0/audit.json')['attention_applicable'] else ARMS[:6]
    for arm in arms:
        generate(arm, 's1')
    for identity in frozen['smoke_ids']:
        b1 = read(image_path('C1',identity,'s1').with_suffix('.json'))
        for arm in arms[3:]:
            rec = read(image_path(arm,identity,'s1').with_suffix('.json'))
            assert rec['final_sha256'] != b1['final_sha256'], (identity,arm,'有效 mask 非零干预无响应')
    write(OUT/'s1/smoke_complete.json', dict(pass_s1=True, arms=arms, n=8,
        protocol_sha256=sha(OUT/'protocol.json'), no_rgb_metric_tuning=True))


def finish(jobs):
    protocol()
    assert read(OUT/'s2/evaluation_complete.json')['complete']
    accounting = subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
    (OUT/'slurm_accounting.txt').write_text(accounting,encoding='utf-8')
    keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
    rows=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    assert set(jobs) == {r['job_id'] for r in rows}
    assert all(r['state'] not in ['RUNNING','PENDING','COMPLETING'] for r in rows)
    gpu_seconds=0
    for row in rows:
        count=re.search(r'gres/gpu=(\d+)',row['resources'])
        gpu_seconds += (int(count.group(1)) if count else 0)*int(row['seconds'])
    write(OUT/'resource_ledger.json',dict(jobs=rows,gpu_hours=gpu_seconds/3600,
        failed_jobs_retained=True,training_updates=0))
    for filename in ['input_hashes.json','model_file_hashes.json']:
        assert all(sha(p)==h for p,h in read(OUT/'manifests'/filename).items())
    code=list(Path('tools').glob('e45*.py'))+[Path('models/e45_pnp_anchor.py'),Path('submit/e45_gpu.sh')]
    include={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.suffix in ['.json','.csv','.png','.txt'] and p.name!='run_manifest.json'}
    large={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.pt'))}
    write(OUT/'run_manifest.json',dict(commit=commit(),protocol_sha256=sha(OUT/'protocol.json'),
        code={str(p):sha(p) for p in code},files=include,server_retained_tensor_files=large,
        decision=read(OUT/'s2/decision.json')))
    archive=OUT/'e45_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',archive,'SHA256',sha(archive),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','smoke','generate','batch','evaluate','finish'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--arms',choices=ARMS,nargs='+')
    parser.add_argument('--stage',choices=['s1','s2'],default='s2')
    parser.add_argument('--jobs',nargs='+')
    args=parser.parse_args()
    if args.action=='prepare':protocol()
    elif args.action=='audit':audit()
    elif args.action=='smoke':smoke()
    elif args.action=='generate':generate(args.arm,args.stage)
    elif args.action=='batch':
        for arm in args.arms:generate(arm,args.stage)
    elif args.action=='evaluate':
        from tools.e45_eval import evaluate
        evaluate()
    else:finish(args.jobs)
