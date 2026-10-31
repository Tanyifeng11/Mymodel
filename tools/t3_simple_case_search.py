"""固定清晰输入的定性案例搜索。保留全部seed；复用现有推理，不训练。"""
import argparse
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from tools.t3_p0p1 import PROJECT, SOURCE, read, write, sha

ROOT = PROJECT/'output_eval/t3_simple_case_search_20261031'
PREVIOUS = PROJECT/'output_eval/t3_p0_p1_20261028'
SPEC = SOURCE/'configs/t3_simple_cases_20261031.json'
SCRIPT = SOURCE/'scripts/t3_simple_case_search.sbatch'


def ledger():
    return read(ROOT/'audit/jobs.json') if (ROOT/'audit/jobs.json').exists() else []


def resources():
    jobs = ledger()
    if not jobs:
        return dict(actual_gpu_seconds=0, maximum_accounted_gpu_seconds=0, jobs=[])
    command = ['sacct', '-n', '-P', '-X', '-j', ','.join(str(x['job']) for x in jobs),
               '--format=JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End,NodeList']
    output = subprocess.check_output(command, text=True)
    records = {r[0]: r for line in output.splitlines() if len(r := line.split('|')) >= 7}
    total = reserved = 0
    states = []
    for job in jobs:
        row = records.get(str(job['job']))
        state = row[1] if row else 'ACCOUNTING_PENDING'
        elapsed = int(row[2]) if row else 0
        match = re.search(r'(?:^|,)gres/gpu=(\d+)', row[3]) if row else None
        allocated = int(match.group(1)) if match else 0
        actual = elapsed*allocated
        final = any(state.startswith(s) for s in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'])
        maximum = actual if final else job['gpus']*job['limit_seconds']
        total += actual
        reserved += maximum
        states.append(dict(job, state=state, elapsed_seconds=elapsed, allocated_gpus=allocated,
                           gpu_seconds=actual, maximum_accounted_gpu_seconds=maximum))
    write(ROOT/'audit/resources.json', dict(actual_gpu_seconds=total, gpu_hours=total/3600,
        maximum_accounted_gpu_seconds=reserved, hard_cap_seconds=5400, jobs=states))
    (ROOT/'audit/resources_sacct.psv').write_text(output, encoding='utf-8')
    return dict(actual_gpu_seconds=total, maximum_accounted_gpu_seconds=reserved, jobs=states)


def submit(role, action, minutes, gpus=0, dependency=None, extra=()):
    old = [x for x in ledger() if x['role'] == role]
    if old:
        print('ALREADY_SUBMITTED', old, flush=True)
        return old[0]['job']
    accounted = resources()['maximum_accounted_gpu_seconds']
    assert accounted+gpus*minutes*60 <= 5400, (accounted, role, minutes, gpus)
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    args = ['sbatch', '--parsable', '-N1', '--ntasks=1', '--partition='+('gpu' if gpus else 'cpu'),
            '--cpus-per-task='+('4' if gpus else '2'), '--mem='+('32G' if gpus else '8G'),
            '--time=00:%02d:00'%minutes, '--job-name=simple_'+role,
            '--output='+str(ROOT/'logs'/(role+'_%j.out'))]
    if gpus:
        args += ['--gpus='+str(gpus)]
    if dependency:
        args += ['--dependency=afterok:'+':'.join(str(j) for j in dependency)]
    result = subprocess.check_output(args+[str(SCRIPT), action]+list(extra), text=True).strip()
    job = int(result.split(';')[0])
    jobs = ledger()
    jobs.append(dict(job=job, role=role, action=action, gpus=gpus, limit_seconds=minutes*60,
                     submitted_at=time.strftime('%Y-%m-%d %H:%M:%S'), command=args+[str(SCRIPT), action]+list(extra)))
    write(ROOT/'audit/jobs.json', jobs)
    print('SUBMITTED', role, job, flush=True)
    return job


def submit_core():
    prepare_job = submit('prepare', 'prepare', 10)
    bf_job = submit('core_bf', 'generate', 25, 1, [prepare_job], ['BF-Fashion'])
    internal_job = submit('core_internal', 'generate', 25, 1, [prepare_job], ['T3', 'T1', 'T2'])
    package_job = submit('package_core', 'package', 10, 0, [bf_job, internal_job], ['--name', 'core_review'])
    print('CORE_CHAIN', prepare_job, bf_job, internal_job, package_job, flush=True)


def prepare():
    spec = read(SPEC)
    assert len(spec['cases']) == 16 and spec['seeds'] == [42, 43, 44]
    for directory in ['protocol', 'inputs', 'outputs', 'native', 'generation', 'audit', 'code', 'metrics']:
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    spec_path = ROOT/'protocol/frozen_spec.json'
    if spec_path.exists():
        assert read(spec_path) == spec
    write(spec_path, spec)
    for relative in ['audit/server_inventory.json', 'audit/BF_provenance.json', 'protocol/cnipa_choice.json']:
        target = ROOT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PREVIOUS/relative, target)
    # 大权重在服务器就地核验；不上传本地替代文件。
    bf = read(ROOT/'audit/BF_provenance.json')
    assert sha(bf['checkpoint_path']) == bf['checkpoint_sha256']
    rows = []
    for case in spec['cases']:
        copied = {}
        for name in ['sketch', 'reference']:
            origin = Path(case[name+'_source'])
            assert sha(origin) == case[name+'_sha256'], origin
            target = ROOT/'inputs'/(case[name+'_sha256']+origin.suffix)
            if not target.exists():
                shutil.copy2(origin, target)
            copied[name] = str(target)
        for seed in spec['seeds']:
            rows.append(dict(case, **copied, id=case['case_id']+'_seed'+str(seed), condition='custom', seed=seed,
                             GT_used_in_inference=False, ground_truth_available=False))
    write(ROOT/'protocol/input_manifest.json', rows)
    paths = ['tools/t3_simple_case_search.py', 'tools/t3_p0p1_generate.py', 'tools/t3_p0p1.py',
             'inference_IMAGGarment-1.py', 'scripts/t3_simple_case_search.sbatch', 'configs/t3_simple_cases_20261031.json']
    code = []
    for relative in paths:
        target = ROOT/'code'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/relative, target)
        code.append(dict(path=relative, sha256=sha(target)))
    write(ROOT/'audit/prepared.json', dict(n_cases=16, n_seed_conditions=len(rows), spec_sha256=sha(SPEC),
        code_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        code=code, training_updates=0, GT_used=False, BF_checkpoint_sha256=bf['checkpoint_sha256']))
    print('PREPARED', len(rows), sha(spec_path), flush=True)


def generate(arms, cases=None):
    import torch
    from tools import t3_p0p1_generate as legacy
    legacy.ROOT = ROOT
    legacy.SOURCE = SOURCE
    spec = read(ROOT/'protocol/frozen_spec.json')
    all_rows = read(ROOT/'protocol/input_manifest.json')
    rows = [r for r in all_rows if cases is None or r['case_id'] in cases]
    if cases:
        assert len(cases) <= spec['maximum_supplementary_cases'] and len(rows) == len(cases)*3
    assert len(rows) in [18, 48] or cases is not None
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    # 记录修复后真正运行的入口；prepare阶段的初版快照继续保留。
    snapshot = ROOT/'code/runtime'/commit/'t3_simple_case_search.py'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    for arm in arms:
        assert arm in spec['core_models']+spec['supplementary_models']
        audit = dict(arm=arm, rows=len(rows), cases=cases, code_commit=commit, torch=torch.__version__,
                     python=sys.version, gpu=torch.cuda.get_device_name(), runner_sha256=sha(__file__))
        write(ROOT/'audit'/('generation_'+arm+'_'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), audit)
        # BF的历史隔离环境使用旧PyTorch，与原生成程序保持相同no_grad上下文。
        with torch.no_grad():
            runner = legacy.bf() if arm == 'BF-Fashion' else legacy.cnipa() if arm == 'ControlNet+IP-Adapter' else legacy.internal(arm)
            for index, saved_row in enumerate(rows):
                row = dict(saved_row)
                # 新seed在GPU上显式生成SD噪声，各内部模型用相同张量；不是历史latent复用声明。
                latent = torch.randn((1,4,64,48), generator=torch.Generator(device='cuda').manual_seed(row['seed']), device='cuda', dtype=torch.float16)
                row['historical_initial_latent_sha256'] = legacy.digest(latent)
                path = ROOT/'outputs'/arm/'custom'/(row['id']+'.png')
                if path.exists():
                    meta = read(path.with_suffix('.json'))
                    assert meta['output_sha256'] == sha(path) and meta['input_reference_sha256'] == row['reference_sha256']
                    assert meta['input_sketch_sha256'] == row['sketch_sha256'] and meta['seed'] == row['seed']
                    continue
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                image, meta = runner(row)
                torch.cuda.synchronize()
                assert image.size == (384,512)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                meta.update(id=row['id'], case_id=row['case_id'], arm=arm, condition='custom', seed=row['seed'],
                    prompt=row['caption'] if meta['text_used'] else None, input_caption=row['caption'], GT_used_in_inference=False,
                    input_sketch_sha256=row['sketch_sha256'], input_reference_sha256=row['reference_sha256'],
                    output_sha256=sha(path), seconds=time.perf_counter()-start, peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    code_commit=commit, frozen_spec_sha256=sha(ROOT/'protocol/frozen_spec.json'))
                write(path.with_suffix('.json'), meta)
                print('OUTPUT', arm, row['id'], index+1, '/', len(rows), flush=True)
            del runner
        gc.collect()
        torch.cuda.empty_cache()
        write(ROOT/'generation'/(arm+'_complete.json'), dict(arm=arm, n=len(rows), cases=cases, complete=True))


def measure():
    from eval.eval_utils import prepare_evaluation_masks, json_safe
    from eval.metrics import compute_structure_preservation, compute_texture_leakage, compute_texture_color_fidelity
    by_id = {r['id']: r for r in read(ROOT/'protocol/input_manifest.json')}
    result = []
    for path in sorted((ROOT/'outputs').glob('*/custom/*.png')):
        meta = read(path.with_suffix('.json'))
        assert meta['output_sha256'] == sha(path)
        row = by_id[path.stem]
        bundle = prepare_evaluation_masks((384,512), sketch_path=row['sketch'], mask_policy='sketch_only')
        kwargs = dict(gen_path=str(path), sketch_path=row['sketch'], mask_bundle=bundle)
        structure = compute_structure_preservation(**kwargs)
        leak = compute_texture_leakage(**kwargs)
        color = compute_texture_color_fidelity(texture_path=row['reference'], **kwargs)
        result.append(json_safe(dict(case_id=row['case_id'], id=row['id'], seed=row['seed'], arm=meta['arm'],
            output_sha256=sha(path), iou=structure.get('sketch_iou'), edge_f1=structure.get('edge_f1'),
            leak=leak.get('leak_colored_frac'), color_lab=color.get('tcf_lab_delta'),
            scope='结构/颜色代理，不是纹元或人类偏好评分；无GT，不计算LPIPS/FID')))
    write(ROOT/'metrics/descriptive.json', result)
    print('CPU_DESCRIPTIVE_METRICS', len(result), flush=True)


def package(name):
    measure()
    resources()
    files = []
    for folder in ['protocol', 'inputs', 'outputs', 'native', 'generation', 'audit', 'code', 'metrics', 'logs']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name != 'bundle_sha256sums.txt'
                  and not p.name.startswith('package_')]
    files = sorted(set(files))
    sums = ROOT/'audit/bundle_sha256sums.txt'
    sums.write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files), encoding='utf-8')
    target = ROOT/(name+'.tar.gz')
    with tarfile.open(target, 'w:gz') as tar:
        for path in files+[sums]:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print('BUNDLE', target, 'BYTES', target.stat().st_size, 'SHA256', sha(target), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['submit-core', 'prepare', 'generate', 'package', 'resources', 'submit-supplement'])
    p.add_argument('arms', nargs='*')
    p.add_argument('--cases', nargs='+')
    p.add_argument('--name', default='review_results')
    args = p.parse_args()
    if args.action == 'submit-core':
        submit_core()
    elif args.action == 'prepare':
        prepare()
    elif args.action == 'generate':
        generate(args.arms, args.cases)
    elif args.action == 'package':
        package(args.name)
    elif args.action == 'resources':
        print(json.dumps(resources()))
    else:
        assert args.cases and len(args.cases) <= 6
        job = submit('supplement', 'generate', 15, 1, extra=['E0', 'E1', 'B', 'C', 'ControlNet+IP-Adapter', '--cases']+args.cases)
        write(ROOT/'protocol/selected_for_supplement.json', dict(cases=args.cases, selection='结果后筛选；三个seed全部补齐；非无偏总体展示'))
        submit('package_final', 'package', 10, 0, [job])


if __name__ == '__main__':
    main()
