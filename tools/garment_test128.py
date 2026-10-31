"""新增128件纯服装、11模型、仅原参考；沿用32件实验的生成与评价代码。"""
import argparse
import getpass
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from tools import garment_test32 as shared
from tools.t3_p0p1 import PROJECT, SOURCE, read, write, sha

ROOT = PROJECT/'output_eval/garment_test128_20261031'
SPEC = SOURCE/'configs/garment_test128_20261031.json'
SCRIPT = SOURCE/'scripts/garment_test128.sbatch'
MODELS = shared.MODELS
CAP = 14400
shared.ROOT, shared.SPEC, shared.SCRIPT, shared.CAP = ROOT, SPEC, SCRIPT, CAP

# 时间上限总和13800 GPU秒，另留600秒；终态按实际分配计费，失败也计入。
PLAN = [('internal_a', 52, ['E0', 'E1', 'B', 'T1']),
        ('internal_b', 38, ['T2', 'C', 'T3']),
        ('bf', 34, ['BF-Fashion']), ('cnipa', 10, ['ControlNet+IP-Adapter']),
        ('sgdiff', 62, ['SGDiff']), ('lots', 18, ['LOTS'])]


def freeze(path, value):
    if path.exists():
        assert read(path) == value, '已冻结文件不可静默覆盖: '+str(path)
    else:
        write(path, value)


def prepare():
    spec = read(SPEC)
    assert spec['original_n'] == 128 and spec['swapped_n'] == 0 and len(spec['cases']) == 128
    for folder in ['audit', 'protocol', 'inputs', 'outputs', 'native', 'code', 'metrics', 'generation']:
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    data = PROJECT.parent/'datasets/BF'
    training = {p.stem.split('_')[0] for folder in ['cloth', 'gt']
                for p in (data/'training'/folder).glob('*') if p.is_file()}
    assert training, '训练编号目录不存在，不可跳过重叠检查'
    bf_asset = read(shared.OLD/'audit/BF_provenance.json')
    train_manifest = Path(bf_asset['training_manifest'])
    assert sha(train_manifest) == bf_asset['training_manifest_sha256']
    bf_training = {Path(r['cloth']).stem.split('_')[0] for r in read(train_manifest)}
    training |= bf_training
    assert sha(bf_asset['checkpoint_path']) == bf_asset['checkpoint_sha256']
    assert sha(bf_asset['vq_path']) == bf_asset['vq_sha256']
    previous = set()
    previous_audit = []
    for source in spec['prior_manifests']:
        path = PROJECT/'output_eval'/source['server_relative']
        assert sha(path) == source['sha256'], str(path)
        previous.update(r['id'].split('_')[0] for r in read(path))
        previous_audit.append(dict(path=str(path), sha256=sha(path)))
    rows, differences, products = [], [], set()
    for cat in spec['categories']:
        cases = [r for r in spec['cases'] if r['category'] == cat]
        assert len(cases) == 32
        for case in cases:
            ident = case['id']
            product = ident.split('_')[0]
            assert product not in training | previous | products, ident
            products.add(product)
            row = dict(id=ident, category=cat, condition='original',
                seed=int(hashlib.sha256(('garment_test128_seed/'+ident).encode()).hexdigest()[:8], 16) % (2**31),
                caption=(data/'test'/cat/'text'/(ident+'.txt')).read_text(encoding='utf-8-sig').strip(),
                GT_used_in_inference=False)
            assert row['caption'], ident
            for key, folder in [('sketch', 'sketch'), ('reference', 'texture'), ('gt', 'gt')]:
                source = data/'test'/cat/folder/(ident+'.jpg')
                digest = sha(source)
                if digest != case[folder+'_sha256']:
                    differences.append(dict(id=ident, category=cat, input=key,
                        server_sha256=digest, local_sha256=case[folder+'_sha256']))
                target = ROOT/'inputs'/(digest+'.jpg')
                if not target.exists():
                    shutil.copy2(source, target)
                row.update({key: str(target), key+'_sha256': digest, key+'_source': str(source)})
            rows.append(row)
    protocol = dict(spec, models=MODELS, n_outputs=1408,
        split='BF/test; 32 each top/outwear/pants/dress; 128 new product roots',
        seed_policy='one fixed identity seed; eight compatible SD arms share explicit initial latent; other architectures keep their native noise shape',
        exposure_limit='current training and previous96/previous32 product roots excluded; historical training/pretraining exposure unknown; SGDiff has prior full-test inference',
        evaluation='same six original-reference metrics as garment_test32; no swapped-reference experiment in this round',
        comparison_limit='modalities, native resolutions, training histories and inference steps differ; LOTS has no reference input',
        display='8-12 post-result selected T3 examples; avoid stripe/polka inputs in highlights; retain all 128 in metrics and full gallery',
        input_version='server BF/test bytes authoritative; input-only review release required before any GPU submission')
    freeze(ROOT/'protocol/input_manifest.json', rows)
    freeze(ROOT/'protocol/controlled_protocol.json', protocol)
    for relative in ['audit/server_inventory.json', 'audit/BF_provenance.json', 'protocol/cnipa_choice.json']:
        shutil.copy2(shared.OLD/relative, ROOT/relative)
    bf_asset.update(training_overlap_ids=[], current_test_product_root_overlap=[],
        current_test_manifest_sha256=sha(ROOT/'protocol/input_manifest.json'))
    write(ROOT/'audit/BF_provenance.json', bf_asset)
    asset = read(shared.SG_OLD/'audit/assets.json')
    asset.update(runner_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        current_test_manifest_sha256=sha(ROOT/'protocol/input_manifest.json'))
    write(ROOT/'audit/assets.json', asset)
    shutil.copy2(shared.SG_OLD/'protocol/resolved_config.py', ROOT/'protocol/resolved_config.py')
    shutil.copy2(PROJECT.parent/'LOTS/assets.json', ROOT/'audit/LOTS_assets.json')
    for relative in ['tools/garment_test128.py', 'scripts/garment_test128.sbatch', 'configs/garment_test128_20261031.json',
                     'tools/garment_test32.py', 'tools/t3_p0p1_generate.py', 'tools/t3_p0p1_evaluate.py',
                     'tools/t3_sgdiff_external.py']:
        target = ROOT/'code'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/relative, target)
    write(ROOT/'audit/local_input_differences.json', differences)
    write(ROOT/'audit/prepared.json', dict(original_n=128, swapped_n=0, n_outputs=1408,
        training_product_roots=len(training), product_root_training_overlap=[], previous_product_overlap=[],
        previous_manifests=previous_audit, code_commit=asset['runner_commit'],
        input_visual_review='actual server input review required before GPU release',
        manifest_sha256=sha(ROOT/'protocol/input_manifest.json')))
    print('PREPARED_128', sha(ROOT/'protocol/input_manifest.json'), flush=True)


def release_inputs():
    write(ROOT/'protocol/input_review_release.json', dict(actual_server_input_review=True,
        reviewer='AI input-only category/pair check of all 128', human_reviewers=0,
        manifest_sha256=sha(ROOT/'protocol/input_manifest.json'), generation_results_seen=False))
    print('INPUTS_RELEASED_128', flush=True)


def submit(role, action, minutes, gpus=0, extra=()):
    existing = [j for j in shared.jobs() if j['role'] == role]
    if existing:
        return existing[0]['job']
    assert shared.resources()['maximum_accounted_gpu_seconds']+minutes*60*gpus <= CAP
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    controller = role == 'controller'
    command = ['sbatch', '--parsable', '-N1', '--ntasks=1', '--partition='+('gpu' if gpus else 'cpu'),
        '--cpus-per-task='+('1' if controller else '4'), '--mem='+('2G' if controller else '48G'),
        '--time=%02d:%02d:00' % (minutes//60, minutes % 60), '--job-name=g128_'+role,
        '--output='+str(ROOT/'logs'/(role+'_%j.out'))]
    if gpus:
        command += ['--gpus='+str(gpus)]
    command += [str(SCRIPT), action]+list(extra)
    job = int(subprocess.check_output(command, text=True).strip().split(';')[0])
    ledger = shared.jobs()
    ledger.append(dict(job=job, role=role, action=action, gpus=gpus, limit_seconds=minutes*60,
        submitted_at=time.strftime('%Y-%m-%d %H:%M:%S'), command=command))
    write(ROOT/'audit/jobs.json', ledger)
    print('SUBMITTED', role, job, flush=True)
    return job


def verify():
    rows = read(ROOT/'protocol/input_manifest.json')
    assert len(rows) == 128 and all(r['condition'] == 'original' for r in rows)
    sd_hashes = {}
    for arm in MODELS:
        for row in rows:
            path = ROOT/'outputs'/arm/'original'/(row['id']+'.png')
            meta = read(path.with_suffix('.json'))
            assert sha(path) == meta['output_sha256'] and meta['seed'] == row['seed']
            assert meta['input_sketch_sha256'] == row['sketch_sha256'] and not meta['GT_used_in_inference']
            assert meta['input_reference_sha256'] == (None if arm == 'LOTS' else row['reference_sha256'])
            assert meta['sketch_used'] == (arm != 'SGDiff')
            assert meta['text_used'] == (arm != 'BF-Fashion')
            assert meta['reference_texture_used'] == (arm != 'LOTS')
            if arm in ['E0', 'E1', 'B', 'T1', 'T2', 'C', 'T3', 'ControlNet+IP-Adapter']:
                sd_hashes.setdefault(row['id'], meta['initial_latent_sha256'])
                assert sd_hashes[row['id']] == meta['initial_latent_sha256']
    assert len(list((ROOT/'outputs').glob('*/*/*.png'))) == 1408
    write(ROOT/'audit/input_noise_checks.json', dict(passed=True, models=11, n_outputs=1408,
        input_sha256s=True, paired_identity_seeds=True, shared_SD_architecture_latents=True, GT_used=False))
    print('VERIFIED_1408', flush=True)


def evaluate():
    verify()
    from tools import t3_p0p1_evaluate as legacy
    legacy.ROOT = ROOT
    legacy.output_path = lambda arm, row: ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
    for arm in MODELS:
        legacy.evaluate(arm, include_swap=True)


def package():
    shared.resources()
    verify()
    for arm in MODELS:
        completed = read(ROOT/'metrics'/arm/'complete.json')
        assert completed['complete'] and completed['missing_generation'] == 0 and completed['n'] == 128
    # 控制作业仍在输出，原始实时日志保留服务器；只打包其他已结束的日志。
    files = sorted({p for folder in ['audit', 'protocol', 'inputs', 'outputs', 'native', 'base64', 'code', 'metrics', 'generation', 'logs']
        for p in (ROOT/folder).rglob('*') if p.is_file() and p.name != 'bundle_sha256sums.txt'
        and not p.name.startswith(('package_', 'controller_'))})
    sums = ROOT/'audit/bundle_sha256sums.txt'
    sums.write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files), encoding='utf-8')
    target = ROOT/'review_results.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for p in files+[sums]:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
    print('BUNDLE_BYTES', target.stat().st_size, 'SHA256', sha(target), flush=True)


def advance():
    release = read(ROOT/'protocol/input_review_release.json')
    assert release['actual_server_input_review'] and release['manifest_sha256'] == sha(ROOT/'protocol/input_manifest.json')
    resource = shared.resources()
    workers = [r for r in resource['jobs'] if r['gpus']]
    failures = [r for r in workers if any(r['state'].startswith(s) for s in
        ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'])]
    if failures:
        write(ROOT/'audit/chain_status.json', dict(status='NEEDS_REVIEW', failures=failures))
        raise RuntimeError('出现失败，先核算预算并人工检查证据，不自动重提')
    completed = {r['role'] for r in workers if r['state'].startswith('COMPLETED')}
    if 'evaluate' in completed:
        write(ROOT/'audit/chain_status.json', dict(status='EVALUATION_COMPLETE', resources=resource))
        package()
        return True
    submitted = {r['role'] for r in resource['jobs']}
    # 账号QOS最多4活动作业；控制作业占1个名额，最多保留3个GPU作业。
    active = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%A'], text=True).splitlines()
    slots = max(0, 4-len(active))
    for role, minutes, arms in PLAN:
        if slots and role not in submitted:
            submit(role, 'generate', minutes, 1, arms)
            slots -= 1
    if all(role in completed for role, _, _ in PLAN) and 'evaluate' not in submitted and slots:
        submit('evaluate', 'evaluate', 16, 1)
    write(ROOT/'audit/chain_status.json', dict(status='ACTIVE', completed=sorted(completed), resources=resource))
    return False


def run_chain():
    while not advance():
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'package_inputs', 'release_inputs', 'submit-controller',
        'run_chain', 'advance', 'generate', 'verify', 'evaluate', 'package', 'resources'])
    parser.add_argument('arms', nargs='*')
    args = parser.parse_args()
    if args.action == 'generate':
        shared.generate(args.arms)
    elif args.action == 'resources':
        print(json.dumps(shared.resources()), flush=True)
    elif args.action == 'package_inputs':
        shared.package_inputs()
    elif args.action == 'submit-controller':
        release = read(ROOT/'protocol/input_review_release.json')
        assert release['actual_server_input_review'] and release['manifest_sha256'] == sha(ROOT/'protocol/input_manifest.json')
        submit('controller', 'run_chain', 1440)
    else:
        globals()[args.action]()
