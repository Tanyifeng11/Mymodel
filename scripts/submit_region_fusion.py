"""分阶段提交并保存全部作业编号；失败重试须显式传 --retry。"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.region_fusion_experiment import ROOT, ARMS, read, write


def submit(stage, arm=None, gpus=0, dependency=None, retry=False):
    path = ROOT / 'submitted_jobs.json'
    jobs = read(path) if path.exists() else {}
    key = stage + ('_' + arm if arm else '')
    previous = jobs.get(key, [])
    if previous and not retry:
        print('EXISTING', key, previous[-1], flush=True)
        return previous[-1]
    partition = 'gpu' if gpus else 'cpu'
    details = subprocess.check_output(['scontrol', 'show', 'partition', partition, '-o'], text=True)
    maximum = re.search(r'\bMaxTime=(\S+)', details).group(1)
    # normal QoS 的已知上限为七天；分区更短时使用其实际值。
    limit = ('7-00:00:00' if maximum == 'UNLIMITED'
             or ('-' in maximum and int(maximum.split('-')[0]) >= 7) else maximum)
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    args = ['sbatch', '--parsable', '-p', partition, '-J', 'raf_' + key, '-N1', '-n1',
            '-c', str(max(4, gpus * 4)), '--mem', '96G' if gpus else '32G', '-t', limit,
            '-o', str(ROOT / 'logs' / (key + '_%j.log'))]
    if gpus:
        args += ['--gres=gpu:' + str(gpus)]
    if dependency:
        args += ['--dependency=afterok:' + str(dependency)]
    args += [str(ROOT / 'source/scripts/region_fusion.sbatch'), stage]
    if stage == 'prepare':
        args += ['--gpus-per-arm', str(options.gpus_per_arm)]
    if arm:
        args += [arm]
    if stage == 'train':
        args += [str(gpus)]
    job = subprocess.check_output(args, text=True).strip()
    assert job.isdigit(), job
    jobs[key] = previous + [job]
    write(path, jobs)
    reservations = ROOT / 'job_resources.json'
    resources = read(reservations) if reservations.exists() else {}
    resources[job] = dict(stage=stage, arm=arm, gpus=gpus, partition=partition,
                           cpus=max(4, gpus * 4), time_limit=limit)
    write(reservations, resources)
    print('SUBMITTED', key, job, flush=True)
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initial', 'reuse', 'train', 'generate', 'evaluate', 'package'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--gpus-per-arm', type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument('--retry', action='store_true')
    options = parser.parse_args()
    if options.stage == 'initial':
        preparation = submit('prepare', retry=options.retry)
        submit('preflight', gpus=1, dependency=preparation, retry=options.retry)
    elif options.stage == 'reuse':
        submit('reuse', retry=options.retry)
    elif options.stage == 'train':
        assert read(ROOT / 'audit/preflight_complete.json')['passed']
        gpus = read(ROOT / 'protocol.json')['gpus_per_arm']
        selected = [options.arm] if options.arm else ['concat', 'region']
        for arm in selected:
            if arm == 'bf':
                raise ValueError('原 BF-SD-T3 基线已训练，必须使用 reuse，禁止重复训练')
            submit('train', arm=arm, gpus=gpus, retry=options.retry)
    elif options.stage == 'generate':
        assert options.arm, '生成时指定已完成训练的 --arm'
        complete = read(ROOT / 'training' / options.arm / 'complete.json')
        assert complete['updates'] == 28365
        if complete.get('outputs_reused'):
            print('REUSED_OUTPUTS', options.arm, flush=True)
            sys.exit(0)
        submit('generate', arm=options.arm, gpus=1, retry=options.retry)
    elif options.stage == 'evaluate':
        submit('evaluate', gpus=1, retry=options.retry)
    else:
        assert read(ROOT / 'evaluation_complete.json')['complete']
        submit('package', retry=options.retry)
