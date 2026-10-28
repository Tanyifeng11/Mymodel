"""T3 P0/P1：只读审计已有资产，新增结果写入独立目录。"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

PROJECT = Path(os.environ.get('T3PP_PROJECT', '/share/home/u2515283058/Mymodel'))
ROOT = PROJECT / 'output_eval/t3_p0_p1_20261028'
SUP = PROJECT / 'output_eval/e2a_sup_comp_01_20261021'
ATTR = PROJECT / 'output_eval/e2a_attr_01_20261019'
CONF = PROJECT / 'output_eval/t3_reference_response_20261027'
SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(argv=args, returncode=result.returncode, output=result.stdout)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_info(path, hash_file=True):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), exists=False)
    return dict(path=str(path), exists=True, size=path.stat().st_size,
                sha256=sha(path) if hash_file else None)


def audit():
    ROOT.mkdir(parents=True, exist_ok=True)
    result = dict(protocol='T3-P0-P1-20261028', time=time.time(), python=platform.python_version(),
                  code=command(['git', 'rev-parse', 'HEAD'], SOURCE), new_training=0,
                  human_reviewers=0, human_blind_scores=None)
    result['resources'] = {name: command(args) for name, args in {
        'partitions': ['sinfo', '-o', '%P %a %l %D %t %G'],
        'gpu_nodes': ['scontrol', 'show', 'nodes'],
        'user_queue': ['squeue', '-u', 'u2515283058', '-o', '%.18i %.12P %.28j %.8T %.10M %R'],
    }.items()}
    models = {
        'E0': PROJECT/'output/phase1_e0_baseline_e5/checkpoint-28365/joint_model.pt',
        'E1': PROJECT/'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt',
        'B': ATTR/'training/seed42/F000/checkpoint-final/joint_model.pt',
        'C': ATTR/'training/seed42/F111/checkpoint-final/joint_model.pt',
        **{arm: SUP/'training/formal'/arm/'checkpoint-final/joint_model.pt' for arm in ['T1', 'T2', 'T3']},
    }
    if not models['E0'].is_file():
        models['E0'] = PROJECT/'output/phase1_e0_baseline_e5/checkpoint-final/joint_model.pt'
    result['models'] = {name: file_info(path) for name, path in models.items()}
    for name, info in result['models'].items():
        print('MODEL', name, info, flush=True)
    result['sources'] = {}
    for name, base in [('internal', SUP), ('confirmation', CONF),
                       ('ablation', CONF/'exploratory_extension_20261027')]:
        paths = [p for p in base.rglob('*.json') if
                 p.name in ['config.json', 'model_registry.json', 'checkpoint_registry.json',
                            'checkpoint_verification.json', 'completion.json', 'decision.json']
                 and 'source' not in p.relative_to(base).parts]
        result['sources'][name] = [file_info(p) for p in sorted(paths)]
    bf = PROJECT.parent/'BF-Fashion-'
    control = PROJECT.parent/'cn_ipa_repro_20261022'
    for name, repo in [('BF-Fashion', bf), ('control_ip', control)]:
        entry = dict(path=str(repo), exists=repo.is_dir())
        if not repo.is_dir():
            result[name] = entry
            continue
        gitroot = repo if (repo/'.git').exists() else repo/'source'
        entry.update(git=command(['git', 'rev-parse', 'HEAD'], gitroot) if gitroot.exists() else None)
        candidates = list(repo.glob('logs/*/checkpoints/last.ckpt')) if name == 'BF-Fashion' else []
        entry['checkpoints'] = [file_info(path) for path in sorted(candidates)]
        entry['training_configs'] = [str(p) for p in sorted(repo.glob('logs/*/configs/*project.yaml'))]
        entry['output_directories'] = [str(p) for p in repo.iterdir() if p.is_dir() and p.name in
                                       ['output', 'output_eval', 'eval_output', 'cn_ipa']]
        if name == 'control_ip':
            entry['selection_files'] = [dict(file_info(p), content=read(p)) for p in
                sorted(repo.glob('selection/*checkpoint.json'))]
            entry['frozen_final_choice'] = read(repo/'selection/frozen_final_choice.json')
            entry['candidate_roots'] = [str(p) for p in repo.iterdir() if p.is_dir()]
        result[name] = entry
    result['conda_environments'] = command(['bash', '-lc', 'ls -1 /share/home/u2515283058/.conda/envs'])
    write(ROOT/'audit/server_inventory.json', result)
    print('AUDIT_COMPLETE', ROOT/'audit/server_inventory.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit'])
    args = parser.parse_args()
    audit()
