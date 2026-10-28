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
        gitroot = repo if (repo/'.git').exists() else repo/'repo'
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


def prepare():
    """先冻结96原参考及32替换输入；不据新生成结果挑样或调参。"""
    import shutil
    from PIL import Image
    inventory = read(ROOT/'audit/server_inventory.json')
    rows = read(SUP/'splits/P2b.json')
    swaps = {row['id']: row for row in read(SUP/'splits/reference_swap_v2.json')}
    original = {row['id']: row for row in rows}
    assert len(rows) == 96 and len(swaps) == 32 and swaps.keys() <= original.keys()
    manifest = []
    for row in rows:
        meta = read(SUP/'evaluation/P2b/T3/images'/(row['id']+'.json'))
        variants = [('original', row['reference'])]
        if row['id'] in swaps:
            variants.append(('swapped', swaps[row['id']]['reference_alternative']))
        for condition, reference in variants:
            source_meta = meta if condition == 'original' else read(SUP/'evaluation/reference_swap/T3/images'/(row['id']+'.json'))
            item = dict(id=row['id'], condition=condition, caption=row['caption'],
                        category=row['stratum']['category'], seed=meta['seed'],
                        historical_initial_latent_sha256=meta['initial_latent_sha256'],
                        reference_alternative=swaps.get(row['id'],{}).get('reference_alternative'))
            for role, path in [('sketch', row['sketch']), ('reference', reference), ('gt', row['gt'])]:
                digest = sha(path)
                assert digest == source_meta['inputs'][role], ('INPUT_CHANGED',row['id'],condition,role)
                target = ROOT/'inputs'/(digest+Path(path).suffix)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    assert sha(target)==digest
                else:
                    shutil.copy2(path,target)
                item[role]=str(target)
                item[role+'_sha256']=digest
                item[role+'_rgb_sha256']=hashlib.sha256(Image.open(target).convert('RGB').tobytes()).hexdigest()
            manifest.append(item)
    protocol=dict(protocol='T3-P0-P1-20261028', new_training=0, internal_stage=['E0','E1'],
        external=['BF-Fashion','ControlNet+IP-Adapter'], original_n=96, swapped_n=32,
        split='previously viewed P2b development, not a fresh confirmation set',
        input_source_sha256=sha(SUP/'splits/P2b.json'),swap_source_sha256=sha(SUP/'splits/reference_swap_v2.json'),
        seed='historical identity-specific seed; paired explicit latent inside each compatible architecture',
        cross_architecture_latents='BF native32x32 float32 differs from SD64x48 float16; same seed numbers do not imply identical noise',
        external_hyperparameters='BF official native DDIM200 scales1.1/1.5; CNIPA uses prior frozen choice, no new tuning',
        smoke='first four input-hash-ranked identities, both conditions where applicable; inspect before full release',
        inference_GT=False, swapped_lpips_and_fid=False, human_reviewers=0,
        metric='original six legacy metrics; swapped DISTS/color Margin plus legacy structure/background proxies',
        DISTS_scope='derived patch-set response; not motif identity, orientation or scale accuracy')
    for relative, value in [('protocol/external_protocol.json',protocol),('protocol/input_manifest.json',manifest)]:
        path=ROOT/relative
        if path.exists():
            assert read(path)==value,('FROZEN_CHANGED',relative)
        else:
            write(path,value)
    csv_write(ROOT/'external_input_manifest.csv',manifest)
    print('PREPARED',len(manifest),flush=True)


def external_audit():
    import torch
    from omegaconf import OmegaConf
    inventory=read(ROOT/'audit/server_inventory.json')
    bf=PROJECT.parent/'BF-Fashion-'
    path=bf/'logs/2026-07-01T22-38-33_mydata_ldm-vq-f8/checkpoints/last.ckpt'
    assert path.is_file()
    saved=list((path.parent.parent/'configs').glob('*project.yaml'))
    assert len(saved)==1, saved
    config=OmegaConf.load(saved[0])
    metadata=OmegaConf.select(config,'data.params.train.params.metadata_path')
    assert metadata is not None, 'saved training metadata path is required'
    trainpath=Path(metadata)
    if not trainpath.is_absolute():
        trainpath=bf/trainpath
    train=read(trainpath)
    ids={row['id'] for row in read(ROOT/'protocol/input_manifest.json')}
    trainids={Path(row['cloth']).stem for row in train}
    overlap=sorted(ids & trainids)
    state=torch.load(path,map_location='cpu')
    vq=bf/'models/first_stage_models/vq-f8/model.ckpt'
    info=dict(checkpoint_path=str(path),checkpoint_sha256=sha(path),epoch=state.get('epoch'),global_step=state.get('global_step'),
        vq_path=str(vq),vq_sha256=sha(vq),training_config_path=str(saved[0]),training_config_sha256=sha(saved[0]),
        training_manifest=str(trainpath),training_manifest_sha256=sha(trainpath),training_samples=len(train),
        training_overlap_ids=overlap,training_config=OmegaConf.to_container(config,resolve=True),
        provenance='historical checkpoint folder, saved config and current manifest reconstruction',
        limitations='no immutable manifest fingerprint from historical training; pretraining exposure unknown; BF repository license not established',
        text_used=False,native_size=[256,256],steps=200,scale_1=1.1,scale_2=1.5,new_training=0)
    del state
    assert not overlap, ('BF_TRAIN_TEST_OVERLAP',overlap)
    write(ROOT/'audit/BF_provenance.json',info)
    cn=PROJECT.parent/'cn_ipa_repro_20261022'
    choice=read(cn/'selection/frozen_final_choice.json')['models']['C2']
    ip=Path(choice['ip_path']);assert sha(ip)==choice['ip_sha256']
    cnpath=Path(choice['cn_path'])
    weight=next(p for p in cnpath.iterdir() if p.suffix in ['.bin','.safetensors'])
    assert sha(weight)==choice['cn_sha256']
    train=read(cn/'manifests/train.json')
    overlap=sorted(ids & {row['id'] for row in train})
    assert not overlap, ('CNIPA_TRAIN_TEST_OVERLAP',overlap)
    write(ROOT/'protocol/cnipa_choice.json',choice)
    write(ROOT/'audit/CNIPA_provenance.json',dict(choice=choice,training_n=len(train),training_overlap_ids=overlap,
        train_manifest_sha256=sha(cn/'manifests/train.json'),source_commit=command(['git','rev-parse','HEAD'],cn/'repo'),
        text_used=True,steps=50,cfg=7.,size=[384,512],new_training=0))
    print('EXTERNAL_ASSET_AUDIT_COMPLETE',flush=True)


def assets():
    audit()
    prepare()
    external_audit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit','prepare','external_audit','assets'])
    args = parser.parse_args()
    globals()[args.action]()
