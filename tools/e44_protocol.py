"""E44 固定实验预算与停止条件，结果不反向修改协议。"""
import random
from pathlib import Path
from tools.e35_freeu_protocol import read, write, sha, commit, bootstrap, E5, SIZE

OUT = Path('output_eval/e44_mimicbrush_drra_20261012')
LAYER = 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor'
ARMS = ['B1', 'B2', 'B3']
E5_SHA = 'aea56b2d7b02492f0e58fc2abc45b6a5a996b9fd962f31e30aa26ff5b0ed28f0'
CONFIG = dict(experiment='E44_MimicBrush_style_DRRA', version=1,
    source_plan_sha256='f2002a5c8506a495a0a893da7fa5f348ac9675edfb64bad24021e58de4d99d08',
    seed_train=42, seed_split=44042, pilot_ids=1024, main_ids=4096,
    pilot_updates=800, main_updates=5000, micro_batch=1, accumulation=8,
    learning_rate=1e-4, weight_decay=.01, grad_clip=1., keep_weight=.1,
    checkpoints_pilot=[0,100,400,800], checkpoints_main=[0,1000,3000,5000],
    checkpoint_selection='fixed final update; no dev checkpoint selection',
    width=384, height=512, ddim_steps=50, cfg=7., sketch_scale=.6,
    texture_scale=1., ipa_scale=1., original_tokens=16, reference_tokens=256,
    layer=LAYER, residual_scale=.1, mask=dict(erosion_pixels=9, feather_pixels=8,
        min_image_coverage=.03, invalid='new branch OFF; retain identity'),
    writer='frozen architecture-matched pretrained SD1.5 U-Net; reimplementation, not official MimicBrush checkpoint',
    writer_time='clean reference VAE mean latent, matched target timestep, empty text, every conditional forward',
    dropout=dict(image=.05,text=.20,both=.05,sketch=0.),
    main_lr_schedule='constant 1e-4, same optimizer rule as pilot',
    safety=dict(iou_ci_lower=-.01, contour_ci_lower=-.02, leak_ci_upper=.005,
        lpips_mean_max=.01, clip_text_mean_min=-.005,
        catastrophic_iou_drop=.15, catastrophic_contour_drop=.15, catastrophic_leak_rise=.10),
    promotion='safety AND paired noncolor HP error CI upper<0 AND LPIPS mean<0; real motif semantics require visual evidence',
    combination='B3 must improve LPIPS and HP mean over both B1 and B2; otherwise no automatic Main',
    seeds_generation=[42,43,44], bootstrap_reps=10000,
    b4='optional after internal positive result and runnable public model',
    blind_human_evaluation='not_run unless actual independent evaluators participate')


def decision(**values):
    path=OUT/'decision.json'
    state=read(path) if path.exists() else dict(P0='not_run',P1='not_run',P2='not_run',
        P3='not_run',P4='not_run',B4='not_run',paper_claim_supported=False)
    state.update(values)
    write(path,state)
    return state


def prepare():
    import torch
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==CONFIG
        return
    assert sha(E5)==E5_SHA
    checkpoint=torch.load(E5,map_location='cpu',weights_only=False)
    meta=checkpoint['meta']
    for key,value in dict(joint_i_drop_rate=.05,joint_t_drop_rate=.20,joint_ti_drop_rate=.05,
        texture_mode='patch_resampled',texture_preprocess_mode='plain_resize').items():
        assert meta[key]==value,(key,meta.get(key))
    write(OUT/'manifests/e5_training_meta.json',meta)
    model_files={str(E5):E5_SHA}
    for root in [Path('models/stable-diffusion-v1-5'),Path('models/clip')]:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ['.bin','.safetensors','.pt','.json']:
                model_files[str(p)]=sha(p)
    write(OUT/'manifests/model_file_hashes.json',model_files)
    old=Path('output_eval/e35_freeu_skip_20261010/splits')
    dev=read(old/'dev32.json'); confirm=read(old/'confirm96.json')
    pool=read('output_eval/e34_sarr_20261010/splits/train.json')
    pool=sorted(pool,key=lambda r:r['id']); random.Random(44042).shuffle(pool)
    held={r['id'] for r in dev+confirm}
    assert not held & {r['id'] for r in pool}
    frozen={str(E5):E5_SHA}
    held_hashes={key:set() for key in ['gt','sketch','reference']}
    for row in dev+confirm:
        for key in held_hashes:
            frozen[row[key]]=sha(row[key]);held_hashes[key].add(frozen[row[key]])
    chosen=[];rejected=[]
    for row in pool:
        current={key:sha(row[key]) for key in held_hashes}
        duplicate=[key for key,h in current.items() if h in held_hashes[key]]
        if duplicate:
            rejected.append(dict(id=row['id'],reasons=duplicate));continue
        chosen.append(row)
        for key,h in current.items():frozen[row[key]]=h
        if len(chosen)==4096:break
    assert len(chosen)==4096
    pool=chosen
    write(OUT/'manifests/duplicate_exclusions.json',rejected)
    for name,rows in [('train1024',pool[:1024]),('train4096',pool[:4096]),
                     ('dev32',dev),('confirm96',confirm)]:
        write(OUT/'splits'/(name+'.json'),rows)
    original=read('data/train_bf_texture.json')
    index={Path(r['cloth']).stem:r for r in original}
    for name in ['train1024','train4096']:
        write(OUT/'splits'/(name+'_records.json'),[index[r['id']] for r in read(OUT/'splits'/(name+'.json'))])
    for p in (OUT/'splits').glob('*.json'):
        frozen[str(p)]=sha(p)
    write(OUT/'manifests/input_hashes.json',frozen)
    # 精确重复来源核查单独存档；旧 E5 训练域不改称独立泛化集。
    hashes={name:{key:sorted({frozen[r[key]] for r in rows}) for key in ['gt','sketch','reference']}
        for name,rows in [('train',pool[:4096]),('dev',dev),('confirm',confirm)]}
    overlap={a+'_'+b:{key:len(set(hashes[a][key])&set(hashes[b][key])) for key in ['gt','sketch','reference']}
        for a,b in [('train','dev'),('train','confirm'),('dev','confirm')]}
    write(OUT/'manifests/split_audit.json',dict(identity_disjoint=True,exact_file_hash_overlap=overlap,
        limitation='dev/confirm are reused development identities; E5 historical training exposure possible',
        strict_unseen_generalization=False,validation500='audit before P4; never assume historical split independent'))
    write(OUT/'protocol.json',CONFIG); decision()
    print('E44 PREPARED',commit(),flush=True)


def verify_inputs():
    expected=read(OUT/'manifests/input_hashes.json')
    assert all(sha(p)==h for p,h in expected.items()),'冻结输入改变'
    write(OUT/'manifests/input_hash_check.json',dict(unchanged=True,n=len(expected)))


if __name__=='__main__': prepare()
