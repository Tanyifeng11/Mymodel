"""固定11身份的两臂验证；旧B400直接复用，分组保留失败例。"""
import gc
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools import e2_state_b_material_report as old
from tools.bf_t3_three_arm_report import paired
from tools.e2_fft_experiment import read, write, sha

ARMS = ('base', 'A', 'B', 'positive')


def archive(root, filename, files):
    path = root / filename
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for f in sorted(set(files)):
            if f.is_file():bundle.write(f, f.relative_to(root))
    print('JOINT_ARCHIVE', filename, path.stat().st_size, sha(path), flush=True)


def diagnostic_archive(args, spec):
    from tools.e2_state_b_joint import probes
    names = ('25_after','34_after','44_after','45_before','49_after','final')
    panels = args.out / 'probe_panels';panels.mkdir(exist_ok=True)
    files = [args.out / n for n in ('input_protocol.json','protocol.json','diagnostics.json',
                                   'target_controls.json','training_reference_panel.jpg')]
    for p,c,group in probes(spec):
        board=Image.new('RGB',(6*124,3*188),'white');draw=ImageDraw.Draw(board)
        for row,mode in enumerate(('raw','zero','outline')):
            d=args.out/'probe'/p['case_id']/(c+'_'+mode)
            for col,name in enumerate(names):
                path=d/(name+'.png')
                if path.exists():
                    draw.text((col*124+2,row*188+2),mode+' '+name,fill='black')
                    board.paste(Image.open(path).resize((120,160)),(col*124,row*188+23))
        panel=panels/(p['case_id']+'_'+c+'.jpg');board.save(panel,quality=95);files.append(panel)
        for arm in ('goal','positive'):
            files.append(args.out/'cases'/p['case_id']/(c+'_'+arm+'.png'))
    archive(args.out,'joint_diagnostics_review.zip',files)


def generate(args,spec):
    from tools.e2_state_b_joint import sample
    from tools import e2_state_b_joint_data as data
    from tools import e2_state_b_pilot as pilot
    from tools.e2_oracle_texture import weights_digest
    complete={a:read(args.out/'training'/a/'complete.json') for a in ('A','B')}
    assert all(v['updates']==400 and v['frozen_unchanged'] and v['protocol_sha256']==sha(args.out/'protocol.json') for v in complete.values())
    assert complete['A']['schedule']==complete['B']['schedule']
    core,pipe,net=pilot.load_core(args);before=weights_digest(net)
    assert before==complete['A']['m0_hash']==complete['B']['m0_hash']
    b=data.initialize(args).eval().requires_grad_(False)
    states={}
    for a in ('A','B'):
        file=args.out/'training'/a/'step400.pt'
        assert sha(file)==complete[a]['checkpoint_sha256']['400']
        states[a]=torch.load(file,map_location='cpu',weights_only=False)['b']
    with torch.inference_mode():
        for p in spec['holdout'][args.shard::2]:
            d=args.out/'cases'/p['case_id'];previous=read(d/'generation_base.json')['rows'];rows=[]
            for seed in range(2):
                pair=dict(p,seed=p['seed']+100000*seed)
                for c in spec['validation_conditions']:
                    key=f'{c}_s{seed}';meta=next(v for v in previous if v['key']==key)
                    assert sha(d/(key+'_base.png'))==meta['output_sha256']['b400']
                    hashes={}
                    for arm in ('A','B'):
                        path=d/f'{key}_{arm}.png'
                        # 完整generation.json存在时才复用；中断中的孤立PNG不作为完成证明。
                        existing=read(d/'generation.json')['rows'] if (d/'generation.json').exists() else []
                        row=next((v for v in existing if v['key']==key),None)
                        if row and arm in row['output_sha256']:
                            assert sha(path)==row['output_sha256'][arm];hashes[arm]=sha(path);continue
                        b.load_state_dict(states[arm])
                        mode=spec.get('arm_modes',{}).get(arm,'raw' if arm=='A' else 'outline')
                        picture,digest,_=sample(args,core,pipe,net,pair,c,b,mode)
                        assert digest==meta['digest'];picture.save(path);hashes[arm]=sha(path)
                    rows.append(dict(key=key,seed=pair['seed'],digest=meta['digest'],output_sha256=hashes))
                print('JOINT_GENERATED',p['case_id'],seed,flush=True)
            write(d/'generation.json',dict(rows=rows))
    assert before==weights_digest(net)
    del b,net,pipe,core;gc.collect();torch.cuda.empty_cache()
    evaluate(args,spec)


def evaluate(args,spec):
    # 复用已经核验的原指标公式，仅更换显式臂名称；所有PNG先校验再评分。
    for p in spec['holdout'][args.shard::2]:
        d=args.out/'cases'/p['case_id']
        for row in read(d/'generation.json')['rows']:
            for arm,h in row['output_sha256'].items():assert sha(d/(row['key']+'_'+arm+'.png'))==h
    previous=old.ARMS;old.ARMS=('base','A','B')
    try:old.evaluate(args,spec)
    finally:old.ARMS=previous
    for p in spec['holdout'][args.shard::2]:
        d=args.out/'cases'/p['case_id'];result=read(d/'result.json')
        mask=np.asarray(Image.open(d/'mask.png'))>0
        inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>=16
        grad=lambda a:np.concatenate((np.pad(a[:,1:]-a[:,:-1],((0,0),(0,1),(0,0))),
                                     np.pad(a[1:]-a[:-1],((0,1),(0,0),(0,0)))),2)
        for key,arms in result['rows'].items():
            if not key.startswith('real'):continue
            seed=key.rsplit('_',1)[1]
            goal,positive,anchor=[old.lab(old.pixels(d/name))/100 for name in
                (key+'_goal.png',key+'_positive.png','real_a_'+seed+'_m0.png')]
            mean=lambda a:a[inside,1:].mean(0)*100
            target_mean=mean(goal);positive_error=float(np.linalg.norm(mean(positive)-target_mean))
            a,b=grad(anchor),grad(positive)
            ea,eb=np.abs(a).mean(2),np.abs(b).mean(2)
            support=np.maximum(ea-eb,0)/(ea+eb+.005)*inside
            for arm,v in arms.items():
                picture=old.lab(old.pixels(d/(key+'_'+arm+'.png')))/100
                v['mean_ab_error']=float(np.linalg.norm(mean(picture)-target_mean))
                v['positive_mean_ab_error']=positive_error
                v['color_hit']=bool(.75*v['positive_spread']<=v['spread']<=1.25*v['positive_spread']
                                    and v['mean_ab_error']<=positive_error+2.)
                v['old_edge_error']=float((np.abs(grad(picture-goal))*support[...,None]).sum()/max(float(support.sum()*6),1e-8))
        write(d/'result.json',result)
    print('JOINT_EVALUATED_SHARD',args.shard,flush=True)


def panel(root,p):
    d=root/'cases'/p['case_id'];names=('reference','base','A','B','goal','positive')
    board=Image.new('RGB',(len(names)*154,4*226),'white');draw=ImageDraw.Draw(board)
    for row,(c,s) in enumerate((c,s) for s in range(2) for c in ('real_a','real_b')):
        for col,arm in enumerate(names):
            name=c+'_ref.png' if arm=='reference' else f'{c}_s{s}_{arm}.png'
            draw.text((col*154+2,row*226+2),f'{c} s{s} {arm}',fill='black')
            board.paste(Image.open(d/name).resize((150,200)),(col*154,row*226+22))
    board.save(d/'real_panel.jpg',quality=96)


def summarize(args,spec):
    completed=[read(args.out/f'metrics{s}/complete.json') for s in (0,1)]
    assert all(v['sanity']['passed'] and v['protocol_sha256']==sha(args.out/'protocol.json') for v in completed)
    cases=[read(args.out/'cases'/p['case_id']/'result.json') for p in spec['holdout']]
    assert sorted(v['case'] for v in cases)==sorted(v for shard in completed for v in shard['cases'])
    train={a:read(args.out/'training'/a/'complete.json') for a in ('A','B')}
    assert all(v['updates']==400 and v['frozen_unchanged'] for v in train.values())
    assert train['A']['schedule']==train['B']['schedule']
    assert train['A']['initial_b_hash']==train['B']['initial_b_hash']==read(args.out/'diagnostics.json')['initial_b_hash']
    fields=('correlation','contrast_error','palette_swd','reference_swd','low_l','old_projection','outside')
    groups={}
    for group in ('synthetic','all_real','low_real','multi_real'):
        real=group!='synthetic'
        selected=[p for p in cases if group in ('synthetic','all_real') or group.startswith(p['stratum'])]
        rows=[{k:v for k,v in p['rows'].items() if k.startswith('real')==real} for p in selected]
        keys=fields+(('mean_ab_error','old_edge_error','gray_margin') if real else ())
        statistics={}
        for arm in ARMS:
            values=lambda key:[v[arm][key] for row in rows for v in row.values() if v[arm][key] is not None]
            stats={k:float(np.mean(values(k))) if values(k) else None for k in keys}
            kinds=('real_response','real_pattern') if real else ('direction','period')
            stats['counts']={k:sum(v[arm] for p in selected for key,v in p['switches'].items() if key.startswith(k)) for k in kinds}
            stats['retain75']=sum(v[arm]['spread']>=.75*v[arm]['positive_spread'] for row in rows for v in row.values())
            if real:stats['color_hits']=sum(v[arm]['color_hit'] for row in rows for v in row.values())
            statistics[arm]=stats
        comparisons={}
        for a,b in (('A','base'),('B','base'),('B','A')):
            comparisons[a+'-'+b]={k:paired([float(np.mean([v[a][k]-v[b][k] for v in row.values()
                if v[a][k] is not None and v[b][k] is not None])) for row in rows
                if any(v[a][k] is not None and v[b][k] is not None for v in row.values())]) for k in keys}
        groups[group]=dict(identities=len(selected),images_per_arm=sum(len(r) for r in rows),arms=statistics,paired=comparisons)
    real=groups['all_real']['arms'];multi=groups['multi_real']['arms'];syn=groups['synthetic']['arms']
    identity=[]
    for p in cases:
        rr=[v for k,v in p['rows'].items() if k.startswith('real')]
        avg=lambda a,k:float(np.mean([r[a][k] for r in rr if r[a][k] is not None]))
        identity.append(dict(case=p['case'],stratum=p['stratum'],
            old_improved=avg('B','old_projection')<=avg('base','old_projection'),
            local_old_improved=avg('B','old_edge_error')<=avg('base','old_edge_error')))
    gates=dict(pattern=syn['B']['counts']['direction']==44 and syn['B']['counts']['period']==44
                       and real['B']['counts']['real_pattern']>real['base']['counts']['real_pattern']
                       and groups['low_real']['arms']['B']['counts']['real_pattern']>=groups['low_real']['arms']['base']['counts']['real_pattern']
                       and multi['B']['counts']['real_pattern']==6,
        color=multi['B']['color_hits']==12 and real['B']['palette_swd']<real['base']['palette_swd'],
        residue=real['B']['old_projection']<real['base']['old_projection'] and real['B']['old_edge_error']<real['base']['old_edge_error']
                and all(p['old_improved'] and p['local_old_improved'] for p in identity if p['stratum']=='multi'),
        light=real['B']['low_l']<=real['base']['low_l'],outside=real['B']['outside']<=real['base']['outside'])
    control_wins=dict(palette=real['B']['palette_swd']<real['A']['palette_swd'],
        residue=real['B']['old_projection']<real['A']['old_projection'] and real['B']['old_edge_error']<real['A']['old_edge_error'],
        pattern=real['B']['counts']['real_pattern']>=real['A']['counts']['real_pattern'] and real['B']['correlation']>real['A']['correlation'])
    write(args.out/'summary.json',dict(groups=groups,gates=gates,all_gates=all(gates.values()),
        control_comparison=control_wins,expansion_supported=all(gates.values()) and all(control_wins.values()),
        training=train,identity=identity,holdout=spec['holdout'],new_generated_images=264,
        scope='same11 reused validation identities and controlled planar targets; A/B full-recipe pilot, not independent final test'))
    for p in spec['holdout']:panel(args.out,p)
    files=[args.out/n for n in ('protocol.json','input_protocol.json','diagnostics.json','target_controls.json','summary.json','training_reference_panel.jpg')]
    files+=list((args.out/'training').glob('*/*.json'))+list((args.out/'training').glob('*/*.jsonl'))
    files+=list(args.out.glob('metrics*/**/*.json'))
    for p in spec['holdout']:files+=list((args.out/'cases'/p['case_id']).iterdir())
    archive(args.out,'joint_b_review.zip',files)
    print('JOINT_GATES',gates,control_wins,flush=True)
