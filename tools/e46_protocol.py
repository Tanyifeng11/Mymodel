"""e46 唯一固定 SARG；训练域校准和开发域评价分离。"""
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from tools.e44_protocol import OUT as OLD,E5,E5_SHA,SIZE,LAYER,read,write,sha,commit

OUT=Path('output_eval/e46_sarg_20261013')
E45=Path('output_eval/e45_b1_pnp_anchor_20261013')
CKPT=OLD/'p1/B1/checkpoints/step0800.pt'
CKPT_SHA='d107dc331c5d8b15079040dc351ec737b0644f9af745834379876e3e524ab84e'
ARMS=['G2','G3','G4']
HISTORY=['G0','G1','H45_C3','H45_C4','H45_C5']
KEYS=['lpips_gt','iou','contour_f1','leak','lab','hp_gray','clip_text','clip_reference','clip_real']
BAD=['172600892','193488637','180633937','204446309']
SMOKE=BAD+['174397224','127609356','53887902','53328647']
CONFIG=dict(experiment='e46',method='SARG',version=1,training_updates=0,new_parameters=0,
    source_plan_sha256='4219394b73c22c66733d0e1d07caeba7101a45f544772530ea0f9a9bab2df155',
    naming='user e46 overrides attached document E47; SARG only',
    alpha=.65,beta=.60,sigma_rgb_px=16.,layer=LAYER,steps=50,cfg=7.,seed=42,size=list(SIZE),
    gate_time='all50 steps',gate_position='after original B1 gate/to_out/dropout/rescale, before adding to original E5 base',
    scales=dict(sketch=.6,texture=1.,ipa=1.),contour='RETR_EXTERNAL; area>=max(64,largest_area*.01)',
    distance='RGB exact Euclidean DIST_L2 / DIST_MASK_PRECISE',downsample='cv2.INTER_AREA 512x384 to32x24',
    calibration='E44 train4096 string-ID ascending first8 valid sketch masks; identity-disjoint dev32/confirm96/validation500',
    lambda_energy='sqrt(sum ideal FP32 SARG residual energies / sum original residual energies) on fixed B1 calibration trajectories',
    energy_tolerance=.003,invalid='inherit E44 B1 OFF; retain32; regional NA; no GT fallback',
    bootstrap=dict(unit='identity',replicates=10000,seed=32042),
    safety=dict(iou_ci_lower=-.01,f1_ci_lower=-.02,leak_ci_upper=.005,tail_f1_drop=-.15,
        lpips_mean_delta_max=0.,retention_min=.5,text_mean_delta_min=-.005),
    lpips_ci='report upper<0 as stronger evidence; not an added hard gate',
    independent_value='G4-G3: meanF1>=.005 and meanLPIPS<=.002 OR meanLPIPS<=-.002 and meanF1>=-.003',
    interpretation='effect-size screening in reused dev32; report paired CI; not confirmatory significance',
    threshold_scan=dict(distance=[12,18,24],brightness=[225,235,245]),
    decisions='tail/appearance failure=>NO_GO; no independent effect=>NO_GO; only safety CI failure=>INCONCLUSIVE; all=>CONDITIONAL_GO',
    p3='optional, not in required round; not_run until separate frozen protocol',
    independent_confirm='only CONDITIONAL_GO plus visual support; audit historical identity/file/near duplicates before any generalization claim')


def protocol():
    path=OUT/'protocol.json'
    if path.exists():
        assert read(path)['config']==CONFIG,'禁止事后调整 e46 配方'
        return read(OUT/'splits/dev32.json'),read(path)
    from tools.e44_runtime import sketch_masks
    from tools.e34_sarr_protocol import BF
    dev=read(OLD/'splits/dev32.json');confirm=read(OLD/'splits/confirm96.json')
    validation=read('output_eval/e34_sarr_20261010/splits/validation_original.json')
    held={r['id'] for r in dev+confirm}|{Path(r['target']).stem for r in validation}
    pool=read(OLD/'splits/train4096.json')
    assert not held&{r['id'] for r in pool}
    selected=[];rejected=[]
    for row in sorted(pool,key=lambda r:r['id']):
        sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        _,_,info=sketch_masks(sketch,'cpu')
        if not info['valid']:
            rejected.append(dict(id=row['id'],mask=info));continue
        selected.append(row)
        if len(selected)==8:break
    assert len(selected)==8
    frozen=read(OLD/'manifests/input_hashes.json')
    inputs={r[k]:frozen[r[k]] for r in dev+selected for k in ['gt','sketch','reference']}
    inputs.update({str(E5):E5_SHA,str(CKPT):CKPT_SHA})
    assert all(sha(p)==h for p,h in inputs.items())
    models=read(OLD/'manifests/model_file_hashes.json')
    assert all(sha(p)==h for p,h in models.items())
    train_ids={r['id'] for r in read(OLD/'splits/train1024.json')}
    original_ids={Path(r['cloth']).stem for r in read('data/train_bf_texture.json')}
    value=dict(config=CONFIG,dev_ids=[r['id'] for r in dev],smoke_ids=SMOKE,
        calibration_ids=[r['id'] for r in selected],dev_split_sha256=sha(OLD/'splits/dev32.json'))
    assert set(SMOKE)<={r['id'] for r in dev}
    for name,rows in [('dev32',dev),('calibration8',selected),('smoke8',[next(r for r in dev if r['id']==x) for x in SMOKE])]:
        write(OUT/'splits'/(name+'.json'),rows)
    # 文件重复只审计，不依据终图质量重新选择校准身份。近重复未审计前不称未见域。
    overlaps={}
    for name,rows in [('dev32',dev),('confirm96',confirm)]:
        overlaps[name]={k:len({sha(r[k]) for r in selected}&{frozen[r[k]] for r in rows})
            for k in ['gt','sketch','reference']}
    val_gt={sha(BF/r['target']) for r in validation}
    overlaps['validation500_gt']=len({inputs[r['gt']] for r in selected}&val_gt)
    write(OUT/'manifests/calibration_audit.json',dict(identity_disjoint=True,
        exact_file_overlap=overlaps,rejected_invalid=rejected,
        membership=[dict(id=r['id'],in_B1_train1024=r['id'] in train_ids,
            in_original_training_manifest=r['id'] in original_ids) for r in selected],
        historical_e5_exposure='possible training domain; no consumed-identity trace',strict_unseen=False,
        near_duplicates='not audited; calibration is training-domain identity-disjoint from held cohorts'))
    write(OUT/'manifests/input_hashes.json',inputs);write(OUT/'manifests/model_file_hashes.json',models)
    write(OUT/'manifests/source_split_hashes.json',{str(p):sha(p) for p in [
        OLD/'splits/train4096.json',OLD/'splits/train1024.json',OLD/'splits/dev32.json',
        OLD/'splits/confirm96.json',Path('output_eval/e34_sarr_20261010/splits/validation_original.json')]})
    write(path,value)
    write(OUT/'manifests/pre_registration.json',dict(commit=commit(),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(path),stage='before e46 calibration and new candidate RGB/metrics'))
    return dev,value


def image_path(arm,identity,stage='p1'):
    return OUT/stage/'images'/arm/(identity+'.png')


def historical(arm,identity):
    if arm in ['G0','G1']:
        return OLD/'p1/images'/('B0' if arm=='G0' else 'B1')/(identity+'.png')
    return E45/'s2/images'/arm.replace('H45_','')/(identity+'.png')
