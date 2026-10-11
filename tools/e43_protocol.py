"""E43：原配对、来源隔离、阈值先锁定；E5与历史结果只读。"""
import argparse
import hashlib
from pathlib import Path
from tools.e35_freeu_protocol import read,write,sha,commit
from tools.e39_protocol import E5,EXPECTED_E5,SIZE

OUT=Path('output_eval/e43_rgb_utility_20261011')
CONFIG=dict(experiment='E43_RRUC',version=1,size=list(SIZE),cfg=7.,steps=50,prefix_steps=10,
    alphas=[.65,1.,1.35],seeds=[42,123],pilot_n=24,train_n=128,val_n=48,holdout_n=128,
    E5_sha256=EXPECTED_E5,model_training_steps=0,controller='two StandardScaler+Ridge(alpha=1), source4fold OOF',
    decision_threshold=.005,threshold_tuning=False,features_n=14,split_seed=43042,bootstrap_seed=43042,bootstrap_reps=2000,
    reference_zero='exact E39 CLIP input + positive BF pooled/patch/CNN inputs zero; learned biases and negative branch retained',
    guided='explicit complete conditional+unconditional paths for both R+ and R0 on current branch z',
    mask=dict(source='existing sketch_only',valid='sketch_flood_fill and not low confidence; >=3% image interior',
        erosion_distance=9,feather_distance=8,latent_downsample='area',invalid='alpha1; retain sample, uncertain ROI metrics NA'),
    metrics=dict(LPIPS='official alex v0.1 calibrated layers; area-downsample sketch garment mask per feature map, sum layer weighted means',
        HP_gray='masked mean absolute error of grayscale highpass sigma3, RGB0..1 units',
        LAB='OpenCV float32 CIELAB mean distance to GT within sketch garment; reference color separately',
        IoU='existing estimated generated foreground vs sketch_only mask',EdgeF1='existing sketch edge F1',
        Leak='existing leak_colored_frac with GT background, unit fraction',FID='not computed at128 sources'),
    safety=dict(iou_drop=.005,edge_drop=.01,leak_rise=.002),
    catastrophic=dict(iou_drop=.15,edge_drop=.15,leak_rise=.10),
    utility=dict(lpips=1.,iou_penalty=2.,edge_penalty=.5,leak_penalty=4.),
    pilot_gate=dict(majority_no_benefit='>50% all fixed samples have no alpha with positive LPIPS gain',
        weak_oracle='mean oracle LPIPS gain<.005 AND safe positive gain fraction<.25',
        enter='at least one safe LPIPS+HP improvement for each non1 alpha; no catastrophic damage; reliable masks>=.80'),
    validation_gate='policy paired LPIPS mean strictly below Best-fixed-val, noncolor improvement and structure safety; otherwise STOP',
    formal_gate=dict(lpips_mean_max=-.005,lpips_ci_upper_max=0.,fixed_ci_upper_max=0.,iou_ci_lower_min=-.005,
        edge_ci_lower_min=-.01,leak_ci_upper_max=.002),
    controls=dict(pilot_full_from_step1_n=4,pilot_unmasked_n=4,ordinary='same2 Ridge; sketch features only',
        token='train-only PCA11 of reference-token mean + sketch area/boundary/confidence, max14 dims',
        shuffled='permute utility source labels inside each training fold only; heldout untouched'),
    evaluation_limit='Historical500 already used in E39–E42: retrospective heldout for controller, never new blind test',
    reporting='local docs only; server JSON/CSV/PNG/NPZ/Pickle and logs; no method novelty or pattern identity claim')


def ranked(rows,tag):
    return sorted(rows,key=lambda r:hashlib.sha256(('E43/43042/'+tag+'/'+r['id']).encode()).hexdigest())


def prepare():
    import cv2,torch,lpips,sklearn
    from tools.e33tmoc_appearance_eval import frozen_lpips,model_hash
    assert sha(E5)==EXPECTED_E5
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'locked_protocol.json').exists():
        assert read(OUT/'locked_protocol.json')==CONFIG
        return
    metric,info=frozen_lpips();info['state_sha256']=model_hash(metric)
    # 指标一致性：相同图像距离为零，区域加权不更换成像素代理。
    from tools.e43_metrics import cloth_lpips
    probe=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
    assert cloth_lpips(metric,probe,probe,torch.ones(1,1,64,64))==0.
    historical=read('output_eval/e42_real_fabric_repeat_gate_20261011/manifest_records.json')
    valrows=[dict(id=r['sample_id'],caption=r['caption'],gt=r['gt_path'],sketch=r['sketch_path'],reference=r['reference_path'],
        group_id=r['reference_sha256'],provenance='historical500 already analyzed') for r in historical]
    pilot=[];seen={r['group_id'] for r in valrows}
    for r in ranked(read('output_eval/e36_dagf_guided_filter_20261010/splits/dev32.json'),'pilot'):
        h=sha(r['reference'])
        if h in seen:continue
        pilot.append(dict(r,group_id=h,provenance='reused development identity'));seen.add(h)
        if len(pilot)==CONFIG['pilot_n']:break
    train=[]
    for r in ranked(read('output_eval/e34_sarr_20261010/splits/train.json'),'train'):
        h=sha(r['reference'])
        if h in seen:continue
        train.append(dict(r,group_id=h,provenance='original existing training pool'));seen.add(h)
        if len(train)==CONFIG['train_n']:break
    unique=[];taken=set()
    for r in ranked(valrows,'validation_holdout'):
        if r['group_id'] in taken:continue
        unique.append(r);taken.add(r['group_id'])
    splits=dict(pilot=pilot,train=train,validation=unique[:CONFIG['val_n']],
        holdout=unique[CONFIG['val_n']:CONFIG['val_n']+CONFIG['holdout_n']])
    for name,count in [('pilot',24),('train',128),('validation',48),('holdout',128)]:assert len(splits[name])==count,(name,len(splits[name]))
    names=list(splits)
    for i,a in enumerate(names):
        for b in names[i+1:]:assert not {r['group_id'] for r in splits[a]}&{r['group_id'] for r in splits[b]}
    hashes={str(E5):EXPECTED_E5}
    for rows in splits.values():
        for r in rows:
            for k in ['reference','gt','sketch']:hashes[r[k]]=sha(r[k])
            assert sha(r['reference'])==r['group_id']
    write(OUT/'locked_protocol.json',CONFIG);write(OUT/'split_sources.json',splits);write(OUT/'audit/input_hashes.json',hashes)
    write(OUT/'metric_initialization.json',info)
    write(OUT/'preparation.json',dict(complete=True,code_commit=commit(),lpips=lpips.__version__ if hasattr(lpips,'__version__') else str(Path(lpips.__file__)),
        torch=torch.__version__,opencv=cv2.__version__,sklearn=sklearn.__version__,sources={k:len(v) for k,v in splits.items()},
        source_disjoint=True,historical500_previously_used=True,E5_training_membership_not_redefined=True))
    write(OUT/'state.json',dict(stage='prepared',precheck='pending',pilot='pending',train='pending',validation='pending',holdout='pending',complete=False))
    print('E43 PREPARED', {k:len(v) for k,v in splits.items()},flush=True)


def verify_inputs():
    hashes=read(OUT/'audit/input_hashes.json')
    assert all(sha(p)==h for p,h in hashes.items()),'冻结输入改变'
    return dict(input_hashes_unchanged=True,n_input_files=len(hashes))


def pilot_permission():
    if read(OUT/'pilot/decision.json')['pass_stage']:
        return True
    path=OUT/'exploratory_continuation.json'
    return path.exists() and read(path).get('skip_gate')=='pilot_only'


def authorize_continuation():
    """用户明确要求继续；只另记Pilot例外，保留原协议和NO_GO文件。"""
    assert read(OUT/'preflight.json')['passed']
    decision=read(OUT/'pilot/decision.json')
    path=OUT/'exploratory_continuation.json'
    if not path.exists():
        write(path,dict(authorization='用户：继续完成后续的实验',skip_gate='pilot_only',
            original_pilot_status=decision['status'],original_pilot_decision_sha256=sha(OUT/'pilot/decision.json'),
            locked_protocol_sha256=sha(OUT/'locked_protocol.json'),code_commit=commit(),
            interpretation='post-Pilot exploratory continuation; original preregistered pipeline remains NO_GO',
            unchanged=['source lists','budgets','generation settings','Ridge settings','validation gate','holdout gate']))
    state=read(OUT/'state.json');state.update(stage='train_ready_exploratory',complete=False,
        exploratory_continuation=True,original_pilot_status=decision['status'])
    write(OUT/'state.json',state)
    print('E43 EXPLORATORY CONTINUATION AUTHORIZED; original Pilot',decision['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continue-exploratory',action='store_true');args=parser.parse_args()
    if args.continue_exploratory:authorize_continuation()
    else:prepare()
