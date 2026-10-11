"""来源等权统计和逐阶段停止规则；不根据测试GT拟合任何决策。"""
import argparse
import csv
import pickle
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from tools.e43_protocol import *
from tools.e43_metrics import safe_gain

KEYS=['LPIPS_cloth','HP_gray','LAB','IoU','EdgeF1','Leak','CLIP_real','CLIP_texture']


def load(phase):
    records=[];rows=read(OUT/'split_sources.json')[phase]
    sidecar=OUT/phase/'supplementary_metrics.json'
    extra=read(sidecar)['values'] if sidecar.exists() else {}
    for row in rows:
        for seed in ([42] if phase in ['pilot','train'] else [42,123]):
            p=OUT/phase/'samples'/row['id']/('seed%d'%seed)/'complete.json';assert p.exists(),p
            r=read(p);assert r['group_id']==row['group_id']
            b1=p.parent/'b1_complete.json'
            if b1.exists():r['outputs'].append(read(b1)['output'])
            for v in r['outputs']:assert sha(v['output_png'])==v['output_png_sha256']
            for v in r['outputs']:v['metrics'].update(extra.get(v['output_png_sha256'],{}))
            records.append(r)
    return records


def bootstrap(values):
    groups={}
    for group,v in values:
        if v is not None:groups.setdefault(group,[]).append(v)
    x=np.asarray([np.mean(v) for v in groups.values()],float)
    if not len(x):return dict(groups=0,mean=None,ci95=None)
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);b=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(groups=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())


def comparison(records,choose,base_choose=lambda r:'alpha1'):
    values={k:[] for k in KEYS};tail=[];fail=[];selected=[]
    for r in records:
        arms={q['arm']:q for q in r['outputs']};name=choose(r);v=arms[name]['metrics'];base=arms[base_choose(r)]['metrics']
        selected.append(name)
        for k in KEYS:
            d=v[k]-base[k] if v.get(k) is not None and base.get(k) is not None else None;values[k].append((r['group_id'],d))
        if v['IoU'] is not None and base['IoU'] is not None:tail.append(v['IoU']-base['IoU'])
        g=safe_gain(base,v)
        if g is not None and g['catastrophic']:fail.append(dict(id=r['id'],seed=r['seed'],arm=name,metrics=v,base=base))
    return dict(metrics={k:bootstrap(v) for k,v in values.items()},fixed_rows=len(records),choices=dict(Counter(selected)),
        delta_iou_p05=float(np.quantile(tail,.05)) if tail else None,catastrophic_flags=fail,catastrophic_fraction=len(fail)/len(records),
        per_seed={str(s):{k:bootstrap([(g,d) for (g,d),r in zip(v,records) if r['seed']==s]) for k,v in values.items()} for s in sorted({r['seed'] for r in records})})


def efficiency(records,choose):
    calls=[];seconds=[];collected_calls=0;collected_seconds=0.;peak=0
    for r in records:
        arms={v['arm']:v for v in r['outputs']};base=arms['alpha1'];selected=arms[choose(r)]
        calls.append(base['unet_calls'] if selected['arm']=='alpha1' else base['prefix_unet_calls']+selected['unet_calls'])
        seconds.append(base['seconds'] if selected['arm']=='alpha1' else base['prefix_seconds']+selected['seconds'])
        for value in r['outputs']:
            collected_calls+=value['unet_calls'];collected_seconds+=value['seconds'];peak=max(peak,value['peak_memory_bytes'])
    return dict(policy_unet_calls_mean=float(np.mean(calls)),original_E5_unet_calls=100,
        policy_seconds_composed_mean=float(np.mean(seconds)),seconds_are_independent_deployment_measurement=False,
        note='共同前缀与所选后缀的录制时间组合估计，未含CPU Ridge和特征文件IO；采集三分支成本另列',
        collection_unet_calls=collected_calls,collection_recorded_generation_seconds=collected_seconds,peak_memory_bytes=peak)


def source_directions(phase,records,choose):
    # caption类别仅作描述性分层，不当作人工服装/纹样真值，也不用于控制器决策。
    import re
    rows={r['id']:r for r in read(OUT/'split_sources.json')[phase]}
    groups={'caption_clothing':{},'reference_path_parent':{}}
    words=['dress','skirt','trousers','pants','jeans','shorts','jacket','coat','sweater','blouse','shirt','top']
    for r in records:
        row=rows[r['id']];caption=row['caption'].lower()
        kind=next((w for w in words if re.search(r'\b'+w+r's?\b',caption)),'unclassified')
        for axis,key in [('caption_clothing',kind),('reference_path_parent',str(Path(row['reference']).parent))]:
            groups[axis].setdefault(key,[]).append(r)
    return dict(label_status='caption/path-derived descriptive groups; not verified semantic labels',
        strata={axis:{key:comparison(values,choose) for key,values in strata.items()} for axis,strata in groups.items()})


def sheets(phase,records):
    rows={r['id']:r for r in read(OUT/'split_sources.json')[phase]};dest=OUT/phase/'contact_sheets';dest.mkdir(exist_ok=True)
    for i in range(0,len(records),6):
        batch=records[i:i+6];sheet=Image.new('RGB',(6*144,len(batch)*220),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            q=rows[r['id']];arms={v['arm']:v for v in r['outputs']}
            paths=[q['reference'],q['sketch'],q['gt']]+[arms[a]['output_png'] for a in ['alpha1','low','high']]
            for k,(name,path) in enumerate(zip(['R','sketch','GT','E5','low','high'],paths)):
                im=Image.open(path).convert('RGB');im.thumbnail((140,187));sheet.paste(im,(k*144,j*220))
                d.text((k*144,j*220+190),name+' '+r['id'],fill='black')
                d.text((k*144,j*220+204),'seed'+str(r['seed']),fill='black')
        sheet.save(dest/('fixed_%03d.jpg'%(i//6)))


def dump_csv(phase,records):
    path=OUT/phase/'rgb_results.csv';keys=['id','group_id','seed','arm','alpha','effective_alpha','output_png','output_png_sha256']+KEYS+['unet_calls','seconds','peak_memory_bytes']
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for r in records:
            for a in r['outputs']:writer.writerow(dict(a,**a['metrics']))


def pilot():
    records=load('pilot');oracle=[];safe_n=0;has_benefit=0;safe_directions=Counter();cat=[];valid=0;cases=[]
    for r in records:
        a={v['arm']:v['metrics'] for v in r['outputs']};base=a['alpha1'];g={name:safe_gain(base,a[name]) for name in ['low','high']}
        good={k:v for k,v in g.items() if v is not None};valid+=bool(good)
        gains=[v['gain'] for v in good.values()];best=max([0.]+gains)
        if good:oracle.append((r['group_id'],best))
        has_benefit+=best>0
        safe_n+=any(v['safe'] and v['gain']>0 for v in good.values())
        for name,v in good.items():
            if v['safe'] and v['gain']>0 and v['noncolor']:safe_directions[name]+=1
            if v['catastrophic']:cat.append(dict(id=r['id'],arm=name))
        cases.append(dict(id=r['id'],group_id=r['group_id'],benefits=g,oracle_lpips_gain=best if good else None))
    upper=bootstrap(oracle);weak=upper['mean'] is None or (upper['mean']<.005 and safe_n/len(records)<.25)
    reasons=[]
    if valid/len(records)<.8:reasons.append('insufficient_reliable_sketch_masks')
    if has_benefit/len(records)<.5:reasons.append('majority_no_positive_LPIPS_gain')
    if weak:reasons.append('insufficient_oracle_and_safe_improvement_space')
    if not all(safe_directions[x]>0 for x in ['low','high']):reasons.append('no_safe_noncolor_gain_in_both_directions')
    if cat:reasons.append('catastrophic_structure_flags')
    decision=dict(pass_stage=not reasons,status='PASS_PILOT' if not reasons else ('INCONCLUSIVE' if 'insufficient_reliable_sketch_masks' in reasons else 'NO_GO'),
        reasons=reasons,fixed_n=len(records),valid_n=valid,oracle_upper_bound=upper,safe_positive_gain_n=safe_n,
        any_positive_gain_n=has_benefit,safe_noncolor_by_alpha=dict(safe_directions),catastrophic_flags=cat,
        limitation='24 retrospective pilot sources, descriptive screening only; oracle is not deployable performance',cases=cases,
        comparisons={arm:comparison(records,lambda r,a=arm:a) for arm in ['alpha1','low','high']},
        controls={arm:comparison(records[:4],lambda r,a=arm:a) for arm in ['full_low','full_high','unmasked_low','unmasked_high']},**verify_inputs())
    write(OUT/'pilot/decision.json',decision);dump_csv('pilot',records);sheets('pilot',records)
    state=read(OUT/'state.json');state.update(precheck='complete',pilot='complete',stage='train_ready' if decision['pass_stage'] else 'stopped_at_pilot',
        stop_reason=reasons,complete=not decision['pass_stage']);write(OUT/'state.json',state)
    print('E43 PILOT',decision['status'],reasons,upper,flush=True)


def dataset(records):
    x=[];tokens=[];y=[];keep=[]
    for r in records:
        folder=OUT/'train/samples'/r['id']/('seed%d'%r['seed']);feature=read(folder/'features.json')
        a={v['arm']:v['metrics'] for v in r['outputs']};g=[safe_gain(a['alpha1'],a[n]) for n in ['low','high']]
        if not feature['mask_valid'] or any(v is None for v in g):continue
        x.append(feature['values']);tokens.append(np.load(folder/'reference_token_mean.npy'));y.append([v['utility'] for v in g]);keep.append(r)
    return np.asarray(x),np.asarray(tokens),np.asarray(y),keep


def fit():
    assert pilot_permission()
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    records=load('train');x,tokens,y,keep=dataset(records);assert len(keep)>=32
    def models():return [make_pipeline(StandardScaler(),Ridge(alpha=1.)) for _ in range(2)]
    def matrix(kind,xx,tt,pca=None,training=False):
        if kind=='time_sketch':return xx[:,[9,10,13]]
        if kind=='token':
            if training:pca=make_pipeline(StandardScaler(),PCA(n_components=min(11,len(tt)-1),svd_solver='full')).fit(tt)
            return np.column_stack([pca.transform(tt),xx[:,[9,10,13]]]),pca
        return xx
    kinds=['utility','time_sketch','token','shuffled'];saved={};oof_report={}
    for kind in kinds:
        predicted=np.zeros_like(y);folds=[]
        for fold,(tr,te) in enumerate(GroupKFold(4).split(x,y,[r['group_id'] for r in keep])):
            if kind=='token':
                a,pca=matrix(kind,x[tr],tokens[tr],training=True);b,_=matrix(kind,x[te],tokens[te],pca=pca)
            else:a=matrix(kind,x[tr],tokens[tr]);b=matrix(kind,x[te],tokens[te])
            target=y[tr].copy()
            if kind=='shuffled':target=target[np.random.default_rng(43042+fold).permutation(len(tr))]
            mods=models()
            for j,m in enumerate(mods):m.fit(a,target[:,j]);predicted[te,j]=m.predict(b)
            train_ids=[keep[i]['group_id'] for i in tr];test_ids=[keep[i]['group_id'] for i in te]
            assert not set(train_ids)&set(test_ids)
            folds.append(dict(fold=fold,train_groups=len(tr),validation_groups=len(te),train_only_standardization=True,
                train_group_ids=train_ids,validation_group_ids=test_ids))
        if kind=='token':a,pca=matrix(kind,x,tokens,training=True)
        else:a=matrix(kind,x,tokens);pca=None
        target=y.copy()
        if kind=='shuffled':target=target[np.random.default_rng(43042).permutation(len(y))]
        mods=models()
        for j,m in enumerate(mods):m.fit(a,target[:,j])
        saved[kind]=dict(models=mods,pca=pca)
        action=np.where(predicted.max(1)>.005,predicted.argmax(1)+1,0);actual=np.where(action>0,y[np.arange(len(y)),np.maximum(action-1,0)],0.)
        arms={r['group_id']:['alpha1','low','high'][int(a)] for r,a in zip(keep,action)}
        oof_report[kind]=dict(mse=float(np.mean((predicted-y)**2)),actual_utility_mean=float(actual.mean()),
            choices=dict(Counter(action.tolist())),oracle_utility_mean=float(np.maximum(0,y.max(1)).mean()),folds=folds,
            oracle_minus_actual_utility=float(np.maximum(0,y.max(1)).mean()-actual.mean()),
            actual_oof_vs_e5=comparison(keep,lambda r:arms[r['group_id']]),
            fixed_low_vs_e5=comparison(keep,lambda r:'low'),fixed_high_vs_e5=comparison(keep,lambda r:'high'),
            rows=[dict(id=r['id'],group_id=r['group_id'],predicted_utility=p.tolist(),actual_utility=t.tolist(),
                chosen_arm=arms[r['group_id']]) for r,p,t in zip(keep,predicted,y)])
    with (OUT/'policy.pkl').open('wb') as f:pickle.dump(saved,f)
    write(OUT/'train/oof.json',dict(n_train_measurable=len(keep),fixed_train_n=len(records),threshold=.005,results=oof_report,
        GT_used_only_for_training_supervision=True,source_disjoint=True))
    state=read(OUT/'state.json');state.update(train='complete',policy_fit='complete',stage='validation_ready',complete=False)
    write(OUT/'state.json',state)
    dump_csv('train',records);sheets('train',records);print('E43 RIDGE FIT',len(keep),flush=True)


def policy_arm(r,kind,policies,phase):
    folder=OUT/phase/'samples'/r['id']/('seed%d'%r['seed']);f=read(folder/'features.json');x=np.asarray(f['values'])[None]
    if not f['mask_valid']:return 'alpha1'
    p=policies[kind]
    if kind=='time_sketch':x=x[:,[9,10,13]]
    if kind=='token':x=np.column_stack([p['pca'].transform(np.load(folder/'reference_token_mean.npy')[None]),x[:,[9,10,13]]])
    estimates=np.asarray([m.predict(x)[0] for m in p['models']])
    return ['low','high'][int(estimates.argmax())] if estimates.max()>.005 else 'alpha1'


def evaluate(phase):
    records=load(phase)
    with (OUT/'policy.pkl').open('rb') as f:policies=pickle.load(f)
    fixed={a:comparison(records,lambda r,arm=a:arm) for a in ['alpha1','low','high']}
    if phase=='validation':
        best=min(fixed,key=lambda a:float('inf') if fixed[a]['metrics']['LPIPS_cloth']['mean'] is None else fixed[a]['metrics']['LPIPS_cloth']['mean'])
        write(OUT/'best_fixed_val.json',dict(arm=best,policy_sha256=sha(OUT/'policy.pkl'),threshold=.005,threshold_adjusted=False))
    else:best=read(OUT/'best_fixed_val.json')['arm']
    report=dict(fixed=fixed,best_fixed_val=best,policies={},
        exploratory_continuation=(OUT/'exploratory_continuation.json').exists(),
        original_pilot_status=read(OUT/'pilot/decision.json')['status'],
        preregistered_pipeline_pass=read(OUT/'pilot/decision.json')['pass_stage'])
    for kind in policies:
        choose=lambda r,k=kind:policy_arm(r,k,policies,phase)
        report['policies'][kind]=dict(vs_e5=comparison(records,choose),vs_best_fixed=comparison(records,choose,lambda r:best))
    write(OUT/phase/'selected_actions.json',dict(policy_sha256=sha(OUT/'policy.pkl'),threshold=.005,
        actions={kind:[dict(id=r['id'],seed=r['seed'],arm=policy_arm(r,kind,policies,phase),
            features_sha256=sha(OUT/phase/'samples'/r['id']/('seed%d'%r['seed'])/'features.json'))
            for r in records] for kind in policies},GT_read_for_decision=False))
    ours_choose=lambda r:policy_arm(r,'utility',policies,phase)
    report['efficiency']=efficiency(records,ours_choose)
    report['source_directions']=source_directions(phase,records,ours_choose)
    report['simple_controls']={kind:comparison(records,ours_choose,lambda r,k=kind:policy_arm(r,k,policies,phase))
        for kind in ['time_sketch','token','shuffled']}
    if all(any(v['arm']=='E36_B1' for v in r['outputs']) for r in records):
        report['E36_B1']=dict(vs_e5=comparison(records,lambda r:'E36_B1'),
            policy_vs_B1=comparison(records,ours_choose,lambda r:'E36_B1'),retrained=False)
    oracle=lambda r:min(r['outputs'][:3],key=lambda v:float('inf') if v['metrics']['LPIPS_cloth'] is None else v['metrics']['LPIPS_cloth'])['arm']
    report['oracle_upper_bound_only']=comparison(records,oracle)
    ours=report['policies']['utility'];m=ours['vs_e5']['metrics'];b=ours['vs_best_fixed']['metrics'];reasons=[]
    sufficient=m['LPIPS_cloth']['groups']>=30
    if not sufficient:reasons.append('insufficient_sources')
    else:
        if phase=='validation':
            if b['LPIPS_cloth']['mean']>=0:reasons.append('not_better_than_best_fixed_val')
            if m['LPIPS_cloth']['mean']>=0:reasons.append('not_better_than_E5')
        else:
            if m['LPIPS_cloth']['mean']>-.005 or m['LPIPS_cloth']['ci95'][1]>=0:reasons.append('LPIPS_gate_failed')
            if b['LPIPS_cloth']['ci95'][1]>=0:reasons.append('best_fixed_gate_failed')
        for key,limit,direction in [('IoU',-.005,'lower'),('EdgeF1',-.01,'lower'),('Leak',.002,'upper')]:
            v=m[key];point=v['mean'] if phase=='validation' else (v['ci95'][0 if direction=='lower' else 1] if v['ci95'] else None)
            if point is None or (point<limit if direction=='lower' else point>limit):reasons.append(key+'_safety_failed')
        if m['HP_gray']['mean'] is None or m['HP_gray']['mean']>=0:reasons.append('no_noncolor_detail_gain')
        if ours['vs_e5']['catastrophic_flags']:reasons.append('catastrophic_structure_flags')
        if phase=='holdout' and any(v['LPIPS_cloth']['mean'] is None or v['LPIPS_cloth']['mean']>=0 for v in ours['vs_e5']['per_seed'].values()):reasons.append('seed_direction_unstable')
    report.update(pass_stage=not reasons,status=('PASS_VALIDATION' if phase=='validation' else 'GO_PROTOTYPE_ONLY') if not reasons else ('INCONCLUSIVE' if not sufficient else 'NO_GO'),reasons=reasons,**verify_inputs())
    if phase=='holdout' and not reasons and report['exploratory_continuation']:
        report['status']='EXPLORATORY_PASS_HOLDOUT'
    write(OUT/phase/'decision.json',report);dump_csv(phase,records);sheets(phase,records)
    state=read(OUT/'state.json');state.update(stage='holdout_ready' if phase=='validation' and not reasons else 'finished_'+phase,
        complete=bool(reasons or phase=='holdout'));state[phase]='complete';write(OUT/'state.json',state)
    print('E43',phase,report['status'],reasons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['continue','pilot','fit','validation','holdout'],required=True);a=p.parse_args()
    if a.phase=='continue':authorize_continuation()
    elif a.phase=='pilot':pilot()
    elif a.phase=='fit':fit()
    else:evaluate(a.phase)
