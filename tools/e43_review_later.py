"""探索性后阶段本地数据核验；只复算统计，不把本地核验称重跑LPIPS。"""
import hashlib
import numpy as np
from PIL import Image
from tools.e43_protocol import OUT,read,write,sha,Path


def verify(root):
    out=Path(root)/OUT;folder=out/'local_review';splits=read(folder/'split_sources.json')
    marker=read(folder/'exploratory_continuation.json')
    assert marker['skip_gate']=='pilot_only'
    assert marker['original_pilot_decision_sha256']==sha(folder/'pilot/decision.json')
    assert marker['locked_protocol_sha256']==sha(folder/'locked_protocol.json')
    stages={};tables={}
    for phase in ['train','validation','holdout']:
        done=(folder/phase/('oof.json' if phase=='train' else 'decision.json')).exists()
        if not done:continue
        records=[];invalid=[];png_n=0;extra_path=folder/phase/'supplementary_metrics.json'
        extra=read(extra_path)['values'] if extra_path.exists() else {}
        for row in splits[phase]:
            for seed in ([42] if phase=='train' else [42,123]):
                sample=folder/phase/'samples'/row['id']/('seed%d'%seed)
                r=read(sample/'complete.json');feature=read(sample/'features.json')
                assert (r['id'],r['seed'],r['group_id'])==(row['id'],seed,row['group_id'])
                assert len(feature['values'])==14 and np.isfinite(feature['values']).all()
                assert feature['feature_steps']==[1,5,10]
                assert feature['values'][13]==feature['mask_confidence']
                arms={v['arm']:v for v in r['outputs']};assert set(arms)=={'alpha1','low','high'}
                base=arms['alpha1'];assert len(base['trajectory_hashes'])==50
                with np.load(sample/'common_prefix.npz') as prefix:
                    assert hashlib.sha256(prefix['initial'].tobytes()).hexdigest()==base['initial_hash']
                    assert hashlib.sha256(prefix['step10'].tobytes()).hexdigest()==base['trajectory_hashes'][9]==base['prefix_latent_hash']
                if not feature['mask_valid']:invalid.append(dict(id=r['id'],seed=seed))
                for arm in ['low','high']:
                    v=arms[arm];assert v['branch_start_hash']==base['prefix_latent_hash'] and len(v['trajectory_hashes'])==40
                    assert v['unet_calls']==(80 if v['effective_alpha']==1 else 160)
                    if not feature['mask_valid']:
                        assert v['effective_alpha']==1 and v['trajectory_hashes']==base['trajectory_hashes'][10:]
                        assert v['output_png_sha256']==base['output_png_sha256']
                b1=sample/'b1_complete.json'
                if b1.exists():
                    b=read(b1);v=b['output'];assert b['adapter_training_steps_this_experiment']==0
                    assert v['initial_hash']==base['initial_hash'] and v['unet_calls']==100
                    r['outputs'].append(v)
                for v in r['outputs']:
                    path=sample/(v['arm']+'.png');assert sha(path)==v['output_png_sha256']
                    assert Image.open(path).size==(384,512);png_n+=1
                    v['metrics'].update(extra.get(v['output_png_sha256'],{}))
                records.append(r)
        for path in (folder/phase).glob('*shard*.json'):
            value=read(path)
            if 'frozen' in value:
                assert value['frozen']['parameters_unchanged'] and value['input_hashes_unchanged']
                original=read(folder/'preflight.json')['frozen']['frozen_modules']
                for name,h in value['frozen']['frozen_modules'].items():
                    if name!='bf_texture_conditioner':assert h==original[name]
        stages[phase]=dict(rows=len(records),fixed_sources=len(splits[phase]),generated_pngs=png_n,invalid_mask_rows=invalid,
            valid_sources=len(splits[phase])-len({r['id'] for r in invalid}),CLIP_rows=len(extra))
        tables[phase]=records

    def check_comparison(records,choose,base_choose,reported):
        for key,expected in reported['metrics'].items():
            by_group={}
            for r in records:
                arms={v['arm']:v['metrics'] for v in r['outputs']};v=arms[choose(r)].get(key);b=arms[base_choose(r)].get(key)
                if v is not None and b is not None:by_group.setdefault(r['group_id'],[]).append(v-b)
            x=np.asarray([np.mean(v) for v in by_group.values()]);assert len(x)==expected['groups']
            if not len(x):assert expected['mean'] is None;continue
            assert np.isclose(x.mean(),expected['mean'],atol=1e-12,rtol=0)
            rng=np.random.default_rng(43042);ci=np.quantile(x[rng.integers(len(x),size=(2000,len(x)))].mean(1),[.025,.975])
            assert np.allclose(ci,expected['ci95'],atol=1e-12,rtol=0)
    if 'train' in tables:
        oof=read(folder/'train/oof.json');assert oof['fixed_train_n']==128
        for kind,result in oof['results'].items():
            actions={r['group_id']:r['chosen_arm'] for r in result['rows']}
            keep=[r for r in tables['train'] if r['group_id'] in actions]
            assert len(keep)==oof['n_train_measurable']
            for fold in result['folds']:assert not set(fold['train_group_ids'])&set(fold['validation_group_ids'])
            check_comparison(keep,lambda r:actions[r['group_id']],lambda r:'alpha1',result['actual_oof_vs_e5'])
    for phase in ['validation','holdout']:
        if phase not in tables:continue
        records=tables[phase];decision=read(folder/phase/'decision.json');actions=read(folder/phase/'selected_actions.json')
        assert decision['original_pilot_status']=='NO_GO' and decision['exploratory_continuation']
        assert actions['policy_sha256']==sha(folder/'policy.pkl') and not actions['GT_read_for_decision']
        maps={kind:{(v['id'],v['seed']):v['arm'] for v in values} for kind,values in actions['actions'].items()}
        for kind,values in actions['actions'].items():
            for v in values:
                feature=folder/phase/'samples'/v['id']/('seed%d'%v['seed'])/'features.json'
                assert sha(feature)==v['features_sha256']
                if not read(feature)['mask_valid']:assert v['arm']=='alpha1'
            choose=lambda r:maps[kind][r['id'],r['seed']]
            check_comparison(records,choose,lambda r:'alpha1',decision['policies'][kind]['vs_e5'])
            check_comparison(records,choose,lambda r:decision['best_fixed_val'],decision['policies'][kind]['vs_best_fixed'])
        for arm,result in decision['fixed'].items():check_comparison(records,lambda r,a=arm:a,lambda r:'alpha1',result)
        ours=decision['policies']['utility'];m=ours['vs_e5']['metrics'];b=ours['vs_best_fixed']['metrics'];reasons=[]
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
        assert reasons==decision['reasons']
        assert decision['pass_stage']==(not decision['reasons'])
    write(out/'later_verification.json',dict(passed=True,stages=stages,source_cluster_bootstrap_verified=True,
        generated_PNG_and_prefix_hashes_verified=True,original_pilot_NO_GO_preserved=True,
        scope='本地复算记录的来源级统计；原始数据、模型和LPIPS计算在服务器核验'))
    print('E43 LATER VERIFIED',stages)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);verify(p.parse_args().root)
