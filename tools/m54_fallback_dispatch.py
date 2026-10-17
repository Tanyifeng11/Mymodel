"""后备独立调度：真实完整性先于筛选，旧轻量作业不再执行。"""
import math
import subprocess
from tools.m54_protocol import *
from tools.m54_dispatch import submit
from tools.m54_selection import share_results,ranking

ARMS=['M_L1','M_L2']


def verify_training(arm,total):
    import torch
    folder=OUT/'training'/arm;done=read(folder/('complete%04d.json'%total))
    sequence=read(folder/('random_sequence%04d.json'%total))
    assert done['updates']==total and len(sequence)==8*total
    assert sha(folder/('random_sequence%04d.json'%total))==done['sequence_sha256']
    assert all(done['nonzero_gradient_windows'].values())
    history=read(folder/'history.json');assert history[-1]['step']==total
    assert all(math.isfinite(v) for row in history for v in row.values())
    checkpoints={}
    for step in ([0,100,400,800] if total==800 else [800,2400]):
        path=folder/'checkpoints'/('step%04d.pt'%step);assert sha(path)==read(path.with_suffix('.json'))['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==step and state['arm']==arm and state['config']==read(OUT/'fallback/protocol.json')
        assert all(torch.isfinite(v).all() for v in state['state_dict'].values())
        assert sum(v.numel() for v in state['state_dict'].values())==done['parameters']
        assert state['optimizer'] is not None and state['scaler'] is not None
        if step:
            assert state['resume']['count']==8*step
            assert all(k in state['resume'] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng','data_rng','order'])
        checkpoints[str(step)]=sha(path);del state
    if total==2400:
        original=read(folder/'random_sequence0800.json');assert sequence[:len(original)]==original
        assert checkpoints['800']==read(OUT/'fallback/audit'/(arm+'_training800.json'))['checkpoints']['800']
    value=dict(passed=True,arm=arm,updates=total,additional_updates=800 if total==800 else 1600,
        exposures=len(sequence),sequence_sha256=done['sequence_sha256'],checkpoints=checkpoints,
        all_history_finite=True,all_gradient_windows_positive=True,commit=commit())
    write(OUT/'fallback/audit'/(arm+'_training'+str(total)+'.json'),value)


def verify_generation(stage,arms):
    rows=read(split_path(stage.split('_u')[0].split('_s')[0]));ids={r['id'] for r in rows}
    assert len(ids)==len(rows)
    for arm in arms:
        done=read(OUT/stage/(arm+'_complete.json'))
        assert done['n']==len(ids) and done['split_sha256']==sha(split_path(stage.split('_u')[0].split('_s')[0]))
        paths=list((OUT/stage/'images'/arm).glob('*.json'))
        metadata=[read(p) for p in paths if not p.stem.endswith('_effects')]
        assert len(metadata)==len(ids) and {m['id'] for m in metadata}==ids
        for meta in metadata:
            path=image_path(stage,arm,meta['id']);assert sha(path)==meta['png_sha256']
            base=read(image_path(stage,'E5',meta['id']).with_suffix('.json'))
            assert len(meta['steps'])==50 and meta['steps']==base['steps']
            assert meta['initial_sha256']==base['initial_sha256'] and meta['mask']['valid']==base['mask']['valid']
            assert meta['seed']==done['seed'] and not meta['gt_used_in_inference']
            effects=read(path.with_name(path.stem+'_effects.json'));assert len(effects)==50
            if arm in ARMS and not meta['mask']['valid']:assert all(not e['enabled'] and e['rms']==0 for e in effects)
        profile=read(OUT/stage/(arm+'_profile.json'));assert profile['n']==2
        for row in profile['records']:
            assert row['full50_steps'] and row['teacher_disabled'] and row['epsilon_trace_hooks_disabled']
            assert row['png_sha256']==sha(image_path(stage,arm,row['id']))
    value=dict(passed=True,stage=stage,arms=arms,identities=len(ids),conditions=len(ids)*len(arms),
        split_sha256=sha(split_path(stage.split('_u')[0].split('_s')[0])),raw_trace_full50='PENDING_FINAL',commit=commit())
    write(OUT/'fallback/audit'/(stage+'_generation.json'),value)


def verify_evaluation(stage,arms):
    from tools.e44_metrics import paired
    keys=['lpips_gt','contour_f1','iou','leak']
    complete=read(OUT/stage/'evaluation_complete.json');assert complete['complete'] and set(complete['arms'])==set(arms)
    rows=read(split_path(stage.split('_u')[0].split('_s')[0]));ids={r['id'] for r in rows};records=read(OUT/stage/'per_image.json')
    assert len(records)==len(ids)*len(arms);gates=read(OUT/stage/'gates.json')
    for arm in arms:
        group=[r for r in records if r['variant']==arm]
        assert len(group)==len(ids) and {r['id'] for r in group}==ids
        assert all(r[k] is not None and math.isfinite(r[k]) for r in group for k in keys)
        for row in group:
            meta=read(image_path(stage,arm,row['id']).with_suffix('.json'))
            assert row['inference_mask_valid']==meta['mask']['valid']
        if arm=='E5':continue
        g=gates[arm];assert g['regional_metrics_complete']
        for baseline in ['E5','B1']:
            for key in keys:
                actual=paired(records,arm,baseline,key);assert actual==g['paired_'+baseline][key] and actual['n']==len(ids)
        d=g['paired_E5'];tail=[r['id'] for r in group if r['contour_f1']-next(b['contour_f1'] for b in records if b['variant']=='E5' and b['id']==r['id'])<-.15]
        assert set(tail)=={r['id'] for r in g['extreme_events']}
        checks=dict(f1=d['contour_f1']['ci95'][0]>=-.020,iou=d['iou']['ci95'][0]>=-.010,
            leak=d['leak']['ci95'][1]<=.005,tail=not tail,lpips=d['lpips_gt']['ci95'][1]<0,
            retention=g['baseline_B1_gain']>0 and -d['lpips_gt']['mean']>=.5*g['baseline_B1_gain'])
        assert checks==g['checks'] and all(checks.values())==g['passed']
    reused=OUT/stage/'reused_results.json'
    if reused.exists():
        source=read(reused)['source'];old=read(OUT/source/'per_image.json')
        for row in records:
            if row['variant'] not in ['E5','B1']:continue
            original=next(r for r in old if r['variant']==row['variant'] and r['id']==row['id'])
            assert all(row[k]==original[k] for k in keys)
    write(OUT/'fallback/audit'/(stage+'_evaluation.json'),dict(passed=True,identities=len(ids),conditions=len(records),
        arms=arms,core_finite=True,paired_n=len(ids),regional_complete=True,bootstrap_core_recomputed_exact=True,
        fixed_six_gates_recomputed_exact=True,commit=commit()))


def select_pilot():
    from tools.m54_fallback_run import require_protocol
    require_protocol();stage='fb_pilot'
    assert read(OUT/'fallback/audit/fb_pilot_evaluation.json')['passed']
    gates=read(OUT/stage/'gates.json');means=read(OUT/stage/'means.json');records=read(OUT/stage/'per_image.json')
    selected=[];rejected={}
    for arm in ARMS:
        g=gates[arm];d=g['paired_E5']
        conditions=dict(lpips=d['lpips_gt']['mean']<0,f1=d['contour_f1']['mean']>=-.05,tail=len(g['extreme_events'])<3)
        if all(conditions.values()):selected.append(arm)
        else:rejected[arm]=conditions
    selected.sort(key=lambda a:ranking(a,means,gates,records))
    return freeze(OUT/'fallback/selection/pilot_selected.json',dict(selected=selected,rejected=rejected,
        policy_sha256=sha(OUT/'fallback/selection/pilot_policy.json'),selection_inputs='Pilot24 only',
        no_qualitative_exception=True,updates_each_if_eligible=2400))


def advance():
    import fcntl
    from tools.m54_fallback_run import require_protocol
    require_protocol()
    with (OUT/'resources/fallback_dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(OUT/'resources/jobs.json');latest={j.get('logical_name',j['name']):j for j in jobs if j['name'].startswith('FB_')}
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State'],text=True)
        states={int(s.split('|')[0]):s.split('|')[1].split()[0] for s in raw.splitlines() if s.split('|')[0].isdigit()}
        bad=[j for j in latest.values() if states.get(j['id']) in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']]
        if bad:print('FALLBACK_ENGINEERING_REPAIR_REQUIRED',[(j['name'],j['id']) for j in bad]);return
        active=[j for j in jobs if states.get(j['id'],'PENDING') in ['RUNNING','PENDING','COMPLETING']]
        slots=[max(0,4-len(active))]
        def enqueue(name,action,extra=(),gpu=True):
            name='FB_'+name
            if name not in latest:
                if slots[0]==0:return False
                identity=submit(name,action,gpu,extra=extra,slurm_options=['--time=12:00:00'] if gpu else ())
                latest[name]=dict(id=identity,name=name);states[identity]='PENDING';slots[0]-=1
            return states.get(latest[name]['id'])=='COMPLETED'
        def finish(outcome,**changes):
            decision(fallback=outcome,fallback_experiments_complete=True,complete=False,**changes)
            finals=[j for j in jobs if j.get('logical_name',j['name'])=='FINAL']
            if not finals:
                if slots[0]:submit('FINAL','finalize',False)
            elif states.get(finals[-1]['id'])=='COMPLETED' and (OUT/'audit/final_complete.json').exists():
                decision(server_experiments='COMPLETE',local_delivery='PENDING_DOWNLOAD_AND_VERIFICATION',complete=False)
        if not enqueue('ROUND0','fallback_audit'):return
        assert read(OUT/'fallback/audit/round0_complete.json')['passed']
        ready=True
        for arm in ARMS:
            done=enqueue('TRAIN_'+arm,'fallback_train',['--arm',arm,'--updates','800'])
            ready=ready and done and (OUT/'training'/arm/'complete0800.json').exists()
        if not ready:return
        for arm in ARMS:
            if not (OUT/'fallback/audit'/(arm+'_training800.json')).exists():verify_training(arm,800)
        def stage_run(label,stage,arms,updates=800,seed=42,reuse=None):
            folder=result_stage(stage,updates,seed)
            if reuse:share_results(reuse,folder,['E5','B1'])
            done=True
            for arm in ['E5','B1']+arms:
                if reuse and arm in ['E5','B1']:continue
                action='fallback_generate' if arm in ARMS else 'generate'
                done=enqueue(label+'_'+arm,action,['--arm',arm,'--stage',stage,'--updates',str(updates),'--seed',str(seed)]) and done
            if not done:return False
            allarms=['E5','B1']+arms
            if not (OUT/'fallback/audit'/(folder+'_generation.json')).exists():verify_generation(folder,allarms)
            if not enqueue(label+'_EVAL','evaluate',['--stage',folder]):return False
            if not (OUT/'fallback/audit'/(folder+'_evaluation.json')).exists():verify_evaluation(folder,allarms)
            return True
        if not stage_run('PILOT','fb_pilot',ARMS,reuse='pilot'):return
        chosen=read(OUT/'fallback/selection/pilot_selected.json') if (OUT/'fallback/selection/pilot_selected.json').exists() else select_pilot()
        selected=chosen['selected']
        if not selected:
            finish('COMPLETE_NO_PILOT_GO');return
        ready=True
        for arm in selected:
            ready=enqueue('DEEPEN_'+arm,'fallback_train',['--arm',arm,'--updates','2400']) and ready
        if not ready:return
        for arm in selected:
            if not (OUT/'fallback/audit'/(arm+'_training2400.json')).exists():verify_training(arm,2400)
        if not stage_run('DEV','fb_dev32',selected,2400,reuse='dev32_u2400'):return
        def passers(stage,name,limit=2):
            path=OUT/'fallback/selection'/(name+'.json')
            if path.exists():return read(path)['selected']
            gates=read(OUT/stage/'gates.json');means=read(OUT/stage/'means.json');records=read(OUT/stage/'per_image.json')
            values=sorted([a for a in selected if a in gates and gates[a]['passed']],key=lambda a:ranking(a,means,gates,records))[:limit]
            freeze(path,dict(selected=values,source_stage=stage,six_gates_required=True,no_combinations=True));return values
        lockarms=passers('fb_dev32_u2400','lock_candidates')
        if not lockarms:
            finish('COMPLETE_NO_DEV_GO');return
        if not (OUT/'audit/input_review/lock_approval.json').exists():
            if not (OUT/'m54_lock_input_review.tar.gz').exists():
                from tools.m54_finalize import input_review_package
                input_review_package('lock')
            decision(fallback='LOCK_INPUT_VISUAL_REVIEW_REQUIRED');return
        require_input_review('lock')
        if not stage_run('LOCK','fb_lock',lockarms,2400):return
        natural=passers('fb_lock_u2400','natural_candidates')
        if not natural:
            finish('COMPLETE_NO_LOCK_GO');return
        if not stage_run('NATURAL','fb_natural',natural,2400):return
        best=passers('fb_natural_u2400','best1',1)
        if not best:
            finish('COMPLETE_NO_NATURAL_GO');return
        ready=True
        for seed in [123,456]:ready=stage_run('SEED'+str(seed),'fb_natural',best,2400,seed) and ready
        if not ready:return
        finish('ALL_ALLOWED_EXPERIMENTS_COMPLETE',fallback_best1=best[0],fallback_seeds='COMPLETE',
            local_delivery='PENDING_FINAL_AND_VERIFICATION')


if __name__=='__main__':advance()
