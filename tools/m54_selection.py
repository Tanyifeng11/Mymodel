"""只按冻结的身份配对结果及用户确认的保守Pilot规则推进。"""
import os
import shutil
from datetime import datetime,timezone
from tools.m54_protocol import *


PILOT_POLICY=dict(version=1,source_plan_sha256=CONFIG['source_plan_sha256'],
    user_authorization='采用上述保守规则（推荐）',
    rules=dict(mean_delta_lpips_lt=0.,mean_delta_f1_ge=-.05,tail_events_lt=3),
    qualitative_exceptions=False,comparison='paired E5;24 identities',maximum_selected=4)


def freeze_pilot_policy():
    path=OUT/'selection/pilot_policy.json'
    if path.exists():
        value=read(path);assert value['policy']==PILOT_POLICY
        assert value['pilot_split_sha256']==sha(split_path('pilot'))
        return value
    # 先登记用户答复，再运行候选Pilot；不得在看过筛选结果后重定门槛。
    assert not list((OUT/'pilot').glob('*_complete.json')),'PILOT_OUTPUTS_ALREADY_EXIST'
    assert not (OUT/'pilot/evaluation_complete.json').exists()
    assert not (OUT/'selection/top4.json').exists()
    return freeze(path,dict(policy=PILOT_POLICY,pilot_split_sha256=sha(split_path('pilot')),
        before_pilot_outputs=True,before_pilot_evaluation=True,
        utc=datetime.now(timezone.utc).isoformat(),commit=commit()))


def ranking(arm,means,gates,records):
    g=gates[arm];d=g['paired_E5'];m=means[arm]
    costs=[r for r in records if r['variant']==arm]
    value=lambda k:next((r[k] for r in costs if r.get(k) is not None),float('inf'))
    f1=d['contour_f1']['ci95'][0];iou=d['iou']['ci95'][0]
    return (len(g['extreme_events']),f1<-.020,-f1,iou<-.010,-iou,m['lpips_gt'],
            value('parameters'),value('latency_seconds'),value('peak_vram_gib'),arm)


def select_pilot(means,gates,records):
    considered=[a for a in gates if family(a) in TRAIN_ARMS+['M7','M8','M10'] and a not in ['M7_A0','M8_T100']]
    rejected={};ambiguous={};eligible=[]
    for arm in considered:
        g=gates[arm];d=g['paired_E5']
        if not g['regional_metrics_complete']:
            raise AssertionError(('INCOMPLETE_PILOT_METRICS',arm))
        lp,f1=(d[k]['mean'] for k in ['lpips_gt','contour_f1'])
        rule=PILOT_POLICY['rules']
        if len(g['extreme_events'])>=rule['tail_events_lt']:rejected[arm]='tail >= 3/24'
        elif lp>=rule['mean_delta_lpips_lt']:rejected[arm]='mean deltaLPIPS >= 0; no qualitative exception'
        elif f1<rule['mean_delta_f1_ge']:rejected[arm]='mean deltaF1 < -0.05; no qualitative exception'
        else:eligible.append(arm)
    ordered=sorted(eligible,key=lambda a:ranking(a,means,gates,records))
    # 每个机制只保留一个Pilot冻结版本，不把M5三scale或M7三位置挤成三个模块。
    selected=[];seen=set()
    for arm in ordered:
        if family(arm) not in seen:selected.append(arm);seen.add(family(arm))
        if len(selected)==4:break
    return dict(selected=selected,rejected=rejected,ambiguous=ambiguous,ready=not ambiguous,
        ranking_rule='tail count; F1 safety/lower CI; IoU safety/lower CI; LPIPS; parameters; latency; VRAM',
        aliases_excluded=['M7_A0 equals original B1','M8_T100 equals original B1'],
        zero_training_promotions_keep_updates_zero=True,selection_inputs='Pilot only',lock_natural_used=False)


def share_results(source,target,arms):
    """相同输入/seed的基线使用已核验文件硬链接，避免重复生成和占用配额。"""
    assert sha(split_path(source.split('_u')[0]))==sha(split_path(target.split('_u')[0]))
    for arm in arms:
        done=OUT/source/(arm+'_complete.json');assert done.exists()
        files=list((OUT/source/'images'/arm).glob('*'))+[done,OUT/source/(arm+'_profile.json')]
        for path in files:
            relative=path.relative_to(OUT/source);dest=OUT/target/relative
            if dest.exists():assert sha(dest)==sha(path);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(path,dest)
            except OSError:shutil.copy2(path,dest)
    write(OUT/target/'reused_results.json',dict(source=source,arms=arms,exact_files=True,source_split_sha256=sha(split_path(source.split('_u')[0]))))


def freeze_top4():
    path=OUT/'selection/pilot_policy.json';policy=read(path)
    assert policy['policy']==PILOT_POLICY,'PILOT_POLICY_NOT_FROZEN'
    assert policy['pilot_split_sha256']==sha(split_path('pilot'))
    assert read(OUT/'pilot/evaluation_complete.json')['complete']
    selected=select_pilot(read(OUT/'pilot/means.json'),read(OUT/'pilot/gates.json'),read(OUT/'pilot/per_image.json'))
    selected['pilot_policy_sha256']=sha(path)
    write(OUT/'selection/pilot_decision.json',selected)
    if selected['ready']:
        selected['recipes']={a:dict(updates=2400 if family(a) in TRAIN_ARMS else 0) for a in selected['selected']}
        freeze(OUT/'selection/top4.json',selected)
    return selected


def freeze_combinations():
    stage='dev32_u2400';gates=read(OUT/stage/'gates.json');means=read(OUT/stage/'means.json');records=read(OUT/stage/'per_image.json')
    chosen=read(OUT/'selection/top4.json')['selected']
    positive=lambda a:gates[a]['regional_metrics_complete'] and not gates[a]['extreme_events'] and (
        gates[a]['paired_E5']['lpips_gt']['ci95'][1]<0 or gates[a]['paired_E5']['contour_f1']['ci95'][0]>0 or gates[a]['paired_E5']['iou']['ci95'][0]>0)
    available=[a for a in chosen if positive(a)]
    appearance=sorted([a for a in available if family(a) in ['M1','M2','M3','M4','M5','M6']],key=lambda a:ranking(a,means,gates,records))
    recipes={}
    for arm in appearance:
        for extra in ['M9','M11']:
            if extra not in available:continue
            if extra=='M9' and not any(gates[extra]['paired_B1'][k]['ci95'][0]>0 for k in ['contour_f1','iou']):continue
            name='C_'+arm+'_'+extra
            recipes[name]=dict(components=[dict(arm=arm,updates=2400),dict(arm=extra,updates=2400)],
                mechanism='sum independently trained residuals on the same original target features; original B1 joint branch OFF; no combination training',
                source='positive dev32 evidence only; M9 additionally requires F1/IoU lower CI >0 vs B1')
            if len(recipes)==4:break
        if len(recipes)==4:break
    return freeze(OUT/'selection/combinations.json',dict(recipes=recipes,dev_evaluation_sha256=sha(OUT/stage/'evaluation_complete.json'),
        total_additional_updates=0,maximum=4,lock_natural_used=False))


def freeze_passers(stage,filename,limit=None):
    gates=read(OUT/stage/'gates.json');means=read(OUT/stage/'means.json');records=read(OUT/stage/'per_image.json')
    arms=[a for a,g in gates.items() if g['passed'] and a not in ['B1','SAFE']]
    arms=sorted(arms,key=lambda a:ranking(a,means,gates,records))
    if limit is not None:arms=arms[:limit]
    return freeze(OUT/'selection'/filename,dict(selected=arms,source_stage=stage,
        evaluation_sha256=sha(OUT/stage/'evaluation_complete.json'),six_gates_required=True))
