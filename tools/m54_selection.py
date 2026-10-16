"""只按冻结的身份配对结果推进；有歧义的Pilot例外留待明确解释，不事后发明门限。"""
import os
import shutil
from tools.m54_protocol import *


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
        lp,f1,iou=(d[k]['mean'] for k in ['lpips_gt','contour_f1','iou'])
        if len(g['extreme_events'])>=3:rejected[arm]='tail >= 3/24'
        elif lp>=0 and f1<=0 and iou<=0:rejected[arm]='nonpositive appearance and no mean F1/IoU improvement'
        elif lp>=0:ambiguous[arm]='文档“明显结构改善”未量化，需明确后才可例外保留'
        elif f1<-.05:ambiguous[arm]='文档“足够大的外观收益”未量化，需明确后才可例外保留'
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
    assert read(OUT/'pilot/evaluation_complete.json')['complete']
    selected=select_pilot(read(OUT/'pilot/means.json'),read(OUT/'pilot/gates.json'),read(OUT/'pilot/per_image.json'))
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
