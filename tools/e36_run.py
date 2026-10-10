"""按 P0→公平 P1→门槛→独立确认推进；失败不增加训练预算。"""
import argparse
import torch
from tools.e36_protocol import *
from tools.e36_infer import Generator
from tools.e36_train import build_adapters


def generate(stage,arms):
    assert read(OUT/'p0/smoke_report.json')['pass_p0']
    if stage=='c1':assert read(OUT/'p1/selection.json')['selected']
    rows=read(OUT/'splits'/('dev32.json' if stage=='p1' else 'confirm96.json'))
    g=Generator();g.adapters,counts=build_adapters(g);folder=OUT/stage
    write(folder/'manifest.json',rows);write(folder/'params.json',{arm:dict(parameters=counts.get(arm,0)) for arm in arms})
    for arm in arms:
        if arm=='A0_E5_OFF':continue
        path=OUT/'checkpoints'/arm/'step0800.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['step']==800 and checkpoint['config']==CONFIG
        g.adapters[arm].load_state_dict(checkpoint['state_dict']);g.adapters[arm].eval()
    manifest=[]
    try:
        for i,row in enumerate(rows):
            hashes=[]
            for arm in arms:
                path=folder/'images'/arm/(row['id']+'.png');g.bridge.reset()
                if path.with_suffix('.json').exists():
                    record=read(path.with_suffix('.json'));assert record['png_sha256']==sha(path)
                else:record,_,_=g.generate(row,arm,path)
                hashes.append(record['initial_latent_sha256']);manifest.append(record)
                if i==0:write(folder/('actuation_'+arm+'.json'),g.bridge.report())
            assert len(set(hashes))==1
            write(folder/'generation_manifest.json',manifest)
            print('GENERATE',stage,i+1,'/',len(rows),flush=True)
    finally:g.close()


def assess(stage):
    from tools.e35_freeu_stats import evaluate,paired,safety,panels
    folder=OUT/stage;evaluate('dev32' if stage=='p1' else 'confirm96',folder=folder)
    records=read(folder/'metrics.json');deltas=read(folder/'paired_deltas.json')
    coverage=read(folder/'metric_protocol.json');arms=list(deltas);audit={}
    index={arm:{r['id']:r for r in records if r['arm']==arm} for arm in ['A0_E5_OFF']+arms}
    for arm in arms:
        checks=safety(deltas[arm])
        catastrophic=[]
        for sid,row in index[arm].items():
            base=index['A0_E5_OFF'][sid]
            for key,limit,direction in [('struct_iou',.15,-1),('struct_edge_f1',.15,-1),('leak_colored_frac',.10,1)]:
                if row.get(key) is not None and base.get(key) is not None and direction*(row[key]-base[key])>limit:
                    catastrophic.append(dict(id=sid,metric=key,delta=row[key]-base[key]))
        gate=dict(checks=checks,catastrophic=catastrophic,pass_safety=all(checks.values()) and not catastrophic)
        comparison=bootstrap([index[arm][sid]['struct_iou']-index['B1_CONV'][sid]['struct_iou']
            for sid in index[arm] if index[arm][sid]['struct_iou'] is not None and index['B1_CONV'][sid]['struct_iou'] is not None])
        iou=deltas[arm]['struct_iou'];edge=deltas[arm]['struct_edge_f1']
        near=all(index[arm][sid]['struct_iou']==index['A0_E5_OFF'][sid]['struct_iou'] and
            sha(folder/'images'/arm/(sid+'.png'))==sha(folder/'images/A0_E5_OFF'/(sid+'.png')) for sid in index[arm])
        passes=(gate['pass_safety'] and not near and coverage['mask_valid_n']/coverage['fixed_n']>=.8
            and iou['mean'] is not None and edge['mean'] is not None)
        if stage=='p1':passes=passes and iou['mean']>=.003 and edge['mean']>=0 and comparison['mean'] is not None and comparison['mean']>=.002
        else:passes=passes and iou['mean']>=.005 and iou['ci95'][0]>0
        audit[arm]=dict(safety=gate,vs_conv_iou=comparison,near_no_edit=near,eligible=bool(passes))
    conflict=bootstrap([index['B3_CONFLICT'][sid]['struct_iou']-index['B2_DAGF_LITE'][sid]['struct_iou']
        for sid in index.get('B3_CONFLICT',{}) if index['B3_CONFLICT'][sid]['struct_iou'] is not None and index['B2_DAGF_LITE'][sid]['struct_iou'] is not None])
    write(folder/'gates.json',dict(arms=audit,conflict_vs_dagf=conflict,coverage=coverage))
    # 固定最差身份集合，所有模型共同展示。
    worst=sorted(records,key=lambda r:r['struct_iou'] if r['struct_iou'] is not None else 2)[:8]
    worst_ids={r['id'] for r in worst};panels(folder,[r for r in read(folder/'manifest.json') if r['id'] in worst_ids],['A0_E5_OFF']+arms,tag='worst_case')
    if stage=='p1':
        candidates=[a for a in ['B2_DAGF_LITE','B3_CONFLICT'] if audit[a]['eligible']]
        selected=None
        if 'B3_CONFLICT' in candidates and conflict['mean'] is not None and conflict['mean']>0 and conflict['ci95'][0]>0:selected='B3_CONFLICT'
        elif 'B2_DAGF_LITE' in candidates:selected='B2_DAGF_LITE'
        elif candidates:selected=candidates[0]
        write(folder/'selection.json',dict(selected=selected,candidates=candidates,conflict_priority_requires_ci_lower_positive=True))
        decision(p1_run_complete=True,p1_dagf_beats_conv=bool(candidates),p1_conflict_beats_dagf=bool(conflict['mean'] is not None and conflict['mean']>0 and conflict['ci95'][0]>0),
            p1_full_rgb_safety_pass=any(audit[a]['safety']['pass_safety'] for a in ['B2_DAGF_LITE','B3_CONFLICT']),
            stop_reason=None if selected else 'P1_fail: no dual-kernel candidate meets all preregistered gates')
        return selected
    selected=read(OUT/'p1/selection.json')['selected'];passed=audit[selected]['eligible']
    decision(c1_confirm96_pass=passed,stop_reason=None if passed else 'C1_fail: locked candidate did not confirm')
    return passed


def run():
    from tools.e34_sarr_protocol import seed_all
    from tools.e36_smoke import run as smoke
    from tools.e36_train import train
    torch.set_num_threads(2);seed_all(42);prepare()
    if not (OUT/'p0/smoke_report.json').exists():smoke()
    for arm in ARMS:
        if not (OUT/'train'/arm/'complete.json').exists():train(arm)
    sequences=[read(OUT/'train'/arm/'complete.json')['sequence_sha256'] for arm in ARMS]
    assert len(set(sequences))==1,'三臂训练随机序列不一致'
    generate('p1',['A0_E5_OFF']+ARMS);selected=assess('p1')
    if selected:
        arms=['A0_E5_OFF','B1_CONV','B2_DAGF_LITE']+(['B3_CONFLICT'] if selected=='B3_CONFLICT' else [])
        generate('c1',arms)
        if assess('c1'):
            # C2 必须另行锁定正式训练及消融清单，不能把 pilot 权重伪报为正式结果。
            decision(stop_reason='C1_pass: C2 formal configuration must be frozen before execution')


if __name__=='__main__':
    try:run()
    except Exception as exc:
        decision(engineering_error=repr(exc));raise
