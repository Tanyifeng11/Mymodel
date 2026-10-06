"""固定carrier/appearance Gate与确定性候选排序，不访问确认集。"""
import argparse
from tools.e33tmif_metrics import bootstrap
from tools.e33tmoc_protocol import *

def evaluate():
    direction=read(OUT/'candidates/appearance_summary.json')
    appearance=read(OUT/'appearance_metrics/summary.json')
    baseline={r['id']:r for r in read(OUT/'appearance_metrics/C0_current/S2_cases.json')}
    results={};eligible=[]
    for name,stages in direction.items():
        s1=stages['S1']['statistics'];s2=stages['S2']['statistics'];p=PROTOCOL['candidate']
        checks=dict(S1_r90=s1['r90_success']['mean']>=p['S1_r90'],S2_r90=s2['r90_success']['mean']>=p['S2_r90'],
            S1_readable=s1['r90_readable']['mean']>=p['S1_readable'],S2_readable=s2['r90_readable']['mean']>=p['S2_readable'],
            S1_r180=s1['r180_success']['mean']>=p['S1_r180'],S2_r180=s2['r180_success']['mean']>=p['S2_r180'],
            S1_S2_drop=s1['r90_success']['mean']-s2['r90_success']['mean']<=p['S1_S2_drop'])
        cases=read(OUT/'appearance_metrics'/name/'S2_cases.json')
        delta=bootstrap([r['Lab_color_distance']-baseline[r['id']]['Lab_color_distance'] for r in cases
            if r['Lab_color_distance'] is not None and baseline[r['id']]['Lab_color_distance'] is not None])
        matched=all((r['Lab_color_distance'] is None)==(baseline[r['id']]['Lab_color_distance'] is None) and
            (r['texture_score'] is None)==(baseline[r['id']]['texture_score'] is None) for r in cases)
        texture=appearance[name]['S2']['statistics']['texture_score']['mean'];base_texture=appearance['C0_current']['S2']['statistics']['texture_score']['mean']
        app_checks=dict(common_valid_cases=matched,color_not_significantly_worse=delta['n']>0 and delta['ci95'][0]<=0,
            texture_retained=texture is not None and base_texture is not None and texture>=.8*base_texture)
        passed=all(checks.values()) and all(app_checks.values())
        results[name]=dict(carrier_checks=checks,carrier_pass=all(checks.values()),appearance_checks=app_checks,
            appearance_pass=all(app_checks.values()),color_distance_increase=delta,texture_ratio=texture/base_texture if texture is not None and base_texture else None,
            eligible=passed,diagnostic_only=all(checks.values()) and not all(app_checks.values()))
        if passed:eligible.append(name)
    # 只在direction相同后比较外观，确保排序在确认数据出现前固定。
    eligible.sort(key=lambda n:(-direction[n]['S2']['statistics']['r90_success']['mean'],
        -direction[n]['S2']['statistics']['r180_success']['mean'],-direction[n]['S2']['statistics']['r90_readable']['mean'],
        -appearance[n]['S2']['statistics']['texture_score']['mean'],appearance[n]['S2']['statistics']['Lab_color_distance']['mean'],n))
    result=dict(candidates=results,eligible_ranked=eligible,selection_split='diagnostic64',confirmation_used=False)
    write(OUT/'candidates/gate.json',result)
    d=read(OUT/'decision_summary.json');d['candidate_gate']=results
    if not eligible:
        all_low=all(s['S1']['statistics']['r90_success']['mean']<.35 for s in direction.values())
        if all_low and d['C1_s1_r90']>=.60:
            route='appearance_geometry_factorization_failure';failure='Hard Stop C: all C2/C3/C4 S1 R90<35%, C1>=60%'
        elif any(r['diagnostic_only'] for r in results.values()):
            route='appearance_preservation_failure';failure='carrier_direction_pass_but_appearance_gate_failed'
        else:route='carrier_level_gate_failure';failure='no_candidate_passes_all_carrier_level_gates'
        d.update(selected_candidate=None,orientation_preserving_carrier_pass=False,primary_failure_mode=failure,next_route=route)
    else:d.update(next_route='freeze_candidate_then_confirmation64')
    write(OUT/'decision_summary.json',d)
    table=read(OUT/'result_table.json');table['appearance_metrics']=appearance;table['candidate_gate']=result
    write(OUT/'result_table.json',table)
    print('[OC candidate Gate]',result,flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    result=evaluate()
    if args.freeze:
        assert read(OUT/'visual_audit/carrier_review.json')['pass']
        ranked=result['eligible_ranked'];assert ranked,'无合格候选'
        candidate=dict(main=ranked[0],fallback=ranked[1] if len(ranked)>1 else None,
            configuration=read(OUT/'protocol/appearance_carrier.json'),selection_split='diagnostic64',
            configuration_sha256=sha(OUT/'protocol/appearance_carrier.json'),git_commit=commit(),training_steps=0)
        p=OUT/'protocol/candidate_freeze.json'
        if p.exists():assert read(p)==candidate
        else:write(p,candidate)
        d=read(OUT/'decision_summary.json');d['selected_candidate']=candidate['main'];d['fallback_candidate']=candidate['fallback']
        write(OUT/'decision_summary.json',d)

if __name__=='__main__':main()
