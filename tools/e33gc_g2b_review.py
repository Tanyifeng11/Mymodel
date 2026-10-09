"""导入两份独立AI评审；先冻结训练身份，不以生成效果选身份。"""
import argparse,csv
from PIL import Image
from tools.e33gc_g2b_protocol import *
from tools.e33gc_g2b_eval import dft_pair
from data.e33gc_g2b_renderer import input_dir

def freeze():
    init(); reviews=read('tools/e33gc_g2b_reviews.json')
    candidates=read(OUT/'G1a_reference_target_audit/candidate16_rows.json')
    accepted=[];disagreement=[]
    for row in candidates:
        a=reviews['reviewer1']['targets'][row['id']];b=reviews['reviewer2']['targets'][row['id']]
        if a['eligible']!=b['eligible']:disagreement.append(row['id'])
        # 两审核一致合格；分歧保守排除，不按生成图重新挑病例。
        if a['eligible'] and b['eligible']:
            folder=OUT/'G1a_controlled_targets'/row['id']
            metrics=dft_pair({arm:Image.open(folder/(arm+'_target.png')).convert('RGB') for arm in ARMS},row['dft_boxes'])
            if metrics['r90_success'] and metrics['r180_success']:accepted.append(row)
    write(OUT/'G1a_reference_target_audit/two_AI_reviews.json',reviews)
    write(OUT/'G1a_reference_target_audit/consensus.json',dict(agreed_eligible_ids=[r['id'] for r in accepted],
        disagreements=disagreement,review_type='two_independent_AI',human_review=None,
        all16_agreement=sum(reviews['reviewer1']['targets'][r['id']]['eligible']==reviews['reviewer2']['targets'][r['id']]['eligible'] for r in candidates)/max(len(candidates),1)))
    old_direction=sum(reviews['reviewer1']['old16'][sid]['direction_valid'] and reviews['reviewer2']['old16'][sid]['direction_valid']
        for sid in reviews['reviewer1']['old16'])
    write(OUT/'G1a_reference_target_audit/old16_AI_direction_gate.json',dict(N=16,agreed_direction_valid_n=old_direction,
        threshold=15,AI_direction_gate_pass=old_direction>=15,original_human_gate=None,
        source_text_issues_retained=True,not_used_as_new_main_training_targets=True))
    assert old_direction>=15,'original direction threshold15/16 not met under two-AI amendment'
    realids=[r['id'] for r in read(OUT/'G0a_real_input_annotations/rows.json')]
    good=[sid for sid in realids if reviews['reviewer1']['real_inputs'][sid]['eligible'] and reviews['reviewer2']['real_inputs'][sid]['eligible']]
    write(OUT/'G0a_real_input_annotations/real_input_eligibility.json',dict(all_fixed64_N=64,AI_eligible_ids=good,AI_eligible_n=len(good),
        human_eligible_n=None,human_gate_24of64=None,old_automatic_eligible_n=19,
        two_AI_agreement=sum(reviews['reviewer1']['real_inputs'][sid]['eligible']==reviews['reviewer2']['real_inputs'][sid]['eligible'] for sid in realids)/64,
        outputs_used=False,review_type='two_independent_AI',reviewer1=reviews['reviewer1']['real_inputs'],reviewer2=reviews['reviewer2']['real_inputs']))
    fields=['id','reviewer_type','eligible','single_axis_pattern','pattern_not_structural_edge','source_patch_purity',
        'reference_R90_valid','padding_or_crop_artifact','text_direction_conflict','target_pattern_support','reason_codes']
    for label in ['reviewer1','reviewer2','consensus']:
        p=OUT/'G0a_real_input_annotations'/('G0a_real_input_annotations_'+label+'.csv')
        with p.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for sid in realids:
                a=reviews['reviewer1']['real_inputs'][sid];b=reviews['reviewer2']['real_inputs'][sid]
                value=dict((a if label!='reviewer2' else b))
                if label=='consensus':
                    value={k:(a[k] and b[k]) for k in fields if k not in ['id','reviewer_type','reason_codes']}
                    value['reason_codes']=sorted(set(a['reason_codes']+b['reason_codes']))
                    for k in ['padding_or_crop_artifact','text_direction_conflict']:value[k]=a[k] or b[k]
                writer.writerow(dict(id=sid,reviewer_type='AI_independent' if label!='consensus' else 'AI_consensus',
                    **{k:(';'.join(value[k]) if k=='reason_codes' else value[k]) for k in fields if k not in ['id','reviewer_type']}))
    if len(accepted)<4:
        decision(main_reference_derived_G2b='not_run',g1a_AI_approved_identity_n=len(accepted),
            main_route_hard_stop='fewer than4 jointly AI-qualified real-crop identities',
            real_input_AI_eligible_n=len(good),next_route='synthetic_reference_control_preparation')
        bundle('routeA_hard_stop');return
    # 原15/16方向阈值保留；主路线另外需要4个真实裁片合格身份。人工认证仍为null。
    fit=[dict(r,label='fit_%02d'%(i+1)) for i,r in enumerate(accepted[:4])]
    probe=[dict(r,label='probe_%02d'%(i+1)) for i,r in enumerate(accepted[4:8])]
    write(OUT/'protocol/g2b_fit_probe_ids.json',dict(fit=fit,probe=probe,selection='reference_SHA_order then independent input-only AI reviews',
        human_gate=None,AI_review_passed_n=len(accepted),newdev_excluded=True))
    files={str(p):sha(p) for r in fit+probe for p in (OUT/'G1a_controlled_targets'/r['id']).glob('*.png')}
    write(OUT/'protocol/input_target_contract.json',dict(route='A',approved_for_AI_amended_training=True,files=files,
        real_source=True,DFT_positive_control_before_training=True,inference_target_RGB=False,
        independent_human_certification=None,user_amendment=CONFIG['review'],
        original15of16_human_gate=None,AI_direction_15of16_pass=True,
        AI_amended_gate='original15of16 direction check plus at least4 jointly AI-qualified real-crop identities; no human certification'))
    decision(g2b_route='A',g2b_train_identity_n=4,g1a_reference_target_consistency_pass=True,
        g1a_text_compatibility_pass=True,g1a_human_direction_gate_pass=None,real_input_AI_eligible_n=len(good),
        g1a_AI_approved_identity_n=len(accepted),next_route='G2b_smoke')
    bundle('frozen_inputs')

def endpoint():
    init();reviews=read('tools/e33gc_g2b_endpoint_reviews.json');keys=read(OUT/'protocol/blind_method_key.json')
    fit=[v for v in keys if v['group']=='fit' and v['method']=='A1_step160']
    fields=['visible_rotation','motif_correspondence','structure_safe','appearance_distinct']
    counts={k:sum(reviews['reviewer1'][v['token']][k] and reviews['reviewer2'][v['token']][k] for v in fit) for k in fields}
    numerical=read(OUT/'G2b_eval/numerical_gate.json');visual=all(n>=3 for n in counts.values())
    passed=numerical['numerical_pass'] and visual
    route=read(OUT/'protocol/input_target_contract.json')['route']
    write(OUT/'G2b_eval/two_AI_blind_review.json',dict(reviews=reviews,counts=counts,AI_visual_pass=visual,human_pass=None,N=4))
    decision(g2b_fit_visible_motif_n_of_4=counts['motif_correspondence'],g2b_fit_structure_and_appearance_gate=visual,
        g2b_fit_pass=passed if route=='A' else None,g2b_AI_amended_fit_pass=passed,
        synthetic_reference_control=('pass' if passed else 'fail') if route=='B' else None,
        original_human_joint_gate_pass=None,
        next_route='future_G2c_G3_protocol_review_only' if passed else 'one_fixed_free_residual_upper_diagnostic')
    bundle('reviewed_endpoint')

def freeze_B():
    init();assert read(OUT/'decision_summary.json')['main_reference_derived_G2b']=='not_run'
    rows=read(OUT/'G1a_synthetic_reference_control/candidate16_rows.json')
    reviews=read('tools/e33gc_g2b_synthetic_reviews.json');approved=[]
    for row in rows:
        sid=row['id'];folder=input_dir(row)
        metric=dft_pair({a:Image.open(folder/(a+'_target.png')).convert('RGB') for a in ARMS},row['dft_boxes'])
        if reviews['reviewer1'][sid]['eligible'] and reviews['reviewer2'][sid]['eligible'] and metric['r90_success'] and metric['r180_success']:
            approved.append(sid)
    # 原16正控制方向门槛仍为15；固定前4身份不得根据读出/生成成功换案例。
    assert len(approved)>=15 and all(r['id'] in approved for r in rows[:4]),'synthetic input contract invalid'
    fit=[dict(r,label='fit_%02d'%(i+1)) for i,r in enumerate(rows[:4])]
    probe=[dict(r,label='probe_%02d'%(i+1)) for i,r in enumerate(rows[4:8])]
    write(OUT/'protocol/g2b_fit_probe_ids.json',dict(fit=fit,probe=probe,selection='first fixed4 SHA input identities; routeB, no output selection'))
    write(OUT/'G1a_synthetic_reference_control/two_AI_reviews.json',dict(reviews=reviews,approved_n=len(approved),human_pass=None))
    files={str(p):sha(p) for r in fit+probe for p in input_dir(r).glob('*.png')}
    write(OUT/'protocol/input_target_contract.json',dict(route='B',approved_for_AI_amended_training=True,files=files,
        real_source=False,source='same independent procedural P drives reference and target',DFT_positive_control_before_training=True,
        inference_target_RGB=False,neutral_caption_protocol=True,old_captions_not_used=True,human_certification=None,
        real_reference_causality_claim_allowed=False,main_reference_derived_G2b='not_run',budget_GPU_hours=6))
    decision(g2b_route='B',g2b_train_identity_n=4,synthetic_reference_control='ready',next_route='synthetic_G2b_smoke')
    bundle('routeB_frozen_inputs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','freeze_B','endpoint']);a=p.parse_args()
    {'freeze':freeze,'freeze_B':freeze_B,'endpoint':endpoint}[a.action]()
