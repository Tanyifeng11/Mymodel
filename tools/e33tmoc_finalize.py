"""按预设Hard Stop结束实验，不把未运行阶段写成通过。"""
import tarfile
import numpy as np
from tools.e33tmoc_protocol import *

def main():
    decision=read(OUT/'decision_summary.json')
    assert read(OUT/'visual_audit/geometry_review.json')['pass']
    if not decision['geometry_mapping_pass']:
        stop='Hard Stop A: synthetic mapping gate failed'
    elif decision.get('C1_upper_bound_reproduction_pass') is False:
        assert read(OUT/'visual_audit/C0_C1_review.json')['pass']
        stop='Hard Stop B: C1 S1 R90 fixed <50%'
    elif (OUT/'candidates/gate.json').exists():
        gate=read(OUT/'candidates/gate.json')
        assert not gate['eligible_ranked'], '有合格候选，必须继续确认阶段'
        assert read(OUT/'visual_audit/carrier_review.json')['pass']
        assert read(OUT/'visual_audit/C0_C1_review.json')['pass']
        stop=decision['primary_failure_mode']
        # 图像与方向状态必须齐全，失败身份也不能缺档。
        ids=read(OUT/'splits/diagnostic64.json')
        for name in ('C0_current','C1_analytic','C2_appearance','C3_support_aware','C4_confidence_blend'):
            for sid in ids:
                folder=OUT/'candidates'/name/'seed42'/sid
                for arm in ('R0','R90','R180'):
                    assert (folder/(arm+'_construction.npz')).exists()
                    with np.load(folder/(arm+'_inputs.npz')) as z:
                        assert 'orientation' in z.files and 'confidence' in z.files
                    for stage in ('S1','S2'):
                        assert (folder/stage/(arm+'.png')).exists()
                        assert (folder/stage/(arm+'_orientation.npz')).exists()
        assert len(read(OUT/'splits/visual_sets.json')['sets'])==5
    else:
        raise RuntimeError('当前Gate允许继续，不能提前结束实验')
    cohort=prepare()
    freeze_carrier_inputs(cohort,read(OUT/'splits/diagnostic64.json'))
    frozen_check()
    skipped=['C2/C3/C4','candidate_freeze','confirmation64','VAE','E5_mini','full254',
             'seed43','seed44','ablations']
    if (OUT/'candidates/gate.json').exists():skipped.remove('C2/C3/C4')
    for key in ('selected_candidate','selected_s1_r90','selected_s2_r90','selected_readable_fraction',
                'selected_r180','selected_texture_similarity','confirmation_pass','vae_survival_pass',
                'mini_image_r90','mini_image_gain_pp','mini_text_delta','mini_contour_delta',
                'seed42_full_pass','seed43_full_pass','seed44_full_pass'):
        decision.setdefault(key,None)
    decision['stop_rule']=stop;decision['skipped_stages']=skipped
    write(OUT/'decision_summary.json',decision)
    write(OUT/'completion_check.json',dict(experiment_execution_complete=True,
        scientific_success=False,stopped_by_prespecified_rule=stop,skipped_stages=skipped,
        geometry_review=read(OUT/'visual_audit/geometry_review.json'),
        C0_C1_review=read(OUT/'visual_audit/C0_C1_review.json') if decision['geometry_mapping_pass'] else None,
        carrier_review=read(OUT/'visual_audit/carrier_review.json') if (OUT/'visual_audit/carrier_review.json').exists() else None,
        input_fingerprints_verified=True, fixed_direction_case_count=64,
        frozen_inputs_pass=read(OUT/'frozen_check.json')['pass'],training_steps=0,git_commit=commit()))
    table=read(OUT/'result_table.json');table['unrun_stages']={name:dict(status='not_run',reason=stop) for name in skipped}
    write(OUT/'result_table.json',table)
    # manifest覆盖全部服务器原始数据；本地审阅包只含数值、图例和曲线。
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.suffix not in ('.gz','.log','.err') and p.name!='artifact_manifest.json'}
    write(OUT/'artifact_manifest.json',dict(files=files,git_commit=commit()))
    with tarfile.open(OUT/'review_bundle.tar.gz','w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and (p.suffix=='.json' or 'visual_audit' in p.parts or 'curves' in p.parts):
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('[OC completed under protocol]',stop,'bundle SHA256',sha(OUT/'review_bundle.tar.gz'),flush=True)

if __name__=='__main__':main()
