"""按预设Hard Stop结束实验，不把未运行阶段写成通过。"""
import tarfile
from tools.e33tmoc_protocol import *

def main():
    decision=read(OUT/'decision_summary.json')
    assert read(OUT/'visual_audit/geometry_review.json')['pass']
    if not decision['geometry_mapping_pass']:
        stop='Hard Stop A: synthetic mapping gate failed'
    elif decision.get('C1_upper_bound_reproduction_pass') is False:
        assert read(OUT/'visual_audit/C0_C1_review.json')['pass']
        stop='Hard Stop B: C1 S1 R90 fixed <50%'
    else:
        raise RuntimeError('当前Gate允许继续，不能提前结束实验')
    frozen_check()
    skipped=['C2/C3/C4','candidate_freeze','confirmation64','VAE','E5_mini','full254',
             'seed43','seed44','ablations']
    decision['stop_rule']=stop;decision['skipped_stages']=skipped
    write(OUT/'decision_summary.json',decision)
    write(OUT/'completion_check.json',dict(experiment_execution_complete=True,
        scientific_success=False,stopped_by_prespecified_rule=stop,skipped_stages=skipped,
        geometry_review=read(OUT/'visual_audit/geometry_review.json'),
        C0_C1_review=read(OUT/'visual_audit/C0_C1_review.json') if decision['geometry_mapping_pass'] else None,
        frozen_inputs_pass=read(OUT/'frozen_check.json')['pass'],training_steps=0,git_commit=commit()))
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.suffix not in ('.gz','.log','.err') and p.name!='artifact_manifest.json'}
    write(OUT/'artifact_manifest.json',dict(files=files,git_commit=commit()))
    with tarfile.open(OUT/'review_bundle.tar.gz','w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix not in ('.gz','.log','.err'):tar.add(p,arcname=str(p.relative_to(OUT)))
    print('[OC completed under protocol]',stop,'bundle SHA256',sha(OUT/'review_bundle.tar.gz'),flush=True)

if __name__=='__main__':main()
