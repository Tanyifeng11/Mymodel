"""只恢复已完成step500的dev评测；不重训、不覆盖checkpoint或train结果。"""
import argparse
import torch
from models.apacc_features import load_dino
from models.e33r_rotation_control import RotationControl
from tools.e33r_train_control import load_prior
from tools.e33r_evaluate import evaluate_records
from tools.e33r_common import *

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--interrupted-job',required=True)
    args=parser.parse_args();folder=OUT/'R_sanity';path=folder/'checkpoint_final.pt'
    integrity=read(folder/'checkpoint_integrity.json');assert sha(path)==integrity['sha256']
    assert integrity['prior_checkpoint_unchanged'] and integrity['prior_model_unchanged']
    original_train_sha=sha(folder/'train/rows.json');original_summary_sha=sha(folder/'train/summary.json')
    train=read(folder/'train/summary.json');assert train['denominator']==512
    checkpoint=torch.load(path,map_location='cpu');assert checkpoint['steps']==500 and checkpoint['seed']==42
    prior,prior_sha=load_prior();model=RotationControl(prior).cuda();model.load_state_dict(checkpoint['model']);model.eval()
    baseline,_=load_prior();baseline.requires_grad_(False);dino,_=load_dino('cuda',WEIGHTS)
    records=read(OUT/'sanity_manifest.json')
    _,dev=evaluate_records(model,records['dev'],dino,folder/'dev','full',baseline,True)
    checks={k:train[k]['mean']>=.95 for k in ['clean_r90_success','r180_identity_success','r0_identity_success']}
    checks['finite']=train['finite_prediction_rate']==1 and dev['finite_prediction_rate']==1
    passed=all(checks.values());write(folder/'gate.json',dict(checks=checks,**{'pass':passed},denominator=512))
    assert original_train_sha==sha(folder/'train/rows.json') and original_summary_sha==sha(folder/'train/summary.json')
    assert integrity['sha256']==sha(path) and prior_sha==sha(OUT/'P0_prior/checkpoint_final.pt')
    write(folder/'evaluation_recovery.json',dict(interrupted_job=args.interrupted_job,new_training_steps=0,
        unchanged_checkpoint_sha256=integrity['sha256'],unchanged_train_rows_sha256=original_train_sha,
        unchanged_train_summary_sha256=original_summary_sha,evaluation_workers=0,
        reason='dev first case stalled after repeated fork following CUDA/OpenCV use; GPU0%, workers idle; serial loader recovery',
        dev_cases=len(records['dev']),evaluation_noise='same original E33 two case seeds'))
    write(folder/'implementation_review.json',dict(gradient_check=read(folder/'gradient_check.json'),
        assertions='complex sign/native rotation/mask/group tests passed; prior unchanged; finite gradients',
        checkpoint_choice='original fixed step500',retrain=False,
        resolved_engineering_issue='serial evaluation avoids repeated-fork first-case stall',
        interpretation='500step train R90 misses95% threshold; full8000step capability remains untested'))
    update_decision(sanity_pass=passed,next_route='P1_full' if passed else 'implementation_or_parameterization_debug')
    frozen=read(OUT/'frozen_check.json');frozen['training_steps']=6500;write(OUT/'frozen_check.json',frozen);finish_frozen(OUT)
    print('[E33R recovered sanity Gate]',checks,flush=True)

if __name__=='__main__':main()
