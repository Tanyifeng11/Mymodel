"""等预算复用融合续训 A，只给 B 加衣内低频 L* 损失。"""
import argparse
import copy
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models.e2_lab_anchor import rgb_to_lab
from tools import e2_f68_e1_a_fused25_continue as fused
from tools import e2_f68_e1_a_roi_ablation as ab
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail


RUN = 'a_fused25_lstar_ab_20261121'
PREPARED_CODE_SHA = '530dbd10575bf28d2406cd3774f4cc882b3b6c11dbfde9840b414b2ca7d0e362'
CONTROL_SHA = 'ed3a53e01201d10fa3c1b4cf0bc545d0b2b81045b6f92685a56471cb67efce1f'
ARM = 'B'
CONTROL = 'continued_A'
STEP = '64'
WEIGHT = .10
SIGMA, RADIUS = 8, 32
BaseObjective = ab.Objective
base_png_path = ab.png_path


class LightingObjective(BaseObjective):
    def __init__(self,args):
        super().__init__(args)
        self.kernel=torch.as_tensor(cv2.getGaussianKernel(2*RADIUS+1,SIGMA),
                                   device=args.device,dtype=torch.float32)

    def blur(self,x):
        y=F.conv2d(x,self.kernel.reshape(1,1,1,-1),padding=(0,RADIUS))
        return F.conv2d(y,self.kernel.reshape(1,1,-1,1),padding=(RADIUS,0))

    def __call__(self,pred,gt,mask,arm):
        loss,terms=super().__call__(pred,gt,mask,arm)
        part=torch.from_numpy(roi.regions(mask,8)[0]).to(device=pred.device,dtype=pred.dtype)[None,None]
        diff=(rgb_to_lab(pred)[:,:1]-rgb_to_lab(gt)[:,:1])/100
        average=self.blur(diff*part)/self.blur(part).clamp_min(1e-6)
        low=ab.weighted_mean(average.abs(),part)
        return loss+WEIGHT*low,dict(terms,low_lstar=low)


def prepare(args):
    review=args.control/'eval64'
    tail.verified_files(review,CONTROL_SHA,('result.json','per_pair.json','pool3.npz'))
    manifest=tail.read(review/'review_manifest.json')
    for name in ('protocol','qualification','selection'):
        assert tail.sha(args.control/f'{name}.json')==manifest[f'{name}.json']
    a_log=args.control/'fused/training.json'
    assert tail.sha(a_log)==manifest['training_fused.json']
    control=tail.read(args.control/'protocol.json')
    log=tail.read(a_log)
    qualified=tail.read(args.control/'qualification.json')
    selection=tail.read(args.control/'selection.json')
    assert log['contract']==selection['contract']==qualified['contract']==control
    assert qualified['start_A_match'] and qualified['pixel_fusion_exact'] and qualified['donors']==160
    assert log['updates']==control['updates']==ab.STEPS==128 and str(selection['step'])==STEP
    assert log['parameter_count']==tail.PARAMETERS and log['frozen_unchanged']
    assert control['lr']==ab.LEARNING_RATE and control['batch']==1 and control['seed']==42
    assert control['blend_interior']==fused.COEFFICIENT==.25 and control['edge_radius']==8 and control['feather_sigma']==2
    assert control['ablation_code']==tail.sha(Path(ab.__file__))
    assert tail.sha(args.parent/'A/step128.pt')==control['start_A_sha256']
    assert tail.sha(args.out/'protocol.json')==control['parent_contract']['training_protocol']
    assert tail.sha(args.baseline/'protocol.json')==control['parent_contract']['evaluation_protocol']
    assert all(tail.sha(args.control/'fused'/f'step{k}.pt')==v['sha256'] for k,v in log['checkpoints'].items())
    assert selection['checkpoints']['fused']==log['checkpoints'][STEP]['sha256']
    train=[p['case_id'] for p in args.trainval['pairs'] if p['split']=='train']
    valid=[p['case_id'] for p in args.trainval['pairs'] if p['split']=='val']
    assert (len(train),len(valid))==(128,32)
    assert [t['case'] for t in log['trace']]==[train[j] for j in np.random.default_rng(42).permutation(128)]
    assert [r['case'] for r in log['checkpoints'][STEP]['rows']]==valid
    donor_context=copy.copy(args)
    donor_context.run,donor_context.contract=args.control,control
    hashes=fused.donor_hashes(donor_context)
    for case,digest in hashes.items():
        assert tail.sha(args.control/'training_donors'/case/'blend.png')==digest
        assert tail.sha(args.parent/'prefix'/(case+'.pt'))==control['prefix'][case]
    contract=dict(control,code=tail.sha(Path(__file__)),fused_code=tail.sha(Path(fused.__file__)),
        compared_arm=CONTROL,selected_A_B_step=int(STEP),control_review_sha256=CONTROL_SHA,
        control_step_sha256=selection['checkpoints']['fused'],donor_hashes=hashes,
        low_lstar=dict(weight=WEIGHT,sigma_px=SIGMA,kernel_radius_px=RADIUS,region_erosion_px=8,scale=100))
    tail.write(args.run/'protocol.json',contract)
    print('LSTAR_PREPARED',len(hashes),selection['checkpoints']['fused'],flush=True)


def training_action(args, action):
    use_action=fused.qualify if action=='qualify' else fused.train
    def donor_path(context,case):
        return context.control/'training_donors'/case
    with patch.object(fused,'donor_hashes',return_value=args.contract['donor_hashes']), patch.object(fused,'donors_dir',donor_path):
        with patch.object(fused,'ARM',ARM), patch.object(ab,'Objective',LightingObjective):
            use_action(args)


def select(args):
    a=tail.read(args.control/'fused/training.json')
    b=tail.read(args.run/ARM/'training.json')
    assert b['contract']==args.contract and a['updates']==b['updates']==128 and b['frozen_unchanged']
    assert [r['case'] for r in a['trace']]==[r['case'] for r in b['trace']]
    for step in ('64','128'):
        assert [r['case'] for r in a['checkpoints'][step]['rows']]==[r['case'] for r in b['checkpoints'][step]['rows']]
    metrics={arm:{k:{m:v[m] for m in ('lpips','inside_lpips','low_lstar') if m in v}
                  for k,v in t['checkpoints'].items()} for arm,t in ((CONTROL,a),(ARM,b))}
    tail.write(args.run/'selection.json',dict(step=int(STEP),validation=metrics,
        checkpoints={ARM:b['checkpoints'][STEP]['sha256']},control_checkpoint_sha256=args.contract['control_step_sha256'],
        contract=args.contract))
    print('LSTAR_MATCHED',STEP,metrics,flush=True)


def candidate_png_path(args, name, case):
    if name == ARM:
        return args.run/'test'/name/(case+'.png')
    return base_png_path(args,name,case)


def generate(args):
    with patch.object(fused,'ARM',ARM):
        fused.generate(args)


def score(args):
    with patch.object(fused,'ARM',ARM):
        if args.action=='score500':
            fused.score(args)
            return
        controls=ab.CONTROLS
        existing=ab.control64
        def additional_control(context,rows,features):
            with patch.object(ab,'CONTROLS',controls):
                existing(context,rows,features)
            saved=tail.read(context.control/'eval64/per_pair.json')
            ids=tail.subset(context.protocol)
            assert [r['index'] for r in saved]==ids
            with np.load(context.control/'eval64/pool3.npz',allow_pickle=False) as f:
                vectors=f['fused']
            for pos,src in enumerate(saved):
                index=src['index']
                row=rows[index]
                assert row['case_id']==src['case_id'] and row['scores']['offline25']==src['scores']['offline25']
                row['scores'][CONTROL]=src['scores']['fused']
                row['texture'][CONTROL]=src['texture']['fused']
                row['pngs'][CONTROL]=src['pngs']['fused']
                features[CONTROL][index]=vectors[pos]
        with patch.object(ab,'CONTROLS',(*controls,CONTROL)), patch.object(ab,'control64',additional_control):
            fused.score(args)


def main():
    actions=dict(prepare=prepare,qualify=lambda args:training_action(args,'qualify'),
                 train=lambda args:training_action(args,'train'),select=select,
                 generate64=generate,generate500=generate,score64=score,score500=score)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    args.project=args.prior.parents[1]
    args.data=args.project.parent/'datasets/BF'
    args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint=args.project/epoch.BASE_CHECKPOINT
    args.baseline=args.prior/tail.distill.BASE
    args.out=args.baseline/tail.distill.RUN
    args.parent=args.baseline/epoch.RUN
    args.control=args.parent/fused.RUN
    args.run=args.parent/RUN
    args.run.mkdir(exist_ok=True)
    args.protocol=tail.verify(argparse.Namespace(out=args.baseline))
    args.trainval=tail.distill.experiment_protocol(args)
    args.m0_result=tail.read(args.baseline/'result.json')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    if args.action!='prepare':
        args.contract=tail.read(args.run/'protocol.json')
        # The initial protocol predates the candidate image path isolation.
        assert args.contract['code'] in (tail.sha(Path(__file__)), PREPARED_CODE_SHA)
        assert args.contract['fused_code']==tail.sha(Path(fused.__file__))
    with patch.object(ab,'png_path',candidate_png_path):
        actions[args.action](args)


if __name__=='__main__':
    main()
