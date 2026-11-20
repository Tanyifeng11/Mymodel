"""固定背景恢复与衣内回混，在融合后用真实成衣监督 A 的末步残差。"""
import argparse
import copy
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from tools import e2_f68_e1_a_roi_ablation as ab
from tools import e2_f68_region_diagnostic as roi
from tools import e2_f68_tail6_e1_epoch10 as epoch
from tools import e2_f68_tail6_real_ab as tail
from tools.e2_f68_rgb_student_roi import pixels_mixed


RUN = 'a_fused25_continue_20261120'
ARM = 'fused'
COEFFICIENT = .25
DONOR_SHARDS = 8
ORIGINAL_SOURCE_SHA = 'dfb43ae4549484e62c223f1f925d035a90b3fc834ff6caa02e2575b9f42e5d8a'


def alpha(mask):
    interior,_,background=(roi.gaussian(zone.astype('float32'),2) for zone in roi.regions(mask,8))
    return background+COEFFICIENT*interior


def integer_st(value):
    return value+(torch.round(value)-value).detach()


def fuse_training(prediction, donor, mask):
    anchor=torch.from_numpy(np.ascontiguousarray(donor.transpose(2,0,1)))[None].to(
        device=prediction.device,dtype=prediction.dtype)
    weight=torch.as_tensor(alpha(mask),device=prediction.device,dtype=prediction.dtype)[None,None]
    image=integer_st(prediction*255)
    return integer_st(image+weight*(anchor-image))/255


def prepare(args):
    # 使用同一 A、训练划分、真实图及旧基线核验，再固定本轮融合合同。
    with tempfile.TemporaryDirectory(prefix='parent_protocol_',dir=args.run) as temp:
        context=copy.copy(args)
        context.run=Path(temp)
        ab.prepare(context)
        contract=tail.read(context.run/'protocol.json')
    contract.update(ablation_code=contract['code'],code=tail.sha(Path(__file__)),
                    mode='real target on final 8-bit fused image',blend_interior=COEFFICIENT,
                    edge_radius=8,feather_sigma=2,fixed_donor_shards=DONOR_SHARDS,
                    training_arm=ARM,expanded_needs_lpips_and_fid_better_than_offline25=True)
    tail.write(args.run/'protocol.json',contract)
    print('FUSED25_PREPARED',len(contract['prefix']),contract['start_A_sha256'],flush=True)


def donors_dir(args, case):
    return args.run/'training_donors'/case


def save_png(path, array):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    Image.fromarray(array).save(temp,format='PNG')
    temp.replace(path)


@torch.no_grad()
def donors(args):
    core,pipe,net,original,frozen=ab.load(args)
    pairs=args.trainval['pairs'][args.shard::DONOR_SHARDS]
    hashes={}
    for count,pair in enumerate(pairs,start=1):
        case=pair['case_id']
        folder=donors_dir(args,case)
        marker=folder/'donor.json'
        output=folder/'blend.png'
        if marker.is_file():
            old=tail.read(marker)
            assert old['base_sha256']==args.contract['parent_contract']['base_sha256']
            assert old['png_sha256']==tail.sha(output) and old['case']==case
        else:
            mask=tail.case_mask(args,pair)
            records,images={},{}
            for name in ('off','on'):
                path=folder/(name+'.png')
                info=path.with_suffix('.json')
                if info.is_file():
                    records[name]=tail.read(info)
                    assert records[name]['png_sha256']==tail.sha(path)
                    images[name]=np.asarray(tail.rgb(path))
                    continue
                picture,noise,tap=tail.joint.sample(args,core,pipe,net,pair,tail.CONDITION,
                    b=original if name=='on' else None,mode='raw',capture=True)
                assert picture.size==tail.SIZE and tap.index==tail.LAST and len(tap.states)==25
                first={k:tail.tensor_sha(tap.states[0][k]) for k in ('zt','epsilon')}
                images[name]=np.asarray(picture)
                save_png(path,images[name])
                records[name]=dict(png_sha256=tail.sha(path),initial_noise_digest=noise,first_b_step=first)
                tail.write(info,records[name])
            assert all(records['off'][key]==records['on'][key] for key in ('initial_noise_digest','first_b_step'))
            blend=tail.mix_pixels(images['off'],images['on'],mask,{'blend':.25})['blend']
            save_png(output,blend)
            tail.write(marker,dict(case=case,seed=pair['seed'],png_sha256=tail.sha(output),
                                   base_sha256=args.contract['parent_contract']['base_sha256'],sources=records))
        hashes[case]=tail.sha(output)
        print('FUSED25_DONOR',args.shard,count,len(pairs),case,flush=True)
    assert tail.tensor_set(frozen)==args.original['frozen_model_hashes']
    tail.write(args.run/f'donor_{args.shard}.json',dict(cases=hashes,frozen=True,contract=args.contract))


def donor_hashes(args):
    covered={}
    for i in range(DONOR_SHARDS):
        row=tail.read(args.run/f'donor_{i}.json')
        assert row['contract']==args.contract and row['frozen'] and not (covered.keys()&row['cases'].keys())
        covered.update(row['cases'])
    assert set(covered)==set(args.contract['prefix']) and len(covered)==160
    return covered


def cached_donor(args, case):
    png=donors_dir(args,case)/'blend.png'
    assert tail.sha(png)==args.donor_hashes[case]
    return np.asarray(tail.rgb(png))


def qualify(args):
    args.donor_hashes=donor_hashes(args)
    models=ab.load(args)
    head,params=ab.head_from_a(args,models[3],True)
    pair=next(p for p in args.trainval['pairs'] if p['split']=='val')
    ref=tail.read(args.parent/'A/training.json')['checkpoints']['128']['rows'][0]
    assert ref['case']==pair['case_id']
    pred,gt,mask=ab.render(args,models,head,pair)
    criterion=ab.Objective(args)
    with torch.no_grad():
        _,terms=criterion(pred,gt,mask,'inside')
        for key in ('lpips','rgb_l1'):
            assert abs(float(terms[key])-ref[key])<5e-6
    donor=cached_donor(args,pair['case_id'])
    fused=fuse_training(pred,donor,mask)
    image=np.rint(pred.detach()[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')
    expected=pixels_mixed(image,donor,alpha(mask))
    actual=np.rint(fused.detach()[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')
    assert np.array_equal(actual,expected)
    loss,terms=criterion(fused,gt,mask,'inside')
    norm=tail.backward(loss,params)
    assert norm>0
    ab.audit_frozen(args,models,head,params)
    tail.write(args.run/'qualification.json',dict(case=pair['case_id'],start_A_match=True,
        pixel_fusion_exact=True,grad_norm=norm,donors=len(args.donor_hashes),contract=args.contract))
    print('FUSED25_QUALIFIED',pair['case_id'],norm,flush=True)


def train(args):
    args.arm=ARM
    args.donor_hashes=donor_hashes(args)
    render=ab.render
    def fused_render(context,models,head,pair):
        prediction,gt,mask=render(context,models,head,pair)
        donor=cached_donor(context,pair['case_id'])
        return fuse_training(prediction,donor,mask),gt,mask
    with patch.object(ab,'render',fused_render), patch.object(ab,'OPTIONS',{ARM:(False,True)}):
        ab.train(args)


def select(args):
    log=tail.read(args.run/ARM/'training.json')
    assert log['contract']==args.contract and log['updates']==ab.STEPS and log['frozen_unchanged']
    values={k:v['inside_lpips'] for k,v in log['checkpoints'].items()}
    step=min(values,key=lambda k:(values[k],-int(k)))
    tail.write(args.run/'selection.json',dict(step=int(step),validation_inside_lpips=values,
        validation_global_lpips={k:v['lpips'] for k,v in log['checkpoints'].items()},
        checkpoints={ARM:log['checkpoints'][step]['sha256']},contract=args.contract))
    print('FUSED25_SELECTION',step,values,flush=True)


def generate(args):
    with patch.object(ab,'ARMS',(ARM,)):
        ab.generate(args)


def score(args):
    read_image=ab.scoring_image
    original_gate=epoch.gate_check
    small=set(tail.subset(args.protocol))
    old={row['index']:row for row in tail.read(args.parent/'eval64/per_pair.json')}

    def read_fused(context,arm,case,rows,index):
        path,image=read_image(context,arm,case,rows,index)
        if arm!=ARM:
            return path,image
        pair=context.protocol['pairs'][index]
        mask=tail.case_mask(context,pair)
        if index in small:
            donor_path=context.parent/'test/blend'/(case+'.png')
            assert old[index]['case_id']==case and tail.sha(donor_path)==old[index]['pngs']['blend']
            donor=np.asarray(tail.rgb(donor_path))
        else:
            off,on=(np.asarray(tail.rgb(ab.png_path(context,name,case))) for name in ('off','on'))
            donor=tail.mix_pixels(off,on,mask,{'blend':.25})['blend']
        fused=pixels_mixed(image,donor,alpha(mask))
        result=context.run/'images'/('eval64' if index in small else 'additional436')/(case+'.png')
        save_png(result,fused)
        rows[index]['raw_A_sha256']=rows[index]['pngs'][ARM]
        rows[index]['pngs'][ARM]=tail.sha(result)
        return result,fused

    def incremental_gate(result,arm):
        checks=original_gate(result,arm)
        scores=result['groups']['all']['arms']
        deltas={m:scores[arm][m][k]-scores['offline25'][m][k]
                for m,k in (('lpips','mean'),('fid','fid'),('ssim','mean'),('contour_f1','mean'))}
        checks['delta_vs_offline25']=deltas
        checks['better_lpips_and_fid']=deltas['lpips']<0 and deltas['fid']<0
        checks['passed']=checks['passed'] and checks['better_lpips_and_fid']
        return checks

    with patch.object(ab,'ARMS',(ARM,)), patch.object(ab,'scoring_image',read_fused):
        with patch.object(epoch,'gate_check',incremental_gate):
            ab.score(args)


def main():
    actions=dict(prepare=prepare,donors=donors,qualify=qualify,train=train,select=select,
                 generate64=generate,generate500=generate,score64=score,score500=score)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=actions)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=4)
    args=parser.parse_args()
    args.project=args.prior.parents[1]
    args.data=args.project.parent/'datasets/BF'
    args.checkpoint=args.project/'output_eval/e2_state_b_pastel_followup_20261115/trust_retry/training/step1.pt'
    args.base_checkpoint=args.project/epoch.BASE_CHECKPOINT
    args.baseline=args.prior/tail.distill.BASE
    args.out=args.baseline/tail.distill.RUN
    args.parent=args.baseline/epoch.RUN
    args.run=args.parent/RUN
    args.run.mkdir(exist_ok=True)
    args.protocol=tail.verify(argparse.Namespace(out=args.baseline))
    args.trainval=tail.distill.experiment_protocol(args)
    args.m0_result=tail.read(args.baseline/'result.json')
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    if args.action!='prepare':
        args.contract=tail.read(args.run/'protocol.json')
        # 原版本只有评分入口的上下文管理器有误。保留已校验的缓存和训练合同。
        assert args.contract['code'] in (tail.sha(Path(__file__)),ORIGINAL_SOURCE_SHA)
        assert args.contract['ablation_code']==tail.sha(Path(ab.__file__))
    actions[args.action](args)


if __name__=='__main__':
    main()
