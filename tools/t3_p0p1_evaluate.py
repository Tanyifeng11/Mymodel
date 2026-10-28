"""统一P2b评价：原参考六指标；换参考只报告响应和结构/背景代理。"""
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tools.t3_p0p1 import PROJECT,ROOT,SOURCE,SUP,read,write,sha,csv_write


def output_path(arm,row):
    if arm in ['B','T1','T2','T3','C']:
        stage='P2b' if row['condition']=='original' else 'reference_swap'
        return SUP/'evaluation'/stage/arm/'images'/(row['id']+'.png')
    return ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')


@torch.inference_mode()
def evaluate(arm):
    from eval.eval_utils import prepare_evaluation_masks,json_safe
    from eval.metrics import compute_structure_preservation,compute_texture_leakage,compute_texture_color_fidelity
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    # 历史helper只向新根写资产审计；历史协议和结果不改写。
    response.ROOT=ROOT
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    rows=read(ROOT/'protocol/input_manifest.json')
    if arm in ['E0','E1','LOTS']:
        rows=[r for r in rows if r['condition']=='original']
    old_protocol=read(SUP/'evaluation/metric_protocol.json')
    assert all(sha(SOURCE/path)==digest for path,digest in old_protocol['evaluation_code'].items())
    if arm!='LOTS':
        dists=response.load_dists('cuda')
        # LOTS无参考输入，不加载或计算参考Margin；其他方法先回放历史DISTS。
        write(ROOT/'audit'/('DISTS_replay_'+arm+'.json'),response.historical_device_check(dists))
    lpips_model,lpips_assets=frozen_lpips()
    by_id={row['id']:row for row in rows if row['condition']=='original'}
    swapped={row['id']:row for row in rows if row['condition']=='swapped'}
    output=[]
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1).copy()/127.5-1)[None].cuda()
    for row in rows:
        target=ROOT/'metrics'/arm/row['condition']/(row['id']+'.json')
        path=output_path(arm,row)
        if not path.is_file():
            record=dict(id=row['id'],arm=arm,condition=row['condition'],valid_generation=False,failure='missing output')
            write(target,record);output.append(record);continue
        if target.is_file():
            saved=read(target)
            if saved.get('output_sha256')==sha(path):
                output.append(saved);continue
        bundle=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
        kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=bundle)
        structure=compute_structure_preservation(**kwargs)
        leak=compute_texture_leakage(**kwargs)
        record=dict(id=row['id'],arm=arm,condition=row['condition'],valid_generation=True,output_sha256=sha(path),
            edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),edge_l1=structure.get('edge_l1'),
            leak=leak.get('leak_colored_frac'),leak_is_legacy_background_proxy=True,
            lpips=None,tcf_lab=None,dists_margin=None,color_margin=None,
            input_reference_sha256=None if arm=='LOTS' else row['reference_sha256'],input_sketch_sha256=row['sketch_sha256'])
        if arm=='LOTS':record.update(reference_texture_used=False,reference_margin_eligible=False,tcf_not_applicable='no reference texture input')
        if row['condition']=='original':
            gt=Image.open(row['gt']).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
            record['lpips']=float(lpips_model(tensor(Image.open(path).convert('RGB')),tensor(gt)).mean())
            if arm!='LOTS':record['tcf_lab']=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs).get('tcf_lab_delta')
        if arm!='LOTS' and row['id'] in swapped:
            old=by_id[row['id']]['reference'];new=swapped[row['id']]['reference']
            old_color=compute_texture_color_fidelity(texture_path=old,**kwargs).get('tcf_lab_delta')
            new_color=compute_texture_color_fidelity(texture_path=new,**kwargs).get('tcf_lab_delta')
            record['color_margin']=old_color-new_color if old_color is not None and new_color is not None else None
            coords,candidates=patches(bundle['garment'])
            record['patch_count']=len(coords)
            if len(coords)>=4:
                feature=response.features(dists,response.array(path),coords)
                old_distance=response.patch_distance(dists,feature,response.reference_features(dists,old))
                new_distance=response.patch_distance(dists,feature,response.reference_features(dists,new))
                record.update(dists_old=old_distance,dists_new=new_distance,dists_margin=old_distance-new_distance)
            else:
                record['dists_invalid_reason']='fewer than four fixed sketch interior patches'
        write(target,json_safe(record));output.append(json_safe(record))
        print('METRIC',arm,row['id'],row['condition'],len(output),'/',len(rows),flush=True)
    csv_write(ROOT/'metrics'/arm/'per_image.csv',output)
    write(ROOT/'metrics'/arm/'complete.json',dict(n=len(output),planned_n=len(rows),
        missing_generation=sum(not row['valid_generation'] for row in output),lpips_assets=lpips_assets,
        source_metric_protocol_sha256=sha(SUP/'evaluation/metric_protocol.json'),complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',nargs='+');args=parser.parse_args()
    for arm in args.arm:
        evaluate(arm)
