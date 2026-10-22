import csv
from math import isfinite
import numpy as np
import torch
from PIL import Image
from cn_ipa.common import *


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


class Scorer:
    def __init__(self):
        from tools.e33tmoc_appearance_eval import frozen_lpips
        self.lpips,self.info=frozen_lpips();self.lpips=self.lpips.cuda()

    @torch.inference_mode()
    def score(self,row,path,arm,seed=42):
        from tools.m54_eval import regional_metrics
        from eval.eval_utils import prepare_evaluation_masks
        image=Image.open(path).convert('RGB')
        gt=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
        result=dict(id=row['id'],variant=arm,seed=seed,**{k:None for k in KEYS})
        result['lpips_gt']=float(self.lpips(tensor(image),tensor(gt)).mean())
        bundle=prepare_evaluation_masks(SIZE,sketch_path=row['sketch'],mask_policy='sketch_only')
        garment=bundle['garment']
        if garment is not None and garment.any():
            result.update(regional_metrics(image,path,row['gt'],bundle,True))
        else:result['na_reason']='original_sketch_only_evaluation_mask_unavailable'
        for key in KEYS:
            if result[key] is not None and not isfinite(result[key]):result[key]=None
        return result


def means(rows):
    return {key:float(np.mean([r[key] for r in rows if r.get(key) is not None]))
        if any(r.get(key) is not None for r in rows) else None for key in KEYS}


def selection_key(scores,baseline,step=0):
    assert all(scores[k] is not None and baseline[k] is not None for k in KEYS)
    limits=CONFIG['scale_selection']
    violations=[max(0.,baseline['contour_f1']+limits['f1_delta_min']-scores['contour_f1'])/.02,
        max(0.,baseline['iou']+limits['iou_delta_min']-scores['iou'])/.01,
        max(0.,scores['leak']-baseline['leak']-limits['leak_delta_max'])/.005]
    return (int(any(x>0 for x in violations)),sum(violations),scores['lpips_gt'],step)


def bootstrap(values):
    a=np.asarray(values,np.float64)
    if not len(a):return dict(n=0,mean=None,ci95=[None,None])
    rng=np.random.default_rng(32042)
    sample=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(sample,[.025,.975]).tolist())


def statistics(records):
    ids=[r['id'] for r in read(ROOT/'manifests/test_A.json')]
    arms=['E5','B1','M10','M10_GAIN','M12','M12_GAIN','C0','C1','C2']
    index={(r['variant'],r['id'],int(r['seed'])):r for r in records}
    aggregated=[]
    for arm in arms:
        for identity in ids:
            samples=[index[arm,identity,s] for s in CONFIG['seeds']]
            aggregated.append(dict(variant=arm,id=identity,**{key:float(np.mean([r[key] for r in samples]))
                if all(r.get(key) is not None for r in samples) else None for key in KEYS}))
    indexed={(r['variant'],r['id']):r for r in aggregated}
    average={arm:{key:bootstrap([r[key] for r in aggregated if r['variant']==arm and r[key] is not None])
        for key in KEYS} for arm in arms}
    comparisons=[(a,b) for a in ['C0','C1','C2'] for b in ['M12','M12_GAIN']]+[('C1','C0'),('C2','C1')]
    paired=[];sensitivity=[]
    for a,b in comparisons:
        for key in KEYS:
            values=[indexed[a,i][key]-indexed[b,i][key] for i in ids
                if indexed[a,i][key] is not None and indexed[b,i][key] is not None]
            paired.append(dict(candidate=a,baseline=b,metric=key,**bootstrap(values)))
            if len(values)>1:
                leave=[(sum(values)-x)/(len(values)-1) for x in values]
                sensitivity.append(dict(candidate=a,baseline=b,metric=key,min=min(leave),max=max(leave)))
    extreme=[]
    for arm in arms:
        for identity in ids:
            deltas=[index[arm,identity,s]['contour_f1']-index['E5',identity,s]['contour_f1']
                for s in CONFIG['seeds'] if index[arm,identity,s].get('contour_f1') is not None
                and index['E5',identity,s].get('contour_f1') is not None]
            if deltas and min(deltas)<-.15:extreme.append(dict(variant=arm,id=identity,worst_delta=min(deltas)))
    per_seed={str(s):{arm:means([r for r in records if r['variant']==arm and int(r['seed'])==s]) for arm in arms}
        for s in CONFIG['seeds']}
    write(ROOT/'metrics/means.json',average);write(ROOT/'metrics/per_seed.json',per_seed)
    for name,rows in [('per_identity',aggregated),('paired_ci',paired),('failure_ids',extreme),('leave_one_out',sensitivity)]:
        write(ROOT/'metrics'/f'{name}.json',rows);csv_write(ROOT/'metrics'/f'{name}.csv',rows)
    return average
