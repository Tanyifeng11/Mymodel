"""复用冻结候选，独立审计R180方向；不读取频率作为Gate。"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import cv2
import numpy as np
from PIL import Image
from data.e32_target_pseudogt import image_at
from data.e33_interventions import nuisance,central_valid_box
from models.local_pattern_field import patch_geometry
from models.pattern_geometry import axial_distance
from tools.e33r_common import *

def measure(rgb,k,seed=None):
    base=rgb;changed=np.rot90(rgb,k).copy();support=np.ones(rgb.shape[:2],bool)
    if seed is not None:
        base,_=nuisance(base,support,seed);changed,support=nuisance(changed,support,seed)
    lo,hi=central_valid_box(support)
    a=patch_geometry(Image.fromarray(base[lo:hi,lo:hi]));b=patch_geometry(Image.fromarray(changed[lo:hi,lo:hi]))
    error=axial_distance(b['orientation'],a['orientation']+90*k)
    return dict(error_deg=float(error),valid=bool(error<=15 and min(a['confidence'],b['confidence'])>=.25),
                before=a,after=b,support_box=[lo,lo,hi,hi])

def worker(item):
    row,group=item;cv2.setNumThreads(1)
    path=OUT/'R0_integrity/cases'/(row['id']+'.json')
    if path.exists():return read(path)
    rgb=np.asarray(image_at(DATASET/row['target']).crop(tuple(row['box'])))
    values={arm:dict(clean=measure(rgb,k),noisy=[measure(rgb,k,audit_seed(row['id'],j)) for j in range(2)])
            for arm,k in [('rot90',1),('rot180',2)]}
    result=dict(id=row['id'],group=group,strict=row['strict'],box=row['box'],source_E33_case_sha256=row['E33_case_sha256'],arms=values)
    write(path,result);return result

def main():
    chosen=prepare_manifest();tasks=[(r,g) for g in GROUPS for r in chosen[g]];rows=[]
    with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,row in enumerate(pool.map(worker,tasks,chunksize=8),1):
            rows.append(row)
            if i%256==0 or i==len(tasks):print('[E33R integrity]',i,'/',len(tasks),flush=True)
    summaries={}
    for group in GROUPS:
        summaries[group]={}
        for subset in ('all','strict'):
            cases=[r for r in rows if r['group']==group and (subset=='all' or r['strict'])]
            summaries[group][subset]={arm:dict(denominator=len(cases),
                 clean=bootstrap([float(r['arms'][arm]['clean']['valid']) for r in cases]),
                 noisy=bootstrap([float(all(x['valid'] for x in r['arms'][arm]['noisy'])) for r in cases]),
                 clean_error=bootstrap([r['arms'][arm]['clean']['error_deg'] for r in cases])) for arm in ('rot90','rot180')}
    checks={g+'/'+setting:summaries[g]['all']['rot180'][setting]['mean']>=threshold
            for g in ('train','dev') for setting,threshold in [('clean',.95),('noisy',.90)]}
    passed=all(checks.values())
    write(OUT/'R0_integrity/summary.json',dict(groups=summaries,checks=checks,**{'pass':passed},
         R180_training_enabled=passed,rotation_only=True,period_gate=False))
    update_decision(r180_integrity_pass=passed,next_route='P0_prior');finish_frozen(OUT)
    print('[E33R integrity decision]',checks,'R180_train',passed,flush=True)

if __name__=='__main__':main()
