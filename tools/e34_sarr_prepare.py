"""CPU 前置审计：固定原验证集，hash 选 train-only dev，不改变任何源文件。"""
import hashlib
import platform
from pathlib import Path
import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from models.e34_sarr_refiner import safe_mask,SARR
from tools.e34_sarr_protocol import *

def run():
    init();seed_all(42);cv2.setNumThreads(1);torch.set_num_threads(2)
    original=read(TRAIN);validation=read(VALIDATION)
    held={Path(r['target']).stem for r in validation}
    assert len(held)==len(validation), '现有验证身份重复，请先核对'
    pool=[];seen=set();duplicates=[]
    for r in original:
        sid=Path(r['cloth']).stem
        if sid in seen:duplicates.append(sid);continue
        seen.add(sid)
        if sid in held:continue
        pool.append(dict(id=sid,gt=str(BF/r['cloth']),sketch=str(BF/r['sketch']),
            reference=str(BF/r.get('texture',r.get('color'))),caption=r['caption']))
    pool.sort(key=lambda r:hashlib.sha256((CONFIG['split_hash_prefix']+r['id']).encode()).hexdigest())
    dev=pool[:128];train=pool[128:];pilot=train[:256]
    assert len(dev)==128 and len(pilot)==256
    assert not ({r['id'] for r in train}&{r['id'] for r in dev}|{r['id'] for r in pool}&held)
    write(OUT/'splits/dev128.json',dev);write(OUT/'splits/pilot256.json',pilot)
    write(OUT/'splits/train.json',train);write(OUT/'splits/validation_original.json',validation)
    frozen={str(p):sha(p) for p in [TRAIN,VALIDATION,E5]}
    for r in dev+pilot:
        for key in ['gt','sketch','reference']:frozen[r[key]]=sha(r[key])
    write(OUT/'frozen_hashes.json',frozen)
    write(OUT/'dataset_audit.json',dict(original_manifest_n=len(original),original_unique_n=len(seen),
        duplicate_ids=duplicates,validation_n=len(validation),validation_overlap_excluded=len(seen&held),
        train_n=len(train),dev_n=len(dev),pilot_n=len(pilot),split_disjoint=True,
        files_modified=False,new_pairs=False,new_annotations=False,
        limitation='identity-disjoint; no original product-group labels; historical E5 exposure not excluded',
        split_sha256={str(p):sha(p) for p in (OUT/'splits').glob('*.json')}))
    audit=[]
    for r in dev:
        sketch=Image.open(r['sketch']).convert('RGB')
        m0,safe,line,info=safe_mask(sketch)
        dest=OUT/'masks'/r['id'];dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'mask.npz',foreground=m0,safe=safe,line=line)
        Image.fromarray(np.round(safe*255).astype(np.uint8)).save(dest/'M_safe.png')
        audit.append(dict(id=r['id'],**info))
    write(OUT/'masks_audit.json',dict(records=audit,n=len(dev),valid_n=sum(r['valid'] for r in audit),
        valid_rate=sum(r['valid'] for r in audit)/len(dev),gt_mask_calls=0,
        coverage_quantiles=np.quantile([r['image_coverage'] for r in audit],[0,.1,.25,.5,.75,.9,1]).tolist(),
        backend='opencv',cv2_version=cv2.__version__,erosion_distance_px=17))
    model=SARR().eval();x=torch.rand(1,3,64,48);line=torch.rand(1,1,64,48);m=torch.rand_like(line)
    with torch.no_grad():
        a=model(x,line,m,torch.rand(1,3,128,128));b=model(x,line,m,torch.rand(1,3,128,128))
        assert torch.equal(a['output'],x) and torch.equal(b['output'],x)
        assert a['q'].shape==(1,128) and not torch.equal(a['q'],b['q'])
        model.delta.bias.fill_(.7)
        assert torch.equal(model(x,line,m*0,torch.rand(1,3,128,128))['output'],x)
        hard=(m>.5).float();out=model(x,line,hard,torch.rand(1,3,128,128))['output']
        assert torch.equal(out*(1-hard),x*(1-hard))
    write(OUT/'engineering_checks.json',dict(step0_exact=True,zero_mask_exact=True,outside_exact=True,
        q_shape=[1,128],different_reference_changes_q=True,
        parameters=sum(p.numel() for p in model.parameters()),no_ref_parameters=sum(p.numel() for p in SARR(True).parameters())))
    write(OUT/'environment.json',dict(python=platform.python_version(),opencv=cv2.__version__,pillow=PIL.__version__,
        numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,git_commit=commit()))
    passed=sum(r['valid'] for r in audit)/len(dev)>=.80
    decision(S1='pass' if passed else 'fail',S0='pending_generation',next_phase='S0',
        note='仍完成固定 dev128 的零训练诊断；未通过 S1 则禁止进入 S2')
    print('PREPARED',len(train),'DEV',len(dev),'MASK_VALID',sum(r['valid'] for r in audit),'PARAMS',sum(p.numel() for p in model.parameters()),flush=True)

if __name__=='__main__':run()
