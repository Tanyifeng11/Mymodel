"""CPU 准备：重新审计候选、冻结身份、缓存历史训练 mask。"""
import argparse
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import subprocess
import tarfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tools.e2a_attr_common import *


def mask_function(source=None):
    text=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8') if source is None else source
    tree=ast.parse(text); node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sketch_to_garment_mask')
    namespace=dict(np=np,Image=Image,ImageFilter=ImageFilter,deque=deque)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-mask>','exec'),namespace)
    return namespace['sketch_to_garment_mask']


def stable(value): return hashlib.sha256(('E2A_ATTR_219042/'+str(value)).encode()).hexdigest()


def all_strings(value):
    if isinstance(value,str): yield value
    elif isinstance(value,dict):
        for item in value.values(): yield from all_strings(item)
    elif isinstance(value,list):
        for item in value: yield from all_strings(item)


def train_paths(rows):
    root=PROJECT.parent/'datasets/BF/training'
    return sorted({str(root/row[k]) for row in rows for k in ['cloth','sketch','texture']})


def prepare(workers):
    freeze(OUT/'config.json',CONFIG)
    if (OUT/'audit/inputs_frozen.json').exists(): return
    historical=read(PROJECT/'wandb/run-20260617_225130-ee7e5n00/files/wandb-metadata.json')
    freeze(OUT/'manifests/historical_command.json',historical['args'])
    dataset=read(TRAIN); assert len(dataset)==45384 and not any(r.get('mask') for r in dataset)
    old_audit=read(HISTORY/'holdout_audit/complete.json')
    assert old_audit['input_sha256']==sha(TRAIN),'训练清单与既有审计不一致'
    cache_path=Path(old_audit['training_signatures']['path'])
    if not cache_path.is_absolute(): cache_path=PROJECT/cache_path
    assert sha(cache_path)==old_audit['training_signatures']['sha256']
    cache=read(cache_path); paths=train_paths(dataset)
    assert set(paths)<=set(cache)
    # 重新核验训练字节，避免使用已过期的签名缓存。
    checked=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p,digest in pool.map(lambda p:(p,sha(p)),paths):
            assert digest==cache[p]['file_sha256'],('TRAINING_IMAGE_CHANGED',p)
            checked+=1
            if checked%5000==0: print('TRAIN HASH',checked,len(paths),flush=True)
    exact={k:{cache[p][k] for p in paths} for k in ['file_sha256','pixel_sha256']}
    refpaths=sorted({str(PROJECT.parent/'datasets/BF/training'/r[k]) for r in dataset for k in ['cloth','texture']})
    matrix=np.array([list(bytes.fromhex(cache[p]['phash'])) for p in refpaths],np.uint8)
    lookup=np.array([bin(i).count('1') for i in range(256)],np.uint8)
    candidates=read(HISTORY/'holdout_audit/eligible.json')
    validation=read(HISTORY/'holdout_audit/validation_signatures.json')
    assert sha(HISTORY/'holdout_audit/validation_signatures.json')==old_audit['validation_signatures_sha256']
    identities={r['id'] for r in candidates}; exposed=set(); registry=[]
    sources=set((PROJECT/'eval/benchmarks').glob('*.json'))
    sources.discard(PROJECT/'eval/benchmarks/bf_validation_resampler_full.json')
    for pattern in ['*/splits/*.json','*/split_sources.json','*/references.json']:
        sources.update(p for p in (PROJECT/'output_eval').glob(pattern) if OUT not in p.parents)
    for path in sorted(sources):
        try: found={Path(v.replace('\\','/')).stem for v in all_strings(read(path))}&identities
        except (ValueError,OSError): continue
        exposed.update(found); registry.append(dict(path=str(path),sha256=sha(path),ids=sorted(found)))
    train_ids={Path(r['cloth']).stem for r in dataset}
    selected=[]; excluded=[]; selected_hashes=set(); selected_phashes=[]
    for row in sorted(candidates,key=lambda r:stable(r['id'])):
        identity=row['id']; reasons=[]
        if identity in train_ids: reasons.append('training_identity')
        if identity in exposed: reasons.append('previous_experiment_exposure')
        if reasons: excluded.append(dict(id=identity,reasons=reasons)); continue
        signatures=[validation[row[k]] for k in ['gt','sketch','reference']]
        for signature in signatures:
            if any(signature[k] in exact[k] for k in exact): reasons.append('training_exact')
            if signature['pixel_sha256'] in selected_hashes: reasons.append('selected_exact')
        probes=[np.array(list(bytes.fromhex(validation[row[k]]['phash'])),np.uint8) for k in ['gt','reference']]
        for probe in probes:
            if int(lookup[matrix^probe].sum(1).min())<=6: reasons.append('training_near')
            if selected_phashes and int(lookup[np.array(selected_phashes)^probe].sum(1).min())<=6:
                reasons.append('selected_near')
        if reasons: excluded.append(dict(id=identity,reasons=sorted(set(reasons)))); continue
        for signature in signatures:
            assert sha(signature['path'])==signature['file_sha256'],('VALIDATION_IMAGE_CHANGED',signature['path'])
            selected_hashes.add(signature['pixel_sha256'])
        selected_phashes.extend(probes); selected.append(row)
        if len(selected)==CONFIG['p1_n']+CONFIG['p2_n']+CONFIG['lock_max_n']: break
    assert len(selected)>=24+32+96,('INDEPENDENT_POOL_SHORTAGE',len(selected))
    freeze(OUT/'splits/P1.json',selected[:24]); freeze(OUT/'splits/P2.json',selected[24:56])
    freeze(OUT/'splits/LOCK_pool.json',selected[56:])
    freeze(OUT/'splits/LOCK_initial.json',selected[56:152])
    freeze(OUT/'audit/exposure_registry.json',registry); write(OUT/'audit/excluded_candidates.json',excluded)
    files={str(p):dict(sha256=sha(p),bytes=Path(p).stat().st_size) for p in [TRAIN,E1]}
    adapter=PROJECT/'output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin'
    files[str(adapter)]=dict(sha256=sha(adapter),bytes=adapter.stat().st_size)
    write(OUT/'checkpoint_manifest.json',files)
    write(OUT/'audit/inputs_frozen.json',dict(passed=True,training_records=len(dataset),verified_training_files=checked,
        P1=24,P2=32,LOCK_initial=96,LOCK_pool=len(selected)-56,code_commit=commit(),
        training_signature_source=str(cache_path),training_signature_sha256=sha(cache_path),
        exact='cross-modal file and RGB-pixel SHA256',near='GT/reference pHash Hamming<=6 conservative exclusion',
        source_pool=str(HISTORY/'holdout_audit/eligible.json'),source_pool_sha256=sha(HISTORY/'holdout_audit/eligible.json'),
        independent_from_training_manifest=True,pretrained_model_exposure_unknown=True))
    state(preparation='INPUTS_FROZEN_MASKS_PENDING')


_mask_function=None
def cache_mask(item):
    global _mask_function
    if _mask_function is None: _mask_function=mask_function()
    index,row=item; root=PROJECT.parent/'datasets/BF/training'
    with Image.open(root/row['cloth']) as cloth: cloth_size=cloth.size
    with Image.open(root/row['sketch']) as im: sketch=im.convert('RGB').resize(cloth_size)
    mask,info=_mask_function(sketch,384,512,return_info=True)
    eroded=mask.filter(ImageFilter.MinFilter(9)); dilated=mask.filter(ImageFilter.MaxFilter(9))
    body=np.array(eroded); boundary=np.array(dilated)-body; outside=255-np.array(dilated)
    base=OUT/'masks/train'/('%03d'%(index//1000)); base.mkdir(parents=True,exist_ok=True)
    for name,image in [('mask',mask),('body',eroded),('boundary',Image.fromarray(boundary)),('outside',Image.fromarray(outside))]:
        path=base/('%05d_%s.png'%(index,name))
        if path.exists():
            with Image.open(path) as old: assert np.array_equal(np.array(old),np.array(image))
        else: image.save(path)
    return dict(index=index,id=Path(row['cloth']).stem,**info,mask_area=float((np.array(mask)>0).mean()),
        body_pixels=int((body>0).sum()),boundary_pixels=int((boundary>0).sum()),outside_pixels=int((outside>0).sum()),
        mask_sha256=sha(base/('%05d_mask.png'%index)))


def review_panels(rows,indices):
    root=PROJECT.parent/'datasets/BF/training'
    for start in range(0,len(indices),20):
        canvas=Image.new('RGB',(4*256,5*210),'white'); draw=ImageDraw.Draw(canvas)
        for j,index in enumerate(indices[start:start+20]):
            row=rows[index]; x=(j%4)*256; y=(j//4)*210
            sketch=Image.open(root/row['sketch']).convert('RGB').resize((128,171))
            mask=np.array(Image.open(OUT/'masks/train'/('%03d'%(index//1000))/('%05d_mask.png'%index)).resize((128,171)))>0
            overlay=np.asarray(sketch).copy(); overlay[mask]=(overlay[mask]*.6+np.array([0,255,80])*.4).astype('uint8')
            canvas.paste(sketch,(x,y+25)); canvas.paste(Image.fromarray(overlay),(x+128,y+25))
            draw.text((x+2,y+4),'%05d %s'%(index,Path(row['cloth']).stem),fill='black')
        path=OUT/'audit/mask_review'/('page_%03d.png'%start); path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def masks(workers):
    if (OUT/'audit/masks_complete.json').exists(): return
    assert read(OUT/'audit/inputs_frozen.json')['passed']
    dataset=read(TRAIN)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records=[]
        for record in pool.map(cache_mask,enumerate(dataset),chunksize=32):
            records.append(record)
            if len(records)%1000==0: print('TRAIN MASK',len(records),len(dataset),flush=True)
    csv_write(OUT/'mask_audit.csv',records)
    indices=sorted(range(len(dataset)),key=stable)[:100]
    original=subprocess.check_output(['git','-C',str(SOURCE),'show',BASE+':train_GAM_texture_joint.py'],text=True)
    old_mask=mask_function(original); root=PROJECT.parent/'datasets/BF/training'
    for index in indices:
        row=dataset[index]
        with Image.open(root/row['cloth']) as cloth: size=cloth.size
        with Image.open(root/row['sketch']) as im: sketch=im.convert('RGB').resize(size)
        cached=Image.open(OUT/'masks/train'/('%03d'%(index//1000))/('%05d_mask.png'%index))
        assert np.array_equal(np.array(old_mask(sketch,384,512)),np.array(cached))
    review_panels(dataset,indices)
    write(OUT/'audit/masks_complete.json',dict(passed=True,n=len(records),exact_historical_comparisons=100,
        review_indices=indices,visual_review='PENDING',fallbacks=sum(r['fallback'] for r in records),
        training_mask_source_sha256=hashlib.sha256(ast.get_source_segment(original,
            next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='sketch_to_garment_mask')).encode()).hexdigest()))
    with tarfile.open(OUT/'mask_review.tar.gz','w:gz') as archive:
        for folder in ['audit','splits']:
            archive.add(OUT/folder,arcname=folder)
        archive.add(OUT/'mask_audit.csv',arcname='mask_audit.csv')
        archive.add(OUT/'config.json',arcname='config.json')
    state(preparation='MASK_VISUAL_REVIEW_PENDING')
    print('PREPARATION COMPLETE; VISUAL REVIEW REQUIRED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','masks','all'])
    parser.add_argument('--workers',type=int,default=8); args=parser.parse_args()
    if args.action in ['prepare','all']: prepare(args.workers)
    if args.action in ['masks','all']: masks(args.workers)
