"""仅复用本地已验收原件；冻结六臂配对与参考图标注输入，不调用生成器。"""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path('F:/fuxian/Mymodel')
ROOT = PROJECT/'output_eval/e2a_texture_eval_01'
NEW = PROJECT/'output_eval/e2a_3d2d_tex_01_20261026/development_review_20261027'
OLD = PROJECT/'output_eval/e2a_sup_comp_01_20261021/review_results_20261026'
ARMS = ['B', 'T1', 'T2', 'C', 'T3', 'T3M']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def fixed_subset(values, maximum):
    # 按行优先的候选网格等距取点，索引不依赖任何模型像素或指标。
    if len(values)<=maximum:return values
    return [values[i] for i in np.linspace(0,len(values)-1,maximum).round().astype(int)]

def prepare():
    assert not (ROOT/'input_manifest.csv').exists(), '禁止覆盖冻结清单'
    portable=read(NEW/'development_review_manifest.json')['portable_inputs']
    inputs={v['sha256']:NEW/v['member'] for v in portable.values()}
    original=read(NEW/'splits/P2b.json'); swap=read(NEW/'splits/reference_swap.json')
    lookup={r['id']:r for r in original}; records=[]; refs={}; pairs=[]; masks=[]; gen_coords={}
    complete={}; oldmetrics={a:{r['id']:r for r in read((OLD if a in ['T1','T2'] else NEW)/f'evaluation/P2b/{a}/per_identity_metrics.json')} for a in ARMS}
    for stage, rows in [('P2b',original),('reference_swap',swap)]:
        sets=[]
        for a in ARMS:
            base=OLD if a in ['T1','T2'] else NEW
            ids={p.stem for p in (base/f'evaluation/{stage}/{a}/images').glob('*.png')}
            sets.append(ids)
        common=set.intersection(*sets)&{r['id'] for r in rows}
        complete[stage]={'planned_n':len(rows),'common_n':len(common),'missing_by_arm':{a:sorted({r['id'] for r in rows}-s) for a,s in zip(ARMS,sets)}}
        for row in rows:
            identity=row['id']
            if identity not in common:continue
            ref_path=row['reference'] if stage=='P2b' else row['reference_alternative']
            ref_info=portable[ref_path]; reference=NEW/ref_info['member']; ref_sha=sha(reference)
            assert ref_sha==ref_info['sha256']
            refs[ref_sha]={'reference_sha256':ref_sha,'reference_path':str(reference),'source_path':ref_path,'reference_identity':Path(ref_path).stem}
            sketch=NEW/portable[row['sketch']]['member']; sketch_sha=sha(sketch)
            assert sketch_sha==portable[row['sketch']]['sha256']
            mask=NEW/f'masks/evaluation/{identity}/garment.png'; mask_sha=sha(mask)
            assert sha(OLD/f'masks/evaluation/{identity}/garment.png')==mask_sha
            if identity not in gen_coords:
                arr=np.asarray(Image.open(mask).convert('L'))>127
                assert arr.shape==(512,384)
                interior=cv2.erode(arr.astype('uint8'),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
                coords=[(x,y,float(interior[y:y+64,x:x+64].mean())) for y in range(0,449,32) for x in range(0,321,32) if interior[y:y+64,x:x+64].mean()>=.9]
                gen_coords[identity]=fixed_subset(coords,8)
                out=ROOT/f'masks/{identity}_interior.png';out.parent.mkdir(exist_ok=True);Image.fromarray(interior.astype('uint8')*255).save(out)
                flags={oldmetrics[a][identity]['mask_low_confidence'] for a in ARMS};assert len(flags)==1
                masks.append(dict(identity=identity,mask_path=str(mask),mask_sha256=mask_sha,interior_path=str(out),interior_sha256=sha(out),erosion='square11x11 (Chebyshev radius5)',mask_low_confidence=flags.pop(),candidate_patch_n=len(coords),selected_patch_n=len(gen_coords[identity]),insufficient_interior_patches=len(coords)<4,mask_policy='sketch_only',mask_source=oldmetrics['C'][identity]['mask_source']))
            paired=None
            for arm in ARMS:
                base=OLD if arm in ['T1','T2'] else NEW
                image=base/f'evaluation/{stage}/{arm}/images/{identity}.png';meta=read(image.with_suffix('.json'))
                assert sha(image)==meta['output_sha256']
                assert meta['inputs']['reference']==ref_sha and meta['inputs']['sketch']==sketch_sha
                assert meta['explicit_latent_verified'] and not meta['gt_used_in_inference']
                binding=(meta['seed'],meta['initial_latent_sha256'],meta['inputs']['gt'],sketch_sha,ref_sha)
                if paired is None:paired=binding
                assert paired==binding
                if stage=='reference_swap':
                    before=read(base/f'evaluation/P2b/{arm}/images/{identity}.json')
                    assert meta['seed']==before['seed'] and meta['initial_latent_sha256']==before['initial_latent_sha256']
                    assert {k:v for k,v in meta['inputs'].items() if k!='reference'}=={k:v for k,v in before['inputs'].items() if k!='reference'}
                assert Image.open(image).size==(384,512)
                records.append(dict(stage='original' if stage=='P2b' else 'swap',arm=arm,identity=identity,reference_path=str(reference),reference_sha256=ref_sha,output_path=str(image),output_sha256=meta['output_sha256'],sketch_path=str(sketch),sketch_sha256=sketch_sha,mask_path=str(mask),mask_sha256=mask_sha,seed=meta['seed'],initial_latent_sha256=meta['initial_latent_sha256'],sampler='DDIM',steps=50,CFG=7,sketch_scale=.6,texture_scale=1,ipa_scale=1,reference_preprocessing='BF: RGB direct384x512 LANCZOS; CLIP: default224 bicubic/center crop; metric uses BF RGB canvas',generation_commit=read(base/f'evaluation/{stage}/{arm}/generation_complete.json')['commit'],gt_used_in_inference=False))
            if stage=='reference_swap':
                old_sha=portable[row['reference']]['sha256'];pairs.append(dict(identity=identity,old_reference_sha256=old_sha,new_reference_sha256=ref_sha))
    csv_write(ROOT/'input_manifest.csv',records);csv_write(ROOT/'mask_manifest.csv',masks)
    refs=sorted(refs.values(),key=lambda r:r['reference_sha256'])
    for i,r in enumerate(refs,1):r['reference_index']=i
    csv_write(ROOT/'reference_only/index.csv',refs);csv_write(ROOT/'reference_pairs.csv',pairs)
    write(ROOT/'generation_patch_coordinates.json',gen_coords);write(ROOT/'input_acceptance.json',dict(passed=True,**complete,output_n=len(records),unique_reference_n=len(refs),inputs_rehashed=True,outputs_rehashed=True,source_masks_rehashed=True,old_artifacts_read_only=True))
    for page,start in enumerate(range(0,len(refs),24),1):
        canvas=Image.new('RGB',(6*212,4*240),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(refs[start:start+24]):
            x=j%6*212;y=j//6*240
            canvas.paste(Image.open(r['reference_path']).convert('RGB').resize((208,208),Image.Resampling.NEAREST),(x,y+28))
            draw.text((x+3,y+4),f"R{r['reference_index']:03} {r['reference_identity']}",fill='black')
        canvas.save(ROOT/f'reference_only/page_{page:02}.png')
    print(json.dumps(read(ROOT/'input_acceptance.json')),flush=True)

if __name__=='__main__':prepare()
