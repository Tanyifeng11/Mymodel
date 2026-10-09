"""只准备train-only目标和Smoke身份；验证集冲突记录，不擅自重划分。"""
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from data.e33gc_renderer import construct,fft_tangent
from tools.e33gc_protocol import *

def order(rows,label):
    return sorted(rows,key=lambda r:hashlib.sha256(('E33GC/'+label+'/'+r['id']).encode()).hexdigest())

def fingerprint(path):
    rgb=np.asarray(Image.open(path).convert('RGB').resize((64,64),Image.Resampling.BICUBIC),np.int16)
    gray=np.asarray(Image.open(path).convert('L').resize((9,8),Image.Resampling.BICUBIC))
    bits=gray[:,1:]>gray[:,:-1]
    return dict(sha256=sha(path),dhash=int.from_bytes(np.packbits(bits).tobytes(),'big'),rgb=rgb)

def duplicate(candidate,held):
    for sid,path,value in held:
        if candidate['sha256']==value['sha256']:return dict(identity=sid,path=path,reason='exact SHA256')
        distance=bin(candidate['dhash']^value['dhash']).count('1')
        if distance<=4:
            mae=float(np.abs(candidate['rgb']-value['rgb']).mean())
            if mae<=4:return dict(identity=sid,path=path,reason='dhash<=4 AND resized RGB MAE<=4',dhash_distance=distance,mae=mae)
    return None

def run():
    prepare();cf=read(RF/'controlled_manifest.json');real=read(RF/'split_manifest.json')
    all_held={r['id']:r for r in real['dev']}
    all_held.update({r['id']:r for r in cf['dev']})
    overlaps={name:sorted(set(read(OUT/'splits'/(name+'.json')))&{r['id'] for r in cf['dev']})
        for name in ['diagnostic64','confirmation64','remaining126']}
    write(OUT/'protocol/original_split_conflict.json',dict(original_controlled_dev128_ids=[r['id'] for r in cf['dev']],
        overlaps=overlaps,counts={k:len(v) for k,v in overlaps.items()},resolution='pending human choice; no validation identities evaluated',
        manifest_sha256={str(p):sha(p) for p in [RF/'controlled_manifest.json',RF/'split_manifest.json']}))
    captions={Path(r['cloth']).stem:r['caption'] for r in read('data/train_bf_texture.json')}
    import re
    neutral=lambda text:re.search(r'\b(horizontal(?:ly)?|vertical(?:ly)?|diagonal(?:ly)?|slanted|oblique)\b',text,re.I) is None
    # 数据去重读取held图像仅作资格审计；不生成验证目标/模型输出，不以GT选超参。
    held=[(sid,str(DATASET/r['target']),fingerprint(DATASET/r['target'])) for sid,r in all_held.items()]
    rejected=[];selected=[];source_audit=[]
    for row in order(cf['train'],'train_target_audit'):
        sid=row['id'];caption=captions.get(sid)
        if sid in all_held or not caption or not neutral(caption):continue
        value=fingerprint(DATASET/row['target']);conflict=duplicate(value,held)
        if conflict:rejected.append(dict(id=sid,**conflict));continue
        rendered=construct(row);inner=rendered['inner'];boxes=[]
        for y in range(0,385,32):
            for x in range(0,257,32):
                if inner[y:y+128,x:x+128].mean()>=.95:boxes.append((x,y,x+128,y+128))
        if not boxes or rendered['meta']['target_pattern_coverage']<.70:
            rejected.append(dict(id=sid,reason='input-only target renderer coverage/interior128 insufficient'));continue
        row=dict(row,caption=caption);selected.append(row)
        folder=OUT/'G1_counterfactual_targets/train_audit16'/sid;folder.mkdir(parents=True,exist_ok=True)
        box=boxes[int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%len(boxes)]
        observed=[fft_tangent(image,box) for image in rendered['targets']]
        errors=[abs((angle-rendered['meta']['analytic_tangent_degrees'][str(arm)]+90)%180-90) for angle,arm in zip(observed,[0,90,180])]
        audit=dict(rendered['meta'],box=list(box),FFT_tangent_degrees=observed,FFT_analytic_errors=errors,
            automatic_pass=max(errors)<=5 and rendered['meta']['target_pattern_coverage']>=.70,
            human_review_pass=None,target_sha256=sha(DATASET/row['target']),sketch_sha256=sha(DATASET/row['sketch']),
            target_supervision='lowpass only; never used in reference,RF2,adapter forward',GT_high_frequency_copied=False)
        for arm,ref,target in zip(['R0','R90','R180'],rendered['references'],rendered['targets']):
            ref.save(folder/(arm+'_reference.png'));target.save(folder/(arm+'_target.png'))
        rendered['sketch'].save(folder/'sketch.png')
        audit['output_sha256']={p.name:sha(p) for p in folder.glob('*.png')};write(folder/'audit.json',audit);source_audit.append(audit)
        print('[GC target]',sid,audit['automatic_pass'],flush=True)
        if len(selected)==16:break
    assert len(selected)==16,'训练池合格目标不足16，禁止用验证身份补足'
    write(OUT/'G1_counterfactual_targets/train_audit16_rows.json',selected)
    write(OUT/'G2_autograd_smoke/train2_rows.json',selected[:2])
    write(OUT/'protocol/smoke_train_identity_audit.json',dict(ids=[r['id'] for r in selected[:2]],
        train_origin=True,all_original_validation_disjoint=True,exact_and_near_target_checks=True,
        heldout_count=len(all_held),rejected=rejected,threshold='dhash<=4 and RGB64 MAE<=4/255 or exact SHA',
        limitation='no catalog product-group labels; near-duplicate screen cannot certify all same-product different-view identities',
        future_controlled_validation_exclude=[r['id'] for r in selected],
        validation_policy='pending user choice; these16 train audit cases cannot become GC validation'))
    pages=[]
    for row in selected:
        sid=row['id'];folder=OUT/'G1_counterfactual_targets/train_audit16'/sid
        panel=Image.new('RGB',(768,216),'white');draw=ImageDraw.Draw(panel);draw.text((2,2),sid+' | '+row['caption'][:100],fill='black')
        for j,name in enumerate(['sketch.png','R0_reference.png','R90_reference.png','R0_target.png','R90_target.png','R180_target.png']):
            draw.text((j*128+2,24),name.replace('.png',''),fill='black');im=Image.open(folder/name).copy();im.thumbnail((124,166));panel.paste(im,(j*128+2,43))
        pages.append(panel)
    dest=OUT/'visual_audit/G1_train_targets';dest.mkdir(parents=True,exist_ok=True)
    for i in range(0,16,4):
        page=Image.new('RGB',(768,864),'white')
        for j,panel in enumerate(pages[i:i+4]):page.paste(panel,(0,216*j))
        page.save(dest/('page%d.png'%(i//4)))
    automatic=sum(r['automatic_pass'] for r in source_audit)
    write(OUT/'G1_counterfactual_targets/automatic_audit.json',dict(count=16,automatic_success=automatic,
        automatic_pass=automatic>=15,human_review_pass=None,independent_readout='DFT+Hann; noRF2/noSobel training proxy',
        contour='all three targets use identical sketch-only support and identical lowpass background',
        actual_validation_not_opened=True))
    frozen_check();bundle('bootstrap')

if __name__=='__main__':run()
