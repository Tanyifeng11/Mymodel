"""100张本地重建试验：U2Net mask、轮廓草图、待视觉确认的简短caption。"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    a.output.mkdir(exist_ok=False,parents=True)
    for name in ['mask','alpha','sketch','text','preview']:
        (a.output/name).mkdir()
    rows=[json.loads(s) for s in a.source.read_text(encoding='utf-8').splitlines()]
    # 固定首50条纹和首50格纹，覆盖好/坏参考，不是新的随机合格率估计。
    rows=rows[:50]+rows[100:150]
    opts=ort.SessionOptions(); opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(a.model),sess_options=opts,providers=['CPUExecutionProvider'])
    result=[]
    for i,r in enumerate(rows):
        stem=Path(r['cloth']).stem
        gt=Image.open(a.root/r['cloth']).convert('RGB')
        arr=np.array(gt.resize((320,320),Image.Resampling.LANCZOS)).astype('float32')
        arr=arr/max(arr.max(),1e-6)
        arr=(arr-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
        pred=session.run(None,{session.get_inputs()[0].name:arr.transpose(2,0,1)[None]})[0][0,0]
        pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-6)
        alpha=Image.fromarray((pred*255).astype('uint8')).resize(gt.size,Image.Resampling.LANCZOS)
        mask=(np.array(alpha)>127).astype('uint8')*255
        # 保留孔洞，不用外轮廓填充；不以白色直接判背景。
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        sketch=np.full(mask.shape,255,'uint8')
        contours=[c for c in contours if cv2.arcLength(c,True)>=12]
        cv2.drawContours(sketch,contours,-1,0,1)
        caption=str(r['caption']).lower()
        category=re.search(r'\b(dress|shirt|blouse|jacket|coat|skirt|pants|trousers|shorts|sweater|cardigan|vest|top|hoodie|sweatshirt|jumpsuit)\b',caption)
        category=category.group(1) if category else 'garment'
        # 这里只生成草稿，后续视觉确认；不编造颜色/材质/领型。
        text=f'A {r["candidate_group"] == "plaid" and "plaid" or "striped"} {category}.'
        paths={k:f'{k}/{stem}.{ "txt" if k=="text" else "png"}' for k in ['mask','alpha','sketch','text']}
        alpha.save(a.output/paths['alpha']); Image.fromarray(mask).save(a.output/paths['mask']); Image.fromarray(sketch).save(a.output/paths['sketch'])
        (a.output/paths['text']).write_text(text+'\n',encoding='utf-8')
        old=Image.open(a.root/'mask'/(stem+'.png')).convert('L')
        overlay=np.array(gt).copy(); valid=mask>0
        overlay[valid]=(overlay[valid]*.65+np.array([0,210,40])*.35).astype('uint8')
        record=dict(r,pilot_index=i,generated=paths,caption_draft=text,review_status='pending',source_hashes={k:sha(a.root/r[k]) for k in ['cloth','texture','sketch']},old_mask_sha256=sha(a.root/'mask'/(stem+'.png')),mask_area=float(valid.mean()),old_mask_area=float((np.array(old)>127).mean()))
        result.append(record)
        if i%10==0: sheet=Image.new('RGB',(1600,1100),'white')
        x,y=i%10%2*800,i%10//2*220
        draw=ImageDraw.Draw(sheet);draw.text((x,y),f'{i:02d} {stem} {text}',fill='black')
        for j,im in enumerate([gt,old.convert('RGB'),Image.fromarray(mask).convert('RGB'),Image.fromarray(overlay),Image.fromarray(sketch).convert('RGB')]):
            im=im.copy(); im.thumbnail((160,180)); sheet.paste(im,(x+j*160,y+25))
        draw.text((x,y+202),'GT / OLD MASK / NEW MASK / OVERLAY / CONTOUR',fill='black')
        if i%10==9:
            sheet.save(a.output/'preview'/f'page_{i//10:02d}.png'); print('completed',i+1,flush=True)
    (a.output/'draft.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    (a.output/'build.json').write_text(json.dumps({'model_sha256':sha(a.model),'source_sha256':sha(a.source),'backend':ort.__version__,'provider':'CPU','limits':['仅轮廓草图：没有可靠重建内部领口/接缝','文本为待复核草稿，没有自动视觉语言模型','参考不变；没有训练或改变原划分','U2Net是显著物体分割，不是服装人工真值']},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('F:/fuxian/dataset/datasets/BF/training'))
    p.add_argument('--source',type=Path,default=Path('eval_outputs/bf_pattern_expansion_200_seed42/reviewed_200.jsonl'))
    p.add_argument('--model',type=Path,default=Path('C:/Users/lenovo/.u2net/u2net.onnx'))
    p.add_argument('--output',type=Path,default=Path('data/processed/bf_rebuild_pilot100_v1'))
    main(p.parse_args())
