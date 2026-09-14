"""全量标注疑点排序；启发式仅用于选样，不自动修正或认证标签。"""
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_bf_full import dump

ROOT = Path('F:/fuxian/dataset/datasets/BF')
OUT = Path('eval_outputs/bf_annotation_review_v1')


def read(rel, gray=False):
    with Image.open(ROOT/rel) as im:
        return np.asarray(im.convert('L' if gray else 'RGB').resize((128,128)))


def screen(r):
    gt, ref, sk = read(r['cloth']), read(r['texture']), read(r['sketch'],True)
    gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,120)>0
    lines = sk<160
    close = cv2.dilate(edges.astype('uint8'),np.ones((5,5),'uint8'))>0
    mismatch = float((lines & ~close).sum()/max(lines.sum(),1))
    m = read(r['mask'],True)>127 if 'mask' in r else None
    # 白色既可能是衣物又可能是背景，绝不据此删除mask。
    white = (gt.min(axis=2)>240)
    white_inside = float(white[m].mean()) if m is not None and m.any() else 0
    outside_edges = float((edges & ~cv2.dilate(m.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)).sum()/max(edges.sum(),1)) if m is not None else 0
    fg = gt[m] if m is not None and m.any() else gt[gray<245]
    color = float(np.linalg.norm(np.median(fg,axis=0)-np.median(ref.reshape(-1,3),axis=0))/441.7) if len(fg) else 0
    cap = r['caption'].lower()
    mixed = float(any(w in cap for w in ('panel','contrasting','trim','patch','graphic','print')) and any(w in cap for w in ('strip','plaid','check')))
    return dict(r, scores=dict(sketch_alignment=mismatch,mask_white=white_inside,
                              mask_missed_edges=outside_edges,reference_color=color,mixed_caption=mixed),
                automatic_status='suspect_ranking_only', mask_available=m is not None)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split in ('training','validation','test'):
        rows.extend(json.loads(s) for s in Path(f'data/processed/bf_full_audit_v1/{split}_clean.jsonl').read_text(encoding='utf-8').splitlines())
    scored=[]
    with ThreadPoolExecutor(max_workers=8) as pool, (OUT/'screening.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(pool.map(screen,rows),1):
            scored.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if i%10000==0: print('screened',i,flush=True)
    selected=[]; seen=set()
    for key in ('sketch_alignment','mask_white','mask_missed_edges','reference_color','mixed_caption'):
        count=0
        for r in sorted(scored,key=lambda r:(-r['scores'][key],r['sample_id'])):
            if r['sample_id'] in seen: continue
            selected.append(dict(r,selection_reason=key,review_index=len(selected)))
            seen.add(r['sample_id']);count+=1
            if count==20: break
    dump(OUT/'selected_100.jsonl',selected)
    for page in range(10):
        sheet=Image.new('RGB',(1536,1350),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected[page*10:page*10+10]):
            x,y=j%2*768,j//2*270
            draw.text((x,y),f"{r['review_index']:02d} {r['sample_id']} {r['selection_reason']}",fill='black')
            gt=Image.open(ROOT/r['cloth']).convert('RGB')
            overlay=np.array(gt)
            if 'mask' in r:
                mask=Image.open(ROOT/r['mask']).convert('L');v=np.array(mask)>127
                overlay[v]=(overlay[v]*.65+np.array([0,210,40])*.35).astype('uint8')
            else: mask=Image.new('L',gt.size,128)
            sk=Image.open(ROOT/r['sketch']).convert('L')
            edge_overlay=np.array(gt);edge_overlay[np.asarray(sk)<160]=[255,0,0]
            ims=[gt,Image.open(ROOT/r['texture']).convert('RGB'),mask,Image.fromarray(overlay),sk,Image.fromarray(edge_overlay)]
            for k,im in enumerate(ims): sheet.paste(im.resize((128,128)),(x+k*128,y+22))
            draw.text((x,y+153),'GT / TEXTURE / MASK / MASK OVERLAY / SKETCH / SKETCH OVERLAY',fill='black')
            draw.multiline_text((x,y+170),'\n'.join(textwrap.wrap(r['caption'],width=112)),fill='black',spacing=2)
        sheet.save(OUT/f'page_{page:02d}.png')
    (OUT/'screening_summary.json').write_text(json.dumps(dict(count=len(scored),selected=100,
        selection='五类疑点各20条，去重；不是随机抽样，不能估计全量错误率',
        limits=['颜色距离不判定纹样一致性','白色区域可能是真实衣物','sketch边缘距离受线条风格影响','测试集没有mask，不对其mask打分','不使用新轮廓sketch或新U2Net mask']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('preview ready',flush=True)


if __name__=='__main__': main()
