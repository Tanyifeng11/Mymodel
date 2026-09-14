"""检查上传mask；仅生成诊断副本，不改源文件。"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path('F:/fuxian/dataset/datasets/BF')
OUT = Path('eval_outputs/bf_uploaded_mask_audit_20260913')


def main():
    OUT.mkdir(exist_ok=False)
    rows = [json.loads(s) for s in Path('eval_outputs/bf_pattern_expansion_200_seed42/original_reference_usable.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['review']['coverage'] == '主要区域一致']
    masks = {p.stem: p for p in (ROOT/'training/mask').iterdir() if p.is_file()}
    result = []
    for i, r in enumerate(rows):
        gt = Image.open(ROOT/'training'/r['cloth']).convert('RGB')
        ref = Image.open(ROOT/'training'/r['texture']).convert('RGB')
        p = masks.get(Path(r['cloth']).stem)
        if not p:
            result.append(dict(r, mask_audit={'missing': True}))
            continue
        original = Image.open(p)
        raw = np.asarray(original.convert('L'))
        # 不擅自推测或反转0/1、0/255之外的语义。
        values = np.unique(raw)
        binary = raw > (0 if raw.max() <= 1 else 127)
        m = Image.fromarray(binary.astype('uint8')*255).resize(gt.size, Image.Resampling.NEAREST)
        inner = m.filter(ImageFilter.MinFilter(9))
        overlay = np.array(gt).copy()
        v = np.asarray(m) > 0
        overlay[v] = (overlay[v]*.65 + np.array([0,200,50])*.35).astype('uint8')
        info = {'path':str(p),'mode':original.mode,'size':original.size,'gt_size':gt.size,'values':values.tolist(),'area_ratio':float(binary.mean()),'inner_area_ratio':float((np.asarray(inner)>0).mean()),'border_foreground_ratio':float(np.concatenate([binary[0],binary[-1],binary[:,0],binary[:,-1]]).mean()),'missing':False}
        result.append(dict(r, mask_audit=info))
        if i % 12 == 0:
            sheet = Image.new('RGB',(1536,1008),'white')
        x,y = i%12%3*512,i%12//3*252
        draw=ImageDraw.Draw(sheet)
        draw.text((x,y),f'{i:02d} source={r["review_index"]} {Path(r["cloth"]).stem}',fill='black')
        for j,im in enumerate([gt,ref,m.convert('RGB'),Image.fromarray(overlay)]):
            im=im.copy(); im.thumbnail((128,192)); sheet.paste(im,(x+j*128,y+25))
        draw.text((x,y+221),f'GT / REF / MASK / OVERLAY area={binary.mean():.3f}',fill='black')
        if i%12==11 or i==len(rows)-1:
            sheet.save(OUT/f'page_{i//12:02d}.png')
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    print('rows',len(result),'missing',sum(r['mask_audit']['missing'] for r in result),flush=True)
    print('modes',Counter(r['mask_audit'].get('mode') for r in result),flush=True)


def detail():
    rows=[json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    sheet=Image.new('RGB',(1536,1152),'white')
    draw=ImageDraw.Draw(sheet)
    for pos,idx in enumerate([4,6,22,39,55,70,76,79]):
        r=rows[idx]; x,y=pos%2*768,pos//2*288
        gt=Image.open(ROOT/'training'/r['cloth']).convert('RGB')
        m=Image.open(r['mask_audit']['path']).convert('L')
        arr=np.array(gt).copy(); v=np.asarray(m)>127
        arr[v]=(arr[v]*.6+np.array([0,220,40])*.4).astype('uint8')
        draw.text((x,y),f'{idx} {Path(r["cloth"]).stem} GT / MASK / OVERLAY',fill='black')
        for k,im in enumerate([gt,m.convert('RGB'),Image.fromarray(arr)]):
            sheet.paste(im,(x+k*256,y+25))
    sheet.save(OUT/'detail.png')


def summarize():
    rows=[json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    # 逐页视觉复核发现的背景空隙填充；索引对应诊断图，而非原清单行号。
    gaps={1,4,6,9,11,14,15,17,19,20,21,22,23,26,27,31,32,36,37,38,39,40,41,42,46,50,54,55,59,60,61,64,65,69,70,74,76,78,79,82}
    for i,r in enumerate(rows):
        r['mask_audit']['visual_review']={
            'reviewer':'Codex视觉诊断，非人工像素级金标准',
            'decision':'needs_gap_review' if i in gaps else 'coarse_roi_candidate',
            'reason':'可见袖身/裤腿空隙或衣物外背景被前景填充，需局部修正或避开该处' if i in gaps else '主体大致覆盖，未在本轮对照图发现突出空隙填充；仅作粗ROI候选，不认证像素级轮廓',
            'training_ready':False}
    (OUT/'reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for name,select in [('needs_gap_review',lambda i:i in gaps),('coarse_roi_candidates',lambda i:i not in gaps)]:
        (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for i,r in enumerate(rows) if select(i)),encoding='utf-8')
    summary={'reviewed':len(rows),'missing':sum(r['mask_audit']['missing'] for r in rows),'all_binary_0_255':all(r['mask_audit']['values']==[0,255] for r in rows),'all_sizes_match':all(r['mask_audit']['size']==r['mask_audit']['gt_size'] for r in rows),'visual_flags':dict(Counter(r['mask_audit']['visual_review']['decision'] for r in rows)),'area_range':[min(r['mask_audit']['area_ratio'] for r in rows),max(r['mask_audit']['area_ratio'] for r in rows)],'limits':['仅83条训练候选完成视觉核查，不代表全部上传mask','没有独立真值，不能报告mask IoU/像素准确率','不修改源mask；未认证可直接训练；9像素腐蚀不能保证移除错误填充的内部孔洞','未验证本次上传mask是否等于先前服务器实验实际加载的mask']}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':
    import sys
    if '--detail' in sys.argv:
        detail()
    elif '--summarize' in sys.argv:
        summarize()
    else:
        main()
