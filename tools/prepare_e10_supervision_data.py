"""固定 E10 v1 冒烟数据；数据路径相对 BF，区域路径相对输出目录。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(args):
    rows=read(args.output/'candidates.jsonl')
    excluded={'175179592':'选区跨腰带/绑带','145754903':'选区跨不同方向条纹拼接','190969765':'选区主要是大格子空白，重复结构不足','157519782':'选区跨拉链及口袋','183153395':'选区跨裤裆/绑带','200648274':'选区跨裤裆','187896623':'选区跨裤裆','204102150':'选区跨裤裆及门襟','125671546':'选区跨裤裆','184303446':'选区跨裤裆','181140607':'选区跨门襟，宽格纹周期不足','121591822':'选区跨纽扣门襟'}
    excluded['184304346']=excluded.pop('184303446')
    assert set(excluded).issubset({r['id'] for r in rows})
    for r in rows:
        r['status']='deferred' if r['id'] in excluded else 'visual_reviewed_smoke'
        r['review_note']=excluded.get(r['id'],'绿色选区位于可见纹样内部；蓝色保持区未见明显侵入衣物；非像素级金标准')
    for name,selected in [('reviewed',rows),('train',[r for r in rows if r['split']=='train' and r['status']=='visual_reviewed_smoke']),('validation',[r for r in rows if r['split']=='validation' and r['status']=='visual_reviewed_smoke']),('deferred',[r for r in rows if r['status']=='deferred'])]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    accepted=[r for r in rows if r['status']=='visual_reviewed_smoke']
    for r in rows:
        for key,h in r['source_sha256'].items():
            assert digest(args.root/r[key])==h
        regions=[np.asarray(Image.open(args.output/r['regions'][k]))>0 for k in ['pattern','keep','ignore']]
        assert np.all(np.stack(regions).sum(axis=0)==1)
        assert all(a.any() for a in regions)
    summary={'accepted_counts':dict(Counter(r['split'] for r in accepted)),'train_patterns':dict(Counter(r['pattern'] for r in accepted if r['split']=='train')),'deferred':len(excluded),'source_files_unchanged':True,'regions_partition_verified':True,'formal_training_ready':False,'blockers':['独立验证仅1条且无格纹，需扩充','尚未完成Pattern指标校验'],'review_scope':'44组区域预览均已视觉复核；当前记录冻结用于冒烟，不代表正式训练金标准'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True))


def main(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'regions').mkdir(); (out/'preview').mkdir()
    train_source=Path('eval_outputs/bf_uploaded_mask_audit_20260913/coarse_roi_candidates.jsonl')
    val_source=Path('eval_outputs/ctd_pattern_review_20260912/validation_accepted.jsonl')
    candidates=[]
    for r in read(train_source):
        candidates.append(dict(id=Path(r['cloth']).stem,split='train',cloth='training/'+r['cloth'],texture='training/'+r['texture'],sketch='training/'+r['sketch'],mask='training/mask/'+Path(r['cloth']).stem+'.png',caption=r['caption'],pattern=r['candidate_group'],sketch_pattern=r['review']['sketch_pattern']))
    for r in read(val_source):
        if 'striped' not in r['review']['pattern_type_verified']:
            continue  # 本版只验证条纹/格纹，不引入无训练对应的波点。
        candidates.append(dict(id=Path(r['target_image']).stem,split='validation',cloth=r['target_image'],texture=r['texture_candidates'][0],sketch=r['sketch'],mask=r['original_mask'],caption=r['caption'],pattern='stripe',sketch_pattern=r['review']['sketch_pattern']))
    rows=[]; seen=set()
    for i,r in enumerate(candidates):
        gt=Image.open(args.root/r['cloth']).convert('RGB')
        mask_im=Image.open(args.root/r['mask']).convert('L')
        assert mask_im.size==gt.size
        mask=np.asarray(mask_im)>127
        pixel_hash=hashlib.sha256(str(gt.size).encode()+gt.tobytes()).hexdigest()
        assert pixel_hash not in seen,'重复GT，需先处理'
        seen.add(pixel_hash)
        safe=cv2.erode(mask.astype('uint8'),np.ones((17,17),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        # 选取完全处于安全内部的64或48像素矩形，不按颜色删除白色纹样。
        box=None
        for size in (64,48,32):
            integral=cv2.integral(safe)
            sums=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
            yy,xx=np.where(sums==size*size)
            if len(xx):
                my,mx=np.where(mask)
                score=(xx+size/2-np.median(mx))**2+(yy+size/2-np.median(my))**2
                k=int(np.argmin(score)); x,y=int(xx[k]),int(yy[k]); box=[x,y,x+size,y+size]; break
        assert box is not None,r['id']
        pattern=np.zeros_like(mask); x,y,x2,y2=box; pattern[y:y2,x:x2]=True
        # 背景以整件服装mask的膨胀外部定义，绝不能用1-pattern。
        keep=cv2.dilate(mask.astype('uint8'),np.ones((25,25),'uint8'))==0
        unknown=~(pattern|keep)
        assert not (pattern&keep).any()
        paths={}
        for name,arr in [('pattern',pattern),('keep',keep),('ignore',unknown)]:
            path=Path('regions')/(r['split']+'_'+r['id']+'_'+name+'.png')
            Image.fromarray(arr.astype('uint8')*255).save(out/path); paths[name]=path.as_posix()
        hashes={key:digest(args.root/r[key]) for key in ['cloth','texture','sketch','mask']}
        rows.append(dict(r,regions=paths,pattern_box_xyxy=box,source_sha256=hashes,target_pixel_sha256=pixel_hash,region_sha256={k:digest(out/v) for k,v in paths.items()},status='pending_visual_review',purpose='smoke_only'))
        if i%12==0: sheet=Image.new('RGB',(1536,1080),'white')
        ox,oy=i%12%3*512,i%12//3*270
        draw=ImageDraw.Draw(sheet); draw.text((ox,oy),f'{i} {r["split"]} {r["id"]} box={box}',fill='black')
        over=np.asarray(gt).copy(); over[keep]=(over[keep]*.65+np.array([30,100,255])*.35).astype('uint8'); over[pattern]=(over[pattern]*.5+np.array([0,230,30])*.5).astype('uint8')
        for j,im in enumerate([gt,Image.open(args.root/r['texture']).convert('RGB'),Image.fromarray(over)]):
            im=im.copy(); im.thumbnail((170,230)); sheet.paste(im,(ox+j*170,oy+28))
        if i%12==11 or i==len(candidates)-1: sheet.save(out/'preview'/f'page_{i//12:02d}.png')
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'build_info.json').write_text(json.dumps({'sources':{str(p):digest(p) for p in [train_source,val_source]},'opencv':cv2.__version__,'mask_margin_px':8,'background_margin_px':12,'counts':dict(Counter(r['split'] for r in rows)),'validation_limit':'只有1条已审条纹验证样本；仅冒烟，不支持泛化评价','path_contract':'cloth/texture/sketch/mask相对BF根目录；regions相对本输出目录','loss_contract':'pattern/keep/ignore互斥且覆盖全图；区域同步几何变换，mask最近邻插值；特征感受野须落在有效区域内'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('prepared',len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('F:/fuxian/dataset/datasets/BF'))
    p.add_argument('--output',type=Path,default=Path('data/processed/e10_supervision_v1'))
    p.add_argument('--finalize',action='store_true')
    args=p.parse_args()
    (finalize if args.finalize else main)(args)
