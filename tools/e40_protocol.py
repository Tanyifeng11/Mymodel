"""E40：冻结身份、来源分组和输入干预，不修改 BF 原始数据。"""
import csv
import math
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tools.e39_protocol import read, write, sha, E5, EXPECTED_E5
from tools.e39_visual_review import LABELS

OUT = Path('output_eval/e40_pattern_identity_20261011')
OLD = Path('output_eval/e39_reference_trajectory_20261011')
CLASSES = ['stripe', 'plaid', 'floral', 'solid']
VARIANTS = ['base', 'color1', 'color2', 'crop1', 'crop2', 'carrier']
BF = Path('/share/home/u2515283058/datasets/BF')


def recolor(image, variant):
    import cv2
    a = np.asarray(image.convert('RGB'))
    y = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    out = np.stack([y, np.full_like(y, 148 if variant == 1 else 108),
                    np.full_like(y, 108 if variant == 1 else 150)], -1)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB))


def controlled(kind, cycles, angle, phase, seed=42):
    y, x = np.mgrid[:256, :256].astype(float)
    t = math.radians(angle)
    u = ((x*np.cos(t)+y*np.sin(t))*cycles/256 + phase + .5) % 1 - .5
    v = ((-x*np.sin(t)+y*np.cos(t))*cycles/256 + phase + .5) % 1 - .5
    if kind == 'solid':
        # 不把素色的不同颜色当独立纹样身份；只有弱非周期照明背景。
        rng = np.random.default_rng(seed)
        a = 168 + 3*np.sin(x/170+rng.uniform(0,6)) + 3*np.cos(y/130+rng.uniform(0,6))
        a = np.rint(a).clip(0,255).astype('uint8')
    else:
        score = abs(u) if kind == 'stripe' else np.minimum(abs(u),abs(v))
        if kind == 'floral':
            score = np.hypot(u,v) / (.28+.10*np.cos(5*np.arctan2(v,u)))
        mask = np.zeros(256*256, bool)
        mask[np.argsort(score.ravel(),kind='stable')[:16384]] = True
        a = np.where(mask.reshape(256,256),48,208).astype('uint8')
    return Image.fromarray(np.repeat(a[:,:,None],3,2))


def variants(image, seed):
    a = image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    yield 'base', a
    yield 'color1', recolor(a,1)
    yield 'color2', recolor(a,2)
    yield 'crop1', a.crop((0,0,192,192)).resize((256,256),Image.Resampling.BILINEAR)
    yield 'crop2', a.crop((64,64,256,256)).resize((256,256),Image.Resampling.BILINEAR)
    rng = np.random.default_rng(seed)
    yy,xx = np.mgrid[:256,:256]
    shading = 1+.06*np.sin(xx/80+.3)*np.cos(yy/93)
    grain = rng.normal(0,2,(256,256,1))
    b = np.asarray(a,float)*shading[:,:,None]+grain
    yield 'carrier', Image.fromarray(np.rint(b).clip(0,255).astype('uint8'))


def prepare():
    assert sha(E5) == EXPECTED_E5
    if (OUT/'protocol.json').exists():
        return
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]; roots=[]; inputs={str(E5):sha(E5)}
    frequencies=[3,5,8,12]
    for kind in CLASSES:
        for fi,f in enumerate(frequencies):
            for angle in [0,30]:
                for pi,phase in enumerate([0.,.17,.33,.51]):
                    id='%s_f%d_a%d_p%d'%(kind,f,angle,pi)
                    roots.append(dict(id=id,domain='controlled',pattern=kind,
                        family='%s_f%d_a%d'%(kind,f,angle),fold=fi,
                        image=controlled(kind,f,angle,phase,seed=fi*100+angle+pi),
                        provenance='procedural, not natural fabric',cycles=f,angle=angle))
    label_file=Path('eval_outputs/e14_candidates/candidates/labels.csv')
    candidates=[]
    if label_file.exists():
        inputs[str(label_file)]=sha(label_file)
        with label_file.open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('confirmed')=='1' and row.get('pattern_visible')=='1' and row.get('pattern') in CLASSES:
                    path=BF/row['texture']
                    if path.exists():
                        candidates.append(dict(id='real_'+row['sample_id'],pattern=row['pattern'],
                            source=str(path),family=row.get('source_group') or row['sample_id'],
                            provenance='historical assistant visual review; '+row.get('annotation_by','unknown')))
    for row in read(OLD/'pairs.json'):
        if LABELS[row['id']] in CLASSES:
            candidates.append(dict(id='real_'+row['id'],pattern=LABELS[row['id']],
                source=row['reference'],family='BF_'+row['id'],
                provenance='E39 input-only single AI surface label'))
    seen=set(); usable=[]
    for row in sorted(candidates,key=lambda r:r['id']):
        digest=sha(row['source'])
        if digest in seen:continue
        seen.add(digest);row['source_sha256']=digest;usable.append(row)
    for ki,kind in enumerate(CLASSES):
        selected=[r for r in usable if r['pattern']==kind]
        order=np.random.default_rng(40042+ki).permutation(len(selected))[:32]
        for i,idx in enumerate(order):
            row=dict(selected[idx]);inputs[row['source']]=row['source_sha256']
            row.update(domain='real',fold=i%4,image=Image.open(row['source']).convert('RGB'))
            roots.append(row)
    # 若历史同来源组重复，强制所有同组样本同折。
    folds={}
    for row in roots:
        key=(row['domain'],row['family'])
        row['fold']=folds.setdefault(key,row['fold'])
    for i,row in enumerate(roots):
        folder=OUT/'references'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        for variant,image in variants(row['image'],40000+i):
            path=folder/(variant+'.png');image.save(path)
            records.append(dict(**{k:v for k,v in row.items() if k!='image'},variant=variant,
                path=str(path),sha256=sha(path)))
    write(OUT/'references.json',records)
    base=[r for r in records if r['variant']=='base']
    # 真实同色异纹只按输入 LAB 选配，绝不看生成结果。
    import cv2
    means={r['id']:cv2.cvtColor(np.asarray(Image.open(r['path']),np.float32)/255,cv2.COLOR_RGB2LAB).mean((0,1)) for r in base}
    pairs=[]
    for r in [v for v in base if v['domain']=='real']:
        pool=[q for q in base if q['domain']=='real' and q['pattern']!=r['pattern'] and q['family']!=r['family']]
        if not pool:continue
        q=min(pool,key=lambda q:float(np.linalg.norm(means[r['id']]-means[q['id']])))
        delta=float(np.linalg.norm(means[r['id']]-means[q['id']]))
        if delta<=5:pairs.append(dict(left=r['id'],right=q['id'],lab_delta=delta))
    write(OUT/'real_color_pairs.json',pairs)
    carriers=read(OLD/'pairs.json')
    order=np.random.default_rng(40043).permutation(len(carriers))[:16]
    generation=[]
    for i in order:
        r=carriers[i]
        inputs[r['sketch']]=sha(r['sketch'])
        for kind in CLASSES:
            image=controlled(kind,7,15,.23)
            for color in [0,1]:
                name='%s_%s_c%d'%(r['id'],kind,color)
                ref=OUT/'generation_references'/(kind+'_c%d.png'%color);ref.parent.mkdir(exist_ok=True)
                (image if color==0 else recolor(image,1)).save(ref)
                noun={'top':'top','bottom':'trousers','outer':'jacket','dress':'dress'}[r['category']]
                generation.append(dict(id=name,carrier=r['id'],pattern=kind,color=color,
                    reference=str(ref),sketch=r['sketch'],
                    caption='a flat lay photograph of a '+noun+' on a white background',
                    reference_template='held-out cycles7 angle15 phase.23'))
    write(OUT/'generation.json',generation);write(OUT/'input_hashes.json',inputs)
    counts={d:{k:sum(r['domain']==d and r['pattern']==k for r in base) for k in CLASSES} for d in ['controlled','real']}
    write(OUT/'protocol.json',dict(seed=42,cfg=7,ddim_steps=50,E5_sha256=sha(E5),generation_updates=0,
        reference_counts=counts,generation_images=len(generation),real_color_pairs=len(pairs),
        folds='four source folds; controlled held-out frequency; all variants of each family stay together',
        H1='luminance approximately preserved, chroma changed; actual LAB and gray drift measured',
        H2='real cross-material identity unavailable; carrier variant is noise/shading stress only',
        C='crop plus resize changes apparent pixel frequency; not a physical-scale identity oracle',
        D='controlled shared carrier only; no verified physical-material labels',
        label_limit='real labels are historical single AI review, provisional sample groups, not verified fabric IDs',
        generation='16 carriers x4 categories x2 colors; neutral category text fixed within each carrier; template held out',
        primary='macro F1, cross-color/crop source grouping, Dpattern-Dcolor paired CI; category is not complete identity',
        gates=dict(classification_macro_f1=.70,max_cross_color_f1_drop=.05,distance_gap_ci_lower=0.,
                   full_go_requires_real_evidence=True,missing_evidence_is_inconclusive=True)))
    cards=OUT/'review';cards.mkdir(exist_ok=True)
    for domain in ['real','controlled']:
        selected=[r for r in base if r['domain']==domain]
        for page in range((len(selected)+31)//32):
            canvas=Image.new('RGB',(8*144,4*172),'white');draw=ImageDraw.Draw(canvas)
            for j,r in enumerate(selected[page*32:page*32+32]):
                x=j%8*144;y=j//8*172
                canvas.paste(Image.open(r['path']).resize((140,140)),(x,y))
                draw.text((x,y+141),r['id'][-20:],fill='black');draw.text((x,y+155),r['pattern'],fill='black')
            canvas.save(cards/('%s_%d.jpg'%(domain,page)))
    print('E40 PREPARE COMPLETE',counts,'generation',len(generation),'real pairs',len(pairs),flush=True)


if __name__=='__main__':prepare()
