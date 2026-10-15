"""模型之外的二维 UV 反事实；GT 只供离线评价，不由自然 GT 逆推几何。"""
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tools.e52_protocol import *

W,H=SIZE
KINDS=['sleeve','hem','neck','legs']
PATTERNS=['stripe','check','symbol','aperiodic']

def texture(seed,kind,variant,size=512):
    rng=np.random.default_rng(seed);y,x=np.mgrid[:size,:size]/size
    period=6+(seed%7);phase=(seed%29)/29
    xx=x*period+phase;yy=y*period+phase*.7
    if kind=='stripe':z=np.sin(2*np.pi*(xx if variant%2==0 else yy)+.35*np.sin(2*np.pi*yy))
    elif kind=='check':
        z=np.sin(2*np.pi*xx)*np.sin(2*np.pi*yy) if variant%2==0 else np.sin(2*np.pi*(xx+yy))*np.sin(2*np.pi*(xx-yy))
    elif kind=='symbol':
        dx=(xx%1)-.5;dy=(yy%1)-.5
        if variant%2==0:z=.28-np.hypot(dx,dy)
        else:z=.24*(1+.25*np.cos(5*np.arctan2(dy,dx)))-np.hypot(dx,dy)
    else:
        other=np.random.default_rng(seed+variant*1000003)
        small=other.standard_normal((48,48)).astype('float32')
        z=cv2.GaussianBlur(cv2.resize(small,(size,size)),(0,0),3.)
    # 所有变体使用完全相同的颜色秩分布，避免单靠配色检索纹样。
    palette=np.array([[37,68,96],[89,119,133],[166,119,89],[207,181,145]],np.uint8)
    palette=np.roll(palette,seed%4,axis=0)
    ranks=np.empty(size*size,np.int32);ranks[np.argsort(z.ravel(),kind='stable')]=np.arange(size*size)
    return palette[np.minimum(3,ranks.reshape(size,size)*4//(size*size))]

def triangles(points,uv):
    return [(np.asarray([points[0],points[i],points[i+1]],np.float32),
        np.asarray([uv[0],uv[i],uv[i+1]],np.float32)) for i in range(1,len(points)-1)]

def mesh(index,change=False):
    rng=np.random.default_rng(520000+index);kind=KINDS[index%4]
    cx=192.+rng.uniform(-10,10);top=83.+rng.uniform(-12,12)
    half=73.+rng.uniform(-8,8);bottom=408.+rng.uniform(-20,20)
    sleeve=58.+rng.uniform(-8,8);hem=half+rng.uniform(-8,10)
    shoulder=top+30.;neck=27.+rng.uniform(-5,5)
    sign=1 if (index//4)%2==0 else -1
    if change:
        if kind=='sleeve':sleeve+=sign*28
        elif kind=='hem':hem+=sign*24;bottom+=sign*25
        elif kind=='neck':neck+=sign*13;shoulder+=sign*15
        else:hem+=sign*19;bottom+=sign*24
    parts=[]
    def panel(name,points,uv):
        parts.extend((name,p,t) for p,t in triangles(points,uv))
    if kind!='legs':
        panel('body',[(cx-half,shoulder),(cx-neck,top),(cx+neck,top),(cx+half,shoulder),(cx+hem,bottom),(cx-hem,bottom)],
            [(0,.10),(.20,0),(.40,0),(.60,.10),(.60,1),(0,1)])
        for direction,name,offset in [(-1,'left_sleeve',.62),(1,'right_sleeve',.81)]:
            outer=cx+direction*(half+sleeve)
            points=[(cx+direction*half,shoulder),(outer,shoulder+24),
                (outer-direction*8,shoulder+95),(cx+direction*(half-4),shoulder+78)]
            panel(name,points,[(offset,0),(offset+.17,0),(offset+.17,.50),(offset,.50)])
    else:
        top=97.;waist=half;join=235.
        panel('waist',[(cx-waist,top),(cx+waist,top),(cx+waist,join),(cx-waist,join)],
            [(0,0),(.6,0),(.6,.3),(0,.3)])
        for direction,name,u0 in [(-1,'left_leg',0),(1,'right_leg',.31)]:
            inner=cx+direction*9
            outer=cx+direction*half
            panel(name,[(inner,join),(outer,join),(cx+direction*(hem+6),bottom),(cx+direction*12,bottom)],
                [(u0,.3),(u0+.28,.3),(u0+.28,1),(u0,1)])
    return parts,dict(kind=kind,sign=sign,index=index,pattern=PATTERNS[(index//4)%4])

def uv_map(parts):
    uv=np.full((H,W,2),-1,np.float32);labels=np.zeros((H,W),np.uint8)
    names=list(dict.fromkeys(name for name,_,_ in parts));yy,xx=np.mgrid[:H,:W]
    for name,points,target in parts:
        mask=np.zeros((H,W),np.uint8);cv2.fillConvexPoly(mask,np.round(points).astype('int32'),1)
        matrix=cv2.getAffineTransform(points,target)
        values=np.stack([xx,yy,np.ones_like(xx)],-1)@matrix.T
        uv[mask>0]=values[mask>0];labels[mask>0]=names.index(name)+1
    return uv,labels,names

def render(atlas,uv):
    size=atlas.shape[0];mask=uv[...,0]>=0
    image=cv2.remap(atlas,(uv[...,0]*(size-1)).astype('float32'),
        (uv[...,1]*(size-1)).astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    image[~mask]=255
    return Image.fromarray(image)

def signed_distance(mask):
    return cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)-cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,5)

def synthesize(stage,n):
    rows=[];qc=[];root=OUT/'synthetic'/stage
    offset=0 if stage=='pilot' else 1000
    for k in range(n):
        index=offset+k;identity=('SP' if stage=='pilot' else 'SL')+'%03d'%k;folder=root/identity;folder.mkdir(parents=True,exist_ok=True)
        p0,meta=mesh(index);p1,_=mesh(index,True)
        u0,m0,parts=uv_map(p0);u1,m1,_=uv_map(p1)
        mask0=m0>0;mask1=m1>0
        # target sketch 仅由预定义网格轮廓、部位接缝构造；没有读取自然 GT。
        sketch=np.full((H,W,3),255,np.uint8)
        for lab in np.unique(m0[m0>0]):
            contours,_=cv2.findContours((m0==lab).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sketch,contours,-1,(0,0,0),2)
        sketch_path=folder/'sketch.png';Image.fromarray(sketch).save(sketch_path)
        atlases=[]
        for t in range(8):
            atlas=texture(52042+index*31+t//2,meta['pattern'],t);atlases.append(atlas)
            Image.fromarray(atlas).save(folder/('atlas%d.png'%t))
        refs={}
        for g,u in [(0,u0),(1,u1)]:
            for t in [0,1]:
                path=folder/('G%dT%d.png'%(g,t));render(atlases[t],u).save(path);refs['G%dT%d'%(g,t)]=str(path)
        sham=Image.open(refs['G0T0']).resize((W*2,H*2),Image.Resampling.BILINEAR).resize(SIZE,Image.Resampling.BILINEAR)
        sham.save(folder/'SHAM.png');refs['SHAM']=str(folder/'SHAM.png')
        np.savez_compressed(folder/'geometry.npz',target_uv=u0,reference_uv0=u0,reference_uv1=u1,
            target_labels=m0,reference_labels1=m1,expected_shift=signed_distance(mask1)-signed_distance(mask0))
        for t in range(8):render(atlases[t],u0).save(folder/('correct%d.png'%t))
        def statistics(path,mask):
            a=np.asarray(Image.open(path),np.float32)
            gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
            return dict(mean_rgb=a[mask].mean(0).tolist(),std_rgb=a[mask].std(0).tolist(),
                foreground_fraction=float(mask.mean()),sharpness=float(cv2.Laplacian(gray,cv2.CV_32F)[mask].var()),
                highpass_rms=float((gray-cv2.GaussianBlur(gray,(0,0),2))[mask].std()))
        stats={key:statistics(path,mask1 if key.startswith('G1') else mask0) for key,path in refs.items()}
        histogram_diffs=[]
        for t in [0,1]:
            a=np.asarray(Image.open(refs['G0T%d'%t]))[mask0]
            b=np.asarray(Image.open(refs['G1T%d'%t]))[mask1]
            histogram_diffs.append(float(np.abs(a.mean(0)-b.mean(0)).max()))
        sham_mae=float(np.abs(np.asarray(sham,np.float32)-np.asarray(Image.open(refs['G0T0']),np.float32)).mean())
        q=dict(id=identity,stats=stats,color_mean_max_difference=max(histogram_diffs),sham_mae=sham_mae,
            canonical_uv_same_texture=True,geometry_mask_iou=float((mask0&mask1).sum()/(mask0|mask1).sum()),
            qc_pass=bool(max(histogram_diffs)<16 and sham_mae<8 and .12<mask0.mean()<.65),
            independent_human_reference_review='not_run')
        assert q['qc_pass'],('RENDER_CONFOUND',q)
        row=dict(id=identity,sketch=str(sketch_path),reference=refs['G0T0'],writer_refs=refs,
            gt=str(folder/'correct0.png'),geometry=str(folder/'geometry.npz'),atlas_folder=str(folder),
            caption='a flat lay '+('pair of trousers' if meta['kind']=='legs' else 'shirt')+' with a textile pattern on a white background',
            **meta,synthetic=True,oracle_texture='offline_only')
        rows.append(row);qc.append(q)
    freeze(OUT/'splits'/(stage+'.json'),rows);freeze(OUT/'audit'/('renderer_'+stage+'.json'),qc)
    write(OUT/'manifest'/('synthetic_'+stage+'_sha256.json'),
        {str(p):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    panel_inputs(rows,stage)
    return rows

def panel_inputs(rows,stage):
    root=OUT/'figures'/stage/'reference_review';root.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4):
        page=rows[start:start+4];canvas=Image.new('RGB',(6*144,len(page)*216),'white');draw=ImageDraw.Draw(canvas)
        for y,row in enumerate(page):
            for x,(key,path) in enumerate([('Sketch',row['sketch'])]+list(row['writer_refs'].items())):
                draw.text((x*144+2,y*216+2),row['id']+' '+key,fill='black')
                canvas.paste(Image.open(path).resize((144,192)),(x*144,y*216+22))
        canvas.save(root/('page%03d.png'%start))

def freeze_confirmation():
    assert read(OUT/'audit/p0_complete.json')['passed']
    assert (OUT/'metrics/pilot/summary.json').exists(),'先完成 pilot 测量核验'
    from tools.e52_texture import calibrate
    calibration=calibrate(read(OUT/'splits/pilot.json'))
    freeze(OUT/'manifest/texture_metric.json',calibration)
    rows=synthesize('lock',64)
    from tools.e52_natural import select
    select()
    provenance=read(OUT/'audit/token_provenance.json')
    arms=A_ARMS+(B_ARMS if provenance['oracle_valid'] else [])+C_ARMS
    freeze(OUT/'manifest/confirmation.json',dict(arms=arms,n=64,natural_arms=CONFIG['natural_arms'],
        geometry_fix=CONFIG['geometry_fix'],routing=provenance['oracle_valid'],
        mapping_sigma=provenance.get('mapping_sigma'),alpha=.5,
        thresholds=dict(f1_mean=.02,texture_noninferiority=.05),
        pilot_statistics=read(OUT/'metrics/pilot/summary.json'),
        human_review='not_run; automatic QC does not substitute independent human review',
        code_commit=commit()))
    decision(P1='objective_QC_pass_human_review_not_run',P3='ready' if provenance['oracle_valid'] else 'NOT_IDENTIFIABLE_WITH_CURRENT_WRITER_TOKENS')

if __name__=='__main__':synthesize('pilot',24)
