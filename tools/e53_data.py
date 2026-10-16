"""E53配对数据和人工核验图；只由程序真值构建，未读取模型结果。"""
from collections import Counter
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tools.e53_protocol import *
from tools.e52_data import mesh,uv_map,render,signed_distance
from tools.e53_mapping import matrices,grid

def texture(seed,kind,variant,size=512):
    rng=np.random.default_rng(seed);y,x=np.mgrid[:size,:size]/size;frequency=2
    if kind=='stripe':z=np.sin(2*np.pi*frequency*x+.23)
    elif kind=='check':z=np.sin(2*np.pi*frequency*x+.23)*np.sin(2*np.pi*frequency*y+.43)
    elif kind=='symbol':
        z=np.zeros((size,size),np.float32)
        for yy in range(100,size,256):
            for xx in range(100,size,256):
                cv2.circle(z,(xx,yy),40,1,-1);cv2.rectangle(z,(xx+10,yy-60),(xx+30,yy-30),.5,-1)
        z=cv2.GaussianBlur(z,(0,0),2.)
    else:
        fx,fy=np.meshgrid(np.fft.fftfreq(size),np.fft.fftfreq(size));radius=np.hypot(fx,fy)*size
        noise=np.fft.fft2(rng.standard_normal((size,size)));spectrum=np.exp(-((radius-2)/1.2)**2)*(radius<=3)
        z=np.fft.ifft2(noise*spectrum).real
    z=(z-z.mean())/max(float(z.std()),1e-8);z=np.clip(z,-2.5,2.5)/2.5
    base=np.array([132,122,111]);color=np.array([67,51,38]);atlas=np.clip(base+z[...,None]*color,0,255).astype('uint8')
    # 循环平移改变相位/排列；像素直方图及每通道Fourier幅度严格不变。
    shift=variant*size//(8*frequency) if kind in ['stripe','check','symbol'] else variant*size//8
    return np.roll(atlas,shift,axis=1)

def synthesize(stage='pilot',n=24):
    rows=[];qc=[];root=OUT/'synthetic'/(stage+'_r2');offset=0 if stage=='pilot' else 1000
    kinds=['sleeve','hem','neck','legs'];patterns=['stripe','check','symbol','aperiodic']
    for k in range(n):
        index=offset+k;identity=('SP' if stage=='pilot' else 'SL')+'%03d'%k;folder=root/identity;folder.mkdir(parents=True,exist_ok=True)
        p0,meta=mesh(index);p1,_=mesh(index,True);pn,_=mesh(3 if meta['kind']=='legs' else 0)
        u0,l0,parts=uv_map(p0);u1,l1,_=uv_map(p1);un,ln,_=uv_map(pn)
        # Pilot使用拉丁式类别分配，确保四种纹样各6；Lock各16。
        pattern=patterns[(k+k//4)%4];meta['pattern']=pattern
        sketch=np.full((512,384,3),255,np.uint8)
        for lab in np.unique(l0[l0>0]):
            contours,_=cv2.findContours((l0==lab).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(sketch,contours,-1,(0,0,0),2)
        Image.fromarray(sketch).save(folder/'sketch.png');refs={}
        for t in range(8):
            atlas=texture(53042+index*31,pattern,t);Image.fromarray(atlas).save(folder/('atlas%d.png'%t));render(atlas,u0).save(folder/('correct%d.png'%t))
            if t<2:
                for g,u in [('0',u0),('1',u1),('N',un)]:
                    p=folder/('G%sT%d.png'%(g,t));render(atlas,u).save(p);refs['G%sT%d'%(g,t)]=str(p)
        np.savez_compressed(folder/'geometry.npz',target_uv=u0,target_labels=l0,uv0=u0,labels0=l0,uv1=u1,labels1=l1,uvN=un,labelsN=ln,expected_shift=signed_distance(l1>0)-signed_distance(l0>0))
        row=dict(id=identity,sketch=str(folder/'sketch.png'),reference=refs['G0T0'],writer_refs=refs,gt=str(folder/'correct0.png'),geometry=str(folder/'geometry.npz'),atlas_folder=str(folder),caption='a flat lay '+('pair of trousers' if meta['kind']=='legs' else 'shirt')+' with a textile pattern on a white background',**meta,synthetic=True)
        a=np.asarray(Image.open(folder/'atlas0.png'),np.float64);b=np.asarray(Image.open(folder/'atlas1.png'),np.float64)
        spectral=np.max(np.abs(np.abs(np.fft.fft2(a,axes=(0,1)))-np.abs(np.fft.fft2(b,axes=(0,1)))))
        entry=dict(id=identity,texture_mean_error=float(np.abs(a.mean((0,1))-b.mean((0,1))).max()),texture_std_error=float(np.abs(a.std((0,1))-b.std((0,1))).max()),texture_fourier_amplitude_max_error=float(spectral),geometries={},human_review='NOT_RUN')
        for g,l in [('0',l0),('1',l1),('N',ln)]:
            yy,xx=np.where(l>0);entry['geometries'][g]=dict(area=len(xx),centroid=[float(xx.mean()),float(yy.mean())])
        assert entry['texture_mean_error']<1e-12 and entry['texture_std_error']<1e-9 and spectral<1e-6
        rows.append(row);qc.append(entry)
    assert all(v==n//4 for v in Counter(x['pattern'] for x in rows).values())
    freeze(OUT/'splits'/(stage+'_r2.json'),rows);freeze(OUT/'audit'/('renderer_'+stage+'_r2.json'),qc)
    write(OUT/'manifest'/('synthetic_'+stage+'_r2_sha256.json'),{str(p):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    return rows

def mapping_review(rows,stage='pilot'):
    folder=OUT/'figures'/(stage+'_r2')/'mapping_review';folder.mkdir(parents=True,exist_ok=True);records=[]
    for row in rows:
        d=np.load(row['geometry']);canvas=Image.new('RGB',(4*288,600),'white');draw=ImageDraw.Draw(canvas)
        target=Image.open(row['sketch']).resize((288,384));canvas.paste(target,(0,35));draw.text((4,4),row['id']+' target; 8 fixed-seed tokens',fill='black')
        m,allow,t,r=matrices(row);valid=np.where(m.sum(1)>.99)[0];rng=np.random.default_rng(identity_seed(row['id']));tokens=rng.choice(valid,min(8,len(valid)),replace=False)
        point_records={str(int(q)):dict(target_uv=t[0][q].tolist(),part=int(t[1][q])) for q in tokens}
        for column,mode in enumerate(['ORACLE','SHIFT','FLIP'],1):
            mm,_,_,_=matrices(row,mode=mode);ref=Image.open(row['writer_refs']['GNT0']).resize((288,384));canvas.paste(ref,(column*288,35));draw.text((column*288+4,4),'GEOMETRIC '+mode,fill='black')
            for num,q in enumerate(tokens):
                j=int(mm[q].argmax());tx=(q%24+.5)/24*288;ty=(q//24+.5)/32*384+35
                # 24->16 adaptive pooling的实际支持中心，不能假设均匀16格中心。
                jy,jx=j//16,j%16;ya=int(np.floor(jy*32/16));yb=int(np.ceil((jy+1)*32/16));xa=int(np.floor(jx*24/16));xb=int(np.ceil((jx+1)*24/16))
                rx=column*288+(xa+xb)/2/24*288;ry=(ya+yb)/2/32*384+35
                point_records[str(int(q))][mode]=dict(reference_token=j,reference_uv=r[0][j].tolist())
                if column==1:draw.ellipse((tx-4,ty-4,tx+4,ty+4),fill='red');draw.text((tx+5,ty),str(num+1),fill='red')
                draw.ellipse((rx-4,ry-4,rx+4,ry+4),fill='blue');draw.text((rx+5,ry),str(num+1),fill='blue')
            residual=float(np.max(np.abs(mm[valid].sum(1)-1)));assert residual<2e-6
            dist=np.linalg.norm(t[0][:,None]-r[0][None],axis=-1);records.append(dict(id=row['id'],mode=mode,coverage=float(len(valid)/max(int(t[2].sum()),1)),row_sum_error=residual,expected_uv_distance=float((mm*dist).sum(1)[valid].mean()),tokens=tokens.tolist(),points=point_records))
        draw.text((4,435),'# part | target UV | ORACLE reference UV | SHIFT reference UV | FLIP reference UV',fill='black')
        for num,q in enumerate(tokens):
            point=point_records[str(int(q))];fmt=lambda x:'(%.3f, %.3f)'%tuple(x)
            draw.text((4,454+num*17),str(num+1)+' '+str(point['part'])+'  '+fmt(point['target_uv'])+'    '+'    '.join(fmt(point[mode]['reference_uv']) for mode in ['ORACLE','SHIFT','FLIP']),fill='black')
        canvas.save(folder/(row['id']+'.png'))
    for start in range(0,len(rows),4):
        sheet=Image.new('RGB',(1152,600),'white')
        for j,row in enumerate(rows[start:start+4]):sheet.paste(Image.open(folder/(row['id']+'.png')).resize((576,300)),((j%2)*576,(j//2)*300))
        sheet.save(folder/('contact_%02d.png'%(start//4)))
    freeze(OUT/'audit'/('mapping_'+stage+'_r2.json'),dict(records=records,semantic_oracle_proven=False,manual_review='NOT_RUN'))
    return records

def transport_validation(rows):
    records=[]
    for row in rows:
        atlas=np.asarray(Image.open(row['atlas_folder']+'/atlas0.png').convert('L'),np.float32)
        for sigma in CONFIG['sigma_sensitivity']:
            m,_,t,r=matrices(row,sigma=sigma);wrong,_,_,_=matrices(row,mode='SHIFT',sigma=sigma);valid=m.sum(1)>.99
            def sample(uv):return cv2.remap(atlas,(uv[:,0]*511).astype('float32')[:,None],(uv[:,1]*511).astype('float32')[:,None],cv2.INTER_LINEAR).ravel()
            truth=sample(t[0]);ref=sample(r[0]);corr=lambda x:float(np.corrcoef(x[valid],truth[valid])[0,1])
            records.append(dict(id=row['id'],pattern=row['pattern'],sigma=sigma,oracle_corr=corr(m@ref),shift_corr=corr(wrong@ref)))
    primary=[x for x in records if x['sigma']==CONFIG['sigma_uv']]
    return dict(passed=all(x['oracle_corr']>.5 and x['oracle_corr']-x['shift_corr']>.05 for x in primary),records=records,criterion='known rendered scalar texture at actual feature-grid coordinates: oracle corr>0.5 and oracle-minus-shift>0.05 for each identity; sigma0.03 retained, not selected from model outputs')

def main():
    assert not (OUT/'images/pilot').exists(),'不能在模型Pilot输出后改变渲染器'
    rows=synthesize();mapping_review(rows)
    from tools.e53_texture import calibrate
    result=calibrate(rows);freeze(OUT/'audit/texture_calibration_r2.json',result);assert result['passed'],'TEXTURE_CALIBRATION_FAIL'
    transport=transport_validation(rows);freeze(OUT/'audit/transport_calibration_r2.json',transport);assert transport['passed'],'UV_TRANSPORT_CALIBRATION_FAIL'
    aliases={'splits/pilot.json':'splits/pilot_r2.json','audit/renderer_pilot.json':'audit/renderer_pilot_r2.json','audit/mapping_pilot.json':'audit/mapping_pilot_r2.json','audit/texture_calibration.json':'audit/texture_calibration_r2.json','manifest/synthetic_pilot_sha256.json':'manifest/synthetic_pilot_r2_sha256.json'}
    for old,new in aliases.items():
        p=OUT/old
        if p.exists() and read(p)!=read(OUT/new):freeze(p.with_name(p.stem+'_r1'+p.suffix),read(p))
        write(p,read(OUT/new))
    write(OUT/'data_revision.json',dict(active=2,reason='known-UV scalar transport aliasing detected before model Pilot; lower texture bandwidth, preserve all24/64 IDs and all controls; HP sigma8 validated on renderer truth',commit=commit(),sigma_uv=CONFIG['sigma_uv']))
    decision(mapping='NUMERIC_PASS_MANUAL_REVIEW_REQUIRED',texture='CONTROL_CALIBRATION_PASS')
    print('E53 DATA PASS',len(rows),Counter(x['pattern'] for x in rows),flush=True)

if __name__=='__main__':main()
