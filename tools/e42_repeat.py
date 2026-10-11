"""E42 CPU重复可观测性：固定几何窗口、Lab高通、自相关及不重叠块对。

这里只测图像像素域的局部重复；低对比不等于没有纹样，不输出密度。
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

OUT = Path('output_eval/e42_real_fabric_repeat_gate_20261011')
CONFIG = dict(sizes=[96,128,160,192], foreground=.95, erosion_at_384=9,
              sigma=3., min_channel_rms=.08, min_image_rms=.20,
              min_shift=8, max_period_fraction=1/3, candidates=12,
              min_pairs=3, pair_limit=12, min_half_shift_difference=.35,
              min_nonoverlap_rois=2, gt_angle_tolerance=20., gt_log_period_tolerance=.35,
              bootstrap=2000, seed=42042,
              signature_angles=36, signature_radii=16,
              signature_log_scale_shifts=[-3,-2,-1,0,1,2,3])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def overlap(a,b):
    x,y,w,h=a;u,v,s,t=b
    return max(x,u)<min(x+w,u+s) and max(y,v)<min(y+h,v+t)


def windows(mask):
    """大窗优先、同行优先的完全几何NMS，GT/G不能按检测分数选窗。"""
    h,w=mask.shape;all_rois=[];chosen=[]
    for s in sorted(CONFIG['sizes'],reverse=True):
        for y in range(0,h-s+1,s//2):
            for x in range(0,w-s+1,s//2):
                if float(mask[y:y+s,x:x+s].mean())>=CONFIG['foreground']:
                    roi=[x,y,s,s];all_rois.append(roi)
                    if not any(overlap(roi,q) for q in chosen):chosen.append(roi)
    return chosen,all_rois


def sketch_geometry(path,size):
    from eval.eval_utils import prepare_evaluation_masks
    b=prepare_evaluation_masks(size,sketch_path=str(path),mask_policy='sketch_only')
    stats=b['stats'];valid=stats['mask_source']=='sketch_flood_fill' and not stats['mask_low_confidence']
    k=max(1,round(CONFIG['erosion_at_384']*min(size[0]/384,size[1]/512)))
    if k%2==0:k+=1
    mask=cv2.erode(b['garment'].astype('uint8'),np.ones((k,k),np.uint8))>0
    rois,all_rois=windows(mask) if valid else ([],[])
    return mask,rois,dict(stats=stats,erosion=k,all_valid_windows=len(all_rois),
        window_hash=digest(rois),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        reason='ok' if len(rois)>=2 else ('NA_insufficient_interior' if valid else 'NA_sketch_mask_low_confidence'))


def highpass(rgb):
    lab=cv2.cvtColor(np.asarray(rgb,dtype=np.float32)/255,cv2.COLOR_RGB2LAB)
    hp=lab-cv2.GaussianBlur(lab,(0,0),CONFIG['sigma'])
    rms=np.sqrt(np.mean(hp**2,axis=(0,1)))
    valid=rms>=CONFIG['min_channel_rms']
    if not valid.any() or float(rms.max())<CONFIG['min_image_rms']:return None,rms.tolist()
    return hp[:,:,valid]/rms[valid],rms.tolist()


def candidates(ac,size):
    c=size//2;yy,xx=np.mgrid[:size,:size];rr=np.hypot(xx-c,yy-c)
    zone=(rr>=8)&(rr<=size/3)&((yy>c)|((yy==c)&(xx>c)))
    bg=ac[zone];median=float(np.median(bg));mad=max(float(np.median(abs(bg-median)))*1.4826,.01)
    peaks=zone&(ac==maximum_filter(ac,size=5,mode='constant'))
    pos=np.argwhere(peaks);order=sorted(pos,key=lambda p:-ac[tuple(p)])
    result=[]
    for y,x in order:
        dx=int(x-c);dy=int(y-c);p=float(np.hypot(dx,dy))
        # 排除沿单缝线方向平移不变的相关脊：真正重复须有离中心谷。
        ts=np.linspace(.20,.85,8)
        trough=float(min(ac[c+round(dy*t),c+round(dx*t)] for t in ts))
        if trough>.35:continue
        if any(np.hypot(dx-a['dx'],dy-a['dy'])<4 for a in result):continue
        result.append(dict(dx=dx,dy=dy,period=p,angle=float(np.degrees(np.arctan2(dy,dx))%180),
            corr=float(ac[y,x]),z=float((ac[y,x]-median)/mad),trough=trough))
        if len(result)>=CONFIG['candidates']:break
    return result


def pair_similarity(hp,dx,dy):
    """固定栅格、全局不重叠块对；不按块相关性择优采样。"""
    h,w,_=hp.shape;b=max(4,min(12,int(np.hypot(dx,dy)/3)))
    occupied=[];values=[]
    for y in range(0,h-b+1,b):
        for x in range(0,w-b+1,b):
            u=x+dx;v=y+dy
            if u<0 or v<0 or u+b>w or v+b>h:continue
            a=[x,y,b,b];q=[u,v,b,b]
            if overlap(a,q) or any(overlap(r,a) or overlap(r,q) for r in occupied):continue
            f=hp[y:y+b,x:x+b].reshape(-1);g=hp[v:v+b,u:u+b].reshape(-1)
            f=f-f.mean();g=g-g.mean();norm=float(np.linalg.norm(f)*np.linalg.norm(g))
            if norm<.02:continue
            values.append(float(np.dot(f,g)/norm));occupied.extend([a,q])
            if len(values)>=CONFIG['pair_limit']:break
        if len(values)>=CONFIG['pair_limit']:break
    # 半周期差排除单色区域的平移一致；不能代替重复块对证据。
    hx=round(dx/2);hy=round(dy/2)
    x0=max(0,-hx);x1=min(w,w-hx);y0=max(0,-hy);y1=min(h,h-hy)
    diff=float(np.sqrt(np.mean((hp[y0:y1,x0:x1]-hp[y0+hy:y1+hy,x0+hx:x1+hx])**2)))
    return dict(ncc=float(np.quantile(values,.25)) if len(values)>=3 else -1.,
        pair_count=len(values),pair_correlations=values,half_shift_difference=diff)


def signature(hp):
    """颜色幅度归一的极坐标频谱，仅作局部结构相容性代理。"""
    n=hp.shape[0];hann=np.outer(np.hanning(n),np.hanning(n))
    power=np.sum(abs(np.fft.fftshift(np.fft.fft2(hp*hann[:,:,None],axes=(0,1)),axes=(0,1)))**2,axis=2)
    angles=np.arange(36)*np.pi/36;radii=np.geomspace(3,n/4,16)
    x=(n//2+np.cos(angles[:,None])*radii).astype('float32')
    y=(n//2+np.sin(angles[:,None])*radii).astype('float32')
    s=np.log1p(cv2.remap(power.astype('float32'),x,y,cv2.INTER_LINEAR))
    s-=s.mean(axis=0,keepdims=True)
    return (s/(np.linalg.norm(s)+1e-9)).tolist()


def compatibility(a,b):
    a=np.asarray(a);b=np.asarray(b);best=-1.
    for shift in CONFIG['signature_log_scale_shifts']:
        left=a[:,max(0,shift):min(16,16+shift)]
        right=b[:,max(0,-shift):min(16,16-shift)]
        for rot in range(36):
            q=np.roll(right,rot,axis=0)
            best=max(best,float(np.sum(left*q)/(np.linalg.norm(left)*np.linalg.norm(q)+1e-9)))
    return best


def measure_window(rgb,roi):
    x,y,w,h=roi;hp,rms=highpass(rgb[y:y+h,x:x+w])
    if hp is None:return dict(roi=roi,valid=False,reason='NA_low_contrast',channel_rms=rms,candidates=[])
    hann=np.outer(np.hanning(w),np.hanning(w))
    ft=np.fft.fft2(hp*hann[:,:,None],axes=(0,1))
    ac=np.fft.fftshift(np.fft.ifft2(abs(ft)**2,axes=(0,1)).real,axes=(0,1)).mean(axis=2)
    ac/=max(ac[w//2,w//2],1e-9)
    proposal=candidates(ac,w)
    power=np.sum(abs(np.fft.fftshift(ft,axes=(0,1)))**2,axis=2)
    yy,xx=np.mgrid[:w,:w];radius=np.hypot(xx-w//2,yy-w//2)
    low_fraction=float(power[(radius>0)&(radius<3)].sum()/max(power[radius>0].sum(),1e-9))
    if low_fraction>=.70 and not proposal:
        return dict(roi=roi,valid=False,reason='NA_insufficient_three_cycles',channel_rms=rms,candidates=[],low_frequency_energy_fraction=low_fraction)
    # B另用无Hann的提议峰，不要求通过A阈值；两代理仍共享预处理。
    raw=np.fft.fft2(hp,axes=(0,1));rawac=np.fft.fftshift(np.fft.ifft2(abs(raw)**2,axes=(0,1)).real,axes=(0,1)).mean(axis=2)
    rawac/=max(rawac[w//2,w//2],1e-9)
    independent=candidates(rawac,w)
    shifts={(q['dx'],q['dy']):dict(q,a_proposal=True,b_proposal=False) for q in proposal}
    for q in independent:
        key=(q['dx'],q['dy'])
        if key in shifts:shifts[key]['b_proposal']=True
        else:
            q=dict(q,a_proposal=False,b_proposal=True)
            # 独立B提议不能绕过A；其A分数在联合规则中不启用。
            shifts[key]=q
    values=[]
    for q in shifts.values():
        q.update(pair_similarity(hp,q['dx'],q['dy']));values.append(q)
    return dict(roi=roi,valid=True,reason='measured_within_8_to_W_over_3',channel_rms=rms,
        candidates=values,signature=signature(hp),
        observability_limit='No label establishes whether an undetected repeat exceeds W/3 or is occluded.')


def decision(window,threshold,mode='joint'):
    if not window['valid']:return []
    passed=[]
    for q in window['candidates']:
        a=q['a_proposal'] and q['corr']>=threshold['corr'] and q['z']>=threshold['z']
        b=(q['ncc']>=threshold['ncc'] and q['pair_count']>=3 and
           q['half_shift_difference']>=CONFIG['min_half_shift_difference'])
        if (a and b) if mode=='joint' else (a if mode=='autocorr_only' else b):passed.append(q)
    return passed


def summarize(ws,threshold,mode='joint'):
    valid=[w for w in ws if w['valid']];found=[w for w in valid if decision(w,threshold,mode)]
    if len(valid)<2:return dict(status='NA',reason='NA_insufficient_measurable_rois',coverage=None,n=len(valid),k=len(found))
    strongest=[max(decision(w,threshold,mode),key=lambda q:q['ncc']) for w in found]
    two_d=any(any(abs((a['angle']-b['angle']+90)%180-90)>25 for a in decision(w,threshold,mode) for b in decision(w,threshold,mode)) for w in found)
    return dict(status='repeat' if len(found)>=2 else 'no_repeat_detected',
        reason='protocol_limited_repeat_observability',coverage=len(found)/len(valid),n=len(valid),k=len(found),
        repeat_1d_or_2d=('2d' if two_d else '1d') if len(found)>=2 else None,
        repeated_rois=[w['roi'] for w in found],period_px=[q['period'] for q in strongest],
        repeat_confidence=min([q['ncc'] for q in strongest]) if strongest else None)


def gt_stable(ws,threshold,mode):
    all_q=[decision(w,threshold,mode) for w in ws];all_q=[q for q in all_q if q]
    if len(all_q)<2:return False
    for i,qs in enumerate(all_q):
        for rs in all_q[i+1:]:
            for q in qs:
                for r in rs:
                    da=abs((q['angle']-r['angle']+90)%180-90)
                    if da<=20 and abs(np.log(q['period']/r['period']))<=.35:return True
    return False


def compatible_rois(left,right,threshold,match_threshold,mode):
    l=[w for w in left if decision(w,threshold,mode)]
    r=[w for w in right if decision(w,threshold,mode)]
    if len(l)<2 or len(r)<2:return dict(compatible=False,scores=[],reason='insufficient_repeat_rois')
    scores=[max(compatibility(a['signature'],b['signature']) for a in l) for b in r]
    return dict(compatible=sum(s>=match_threshold for s in scores)>=2,scores=scores,
        reason='local_color_normalized_structure_proxy')
