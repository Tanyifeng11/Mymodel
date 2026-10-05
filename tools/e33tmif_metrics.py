"""同身份、同GT支持的阶段比较；不可读保留在固定分母中。"""
import cv2
import numpy as np

def bootstrap(values):
    values=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(values): return dict(mean=None,ci95=None,n=0)
    draws=np.random.default_rng(32042).choice(values,(2000,len(values)),replace=True).mean(1)
    return dict(mean=float(values.mean()),ci95=np.percentile(draws,[2.5,97.5]).tolist(),n=len(values))

def axial(a,b):
    dot=np.sum(a*b,axis=0)
    return np.rad2deg(np.arccos(np.clip(dot,-1,1)))/2

def arm_geometry(geometry,support,weight):
    valid=support.astype(bool)&(geometry[3]>=.25)
    if valid.any():
        direction=np.average(geometry[:2,valid],axis=1,weights=weight[valid])
        angle=float(np.rad2deg(np.arctan2(direction[1],direction[0]))/2%180)
        confidence=float(np.average(geometry[3,valid],weights=weight[valid]))
    else: angle=confidence=None
    return dict(readable=bool(valid.any()),support_fraction=float(valid.sum()/max(support.sum(),1)),
                orientation_angle=angle,orientation_confidence=confidence,support_cells=int(valid.sum()))

def pair(geometries,support,weight,field=False):
    result={}
    for arm,prefix,sign in [('R90','r90',-1),('R180','r180',1)]:
        a,b=geometries['R0'],geometries[arm]
        valid=support.astype(bool) if field else support.astype(bool)&(a[3]>=.25)&(b[3]>=.25)
        error=float(np.average(axial(b[:2],sign*a[:2])[valid],weights=weight[valid])) if valid.any() else None
        result.update({prefix+'_error':error,prefix+'_readable':bool(valid.any()),
            prefix+'_coverage':float(valid.sum()/max(support.sum(),1)),
            prefix+'_success':bool(error is not None and error<=15)})
    result['fixed_support_cells']=int(support.sum())
    return result

def summarize(rows):
    assert len(rows)==len({r['id'] for r in rows})
    result=dict(case_count=len(rows),statistics={})
    keys=sorted({k for r in rows for k,v in r.items() if k not in ('id','arms','stage') and isinstance(v,(int,float,bool))})
    for k in keys: result['statistics'][k]=bootstrap([r.get(k) for r in rows])
    for prefix in ('r90','r180'):
        result['statistics'][prefix+'_error']=bootstrap([r[prefix+'_error'] for r in rows])
        readable=[r for r in rows if r[prefix+'_readable']]
        result['statistics'][prefix+'_success_readable']=bootstrap([r[prefix+'_success'] for r in readable])
        result[prefix+'_readable_N']=len(readable)
    result['arm_statistics']={}
    for arm in ('R0','R90','R180'):
        names=sorted({k for r in rows for k,v in r['arms'][arm].items() if isinstance(v,(int,float,bool))})
        result['arm_statistics'][arm]={k:bootstrap([r['arms'][arm].get(k) for r in rows]) for k in names}
    return result

def compare(before,after,threshold=.20):
    lookup={r['id']:r for r in after}
    assert set(lookup)=={r['id'] for r in before}
    pairs=[(r,lookup[r['id']]) for r in before]
    stats={key:bootstrap([a[key]-b[key] for a,b in pairs]) for key in
           ('r90_success','r180_success','r90_readable','r180_readable','r90_coverage','r180_coverage')}
    stats['response_error_increase']=bootstrap([b['r90_error']-a['r90_error'] for a,b in pairs
                                               if a['r90_readable'] and b['r90_readable']])
    lost=[(a,b) for a,b in pairs if a['r90_success'] and not b['r90_success']]
    newly_unreadable=sum(not b['r90_readable'] for a,b in lost)
    stats.update(lost_success_N=len(lost),newly_unreadable_N=newly_unreadable,
        still_readable_incorrect_N=len(lost)-newly_unreadable,
        readability_loss=bool(lost and newly_unreadable>len(lost)/2),
        significant_drop=bool(stats['r90_success']['mean']>=threshold and stats['r90_success']['ci95'][0]>0))
    # 条件均值不同可读分母时，不把两个条件误差均值直接相减。
    return stats

def frequency_diagnostics(image,mask):
    gray=np.asarray(image.convert('L'),dtype=np.float32)/255
    inner=cv2.erode((np.asarray(mask)>0).astype(np.uint8),np.ones((17,17),np.uint8))>0
    if not inner.any(): return dict(high_frequency_energy=None,local_contrast=None,fft_radial=None)
    low=cv2.GaussianBlur(gray,(0,0),2)
    contrast=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),2)-low*low,0))
    centered=(gray-gray[inner].mean())*inner
    window=np.hanning(gray.shape[0])[:,None]*np.hanning(gray.shape[1])[None,:]
    power=np.abs(np.fft.fftshift(np.fft.fft2(centered*window)))**2
    fy=np.fft.fftshift(np.fft.fftfreq(gray.shape[0]))[:,None]
    fx=np.fft.fftshift(np.fft.fftfreq(gray.shape[1]))[None,:]
    radius=np.sqrt(fx*fx+fy*fy)
    radial=np.histogram(radius,bins=np.linspace(0,.71,33),weights=power)[0]
    radial=radial/max(radial.sum(),1e-12)
    return dict(high_frequency_energy=float(power[radius>=.15].sum()/max(power.sum(),1e-12)),
                local_contrast=float(contrast[inner].mean()),fft_radial=radial.tolist())
