"""冻结的纹样描述子。频谱/方向/周期是诊断代理，不是纹样身份分类器。"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks

PROTOCOL = dict(version='PATTERN-FIDELITY-20261029-v1', new_training=0,
    scope='all analyses on previously viewed development data; no new confirmation claim',
    main_split='P2b_swap32', supplemental_split='viewed128',
    primary='gray_fft_new: symmetric nearest-patch Jensen-Shannon divergence on reference-eligible regular-pattern identities',
    primary_models=['B','T1','T2','T3'], primary_family='three T3-baseline comparisons; Holm',
    secondary_family='all remaining main-split internal endpoints and three baselines jointly Holm',
    supplemental_families='C singleton per endpoint; two external contrasts per endpoint; two ablations per endpoint, separately by split',
    statistics=dict(unit='identity; aggregate patches first',bootstrap=10000,signflip=10000,seed=291042,
        CI='paired percentile 95%, descriptive unadjusted CI',missing='pairwise complete; no imputation; retain failures; publish per-model and pairwise denominators'),
    mask='frozen sketch-only garment mask; evaluation only; no generated-mask selection',
    patches=dict(size=64,stride=32,erosion_kernel=11,occupancy_min=1.0,minimum=4,maximum=8,
        selection='row-major candidate grid then equally spaced rounded indices; no output inspected'),
    reference='OpenCV RGB decode; LANCZOS384x512; fixed4x4 grid64; apparent image scale, not physical textile scale',
    orientation=dict(bins=24,gradient_sigma=1,tensor_sigma=2,angle='line tangent modulo180',
        confidence_min=.35,gray_std_min=2/255.,histogram='energy*coherence weighted; JS base2; angle from unbinned pixel circular moment'),
    spectrum=dict(size=64,window='2D Hann',demean=True,power=True,DC_radius_removed=1,
        smoothing_sigma=1,distance='JS base2 [0,1]',views=['RGB channel-summed power','luminance-normalized gray']),
    periodicity=dict(lags=[3,31],autocorrelation_min=.2,prominence_min=.1,
        reliable_patch_fraction=.5,relative_IQR_max=.35,
        classes=['stripe','plaid_grid','geometric_repeat'],
        interpretation='apparent normalized image period; no physical calibration'),
    category_rules=dict(solid='no visible motif; nearly uniform',stripe='dominant parallel repeated bands',
        plaid_grid='two intersecting regular band systems',floral_irregular='floral/organic or nonregular print',
        geometric_repeat='visible regular repeated geometric motifs',text_logo='letters, numerals or logos',
        natural_stochastic='denim, weave or random nonperiodic texture',ambiguous='insufficient evidence/mixed class; retain notes'),
    labels='AI reference-input annotation only; never preference ratings; ambiguous retained; labels frozen before model metrics',
    minimum_meaningful_difference=None,
    interpretation='do not import DISTS .005 threshold to new metrics; effect sizes and sensitivity first',
    STSIM='optional; N/A unless implementation, equivalence and license verified; no SSIM substitution',
    motif='no validated motif classifier or OCR; N/A, no invented identity accuracy',
    LOTS='text+sketch supplement only; no reference-pattern ranking')


def gray(rgb):
    return np.asarray(rgb,dtype=np.float64)@np.array([.2126,.7152,.0722])/255.


def gray_rgb(rgb):
    y=gray(rgb);normalized=np.clip(.5+.18*(y-y.mean())/max(y.std(),2/255.),0,1)
    return np.repeat(np.round(normalized[...,None]*255).astype('uint8'),3,axis=2)


def js(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a/max(a.sum(),1e-20);b=b/max(b.sum(),1e-20)
    m=(a+b)/2
    return float(.5*np.sum(a*np.log2((a+1e-20)/(m+1e-20)))+.5*np.sum(b*np.log2((b+1e-20)/(m+1e-20))))


def spectrum(rgb, color=False):
    x=np.asarray(rgb,float)/255. if color else gray(rgb)[...,None]
    window=np.outer(np.hanning(64),np.hanning(64))
    x=x-x.mean((0,1),keepdims=True)
    power=np.fft.fftshift(abs(np.fft.fft2(x*window[...,None],axes=(0,1)))**2,axes=(0,1)).sum(2)
    yy,xx=np.mgrid[-32:32,-32:32];power[xx*xx+yy*yy<=1]=0
    power=gaussian_filter(power,1);power[xx*xx+yy*yy<=1]=0
    if power.sum()<1e-12:return None
    return (power/power.sum()).astype('float32')


def direction(rgb):
    y=gray(rgb);gy=gaussian_filter(y,1,order=(1,0));gx=gaussian_filter(y,1,order=(0,1))
    jxx=gaussian_filter(gx*gx,2);jyy=gaussian_filter(gy*gy,2);jxy=gaussian_filter(gx*gy,2)
    energy=jxx+jyy;coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(energy+1e-20)
    angle=(.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2)%np.pi
    weight=(energy*coherence)[4:-4,4:-4];theta=angle[4:-4,4:-4]
    hist=np.histogram(theta,bins=24,range=(0,np.pi),weights=weight)[0]
    if y.std()<2/255. or hist.sum()<1e-12:return dict(hist=None,angle=None,confidence=0.)
    hist=hist/hist.sum()
    moment=np.sum(weight*np.exp(2j*theta))/weight.sum();confidence=abs(moment)
    return dict(hist=hist.astype('float32'),angle=float(np.angle(moment)/2%np.pi*180/np.pi),confidence=float(confidence))


def peak_period(profile):
    profile=np.asarray(profile,float)
    if profile[0]<=1e-12:return (None,0.)
    profile=profile/profile[0]
    peaks,props=find_peaks(profile[2:33],height=.2,prominence=.1)
    peaks=peaks+2
    keep=[i for i,p in enumerate(peaks) if 3<=p<=31]
    if not keep:return (None,0.)
    i=max(keep,key=lambda i:props['prominences'][i]);p=int(peaks[i])
    return p/64.,float(props['prominences'][i])


def period(rgb):
    y=gray(rgb);y=y-y.mean()
    if y.std()<2/255.:return dict(value=None,x=None,y=None,confidence=0.)
    # 非循环线性自相关，补零至128避免边缘首尾产生假峰。
    win=np.outer(np.hanning(64),np.hanning(64));signal=y*win
    ac=np.fft.fftshift(np.fft.ifft2(abs(np.fft.fft2(signal,s=(128,128)))**2).real)
    d=direction(rgb);line=(d['angle'] or 0)*np.pi/180;normal=line-np.pi/2
    lags=np.arange(34);profile=map_coordinates(ac,[64+lags*np.sin(normal),64+lags*np.cos(normal)],order=1)
    value,confidence=peak_period(profile)
    x,cx=peak_period(ac[64,64:98]);yy,cy=peak_period(ac[64:98,64])
    return dict(value=value,x=x,y=yy,confidence=confidence,confidence_x=cx,confidence_y=cy)


def descriptors(images):
    return [dict(gray_fft=spectrum(im),rgb_fft=spectrum(im,True),orientation=direction(im),period=period(im)) for im in images]


def nearest(first,second,fn):
    a=[x for x in first if x is not None];b=[x for x in second if x is not None]
    if not a or not b:return None
    matrix=np.array([[fn(x,y) for y in b] for x in a])
    return float(.5*(matrix.min(0).mean()+matrix.min(1).mean()))


def stable_period(desc,axis='value'):
    values=[d['period'][axis] for d in desc if d['period'].get(axis) is not None]
    if len(values)<max(4,int(np.ceil(len(desc)*.5))):return None
    med=float(np.median(values));iqr=float(np.subtract(*np.percentile(values,[75,25])))
    return med if iqr/max(med,1e-20)<=.35 else None


def compare(first,second,regular=False,directional=False):
    result={key:nearest([d[key] for d in first],[d[key] for d in second],js) for key in ['gray_fft','rgb_fft']}
    result['orientation_hist']=nearest([d['orientation']['hist'] for d in first],[d['orientation']['hist'] for d in second],js) if directional else None
    angles=lambda data:[d['orientation']['angle'] if d['orientation']['confidence']>=.35 else None for d in data]
    result['orientation_angle']=nearest(angles(first),angles(second),lambda a,b:min(abs(a-b),180-abs(a-b))) if directional else None
    for axis in ['value','x','y']:
        a=stable_period(first,axis);b=stable_period(second,axis)
        result['period_'+axis]=abs(float(np.log2(a/b))) if regular and a and b else None
        result['period_gen_'+axis]=a;result['period_ref_'+axis]=b
    return result
