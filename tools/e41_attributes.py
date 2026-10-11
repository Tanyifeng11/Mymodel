"""E41 无训练属性读出。像素单位仅表示图像坐标，不代表真实布料尺度。"""
import math
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

ATTRS = ['orientation', 'period', 'scale', 'density']
LEVELS = dict(orientation=[0.,30.,45.,90.], period=[32.,48.,64.],
              scale=[12.,18.,24.], density=[1/3,2/3,1.])
BASE = dict(orientation=30., period=48., scale=18., density=2/3)


def render(params, phase=0, color=0, shape='ellipse', size=256):
    """固定方格晶格；方向/单元大小/占位率分别变化，缺位结构单独审计。"""
    p=params['period'];s=params['scale'];a=math.radians(params['orientation'])
    y,x=np.mgrid[:size,:size].astype(float)
    x+=phase*p;y+=phase*p*.73
    ix=np.floor(x/p).astype(int);iy=np.floor(y/p).astype(int)
    u=(x/p%1-.5)*p;v=(y/p%1-.5)*p
    q=u*math.cos(a)+v*math.sin(a);r=-u*math.sin(a)+v*math.cos(a)
    # 每三个晶格单元固定占位1/2/3个；只改变占位，不改变单元轮廓。
    occupied=(ix+2*iy)%3 < round(params['density']*3)
    signed=(q/(s/2))**2+(r/(s/4))**2
    if shape=='leaf':signed += .16*np.sin(q/max(s,1)*math.pi*2)*r/(s/4)
    alpha=np.clip((1-signed)*s/2+.5,0,1)*occupied
    colors=[([48,48,48],[208,208,208]),([120,70,30],[218,200,170])]
    fg,bg=map(np.asarray,colors[color])
    return Image.fromarray(np.rint(bg[None,None,:]+alpha[:,:,None]*(fg-bg)).clip(0,255).astype('uint8'))


def angular_error(a,b):
    return abs((float(a)-float(b)+90)%180-90)


def error(a,b,attribute):
    if a is None or b is None:return None
    if attribute=='orientation':return angular_error(a,b)
    if attribute in ['period','scale']:
        return abs(math.log(max(float(a),1e-8)/max(float(b),1e-8)))
    return abs(float(a)-float(b))


def frequency_scale(component):
    """孤立单元的多尺度频率响应；用固定几何标准查表换算像素直径。"""
    return _frequency_scale(component.shape,component.astype('uint8').tobytes())


@lru_cache(maxsize=8192)
def _frequency_scale(shape,content):
    component=np.frombuffer(content,dtype='uint8').reshape(shape)
    a=component.astype(float);n=64
    a=cv2.copyMakeBorder(a,16,16,16,16,cv2.BORDER_CONSTANT,value=0)
    if max(a.shape)>n:a=cv2.resize(a,(n,n))
    a=np.pad(a,((0,max(0,n-a.shape[0])),(0,max(0,n-a.shape[1]))))[:n,:n]
    power=abs(np.fft.fft2(a))**2
    fy,fx=np.meshgrid(np.fft.fftfreq(n),np.fft.fftfreq(n),indexing='ij');r=np.hypot(fx,fy)
    sigmas=np.geomspace(.6,12,48)
    responses=[float(np.sum(power*((2*np.pi*s*r)**2*np.exp(-2*np.pi**2*s*s*r*r))**2)) for s in sigmas]
    return float(sigmas[int(np.argmax(responses))])


@lru_cache(None)
def scale_standards():
    diameters=np.arange(6.,33.,1.);values=[]
    yy,xx=np.mgrid[:48,:48];xx=xx-24;yy=yy-24
    for d in diameters:
        a=((xx/(d/2))**2+(yy/(d/4))**2<=1).astype(float)
        values.append(frequency_scale(a[12:37,7:42]))
    # 无拟合、无模型优化；并列频率网格取标准直径中位数。
    u=sorted(set(values));ds=[float(np.median(diameters[np.asarray(values)==v])) for v in u]
    return np.array(u),np.array(ds)


def extract(image, mask=None):
    a=np.asarray(image.convert('RGB'));gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(float)/255
    h,w=gray.shape;valid=np.ones((h,w),bool) if mask is None else mask.astype(bool)
    result={k:None for k in ATTRS};result.update(valid_pixels=int(valid.sum()),method='FFT angular/radial; isolated-unit multiscale spectral width; component occupancy and edge density')
    if valid.sum()<4096:result['status']='insufficient_region';return result
    # 避免掩码边界伪边缘；窗函数与均值只作用于内部。
    interior=cv2.erode(valid.astype('uint8'),np.ones((5,5),np.uint8))>0
    centered=(gray-float(gray[valid].mean()))*valid
    power=abs(np.fft.fft2(centered*np.outer(np.hanning(h),np.hanning(w))))**2
    fy,fx=np.meshgrid(np.fft.fftfreq(h),np.fft.fftfreq(w),indexing='ij');radius=np.hypot(fx,fy)
    band=(radius>=.06)&(radius<=.35)
    gx=float(np.sum(power[band]*(fx[band]**2-fy[band]**2)))
    gy=float(np.sum(power[band]*2*fx[band]*fy[band]))
    total=float(np.sum(power[band]*radius[band]**2));conf=math.hypot(gx,gy)/max(total,1e-12)
    result['orientation_confidence']=conf
    if conf>=.15:result['orientation']=(math.degrees(math.atan2(gy,gx))/2+90)%180
    radii=np.arange(1,61)/max(h,w)
    radial=np.array([power[abs(radius-r)<.5/max(h,w)].sum() for r in radii])
    eligible=(radii>=1/80)&(radii<=1/26)
    peaks,_=find_peaks(radial)
    peaks=[i for i in peaks if eligible[i]]
    if peaks:
        strongest=max(radial[i] for i in peaks)
        selected=min((i for i in peaks if radial[i]>=strongest*.40),key=lambda i:radii[i])
        result['period']=float(1/radii[selected]);result['period_peak_fraction']=float(radial[selected]/max(radial[eligible].sum(),1e-12))
    else:result['period_peak_fraction']=0.
    result['radial_period_unrestricted']=result['period']
    # 缺位的3x3超晶格会产生更低的径向峰。固定轴向晶格投影用作消歧，
    # 与全径向读出一起保留；自然织物上的轴向假设并非已验证真值。
    fxline=np.fft.fftfreq(w);fyline=np.fft.fftfreq(h)
    candidates=[]
    for frequencies,axis_power in [(fxline,power[0,:]),(fyline,power[:,0])]:
        indices=np.flatnonzero((frequencies>=1/80)&(frequencies<=1/26))
        i=int(indices[np.argmax(axis_power[indices])]);offset=0.
        logs=np.log(axis_power[max(0,i-1):i+2]+1e-12)
        if len(logs)==3 and abs(logs[0]-2*logs[1]+logs[2])>1e-10:
            offset=float(np.clip(.5*(logs[0]-logs[2])/(logs[0]-2*logs[1]+logs[2]),-.5,.5))
        candidates.append((float(axis_power[i]),float(len(frequencies)/(i+offset))))
    if max(v[0] for v in candidates)>1e-8:result['period']=max(candidates)[1]
    result['period_axis_assumption']=True
    # 无训练前景定位；自然生成上仅为代理，不将检测到的斑块宣称为语义单元。
    quant=np.rint(gray*255).astype('uint8');threshold,_=cv2.threshold(quant[valid],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    foreground=(quant<threshold+1)&interior
    count,labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype('uint8'),8)
    widths=[];components=0;standards,diameters=scale_standards()
    for i in range(1,count):
        x,y,bw,bh,area=stats[i]
        if not (6<=area<=1500 and 3<=bw<=48 and 3<=bh<=48):continue
        if x<3 or y<3 or x+bw>w-3 or y+bh>h-3:continue
        patch=(labels[y:y+bh,x:x+bw]==i).astype(float)
        spectral=frequency_scale(patch)
        widths.append(float(np.interp(spectral,standards,diameters)));components+=1
    result['components']=components
    result['edge_density']=float((cv2.Canny(quant,50,120)>0)[interior].mean()) if interior.any() else None
    result['foreground_fraction']=float(foreground.sum()/valid.sum())
    if len(widths)>=3:result['scale']=float(np.median(widths))
    if result['period'] is not None and components>=3:
        result['density']=float(components*result['period']**2/valid.sum())
    result['status']='measured_proxy';return result


def latent_readout(array):
    """保留多通道空间频谱；仅由独立编码标准训练的 probe 解读，不直接对latent通道作RGB FFT。"""
    a=np.asarray(array,float).reshape(-1,*array.shape[-2:]);out=[]
    for ch in a:
        power=np.log1p(abs(np.fft.fftshift(np.fft.fft2(ch-ch.mean())))**2)
        out.extend(cv2.resize(power,(16,16),interpolation=cv2.INTER_AREA).ravel())
        out.extend(cv2.resize(ch,(8,8),interpolation=cv2.INTER_AREA).ravel())
    return np.asarray(out,np.float32)
