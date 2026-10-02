"""自动top1 self-reference；严格/放宽仅由train覆盖率决策。"""
import cv2
import numpy as np
from PIL import Image
from models.local_pattern_field import patch_geometry
from tools.e33_protocol import PROTOCOL


def structural_proxy(sketch,foreground):
    gray=cv2.cvtColor(np.asarray(sketch),cv2.COLOR_RGB2GRAY)
    strong=(gray<64).astype(np.uint8)
    edge=cv2.Canny(gray,40,120)>0
    long=np.zeros(strong.shape,np.uint8)
    lines=cv2.HoughLinesP(strong*255,1,np.pi/180,threshold=60,minLineLength=64,maxLineGap=5)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:cv2.line(long,(int(x1),int(y1)),(int(x2),int(y2)),1,1)
    for k in ((1,61),(61,1)):
        long|=cv2.morphologyEx(strong,cv2.MORPH_OPEN,np.ones(k,np.uint8))
    neighbors=cv2.filter2D(edge.astype(np.uint8),cv2.CV_16U,np.ones((3,3),np.uint8))
    junction=edge&(neighbors>=5)
    distance=cv2.distanceTransform(foreground.astype(np.uint8),cv2.DIST_L2,5)
    ridge=(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-.01)&(distance>5)&edge
    # holes的边界已由silhouette distance保护；额外面积计入Qstruct。
    filled=foreground.astype(np.uint8).copy();flood=filled.copy()
    cv2.floodFill(flood,np.zeros((514,386),np.uint8),(0,0),1)
    holes=(flood==0)&~foreground
    proxy=cv2.dilate((long.astype(bool)|ridge|junction|holes).astype(np.uint8),np.ones((5,5),np.uint8))>0
    return proxy,edge,junction,holes,distance


def integral(arr):
    return cv2.integral(np.asarray(arr,dtype=np.float64))


def region(ii,x,y,n):
    return (ii[y+n,x+n]-ii[y,x+n]-ii[y+n,x]+ii[y,x])/(n*n)


def select_reference(target,sketch,foreground,geometry,mode='strict',exclude_structure=True):
    cfg=PROTOCOL[mode];rgb=np.asarray(target)
    proxy,edge,junction,holes,distance=structural_proxy(sketch,foreground)
    # 对现有E32的dense pseudo-field池化，避免给候选另拟合一个估计器。
    g=cv2.resize(geometry.transpose(1,2,0),(384,512),interpolation=cv2.INTER_LINEAR)
    ori=g[...,:2];ori/=np.maximum(np.linalg.norm(ori,axis=-1,keepdims=True),1e-8)
    c=g[...,3];logf=g[...,2]
    signals=[foreground,distance>=5,c,c*ori[...,0],c*ori[...,1],c*logf,c*logf*logf,
             proxy,edge,junction,holes]
    ii=[integral(v) for v in signals];candidates=[]
    for n in PROTOCOL['candidate_sizes']:
        for y in range(0,513-n,16):
            for x in range(0,385-n,16):
                v=[region(a,x,y,n) for a in ii]
                purity,interior,conf=v[:3]
                if purity<.98 or interior<1-1e-8 or conf<cfg['confidence']:continue
                coherence=float(np.hypot(v[3],v[4])/max(conf,1e-8))
                consistency=float(np.exp(-np.sqrt(max(v[6]/conf-(v[5]/conf)**2,0))))
                if coherence<cfg['orientation_consistency'] or consistency<cfg['period_consistency']:continue
                if exclude_structure and (v[7]>.10 or v[8]>.10):continue
                patch=rgb[y:y+n,x:x+n]
                local=cv2.resize(patch,(4,4),interpolation=cv2.INTER_AREA)/255
                hom=float(np.exp(-local.std((0,1)).mean()/.25))
                gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(np.float64);gray-=gray.mean()
                variance=max(float(np.mean(gray*gray)),1e-8)
                sim=max(float(np.mean(gray[:,lag:]*gray[:,:-lag])/variance) for lag in range(4,min(25,n//2)))
                struct=float(v[7]+v[9]+v[10])
                score=.30*coherence+.25*consistency+.20*hom+.15*interior+.10*np.clip(sim,0,1)-.20*struct
                candidates.append(dict(box=[x,y,x+n,y+n],score=float(score),confidence=float(conf),
                    orientation_consistency=coherence,period_consistency=consistency,homogeneity=hom,
                    self_similarity=float(np.clip(sim,0,1)),structural_occupancy=float(v[7]),edge_density=float(v[8]),
                    junction_density=float(v[9]),hole_density=float(v[10])))
    candidates.sort(key=lambda q:(-q['score'],q['box']))
    for q in candidates:
        native=patch_geometry(target.crop(tuple(q['box'])))
        if native['confidence']>=cfg['confidence']:
            return dict(controlled_available=True,mode=mode,candidate_count=len(candidates),**q,native_geometry=native)
    return dict(controlled_available=False,mode=mode,candidate_count=len(candidates))
