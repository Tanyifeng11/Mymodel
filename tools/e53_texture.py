"""固定目标UV上的可读性、纹样身份与负对照；白图不能靠argmax成为成功。"""
import cv2
import numpy as np
from PIL import Image

def project(image,uv,labels,size=128):
    a=np.asarray(image.convert('RGB'),np.float32)/255
    mask=np.zeros(labels.shape,bool)
    for p in np.unique(labels[labels>0]):mask|=cv2.erode((labels==p).astype('uint8'),np.ones((7,7),np.uint8))>0
    ids=np.minimum(size-1,np.maximum(0,(uv[mask]*(size-1)).astype(int)))
    flat=ids[:,1]*size+ids[:,0];count=np.bincount(flat,minlength=size*size).reshape(size,size)
    out=np.stack([np.bincount(flat,weights=a[mask,c],minlength=size*size).reshape(size,size) for c in range(3)],-1)
    out/=np.maximum(count[...,None],1)
    return out.astype('float32'),count>0

def descriptor(rgb,support):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);weight=support.astype('float32')
    # 受16×16网格采样带宽约束的纹样使用8格高通尺度；只由渲染真值校准。
    mean=cv2.GaussianBlur(gray*weight,(0,0),8)/np.maximum(cv2.GaussianBlur(weight,(0,0),8),1e-8)
    hp=gray-mean;x=hp[support]
    readable=bool(len(x)>64 and np.std(gray[support])>=.015 and np.std(x)>=.008)
    x=x-x.mean();return x/max(float(np.linalg.norm(x)),1e-12),readable,float(np.std(x))

def score(image,row):
    d=np.load(row['geometry']);uv,lab=d['target_uv'],d['target_labels'];generated,support=project(image,uv,lab)
    v,valid,contrast=descriptor(generated,support)
    candidates=[]
    for k in range(8):
        rgb,ss=project(Image.open(row['atlas_folder']+'/correct%d.png'%k),uv,lab)
        assert np.array_equal(ss,support);u,_,_=descriptor(rgb,support);candidates.append(float(np.dot(v,u)))
    if not valid:return dict(Q_tex_corr=0.,Q_tex_identity=0,Q_tex_margin=0.,predicted_texture=None,readable=False,hp_std=contrast,scores=candidates,reason='NO_READABLE_PATTERN')
    prediction=int(np.argmax(candidates));return dict(Q_tex_corr=candidates[0],Q_tex_identity=int(prediction==0),Q_tex_margin=candidates[0]-max(candidates[1:]),predicted_texture=prediction,readable=True,hp_std=contrast,scores=candidates,reason=None)

def calibrate(rows):
    records=[]
    for row in rows:
        true=score(Image.open(row['gt']),row);wrong=score(Image.open(row['atlas_folder']+'/correct1.png'),row)
        white=score(Image.new('RGB',(384,512),'white'),row)
        records.append(dict(id=row['id'],true=true,wrong=wrong,white=white))
    passed=all(r['true']['Q_tex_identity']==1 and r['true']['Q_tex_corr']>.99 and r['wrong']['Q_tex_identity']==0 and r['wrong']['predicted_texture']==1 and not r['white']['readable'] and r['white']['Q_tex_identity']==0 for r in records)
    return dict(passed=passed,n=len(rows),records=records,scope='renderer truth, wrong identity and blank negative controls; not a guarantee of semantic readout on generated generic clothes')
