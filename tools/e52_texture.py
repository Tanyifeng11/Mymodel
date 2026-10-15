"""预先校准的灰度纹样检索；明确记录重采样与几何偏移敏感性。"""
import cv2
import numpy as np
from PIL import Image
from tools.e52_protocol import read,write,OUT

def descriptor(image,mask):
    gray=np.asarray(image.convert('L'),np.float32)/255
    gray=cv2.resize(gray,(96,128),interpolation=cv2.INTER_AREA)
    mask=cv2.resize(mask.astype('uint8'),(96,128),interpolation=cv2.INTER_NEAREST)>0
    mask=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
    if mask.sum()<32:return None
    channels=[gray-cv2.GaussianBlur(gray,(0,0),sigma) for sigma in [1.,3.,8.]]
    channels.append(gray)
    parts=[]
    for value in channels:
        x=value[mask];x=(x-x.mean())/max(float(x.std()),.01);parts.append(x)
    vector=np.concatenate(parts);return vector/max(float(np.linalg.norm(vector)),1e-8)

def score(image,row,correct):
    root=row['atlas_folder'];geometry=np.load(row['geometry']);mask=geometry['target_labels']>0
    descriptor_generated=descriptor(image,mask)
    if descriptor_generated is None:return dict(texture_retrieval_correct=None,texture_margin=None,texture_distances=None)
    library=[descriptor(Image.open(root+'/correct%d.png'%k),mask) for k in range(8)]
    distances=[float(1-np.dot(descriptor_generated,candidate)) for candidate in library]
    wrong=min(distances[k] for k in range(8) if k!=correct)
    # 方向、周期和相位只有条纹有清晰可读性时再报告；不能给复杂纹样编造角度。
    from models.local_pattern_field import patch_geometry
    box=(120,170,265,315)
    generated=patch_geometry(image.crop(box));truth=patch_geometry(Image.open(root+'/correct%d.png'%correct).crop(box))
    angle=period=None
    if row['pattern']=='stripe' and generated['valid'] and truth['valid'] and generated['confidence']>=.25:
        delta=abs(generated['orientation']-truth['orientation'])%180;angle=float(min(delta,180-delta))
        if generated['frequency']>0 and truth['frequency']>0:period=float(abs(1/generated['frequency']-1/truth['frequency']))
    return dict(texture_retrieval_correct=int(int(np.argmin(distances))==correct),
        texture_margin=wrong-distances[correct],texture_distances=distances,
        orientation_error=angle,period_error=period,phase_error=None,
        phase_na_reason='no validated generated-to-UV phase recovery; not inferred from LPIPS')

def calibrate(rows):
    records=[]
    for row in rows:
        for correct in [0,1]:
            image=Image.open(row['atlas_folder']+'/correct%d.png'%correct)
            same=score(image,row,correct)
            resized=image.resize((768,1024),Image.Resampling.BILINEAR).resize((384,512),Image.Resampling.BILINEAR)
            blur=Image.fromarray(cv2.GaussianBlur(np.asarray(image),(0,0),1.))
            shifted=Image.fromarray(cv2.warpAffine(np.asarray(image),np.float32([[1,0,3],[0,1,3]]),(384,512),borderValue=(255,255,255)))
            records.append(dict(id=row['id'],correct=correct,ideal=same,
                resampled=score(resized,row,correct),blurred=score(blur,row,correct),
                translated_3px=score(shifted,row,correct)))
    ideal=np.mean([r['ideal']['texture_retrieval_correct'] for r in records])
    resampled=np.mean([r['resampled']['texture_retrieval_correct'] for r in records])
    assert ideal>=.95 and resampled>=.90,('TEXTURE_METRIC_CALIBRATION_FAIL',ideal,resampled)
    return dict(validated_on_synthetic_controls=True,ideal_accuracy=float(ideal),resampled_accuracy=float(resampled),
        blurred_accuracy=float(np.mean([r['blurred']['texture_retrieval_correct'] for r in records])),
        shifted_accuracy=float(np.mean([r['translated_3px']['texture_retrieval_correct'] for r in records])),
        library_n=8,distance='1-cosine of standardized grayscale/highpass at sigma1,3,8 and grayscale,128x96',
        geometry_sensitive=True,not_a_natural_texture_identity_validator=True,records=records)
