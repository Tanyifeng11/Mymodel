"""外轮廓与内部灰度纹理分开测量；统计以服装身份为单位。"""
import cv2
import numpy as np
from PIL import Image
from tools.e44_protocol import bootstrap


def external_contour(mask):
    result=np.zeros(mask.shape,np.uint8)
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result,contours,-1,1,1)
    return result.astype(bool)


def contour_f1(generated,sketch,tolerance=3):
    a,b=external_contour(generated),external_contour(sketch)
    if not b.any():return None
    if not a.any():return 0.
    near_a=cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,5)<=tolerance
    near_b=cv2.distanceTransform((~b).astype('uint8'),cv2.DIST_L2,5)<=tolerance
    precision=float(near_b[a].mean());recall=float(near_a[b].mean())
    return 2*precision*recall/max(precision+recall,1e-12)


def highpass_error(a,b,mask):
    def descriptor(image):
        gray=np.asarray(image.convert('L'),np.float32)/255
        std=float(gray[mask].std())
        if std<.02:return None
        hp=gray-cv2.GaussianBlur(gray,(0,0),3)
        return hp/std
    aa,bb=descriptor(a),descriptor(b)
    if aa is None or bb is None:return None
    return float(np.abs(aa-bb)[mask].mean())


def calibration():
    from models.local_pattern_field import patch_geometry
    x,y=np.meshgrid(np.arange(128),np.arange(128))
    stripe=(np.sin(x*2*np.pi/16)>0).astype('uint8')*140+50
    rgb=Image.fromarray(np.repeat(stripe[:,:,None],3,axis=2));rot=rgb.transpose(Image.Transpose.ROTATE_90)
    mask=np.zeros((128,128),bool);mask[20:-20,20:-20]=True
    same=highpass_error(rgb,rgb,mask);wrong=highpass_error(rgb,rot,mask)
    color=Image.fromarray(np.asarray(rgb)//2+25)
    changed=highpass_error(rgb,color,mask)
    assert same==0 and wrong>0 and changed<1e-5
    # 外轮廓 F1 对内部线条数量不敏感。
    assert contour_f1(mask,mask)==1
    geometry=[]
    for direction,coordinate in [(90,x),(0,y)]:
        for period in [8,16]:
            image=Image.fromarray(((np.sin(coordinate*2*np.pi/period)>0)*140+50).astype('uint8'))
            value=patch_geometry(image)
            assert value['valid'] and abs(value['orientation']-direction)<1 and abs(1/value['frequency']-period)<1
            geometry.append(dict(expected_direction=direction,expected_period=period,measured=value))
    return dict(same_error=same,rotated_error=wrong,affine_gray_color_error=changed,
        stripe_direction_period_controls=geometry,
        calibrated_scope='synthetic stripe, affine gray intensity; not a validated real motif identity metric',
        contour='RETR_EXTERNAL of estimated generated/sketch foreground, 3px tolerance; no interior edges')


def patch_readout(image,boxes):
    from models.local_pattern_field import patch_geometry
    values=[dict(box=box,**patch_geometry(image.crop(tuple(box)))) for box in boxes]
    valid=[r for r in values if r['valid'] and r['confidence']>=.25]
    result=dict(patches=values,n=len(values),readable_n=len(valid),orientation=None,period=None)
    if valid:
        angles=np.deg2rad([r['orientation']*2 for r in valid])
        z=np.average(np.exp(1j*angles),weights=[r['confidence'] for r in valid])
        result['orientation']=float((np.degrees(np.angle(z))/2)%180)
        result['period']=float(np.median([1/r['frequency'] for r in valid]))
        result['axial_concentration']=float(abs(z))
    return result


def paired(records,a,b,key):
    index={r['id']:r for r in records if r['variant']==b}
    grouped={}
    for row in records:
        if row['variant']!=a:continue
        # 此函数仅用于单seed Pilot；正式多seed由独立聚合入口处理。
        base=index[row['id']]
        if row.get(key) is not None and base.get(key) is not None:
            grouped.setdefault(row['id'],[]).append(row[key]-base[key])
    return bootstrap([float(np.mean(v)) for v in grouped.values()])
