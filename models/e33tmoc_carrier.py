"""真实source patch旋转+固定overlap-add；RF仅控制目标方向。"""
import hashlib
import cv2
import numpy as np
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from models.local_pattern_field import patch_geometry

CONFIG=dict(version=1,source_patch=96,output_patch=64,stride=32,q_min=.25,
    variance_min=.0004,foreground_min=.95,source_boundary_erosion=8,
    boundary_distance=8,reliability_threshold=.25,contrast_normalizer=.04,
    fallback='single reference-foreground mean RGB fill; no sampled background',
    patch_selection='SHA256 OC/tile/id/x/y modulo deterministic row-major valid bank; shared rule across arms',
    rotation='cv2 source tangent minus RF target tangent; center-crop64 from96, no rotation padding enters core',
    blending='fixed separable Hann64; no phase or period fitting',
    C2='valid-readable real patches, full-canvas raw oriented overlap-add; original garment mask atS2',
    C3='C2 multiplied source reliability, RF confidence>=.25, garment support and distance falloff',
    C4='C3 reliability with clipped((RF confidence-.25)/.75) soft blend',
    training_steps=0)

def patch_bank(reference):
    rgb=np.asarray(reference);fg=np.asarray(estimate_cloth_foreground_mask(reference,*reference.size)[0])>127
    interior=cv2.erode(fg.astype(np.uint8),np.ones((17,17),np.uint8))>0
    values=rgb[fg] if fg.any() else rgb.reshape(-1,3)
    prototype=dict(mean_RGB=values.mean(0).tolist(),RGB_covariance=np.cov(values.astype(float),rowvar=False).tolist())
    valid=np.zeros(fg.shape,bool);bank=[]
    for y in range(0,reference.height-95,32):
        for x in range(0,reference.width-95,32):
            foreground=float(interior[y:y+96,x:x+96].mean())
            patch=reference.crop((x,y,x+96,y+96));core=patch.crop((16,16,80,80));geometry=patch_geometry(core)
            std=float(np.asarray(core.convert('L'),float).std()/255)
            if foreground<.95 or geometry['confidence']<.25 or std*std<.0004:continue
            confidence=geometry['confidence'];reliability=float(confidence*min(1,std/.04)*foreground)
            bank.append(dict(box=[x,y,x+96,y+96],orientation=geometry['orientation'],confidence=confidence,
                variance=std*std,foreground_ratio=foreground,reliability=reliability,pixels=np.asarray(patch)))
            valid[y:y+96,x:x+96]=True
    return bank,valid,fg,prototype

def construct(reference,orientation,confidence,mask,sid,kind):
    bank,source_valid,source_fg,prototype=patch_bank(reference)
    width,height=mask.size;inside=np.asarray(mask)>0
    dense=cv2.resize(np.moveaxis(orientation,0,-1),(width,height),interpolation=cv2.INTER_LINEAR)
    theta=np.arctan2(dense[...,1],dense[...,0])/2
    rf_conf=cv2.resize(np.asarray(confidence).squeeze(),(width,height),interpolation=cv2.INTER_LINEAR)
    fallback=np.float32(prototype['mean_RGB']);accum=np.zeros((height,width,3),np.float32)
    norm=np.zeros((height,width),np.float32);source_reliability=np.zeros_like(norm);selected=[]
    window=np.outer(np.hanning(64),np.hanning(64)).astype(np.float32)
    for y in range(-32,height,32):
        for x in range(-32,width,32):
            if not bank:continue
            index=int(hashlib.sha256(('OC/tile/%s/%d/%d'%(sid,x,y)).encode()).hexdigest(),16)%len(bank)
            p=bank[index];cy,cx=np.clip(y+32,0,height-1),np.clip(x+32,0,width-1)
            angle=np.degrees(theta[cy,cx]);rotation=(p['orientation']-angle+90)%180-90
            transform=cv2.getRotationMatrix2D((47.5,47.5),float(rotation),1.)
            patch=cv2.warpAffine(p['pixels'],transform,(96,96),flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=tuple(prototype['mean_RGB']))[16:80,16:80].astype(np.float32)
            x0,y0=max(0,x),max(0,y);x1,y1=min(width,x+64),min(height,y+64)
            w=window[y0-y:y1-y,x0-x:x1-x]
            accum[y0:y1,x0:x1]+=patch[y0-y:y1-y,x0-x:x1-x]*w[...,None]
            norm[y0:y1,x0:x1]+=w
            source_reliability[y0:y1,x0:x1]+=p['reliability']*w
            selected.append(dict(target_box=[x,y,x+64,y+64],source_bank_index=index,target_tangent_degrees=float(angle),rotation=float(rotation)))
    oriented=np.where((norm>1e-8)[...,None],accum/np.maximum(norm[...,None],1e-8),fallback)
    source_reliability/=np.maximum(norm,1e-8)
    distance=cv2.distanceTransform(inside.astype(np.uint8),cv2.DIST_L2,5)
    boundary=np.clip(distance/8,0,1)*inside
    if kind=='C2_appearance':weight=np.ones_like(norm) if bank else np.zeros_like(norm)
    elif kind=='C3_support_aware':weight=source_reliability*boundary*(rf_conf>=.25)
    elif kind=='C4_confidence_blend':weight=source_reliability*boundary*np.clip((rf_conf-.25)/.75,0,1)
    else:raise ValueError(kind)
    if kind!='C2_appearance':weight=np.where(weight>=.25,weight,0)
    raw=oriented*weight[...,None]+fallback*(1-weight[...,None])
    S1=Image.fromarray(np.uint8(np.clip(np.round(raw),0,255)))
    rgb=np.asarray(S1).copy();rgb[~inside]=255;S2=Image.fromarray(rgb)
    state=dict(source_valid_support=source_valid,source_foreground=source_fg,target_support=inside,
        RF_orientation=orientation,RF_confidence=confidence,dense_orientation=np.degrees(theta)%180,
        dense_RF_confidence=rf_conf,source_reliability=source_reliability,boundary_weight=boundary,blend_weight=weight)
    metadata=dict(config=CONFIG,prototype=prototype,bank=[{k:v for k,v in p.items() if k!='pixels'} for p in bank],
        selected_patches=selected,empty_bank=len(bank)==0)
    return S1,S2,state,metadata,bank
