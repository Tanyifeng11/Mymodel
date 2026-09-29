"""E27 analytic CALPC；接口只接收 RGB reference 和 target mask，不读取人工标注。"""

import cv2
import numpy as np
from PIL import Image

from models.local_pattern_field import patch_geometry, rectify_reference
from tools.e27_correspondence import descriptor, similarity, target_parts


VERSION='e27_analytic_geometry_descriptor_v1'
MODES=('local_no_confidence','confidence_no_rectification','geometry_only','full_CALPC')


def foreground(image):
    rgb=np.asarray(image)
    rough=((rgb.min(-1)<243)|(rgb.max(-1)-rgb.min(-1)>12)).astype(np.uint8)
    rough=cv2.morphologyEx(rough,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(rough)
    owner=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    mask=(labels==owner).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask,contours,-1,1,cv2.FILLED)
    return mask>0


def select_crop(image,area):
    """按区域前景覆盖和局部纹理置信度选 canonical patch；32/48/64px。"""
    ys,xs=np.where(area)
    candidates=[]
    for size in (64,48,32,16):
        if not len(xs):break
        for y in range(max(0,int(ys.min())),min(image.height-size,int(ys.max()))+1,8):
            for x in range(max(0,int(xs.min())),min(image.width-size,int(xs.max()))+1,8):
                coverage=float(area[y:y+size,x:x+size].mean())
                if coverage<.95:continue
                patch=image.crop((x,y,x+size,y+size));info=patch_geometry(patch)
                # 可读纹样优先；floral/plaid 的 identity 不依赖单轴 coherence。
                energy=np.asarray(patch.convert('L'),dtype=float).std()/255
                score=info['confidence']*.25+min(energy,.20)+size/512
                candidates.append((score,[x,y,x+size,y+size],info,coverage))
        if candidates:break
    if candidates:
        _,box,info,coverage=max(candidates,key=lambda q:q[0])
        return box,info,coverage
    # 窄片没有可用内部 patch，选最大内距中心，并显式降置信度。
    distance=cv2.distanceTransform(area.astype(np.uint8),cv2.DIST_L2,5)
    y,x=np.unravel_index(distance.argmax(),distance.shape)
    radius=max(4,min(8,int(distance[y,x])))
    box=[int(max(0,x-radius)),int(max(0,y-radius)),int(min(image.width,x+radius)),int(min(image.height,y+radius))]
    return box,patch_geometry(image.crop(box)),.2


def infer_regions(image):
    mask=foreground(image)
    labels,u,v=target_parts(mask)
    regions=[]
    body=labels=='body'
    left,right=body&(u<.5),body&(u>=.5)
    upper,lower=body&(v<.5),body&(v>=.5)
    def feature(area):
        box,info,coverage=select_crop(image,area)
        return image.crop(box)
    # appearance/selfsim discontinuity 驱动 body 二分，不能读取 case ID/family/人工 panel。
    vertical=similarity(feature(left),feature(right))[0]
    horizontal=similarity(feature(upper),feature(lower))[0]
    if vertical<.78 and vertical<horizontal:
        regions += [('body_left',left),('body_right',right)]
        split='vertical_descriptor_discontinuity'
    elif horizontal<.78:
        regions += [('body_upper',upper),('body_lower',lower)]
        split='horizontal_descriptor_discontinuity'
    else:
        regions += [('body',body)];split='single_body'
    for name in ('left_sleeve','right_sleeve','collar'):
        area=labels==name
        if area.sum()>256:regions.append((name,area))
    records=[]
    for name,area in regions:
        box,geometry,coverage=select_crop(image,area)
        y,x=np.where(area)
        records.append({'name':name,'area':area,'crop':box,'geometry':geometry,'coverage':coverage,
                        'source_box':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]})
    return mask,records,{'body_split':split,'vertical_similarity':vertical,'horizontal_similarity':horizontal}


def build_scaffold(image,target_mask,mode='full_CALPC',region_proposer=None):
    assert mode in MODES
    source_mask,regions,decision=(region_proposer or infer_regions)(image)
    target_labels,u,v=target_parts(target_mask)
    gy,gx=np.indices(target_mask.shape,dtype=np.float32)
    rgb_samples=[];supports=[];confidence=[];uvs=[];records=[];target_masks=[]
    existing={r['name'] for r in regions}
    body_target=target_labels=='body'
    if region_proposer is not None:
        for extra in ('left_sleeve','right_sleeve','collar'):
            if extra not in existing:body_target|=target_labels==extra
    for region in regions:
        name=region['name'];area=region['area'];crop=image.crop(region['crop'])
        if 'target_u_range' in region:
            lo,hi=region['target_u_range'];target=body_target&(u>=lo)&((u<=hi) if hi==1 else (u<hi))
        elif 'target_v_range' in region:
            lo,hi=region['target_v_range'];target=body_target&(v>=lo)&((v<=hi) if hi==1 else (v<hi))
        elif name=='body':target=body_target
        elif name=='body_left':target=(target_labels=='body')&(u<.5)
        elif name=='body_right':target=(target_labels=='body')&(u>=.5)
        elif name=='body_upper':target=(target_labels=='body')&(v<.5)
        elif name=='body_lower':target=(target_labels=='body')&(v>=.5)
        else:target=target_labels==name
        if name.startswith('body') and region_proposer is None:
            for extra in ('left_sleeve','right_sleeve','collar'):
                if extra not in existing:target|=target_labels==extra
        if not target.any():continue
        target_masks.append(target)
        rect_info=None;canonical=crop
        if mode not in ('confidence_no_rectification','geometry_only') and region['geometry']['valid'] and min(crop.size)>=32:
            canonical,_,rect_info=rectify_reference(crop)
        identity_preservation=similarity(crop,canonical)[0]
        # rectification 明显伤害 descriptor 时回退 raw appearance。
        if identity_preservation<.85:
            canonical=crop;rect_info={'rejected_identity_change':True}
            identity_preservation=1.
        if mode=='geometry_only':
            yy,xx=np.indices((crop.height,crop.width))
            theta=np.deg2rad(region['geometry']['orientation']-90)
            f=max(region['geometry']['frequency'],2/min(crop.size))
            pattern=np.sin(2*np.pi*f*(xx*np.cos(theta)+yy*np.sin(theta)))>0
            pixels=np.asarray(crop).reshape(-1,3)
            colors=np.percentile(pixels,(20,80),axis=0)
            canonical=Image.fromarray(np.where(pattern[...,None],colors[0],colors[1]).astype(np.uint8))
        y,x=np.where(target);box=region['source_box']
        uvx=np.mod((gx-x.min())/max(x.max()-x.min(),1)*(box[2]-box[0]-1),canonical.width)
        uvy=np.mod((gy-y.min())/max(y.max()-y.min(),1)*(box[3]-box[1]-1),canonical.height)
        sample=cv2.remap(np.asarray(canonical),uvx.astype(np.float32),uvy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP).astype(float)
        preservation=identity_preservation*region['coverage']
        c=float(np.clip(preservation*(.4+.6*region['geometry']['confidence']),0,1))
        if mode in ('local_no_confidence','geometry_only'):c=1.
        low_frequency=cv2.GaussianBlur(np.asarray(crop),(0,0),2.)
        fallback=cv2.remap(low_frequency,uvx.astype(np.float32),uvy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        sample=c*sample+(1-c)*fallback
        support=cv2.GaussianBlur(target.astype(np.float32),(0,0),1.5)
        rgb_samples.append(sample);supports.append(support);confidence.append(c)
        uvs.append(np.stack((uvx+region['crop'][0],uvy+region['crop'][1]),-1))
        records.append({k:value for k,value in region.items() if k!='area'})
        records[-1].update(rectification=rect_info,identity_preservation=identity_preservation,blend_confidence=c)
    # softmax 在 seam 的重叠 support 中组合，interior 仍由单一 owner 控制。
    score=np.stack([np.log(np.maximum(s,1e-12))+4*c for s,c in zip(supports,confidence)])
    weights=np.exp(score-score.max(0));weights/=weights.sum(0)
    rgb=(weights[...,None]*np.stack(rgb_samples)).sum(0);rgb[~target_mask]=255
    conf=(weights*np.array(confidence)[:,None,None]).sum(0);conf[~target_mask]=0
    arrays={'source_mask':source_mask,'automatic_source_masks':np.stack([r['area'] for r in regions]),
            'automatic_target_masks':np.stack(target_masks),
            'automatic_target_supports':np.stack(supports),'automatic_patch_uv':np.stack(uvs),
            'composition_weights':weights,'confidence':conf}
    info={'version':VERSION,'mode':mode,'source_access':'RGB image only; no manual boxes/masks/IDs/family labels',
          'region_rule':decision,'regions':records,'confidence_coverage':float((conf[target_mask]>.5).mean())}
    if region_proposer is not None:
        assert np.all(np.stack(target_masks).sum(0)[target_mask]==1)
        info['target_graph_partition_pass']=True
    return Image.fromarray(np.clip(np.rint(rgb),0,255).astype(np.uint8)),arrays,info
