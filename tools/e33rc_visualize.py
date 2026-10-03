"""固定near worst/best、rot90 worst、hash各16；所有组完整并集，无人工重选。"""
import hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from tools.e33r_visualize import direction,scalar,fit_panel
from tools.e33rc_common import *

def real_rotation(image):
    matrix=cv2.getRotationMatrix2D((191.5,255.5),90,1.)
    return Image.fromarray(cv2.warpAffine(np.asarray(image),matrix,image.size,flags=cv2.INTER_LINEAR,borderValue=(255,255,255)))

def create_panels(folder):
    rows=read(folder/'real/rows.json');valid=[r for r in rows if r['near_advantage'] is not None]
    near=sorted(valid,key=lambda r:(r['near_advantage'],r['id']))
    rotation=sorted(valid,key=lambda r:(-r['rot90_response_error'],r['id']))
    fixed=sorted(rows,key=lambda r:hashlib.sha256(('E33RC/visual/'+r['id']).encode()).hexdigest())
    groups=dict(near_worst16=[r['id'] for r in near[:16]],near_best16=[r['id'] for r in near[-16:]],
        rot90_worst16=[r['id'] for r in rotation[:16]],fixed_hash16=[r['id'] for r in fixed[:16]])
    ids=sorted(set(sum(groups.values(),[])));visual=folder/'visual_audit';visual.mkdir(exist_ok=True)
    write(visual/'selection.json',dict(groups=groups,unique_ids=ids,
        unavailable_near=[r['id'] for r in rows if r['near_advantage'] is None],
        ordering='metric extremes only GT-readable217; hash over all256 including emptyGT; same hash IDs all stages'))
    lookup={r['id']:r for r in read(OUT/'split_manifest.json')['dev']};metrics={r['id']:r for r in rows}
    cf_records=read(OUT/'controlled_manifest.json')['dev'];cfmap={r['id']:r for r in cf_records}
    # 无自身controlled候选的真实target，显示预先固定hash probe，不重新挑成功病例。
    fallback=hash_order(cf_records,'E33RC/visual/controlled-probe')[0]['id']
    for sid in ids:
        row=lookup[sid];metric=metrics[sid];wrong=metric['wrong_references']['color_near']
        wrongrow=next(r for r in read(OUT/'split_manifest.json')['dev']+read(OUT/'split_manifest.json')['confirmation_all'] if r['id']==wrong)
        with np.load(folder/'real/fields'/(sid+'.npz')) as z:
            ori,q,gt,conf,support=[np.asarray(z[k]) for k in ['orientation','q','gt','confidence_logits','support']]
        match=image_at(DATASET/row['reference']);probe=sid if sid in cfmap else fallback
        with np.load(folder/'controlled/fields'/(probe+'.npz')) as z:
            cf_ori=np.asarray(z['orientation']);cf_support=np.asarray(z['support'])
            cf_diff=np.degrees(np.arccos(np.clip(-(cf_ori[0]*cf_ori[1]).sum(0),-1,1)))/2
        diff=np.degrees(np.arccos(np.clip((ori[0]*ori[1]).sum(0),-1,1)))/2
        diff_image=np.asarray(scalar(diff,90)).copy();diff_image[~support]=235
        probe_image=np.asarray(scalar(cf_diff,90)).copy();probe_image[~cf_support]=235
        items=[('target sketch',image_at(DATASET/row['sketch'])),('GT orientation',direction(gt[:2],support)),
            ('matched reference',match),('near reference '+wrong,image_at(DATASET/wrongrow['reference'])),
            ('matched prediction',direction(ori[0],support)),('near prediction',direction(ori[1],support)),
            ('real rot90 reference',real_rotation(match)),('rot90 prediction',direction(ori[3],support)),
            ('match-near difference 0..90',Image.fromarray(diff_image)),('Q_ref match',direction(q[0],support)),
            ('C_ref match',scalar(1/(1+np.exp(-np.clip(conf[0,0],-60,60))))),
            ('controlled response error '+probe,Image.fromarray(probe_image))]
        canvas=Image.new('RGB',(6*210,2*275+65),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),'E33-RC '+sid+' near advantage='+str(metric['near_advantage'])+' real rot90 error='+str(metric['rot90_response_error']),fill='black')
        draw.text((8,24),'Axial hue [0,180); gray outside readable GT; learned confidence is not independent truth',fill='black')
        for i,(label,pic) in enumerate(items):
            x,y=(i%6)*210+5,(i//6)*275+65;draw.text((x,y),label,fill='black');canvas.paste(fit_panel(pic),(x,y+20))
        canvas.save(visual/(sid+'.png'))
    return ids
