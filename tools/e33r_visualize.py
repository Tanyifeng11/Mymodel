"""固定dev worst/best/hash各16；从最终字段绘图，不重新挑候选。"""
import cv2
import numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from tools.e33r_common import OUT,DATASET,read,write,hash_order

def direction(field,support=None):
    h,w=field.shape[1:];angle=(np.arctan2(field[1],field[0])/2)%np.pi
    hsv=np.zeros((h,w,3),np.uint8);hsv[...,0]=angle/np.pi*179;hsv[...,1:]=255
    rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    if support is not None:rgb[~support]=235
    return Image.fromarray(rgb)

def scalar(field,maximum=1):
    rgb=cv2.applyColorMap(np.uint8(np.clip(field/maximum,0,1)*255),cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))

def create_panels(folder,records):
    rows=read(folder/'dev/rows.json');lookup={r['id']:r for r in records}
    ordered=sorted(rows,key=lambda r:(r['r90_response_error'] if r['r90_response_error'] is not None else 999,r['id']))
    selections=dict(best16=[r['id'] for r in ordered[:16]],worst16=[r['id'] for r in ordered[-16:]],
                    fixed_hash16=[r['id'] for r in hash_order(rows,'E33R/visual/dev')[:16]])
    unique=sorted(set(sum(selections.values(),[])))
    visual=folder/'visual';visual.mkdir(exist_ok=True)
    write(visual/'selection.json',dict(groups=selections,unique_ids=unique,overlap_count=48-len(unique),
        ordering='best/worst by clean R90 response error; empty support ranks worst; fixed hash unaffected by outcomes'))
    metrics={r['id']:r for r in rows}
    for sid in unique:
        row=lookup[sid]
        with np.load(folder/'dev/fields'/(sid+'.npz')) as z:
            ori,q,gt,prior,support,conf,src=[np.asarray(z[k]) for k in
                   ('orientation','q','gt','prior','support','confidence','source_geometry')]
        rgb=np.asarray(image_at(DATASET/row['target']).crop(tuple(row['box'])))
        e90=np.degrees(np.arccos(np.clip(-(ori[1]*ori[0]).sum(0),-1,1)))/2
        items=[('sketch',image_at(DATASET/row['sketch'])),('R0 native',Image.fromarray(rgb)),
               ('R90 native',Image.fromarray(np.rot90(rgb).copy())),('R180 native',Image.fromarray(np.rot90(rgb,2).copy())),
               ('source E26 R0',direction(src[0,:,:,:2].transpose(2,0,1))),('GT orientation',direction(gt[:2],support)),
               ('frozen prior',direction(prior,support)),('R0 prediction',direction(ori[0],support)),
               ('R90 prediction',direction(ori[1],support)),('R180 prediction',direction(ori[2],support)),
               ('response error 0..90deg',scalar(e90,90)),('Q90 physical rotation',direction(q[1],support)),
               ('C_ref R0',scalar(conf[0,0])),('M_cf',Image.fromarray(np.uint8(support)*255).convert('RGB'))]
        canvas=Image.new('RGB',(7*210,2*275+60),'white');draw=ImageDraw.Draw(canvas)
        metric=metrics[sid]
        draw.text((8,5),'E33-R '+sid+' strict='+str(row['strict'])+' error='+str(metric['r90_response_error']),fill='black')
        draw.text((8,23),'Hue is physical axial angle [0,180); error heatmap [0,90deg]; gray means outside M_cf',fill='black')
        for i,(label,image) in enumerate(items):
            x=(i%7)*210;y=60+(i//7)*275;draw.text((x+5,y),label,fill='black')
            image=image.copy();image.thumbnail((200,245));canvas.paste(image,(x+5,y+22))
        canvas.save(visual/(sid+'.png'))
    return unique

def prior_panels():
    """P0未过时仍提供固定hash16误差图，不绘制未训练的control预测。"""
    from data.e33r_group_dataset import jitter_structure
    from tools.e33r_train_prior import PriorDataset
    from models.e33r_rotation_control import SketchPrior
    import torch
    folder=OUT/'P0_prior';model=SketchPrior().cuda().eval()
    model.load_state_dict(torch.load(folder/'checkpoint_final.pt',map_location='cpu')['model'])
    records=hash_order(read(OUT/'split_manifest.json')['dev'],'E33R/visual/P0')[:16]
    visual=folder/'visual';visual.mkdir(exist_ok=True)
    write(visual/'selection.json',dict(ids=[r['id'] for r in records],selection='fixed hash16 full256dev'))
    with torch.no_grad():
        for row in records:
            case=PriorDataset([row])[0];gt=case['supervision_geometry'].numpy()
            support=(case['supervision_interior'].numpy()>=.95)&(gt[3]>=.25)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(case['structure'][None].cuda())['orientation'][0]
            pred=pred.float().cpu().numpy();error=np.degrees(np.arccos(np.clip((pred*gt[:2]).sum(0),-1,1)))/2
            items=[('sketch',image_at(DATASET/row['sketch'])),('GT',direction(gt[:2],support)),
                   ('prior',direction(pred,support)),('error 0..90deg',scalar(error,90))]
            canvas=Image.new('RGB',(840,310),'white');draw=ImageDraw.Draw(canvas);draw.text((5,5),row['id'],fill='black')
            for i,(label,image) in enumerate(items):
                draw.text((i*210+5,30),label,fill='black');image.thumbnail((200,245));canvas.paste(image,(i*210+5,55))
            canvas.save(visual/(row['id']+'.png'))
