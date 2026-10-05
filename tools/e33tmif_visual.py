"""固定field-success名单的阶段图例；不按生成结果挑样本。"""
import argparse
import tarfile
import numpy as np
from PIL import Image,ImageDraw
from tools.e33tmif_protocol import *

def orientation_image(path):
    with np.load(path) as z:
        angle=z['orientation_angle'];confidence=z['orientation_confidence'];support=z['support_mask']
    hsv=np.stack([np.uint8(angle/180*255),np.full_like(angle,220,dtype=np.uint8),
                  np.uint8(np.clip(confidence,0,1)*230)],-1)
    return Image.fromarray(hsv,'HSV').convert('RGB'),Image.fromarray(support.astype(np.uint8)*255)

def panel(sid,group,folder):
    dest=OUT/group/'seed42'/sid
    case=read(dest/'case.json')
    stages=['S0','S1','S2','S4','x0_01','x0_04','x0_08','S9']
    width=(3+2*len(stages))*100;canvas=Image.new('RGB',(width,465),'white');draw=ImageDraw.Draw(canvas)
    cohort=read(TM/'manifests/cohorts.json');row=next(r for r in cohort['primary'] if r['id']==sid)
    draw.text((4,2),sid+' | '+row['caption'][:190],fill='black')
    from data.e32_target_pseudogt import image_at
    from data.e33rf_real_rotation_dataset import rotate
    reference=image_at(DATASET/row['reference'])
    for i,(name,image) in enumerate([('Sketch',image_at(DATASET/row['sketch'])),('R0 ref',reference),('R90 ref',rotate(reference,90))]):
        draw.text((i*100+2,22),name,fill='black');image.thumbnail((96,128));canvas.paste(image,(i*100+2,40))
    for si,stage in enumerate(stages):
        for ai,arm in enumerate(('R0','R90')):
            x=(3+si*2+ai)*100
            draw.text((x+2,22),stage+' '+arm,fill='black')
            angle,support=orientation_image(dest/stage/(arm+'_orientation.npz'))
            rgb=Image.open(dest/stage/(arm+'.png')) if stage!='S0' else angle
            for image,y in [(rgb,40),(angle,170),(support,300)]:
                image=image.convert('RGB');image.thumbnail((96,128));canvas.paste(image,(x+2,y))
        r=case['stages'][stage]
        error='NA' if r['r90_error'] is None else '%.1f'%r['r90_error']
        draw.text((300+si*200+2,432),'R90 '+error+' deg, '+str(r['r90_success']),fill='black')
        draw.text((300+si*200+2,447),'support %.1f%%'%(100*r['r90_coverage']),fill='black')
    folder.mkdir(parents=True,exist_ok=True);canvas.save(folder/(sid+'.png'))
    return canvas

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    group='smoke' if args.smoke else 'stage_survival'
    ids=read(OUT/'splits/field_success_fixed_hash16.json')
    folder=OUT/'visual_audit'/('smoke' if args.smoke else 'field_success_fixed_hash16')
    images=[panel(sid,group,folder) for sid in ids]
    for i in range(0,len(images),4):
        page=Image.new('RGB',(images[0].width,4*images[0].height),'white')
        for j,image in enumerate(images[i:i+4]):page.paste(image,(0,j*image.height))
        page.save(folder/('page%d.png'%(i//4)))
    write(folder/'manifest.json',dict(ids=ids,group=group,stages=['S0','S1','S2','S4','x0_01','x0_04','x0_08','S9'],
        top='RGB',middle='orientation/confidence',bottom='paired GT-supported readability mask'))
    if args.smoke:
        checks=[]
        for sid in ids:
            case=read(OUT/'smoke/seed42'/sid/'case.json')
            for arm in ARMS:
                proof=read(OUT/'smoke/seed42'/sid/(arm+'_observer_check.json'))
                assert proof['pixels_equal'] and proof['noise_equal']
                assert case['schedules'][arm]['effective_steps']==8
                checks.append(dict(id=sid,arm=arm,steps=8,start=case['schedules'][arm]['start'],**proof))
        write(OUT/'visual_audit/smoke_numeric.json',dict(checks=checks,**{'pass':True}))
        with tarfile.open(OUT/'smoke_review.tar.gz','w:gz') as tar:
            for p in sorted((OUT/'visual_audit').rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(OUT)))
            for p in (OUT/'protocol/protocol.json',OUT/'splits/field_success_fixed_hash16.json'):
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('[IF visual]',group,len(ids),'complete',flush=True)

if __name__=='__main__':main()
