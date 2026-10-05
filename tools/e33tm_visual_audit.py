"""固定名单、最好/最差和各干预图，保存可下载人工检查面板。"""
import colorsys
import textwrap
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from tools.e33tm_protocol import *

def field_image(path,key='orientation'):
    if not path.exists(): return Image.new('RGB',(96,128),'white')
    with np.load(path) as z: v=z[key]
    angle = (np.arctan2(v[1],v[0])/2)%np.pi
    hue = np.uint8(angle/np.pi*255)
    hsv = np.stack([hue,np.full_like(hue,220),np.full_like(hue,230)],-1)
    return Image.fromarray(hsv,'HSV').convert('RGB')

def panel(cohort,sid,destination,setting='full',seed=42):
    row = next(r for r in cohort['dev'] if r['id']==sid)
    base = OUT/'baseline_e5'/sid/'d42'
    if setting.startswith('conflict_'): base=OUT/'conflict_test/baseline_e5'/sid/'d42'
    if setting=='full': full = OUT/('rf2_seed%d'%seed)/sid/'d42'
    elif setting=='near': full = OUT/'text_compatible_near'/sid/'d42'
    elif setting.startswith('conflict_'): full=OUT/'conflict_test'/setting.replace('conflict_','')/sid/'d42'
    else: full = OUT/'ablations'/setting/sid/'d42'
    if not (full/'pair.json').exists(): return None
    refs = [Image.open(DATASET/row['reference']).convert('RGB').resize((384,512))]
    from data.e33rf_real_rotation_dataset import rotate
    refs += [rotate(refs[0],90),rotate(refs[0],180)]
    entries = [('Sketch',Image.open(DATASET/row['sketch']))]
    entries += [(a+' ref',r) for a,r in zip(('R0','R90','R180'),refs)]
    if setting in ('near','wrong_texture'):
        used=read(full/'R0.json')
        entries.append(('Donor '+str(used['donor']),Image.open(used['source_reference'])))
    for label,path in [('E5 R0',base/'R0.png'),('E5 R90',base/'R90.png')]:
        if path.exists(): entries.append((label,Image.open(path)))
    records=[]
    for arm in ('R0','R90','R180'):
        record=read(full/(arm+'.json'));records.append(record)
        entries.append((setting+' '+arm,Image.open(full/(arm+'.png'))))
    entries += [('RF field R0',field_image(full/'R0_field.npz')),
                ('RF field R90',field_image(full/'R90_field.npz')),
                ('E26 image R0',field_image(full/'R0_orientation.npz','geometry')),
                ('E26 image R90',field_image(full/'R90_orientation.npz','geometry'))]
    width=len(entries)*100
    canvas=Image.new('RGB',(width,252),'white');draw=ImageDraw.Draw(canvas)
    caption=row['caption'].encode('ascii','replace').decode()
    draw.text((4,2),sid+' | '+setting+' RF'+str(seed),fill='black')
    draw.text((4,16),caption[:190],fill='black')
    for i,(label,image) in enumerate(entries):
        draw.text((i*100+2,35),label[:16],fill='black')
        image=image.convert('RGB');image.thumbnail((96,128))
        canvas.paste(image,(i*100+2,50))
    pair=read(full/'pair.json')
    line='R90 error={} success={} readable={:.1%}; R180 error={} success={}; fixed support={}'.format(
        None if pair['r90_error'] is None else round(pair['r90_error'],2),pair['r90_success'],pair['r90_coverage'],
        None if pair['r180_error'] is None else round(pair['r180_error'],2),pair['r180_success'],pair['fixed_support_cells'])
    draw.text((4,182),line,fill='black')
    for i,record in enumerate(records):
        m=record['metrics']
        draw.text((4,196+14*i),'{}: Contour {:.4f} Text {:.4f} Texture {:.4f} | text used: {}'.format(
            record['arm'],m['contour_f1'],m['text_score'],m['texture_score'],record['text_used'][:125]),fill='black')
    destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)
    return destination

def grouped(cohort,ids,folder,setting='full'):
    paths=[]
    for sid in ids:
        path=panel(cohort,sid,folder/(sid+'.png'),setting)
        if path: paths.append(path)
    for start in range(0,len(paths),4):
        images=[Image.open(p) for p in paths[start:start+4]]
        page=Image.new('RGB',(max(i.width for i in images),sum(i.height for i in images)),'white')
        y=0
        for image in images: page.paste(image,(0,y));y+=image.height
        page.save(folder/('page%d.png'%(start//4)))
    write(folder/'manifest.json',dict(ids=[p.stem for p in paths],requested_ids=ids,setting=setting))

def smoke(cohort):
    grouped(cohort,cohort['smoke16'],OUT/'smoke_audit/panels')
    with zipfile.ZipFile(OUT/'smoke_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in (OUT/'smoke_audit').rglob('*'):
            if path.is_file(): archive.write(path,str(path.relative_to(OUT)))
        for path in (OUT/'caption_audit/caption_audit_summary.json',OUT/'field_precheck/summary.json'):
            archive.write(path,str(path.relative_to(OUT)))

def final_audit():
    cohort=read(OUT/'manifests/cohorts.json')
    if (OUT/'manifests/intervention_data.json').exists():
        interventions=read(OUT/'manifests/intervention_data.json')
        cohort['text_compatible_near']=interventions['text_compatible_near']
    rows=read(OUT/'rf2_seed42/rows.json')
    ranked=sorted(rows,key=lambda r:(r['r90_error'] if r['r90_error'] is not None else 999.,r['id']))
    grouped(cohort,[r['id'] for r in ranked[:16]],OUT/'visual_audit/best16_R90')
    grouped(cohort,[r['id'] for r in ranked[-16:]],OUT/'visual_audit/worst16_R90')
    grouped(cohort,cohort['hash16'],OUT/'visual_audit/fixed_hash16')
    for setting in ('no_text','no_sketch','no_texture'):
        grouped(cohort,cohort['hash16'],OUT/'visual_audit'/(setting+'16'),setting)
    near=[r for r in cohort['primary'] if cohort['text_compatible_near'][r['id']]['available']]
    grouped(cohort,[r['id'] for r in order(near,'E33TM/near_visual')[:16]],
            OUT/'visual_audit/text_compatible_near16','near')
    conflict=order(cohort['conflict'],'E33TM/conflict_visual')[:16]
    for setting in ('conflict_C0','conflict_C1','conflict_C2'):
        grouped(cohort,[r['id'] for r in conflict],OUT/'visual_audit/text_conflict16'/setting,setting)

if __name__=='__main__': final_audit()
