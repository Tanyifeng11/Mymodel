"""G0a/G1a CPU准备：历史只读、划分去重、预渲染目标、全64输入审核材料。"""
import csv,hashlib,re,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from tools.e33gc_g2b_protocol import *
from tools.e33gc_bootstrap import fingerprint,duplicate
from data.e32_target_pseudogt import image_at
from data.e33gc_g2b_renderer import construct
from data.e33gc_renderer import support,fft_tangent
from data.e33rf_real_rotation_dataset import rotate
from models.local_pattern_field import patch_geometry

def panels(records,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for page in range(0,len(records),4):
        canvas=Image.new('RGB',(1280,4*286),'white');draw=ImageDraw.Draw(canvas)
        for j,(sid,caption,items) in enumerate(records[page:page+4]):
            top=j*286; draw.text((3,top+3),sid+' | '+caption[:170],fill='black')
            for k,(name,im) in enumerate(items):
                copy=im.copy();copy.thumbnail((154,220));x=k*160+3
                draw.text((x,top+25),name,fill='black');canvas.paste(copy,(x,top+45))
        canvas.save(folder/('page%02d.png'%(page//4)))

def axis(a,b):return float(abs((a-b+90)%180-90))

def run():
    init();cv2.setNumThreads(1)
    cf=read(RF/'controlled_manifest.json'); real=read(RF/'split_manifest.json')
    held={r['id']:r for key in ['dev','causal','independent'] for r in real.get(key,[])}
    for key,rows in cf.items():
        if key!='train' and isinstance(rows,list):held.update({r['id']:r for r in rows})
    olddev={r['id']:r for r in cf['dev']}; collisions=[]
    for name in ['diagnostic64','confirmation64','remaining126']:
        for sid in sorted(set(read(OLD/'splits'/(name+'.json')))&set(olddev)):
            r=olddev[sid];collisions.append(dict(id=sid,locked_split=name,
                reference_sha256=sha(DATASET/r['reference']),target_sha256=sha(DATASET/r['target']),
                same_identity=True,exact_reference=True,exact_target=True,dhash_distance=0,color_MAE=0))
    dest=OUT/'G0a_split';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'split_collision_matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(collisions[0]));writer.writeheader();writer.writerows(collisions)
    write(dest/'G0a_split_audit.json',dict(old_dev_ids=list(olddev),collision_counts={name:sum(c['locked_split']==name for c in collisions)
        for name in ['diagnostic64','confirmation64','remaining126']},old_dev_excluded=True,
        missing_product_group_labels=True,near_duplicate_rule='dHash<=4 AND RGB64 MAE<=4/255; exact SHA',
        scope='GC adapter train-independent; historical E5/RF2 exposure not removed'))
    held_f=[]
    for sid,row in held.items():
        for key in ['reference','target']:
            held_f.append((sid,str(DATASET/row[key]),fingerprint(DATASET/row[key])))
    captions={Path(r['cloth']).stem:r['caption'] for r in read('data/train_bf_texture.json')}
    # SHA 基于实际reference文件；输入单轴筛选只读reference/sketch，不读生成图。
    rows=[dict(r,caption=captions[r['id']],reference_sha256=sha(DATASET/r['reference']))
        for r in cf['train'] if r['id'] not in held and r['id'] in captions]
    rows.sort(key=lambda r:(r['reference_sha256'],r['id']))
    qualified=[];reject=[];seen=set();start=time.monotonic()
    for r in rows:
        caption=r['caption']
        if re.search(r'\b(horizontal|vertical|diagonal|plaid|checked|checkered|floral|polka)\b',caption,re.I):continue
        ref=image_at(DATASET/r['reference']);p=ref.crop(CONFIG['crop']);pg=patch_geometry(p)
        gray=np.asarray(p.convert('L'),float)/255
        if not pg['valid'] or pg['confidence']<.35 or gray.std()<.025 or ((gray>.97)|(gray<.03)).mean()>.20:continue
        sketch=image_at(DATASET/r['sketch']);mask,inner,falloff=support(sketch)
        boxes=[[x,y,x+64,y+64] for y in range(0,512,64) for x in range(0,384,64) if inner[y:y+64,x:x+64].mean()>=.95]
        if len(boxes)<3 or (falloff>=.99).sum()/max(inner.sum(),1)<.70:continue
        fp=[fingerprint(DATASET/r[key]) for key in ['reference','target']]
        collision=None
        for value in fp:
            collision=duplicate(value,held_f)
            if collision:break
        if collision or any(v['sha256'] in seen for v in fp):
            reject.append(dict(id=r['id'],reason=collision or 'exact duplicate selected candidate'));continue
        r.update(source_geometry=pg,dft_boxes=boxes,source_crop_sha256=hashlib.sha256(np.asarray(p).tobytes()).hexdigest())
        qualified.append(r);seen.update(v['sha256'] for v in fp)
        # 首128个合格输入将预留为新dev；剩余候选也须排除与这128个的视觉近重复。
        if len(qualified)<=128:
            held_f.extend((r['id'],str(DATASET/r[key]),value) for key,value in zip(['reference','target'],fp))
        if len(qualified)%20==0:print('QUALIFIED',len(qualified),'seconds',round(time.monotonic()-start),flush=True)
    dev=qualified[:128] if len(qualified)>=144 else []
    train=[r for r in qualified if r not in dev]
    write(OUT/'protocol/gc_controlled_dev128_new_ids.json',[r['id'] for r in dev])
    write(OUT/'protocol/gc_train_pool_ids.json',[r['id'] for r in train])
    write(dest/'input_qualified_rows.json',qualified);write(dest/'rejected_collisions.json',reject)
    write(dest/'pool_audit.json',dict(input_qualified=len(qualified),new_dev=len(dev),gc_train=len(train),elapsed_seconds=time.monotonic()-start,
        train_pool_real_and_controlled_holdouts_excluded=True,original_train_count=len(cf['train']),
        limitation='selected pool exact dedup; visual near dedup against locked identities, no product group labels'))
    candidates=train[:16]
    write(OUT/'G1a_reference_target_audit/candidate16_rows.json',candidates)
    materials=[];audits=[]
    for r in candidates:
        value=construct(r);folder=OUT/'G1a_controlled_targets'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        image_at(DATASET/r['reference']).save(folder/'original_reference.png')
        value['source_crop'].save(folder/'source_crop.png');value['sketch'].save(folder/'sketch.png')
        for arm,ref,target in zip(ARMS,value['references'],value['targets']):
            ref.save(folder/(arm+'_reference.png'));target.save(folder/(arm+'_target.png'))
        angles=[fft_tangent(target,tuple(r['dft_boxes'][0])) for target in value['targets']]
        audit=dict(id=r['id'],source_sha256=r['reference_sha256'],crop=CONFIG['crop'],source_crop_sha256=r['source_crop_sha256'],
            direction_angles=angles,r90_error=axis(angles[1],angles[0]-90),r180_error=axis(angles[2],angles[0]),
            coverage=float((value['falloff']>=.99).sum()/value['inner'].sum()),
            target_only_GT_lowpass=True,forward_GT=False,reference_from_real_crop=True,
            matrices={a:[[int(round(np.cos(np.deg2rad(i*90)))),int(round(np.sin(np.deg2rad(i*90))))],
                [-int(round(np.sin(np.deg2rad(i*90)))),int(round(np.cos(np.deg2rad(i*90))))]] for i,a in enumerate(ARMS)},
            renderer_sha256=sha('data/e33gc_g2b_renderer.py'),files={p.name:sha(p) for p in folder.glob('*.png')})
        write(folder/'audit.json',audit);audits.append(audit)
        materials.append((r['id'],r['caption'],[('source crop',value['source_crop']),('sketch',value['sketch'])]+
            [(a+' ref',v) for a,v in zip(ARMS,value['references'])]+[(a+' target',v) for a,v in zip(ARMS,value['targets'])]))
    panels(materials,OUT/'visual_audit/G1a_candidate16')
    write(OUT/'G1a_reference_target_audit/automatic_audit.json',dict(records=audits,human_review_pass=None))
    oldrows=read(OLD/'G1_counterfactual_targets/train_audit16_rows.json')
    write(OUT/'G1a_reference_target_audit/old16_source_contract.json',dict(ids=[r['id'] for r in oldrows],
        renderer_sha256=sha('data/e33gc_renderer.py'),procedural_source=True,reference_and_target_share_hash_generated_pattern=True,
        not_derived_from_original_real_reference=True,old_human_gate=None,
        issue='actual old controlled input reference IS hash-generated pattern, but caption inherited real garment; appearance/text conflicts possible',
        new_route='real source crop; no reuse of old16 as main G2b labels'))
    realrows=[r for r in read(TM/'manifests/cohorts.json')['primary'] if r['id'] in read(OLD/'splits/diagnostic64.json')]
    records=[]
    for r in realrows:
        ref=image_at(DATASET/r['reference']);sketch=image_at(DATASET/r['sketch'])
        refs=[ref,rotate(ref,90),rotate(ref,180)]
        records.append((r['id'],r['caption'],[('sketch',sketch)]+[(a,v) for a,v in zip(ARMS,refs)]+
            [('R0 crop',refs[0].crop(CONFIG['crop'])),('R90 crop',refs[1].crop(CONFIG['crop']))]))
    panels(records,OUT/'visual_audit/G0a_all64_inputs')
    write(OUT/'G0a_real_input_annotations/rows.json',realrows)
    for reviewer in [1,2]:
        p=OUT/'G0a_real_input_annotations'/('reviewer%d_template.csv'%reviewer)
        with p.open('w',newline='') as f:
            keys=['id','single_axis_pattern','pattern_not_structural_edge','source_patch_purity','reference_R90_valid',
                'padding_or_crop_artifact','text_direction_conflict','target_pattern_support','reason_codes']
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([dict(id=r['id']) for r in realrows])
    (dest/'frozen_split_sha256.txt').write_text('\n'.join(str(p)+': '+sha(p) for p in [
        OUT/'protocol/gc_controlled_dev128_new_ids.json',OUT/'protocol/gc_train_pool_ids.json']),encoding='utf-8')
    decision(gc_controlled_dev_new_frozen=len(dev)==128,next_route='two_AI_input_and_target_reviews',
        g1a_real_crop_candidate_n=len(candidates))
    verify_frozen();bundle('preparation')

if __name__=='__main__':run()
