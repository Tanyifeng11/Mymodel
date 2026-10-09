"""仅输入/参考/Sketch的预选审计；自动资格不是人工纹样真实性认证。"""
import hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
from data.e32_target_pseudogt import image_at
from data.e33rf_real_rotation_dataset import rotate
from data.e33gc_renderer import support,fft_tangent
from models.local_pattern_field import patch_geometry
from tools.e33gc_protocol import *

def axis_error(a,b):return float(abs((a-b+90)%180-90))

def run():
    prepare();ids=read(OUT/'splits/diagnostic64.json')
    cohort=[r for r in read(TM/'manifests/cohorts.json')['primary'] if r['id'] in ids];assert len(cohort)==64
    manifest=read(IF/'artifact_manifest.json')['files'];records=[];visual={}
    for row in cohort:
        sid=row['id'];ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch'])
        mask,inner,_=support(sketch);refs=[ref,rotate(ref,90),rotate(ref,180)]
        box=(128,192,256,320)
        patches=[r.crop(box) for r in refs];pg=[patch_geometry(p) for p in patches]
        fft=[fft_tangent(p,(0,0,128,128)) for p in patches]
        rotation_errors=[axis_error(fft[1],fft[0]-90),axis_error(fft[2],fft[0])]
        # 相同中心方形的真实rotation；padding/crop不进入此source资格读出。
        native90=np.array_equal(np.rot90(np.asarray(patches[0])),np.asarray(patches[1]))
        native180=np.array_equal(np.rot90(np.asarray(patches[0]),2),np.asarray(patches[2]))
        field_path=IF/'reproduction/seed42/fields'/(sid+'.npz')
        assert sha(field_path)==manifest[str(field_path.relative_to(IF))]
        with np.load(field_path) as z:confidence=z['confidence'][0,0].copy()
        reduced=cv2.resize(inner.astype(np.float32),(48,64),interpolation=cv2.INTER_AREA)
        cells=int(((reduced>=.95)&(confidence>=.25)).sum())
        gray=np.asarray(patches[0].convert('L'),float)/255
        blank=float(((gray>.97)|(gray<.03)).mean())
        full_blank=[float((np.asarray(r.convert('L'),float)>247).mean()) for r in refs]
        boundary_sensitive=bool(abs(full_blank[1]-full_blank[0])>.10)
        eligible=bool(all(p['confidence']>=.25 for p in pg) and max(rotation_errors)<=5 and native90 and native180 and cells>=16)
        item=dict(id=sid,input_only=True,source_box=list(box),source_patch_geometry=pg,FFT_tangent=fft,rotation_error=rotation_errors,
            sketch_RF2_supported_cells=cells,source_extreme_pixels=blank,full_reference_blank=full_blank,
            boundary_sensitive_proxy=boundary_sensitive,source_gray_std=float(gray.std()),
            native_centre_rotation_exact=[native90,native180],automatic_eligible=eligible,human_review=None,
            inputs={str(DATASET/row[k]):sha(DATASET/row[k]) for k in ['reference','sketch']},field_sha256=sha(field_path),
            reference_preprocess='original native np.rot90 centre crop/pad at384x512; original E5 plain_resize appearance path',
            review_required=['seams','folds','buttons','multiple motifs','non-tileable pattern','crop-boundary sensitivity'])
        records.append(item);visual[sid]=(row,refs,sketch)
    fixed=[sid for sid in read(OUT/'splits/field_success_fixed_hash16.json') if sid in ids];assert len(fixed)==16
    ranked=sorted(records,key=lambda r:(-r['sketch_RF2_supported_cells'],r['id']))
    hash_order=lambda rr:sorted(rr,key=lambda r:hashlib.sha256(('E33GC/input_groups/'+r['id']).encode()).hexdigest())
    groups=dict(high_support16=[r['id'] for r in ranked[:16]],low_support16=[r['id'] for r in ranked[-16:]],
        blank_reference16=[r['id'] for r in hash_order([r for r in records if r['source_gray_std']<.02 or r['source_extreme_pixels']>.65])[:16]],
        boundary_sensitive16=[r['id'] for r in hash_order([r for r in records if r['boundary_sensitive_proxy']])[:16]],
        single_pattern16=[r['id'] for r in hash_order([r for r in records if r['automatic_eligible']])[:16]])
    eligible=[r['id'] for r in records if r['automatic_eligible']]
    write(OUT/'G0_reproduction/input_audit.json',dict(records=records,fixed16=fixed,groups=groups,group_N={k:len(v) for k,v in groups.items()},
        automatic_eligible_ids=eligible,automatic_eligible_n=len(eligible),fully_audited_eligible_ids=None,independent_human_review=None,
        limitation='frozen RF2 sketch support and automatic source measures cannot distinguish seams/folds from true pattern; manual input review pending',
        reference_ground_truth=False,generated_outputs_used=False))
    decision(input_audited_eligible_n=None,intervention_validity_pass=None)
    panels=[]
    for sid in fixed:
        row,refs,sketch=visual[sid];r=next(v for v in records if v['id']==sid)
        panel=Image.new('RGB',(768,216),'white');draw=ImageDraw.Draw(panel)
        draw.text((2,2),sid+' | input-only auto='+str(r['automatic_eligible'])+' cells='+str(r['sketch_RF2_supported_cells']),fill='black')
        for i,(name,image) in enumerate(zip(['Sketch','R0','R90','R180','R0 centre','R90 centre'],[sketch,*refs,refs[0].crop(box),refs[1].crop(box)])):
            image=image.copy();image.thumbnail((124,166));panel.paste(image,(i*128+2,43));draw.text((i*128+2,24),name,fill='black')
        panels.append(panel)
    folder=OUT/'visual_audit/real_input_fixed16';folder.mkdir(parents=True,exist_ok=True)
    for i in range(4):
        page=Image.new('RGB',(768,864),'white')
        for j,panel in enumerate(panels[4*i:4*i+4]):page.paste(panel,(0,j*216))
        page.save(folder/('page%d.png'%i))
    frozen_check();bundle('input_audit');print('[GC input audit] automatic eligible',len(eligible),'human gate pending',flush=True)

if __name__=='__main__':run()
