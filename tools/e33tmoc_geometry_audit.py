"""当前remap的合成/局部/真实坐标审计；失败不运行外观候选。"""
import argparse
import tarfile
import cv2
import numpy as np
from PIL import Image,ImageDraw
from models.e33tm_generation_wrapper import spatial_carrier
from data.e32_target_pseudogt import FrozenFeatures,image_at,masks,SIZE
from tools.e33tmoc_protocol import *

def stripe(theta, uv=None, size=SIZE):
    if uv is None:
        y,x=np.indices((size[1],size[0]),dtype=np.float32)
        x-=(size[0]-1)/2; y-=(size[1]-1)/2
    else:
        x=uv[...,0]-(size[0]-1)/2;y=uv[...,1]-(size[1]-1)/2
    gray=np.uint8(np.clip(128+80*np.cos(2*np.pi*(-np.sin(theta)*x+np.cos(theta)*y)/16),0,255))
    return Image.fromarray(np.repeat(gray[...,None],3,axis=-1))

def extract(image):
    return FrozenFeatures(None,'cpu').extract(image,perceptual=False)['geometry']

def metrics(geometry,target,eligible=None):
    if eligible is None: eligible=np.ones(geometry.shape[:2],bool)
    theta=np.degrees(np.arctan2(geometry[...,1],geometry[...,0]))/2%180
    error=np.abs((theta-target+90)%180-90)
    readable=eligible&(geometry[...,3]>=.25)
    return dict(cell_count=int(eligible.sum()),readable_cells=int(readable.sum()),
        readable_fraction=float(readable.sum()/max(eligible.sum(),1)),
        mean_axial_error=float(error[readable].mean()) if readable.any() else None,
        all_cell_error=float(error[eligible].mean()) if eligible.any() else None)

def montage(images,path):
    width=192; height=282
    canvas=Image.new('RGB',(width*len(images),height),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,image) in enumerate(images):
        draw.text((i*width+3,3),label,fill='black');image=image.copy();image.thumbnail((width-4,256))
        canvas.paste(image,(i*width+2,24))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def synthetic(folder):
    folder.mkdir(parents=True,exist_ok=True)
    source=stripe(0.);source.save(folder/'procedural_source.png')
    source_metric=metrics(extract(source),0.)
    mask=Image.new('L',SIZE,255);rows=[]
    for angle in PROTOCOL['synthetic']['angles']:
        radians=np.deg2rad(angle);field=np.empty((2,64,48),np.float32)
        field[0]=np.cos(2*radians);field[1]=np.sin(2*radians)
        _,coordinate=spatial_carrier(source,field,mask)
        raw=Image.fromarray(cv2.remap(np.asarray(source),coordinate.uv[...,0],coordinate.uv[...,1],
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
        direct=stripe(radians);unbounded=stripe(0.,uv=coordinate.uv)
        raw.save(folder/('uniform_%g_S1.png'%angle));direct.save(folder/('uniform_%g_direct.png'%angle))
        g=extract(raw);m=metrics(g,angle)
        row=dict(angle=angle,formal=m,direct_control=metrics(extract(direct),angle),
            unbounded_uv_control=metrics(extract(unbounded),angle),
            uv_out_of_bounds_fraction=float(((coordinate.uv[...,0]<0)|(coordinate.uv[...,0]>383)|
                (coordinate.uv[...,1]<0)|(coordinate.uv[...,1]>511)).mean()))
        row['pass']=bool(m['mean_axial_error'] is not None and m['mean_axial_error']<=3 and m['readable_fraction']>=.95)
        rows.append(row);np.savez_compressed(folder/('uniform_%g.npz'%angle),field=field,uv=coordinate.uv,geometry=g)
        montage([('current raw S1',raw),('direct control',direct),('unbounded UV control',unbounded)],folder/('uniform_%g_review.png'%angle))
        print('[OC uniform]',angle,row,flush=True)
    # 局部场在同一全画布中定义，不把边界跨区patch当纯方向patch。
    gy,gx=np.indices((64,48));local_fields={
        'left0_right90':np.where(gx<24,0.,90.),
        'top45_bottom135':np.where(gy<32,45.,135.),
        'quadrants':np.where(gy<32,np.where(gx<24,0.,45.),np.where(gx<24,90.,135.))}
    local=[]
    for name,angle in local_fields.items():
        theta=np.deg2rad(angle);field=np.stack([np.cos(2*theta),np.sin(2*theta)]).astype(np.float32)
        _,coordinate=spatial_carrier(source,field,mask)
        raw=Image.fromarray(cv2.remap(np.asarray(source),coordinate.uv[...,0],coordinate.uv[...,1],
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
        target=np.zeros((16,12),float);eligible=np.ones((16,12),bool)
        for y in range(16):
            for x in range(12):
                cy,cx=round((y+.5)*512/16),round((x+.5)*384/12)
                y0,x0=min(max(cy-32,0),448),min(max(cx-32,0),320)
                region=cv2.resize(angle,SIZE,interpolation=cv2.INTER_NEAREST)[y0:y0+64,x0:x0+64]
                target[y,x]=region[32,32];eligible[y,x]=np.all(region==region[32,32])
        geometry=extract(raw);direct=stripe(cv2.resize(theta,SIZE,interpolation=cv2.INTER_NEAREST))
        row=dict(name=name,all_cells=metrics(geometry,target),region_interior=metrics(geometry,target,eligible),
            direct_control=metrics(extract(direct),target,eligible));local.append(row)
        raw.save(folder/(name+'_S1.png'));montage([('current local raw',raw),('direct local control',direct)],folder/(name+'_review.png'))
        np.savez_compressed(folder/(name+'.npz'),field=field,uv=coordinate.uv,geometry=geometry,target=target,eligible=eligible)
    # 主Gate保留全部8角度；另报 pooled，不用平均掩盖某一方向失败。
    total=sum(r['formal']['cell_count'] for r in rows)
    count=sum(r['formal']['readable_cells'] for r in rows)
    pooled_error=sum(r['formal']['mean_axial_error']*r['formal']['readable_cells'] for r in rows if r['formal']['mean_axial_error'] is not None)/max(count,1)
    result=dict(uniform=rows,local=local,source_control=source_metric,
        pooled_readable_fraction=count/total,pooled_mean_axial_error=pooled_error,
        geometry_mapping_pass=all(r['pass'] for r in rows))
    write(folder/'synthetic_results.json',result)
    return result

def overlay(image,field):
    result=image.copy();draw=ImageDraw.Draw(result)
    h,w=field.shape[1:]
    for y in range(2,h,5):
        for x in range(2,w,5):
            theta=np.arctan2(field[1,y,x],field[0,y,x])/2
            px=(x+.5)*image.width/w;py=(y+.5)*image.height/h
            dx=8*np.cos(theta);dy=8*np.sin(theta)
            draw.line((px-dx,py-dy,px+dx,py+dy),fill=(255,0,0),width=2)
    return result

def real_audit(cohort):
    diagnostic=set(read(OUT/'splits/diagnostic64.json'));records=[]
    for row in cohort:
        if row['id'] not in diagnostic:continue
        native=Image.open(DATASET/row['sketch']);reference_native=Image.open(DATASET/row['reference'])
        sketch=image_at(DATASET/row['sketch']);reference=image_at(DATASET/row['reference'])
        input_mask,info=masks(sketch);mask=Image.fromarray(input_mask.astype(np.uint8)*255)
        with np.load(IF/'reproduction/seed42/fields'/(row['id']+'.npz')) as z:
            orientation=z['orientation'][0].copy();confidence=z['confidence'][0].copy()
        _,coordinate=spatial_carrier(reference,orientation,mask)
        raw=Image.fromarray(cv2.remap(np.asarray(reference),coordinate.uv[...,0],coordinate.uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
        old=IF/'stage_survival/seed42'/row['id']/'S1/R0.png'
        assert np.array_equal(np.asarray(raw),np.asarray(Image.open(old))), 'OC C0 differs from frozen IF S1'
        blank=float((np.all(np.asarray(reference)>=245,axis=-1)).mean())
        boundary=input_mask&~(cv2.erode(input_mask.astype(np.uint8),np.ones((17,17),np.uint8))>0)
        record=dict(id=row['id'],original_sketch_size=list(native.size),original_reference_size=list(reference_native.size),
            reference_crop_size=list(reference.size),reference_crop_offset=[0,0],reference_padding_offset=[0,0],
            target_canvas_size=list(SIZE),field_grid_hw=list(orientation.shape[1:]),E26_grid_hw=[16,12],
            mask_size=list(mask.size),resize_scale_reference=[384/reference_native.width,512/reference_native.height],
            resize_scale_sketch=[384/native.width,512/native.height],bbox=list(coordinate.bbox),
            bbox_offset=list(coordinate.bbox[:2]),width_height='PIL W,H; ndarray H,W; UV[...,0]=x,UV[...,1]=y',
            y_axis='down',angle='texture tangent, positive toward image y',source_angle=coordinate.orientation[0],
            target_support_fraction=float(input_mask.mean()),blank_reference_fraction=blank,
            boundary_fraction=float(boundary.sum()/max(input_mask.sum(),1)),mask_info=info,
            C0_raw_pixels_equal_IF=True)
        records.append(record)
        folder=OUT/'geometry_audit/real'/row['id'];folder.mkdir(parents=True,exist_ok=True)
        write(folder/'mapping.json',record)
        np.savez_compressed(folder/'mapping.npz',orientation=orientation,confidence=confidence,uv=coordinate.uv,target_support=input_mask)
    write(OUT/'geometry_audit/real_mapping.json',records)
    sets={name:[r['id'] for r in sorted(records,key=lambda r:(sign*r[key],r['id']))[:16]] for name,key,sign in
        [('high_support16','target_support_fraction',-1),('low_support16','target_support_fraction',1),
         ('boundary_sensitive16','boundary_fraction',-1),('blank_reference16','blank_reference_fraction',-1)]}
    sets['field_success_fixed16']=read(OUT/'splits/field_success_fixed_hash16.json')
    write(OUT/'splits/visual_sets.json',dict(sets=sets,selection='input-only metrics before any OC candidate; diagnostic64; fixed16 ties by identity'))
    lookup={r['id']:r for r in cohort};pages=[]
    for sid in sets['field_success_fixed16']:
        row=lookup[sid];sketch=image_at(DATASET/row['sketch']);ref=image_at(DATASET/row['reference'])
        with np.load(OUT/'geometry_audit/real'/sid/'mapping.npz') as z:field=z['orientation'].copy()
        raw=Image.open(IF/'stage_survival/seed42'/sid/'S1/R0.png').convert('RGB')
        path=OUT/'visual_audit/geometry_fixed16'/(sid+'.png')
        montage([('reference',ref),('target + RF tangent',overlay(sketch,field)),('current S1 + RF tangent',overlay(raw,field))],path)
        pages.append(Image.open(path).copy())
    for i in range(0,16,4):
        page=Image.new('RGB',(pages[0].width,pages[0].height*4),'white')
        for j,image in enumerate(pages[i:i+4]):page.paste(image,(0,j*image.height))
        page.save(OUT/'visual_audit/geometry_fixed16'/('page%d.png'%(i//4)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic-only',type=Path);args=parser.parse_args()
    if args.synthetic_only:
        synthetic(args.synthetic_only);return
    cohort=prepare();result=synthetic(OUT/'geometry_audit/synthetic');real_audit(cohort);frozen_check()
    decision=dict(geometry_mapping_pass=result['geometry_mapping_pass'],selected_candidate=None,
        confirmation_pass=None,vae_survival_pass=None,seed42_full_pass=None,seed43_full_pass=None,seed44_full_pass=None,
        orientation_preserving_carrier_pass=False if not result['geometry_mapping_pass'] else None,
        primary_failure_mode='synthetic_mapping_gate_failed' if not result['geometry_mapping_pass'] else None,
        next_route='coordinate_mapping_bug' if not result['geometry_mapping_pass'] else 'C0_C1_carrier_validation')
    write(OUT/'decision_summary.json',decision)
    write(OUT/'result_table.json',dict(geometry_audit=result))
    write(OUT/'completion_check.json',dict(OC0_complete=True,experiment_execution_complete=False,
        awaiting='fixed16_visual_audit_then_geometry_gate_routing',training_steps=0))
    entries={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.gz','.log','.err') and p.name!='artifact_manifest.json'}
    write(OUT/'artifact_manifest.json',dict(files=entries,git_commit=commit()))
    with tarfile.open(OUT/'geometry_review.tar.gz','w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix not in ('.gz','.log','.err'):tar.add(p,arcname=str(p.relative_to(OUT)))
    print('[OC0 complete]',decision,'bundle SHA256',sha(OUT/'geometry_review.tar.gz'),flush=True)

if __name__=='__main__':main()
