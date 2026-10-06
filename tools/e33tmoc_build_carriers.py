"""先验证C0/C1；未通过合成Gate不启动真实载体比较。"""
import argparse
import cv2
import numpy as np
from PIL import Image
from models.e33tm_generation_wrapper import spatial_carrier
from data.e33rf_real_rotation_dataset import rotate
from data.e32_field_dataset import cache_path
from tools.e33rf_common import E32
from tools.e33tmif_metrics import pair,arm_geometry,summarize
from tools.e33tmoc_geometry_audit import stripe,extract,montage
from tools.e33tmoc_protocol import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--appearance',action='store_true');args=parser.parse_args()
    assert read(OUT/'decision_summary.json')['geometry_mapping_pass'], 'OC0 Gate失败，停止'
    assert read(OUT/'visual_audit/geometry_review.json')['pass'], '须先审阅固定16例坐标叠图'
    cohort=prepare();ids=read(OUT/'splits/diagnostic64.json');rows=[r for r in cohort if r['id'] in ids]
    input_hashes=freeze_carrier_inputs(cohort,ids)
    if args.appearance:
        assert read(OUT/'decision_summary.json')['C1_upper_bound_reproduction_pass'], 'C1未恢复，禁止开发外观载体'
        from models.e33tmoc_carrier import construct,CONFIG
        path=OUT/'protocol/appearance_carrier.json'
        if path.exists():assert read(path)==CONFIG
        else:write(path,CONFIG)
    names=('C2_appearance','C3_support_aware','C4_confidence_blend') if args.appearance else ('C0_current','C1_analytic')
    results={name:{s:[] for s in ('S1','S2')} for name in names}
    from data.e32_target_pseudogt import image_at,masks,upsample_geometry
    for row in rows:
        sid=row['id'];sketch=image_at(DATASET/row['sketch']);reference=image_at(DATASET/row['reference'])
        mask=Image.fromarray(masks(sketch)[0].astype(np.uint8)*255)
        with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:
            orientation=z['orientation'][:3].copy();confidence=z['confidence'][:3].copy()
        with np.load(cache_path(E32,sid)) as z:
            gt=z['supervision_geometry'].copy();support=(z['supervision_interior']>=.95)&(gt[3]>=.25)
        inputs=dict(zip(('R0','R90','R180'),(reference,rotate(reference,90),rotate(reference,180))))
        old=read(IF/'stage_survival/seed42'/sid/'case.json')
        for name in results:
            dest=OUT/'candidates'/name/'seed42'/sid;geo={s:{} for s in ('S1','S2')};scores={s:{} for s in geo}
            for ai,(arm,ref) in enumerate(inputs.items()):
                if name=='C0_current':
                    S2,coordinate=spatial_carrier(ref,orientation[ai],mask)
                    S1=Image.fromarray(cv2.remap(np.asarray(ref),coordinate.uv[...,0],coordinate.uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
                elif name=='C1_analytic':
                    dense=cv2.resize(np.moveaxis(orientation[ai],0,-1),mask.size,interpolation=cv2.INTER_LINEAR)
                    theta=np.arctan2(dense[...,1],dense[...,0])/2
                    S1=stripe(theta,size=mask.size);rgb=np.asarray(S1).copy();rgb[np.asarray(mask)==0]=255
                    S2=Image.fromarray(rgb)
                else:
                    S1,S2,state,metadata,bank=construct(ref,orientation[ai],confidence[ai],mask,sid,name)
                    dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/(arm+'_construction.npz'),**state)
                    write(dest/(arm+'_patch_bank.json'),metadata)
                    patch_dir=dest/'source_patches'/arm;patch_dir.mkdir(parents=True,exist_ok=True)
                    for bi,p in enumerate(bank):Image.fromarray(p['pixels']).save(patch_dir/('%03d.png'%bi))
                for stage,image in [('S1',S1),('S2',S2)]:
                    folder=dest/stage;folder.mkdir(parents=True,exist_ok=True);image.save(folder/(arm+'.png'))
                    geometry=upsample_geometry(extract(image)).transpose(2,0,1)
                    if name=='C0_current':
                        assert np.array_equal(np.asarray(image),np.asarray(Image.open(IF/'stage_survival/seed42'/sid/stage/(arm+'.png'))))
                    geo[stage][arm]=geometry;scores[stage][arm]=arm_geometry(geometry,support,gt[3])
                    np.savez_compressed(folder/(arm+'_orientation.npz'),geometry=geometry,orientation_angle=np.degrees(np.arctan2(geometry[1],geometry[0]))/2%180,
                        orientation_confidence=geometry[3],support_mask=support&(geometry[3]>=.25))
                np.savez_compressed(dest/(arm+'_inputs.npz'),orientation=orientation[ai],confidence=confidence[ai],target_support=np.asarray(mask)>0,gt_support=support)
            case={}
            for stage in geo:
                record=dict(id=sid,stage=stage,arms=scores[stage],**pair(geo[stage],support,gt[3]))
                if name=='C0_current':
                    for key in ('r90_success','r180_success','r90_readable','r180_readable'):
                        assert record[key]==old['stages'][stage][key], 'C0评测不一致'
                    for key in ('r90_error','r180_error'):
                        assert record[key] is None and old['stages'][stage][key] is None or abs(record[key]-old['stages'][stage][key])<1e-5
                results[name][stage].append(record);case[stage]=record
            write(dest/'case.json',case)
        print('[OC carrier]',sid,flush=True)
    table={name:{stage:summarize(records) for stage,records in stages.items()} for name,stages in results.items()}
    for name,stages in results.items():
        for stage,records in stages.items():write(OUT/'candidates'/name/(stage+'_rows.json'),records)
    label='appearance' if args.appearance else 'initial'
    write(OUT/'candidates'/(label+'_summary.json'),table)
    ids=read(OUT/'splits/field_success_fixed_hash16.json');panels=[]
    lookup={r['id']:r for r in cohort}
    for sid in ids:
        ref=image_at(DATASET/lookup[sid]['reference'])
        images=[('reference',ref)]+[(name+' '+stage,Image.open(OUT/'candidates'/name/'seed42'/sid/stage/'R90.png'))
            for name in results for stage in ('S1','S2')]
        path=OUT/'visual_audit'/(label+'_fixed16')/(sid+'.png');montage(images,path);panels.append(Image.open(path).copy())
    for i in range(0,16,4):
        page=Image.new('RGB',(panels[0].width,panels[0].height*4),'white')
        for j,p in enumerate(panels[i:i+4]):page.paste(p,(0,j*p.height))
        page.save(OUT/'visual_audit'/(label+'_fixed16')/('page%d.png'%(i//4)))
    d=read(OUT/'decision_summary.json')
    if not args.appearance:
        for name,tag in [('C0_current','C0'),('C1_analytic','C1')]:
            d[tag+'_s1_r90']=table[name]['S1']['statistics']['r90_success']['mean']
        d['C0_reproduction_pass']=True
        d['C1_upper_bound_reproduction_pass']=d['C1_s1_r90']>=.50
        if not d['C1_upper_bound_reproduction_pass']:
            d.update(orientation_preserving_carrier_pass=False,primary_failure_mode='analytic_upper_bound_reproduction_failed',next_route='analytic_carrier_pipeline_revisit')
        else:d['next_route']='C2_C3_C4_appearance_carriers'
    else:
        for name,tag in zip(names,('C2','C3','C4')):d[tag+'_s1_r90']=table[name]['S1']['statistics']['r90_success']['mean']
        d['next_route']='appearance_metrics_and_candidate_gate'
    write(OUT/'decision_summary.json',d)
    result=read(OUT/'result_table.json');result['carrier_'+label]=table;write(OUT/'result_table.json',result)
    frozen_check()
    assert {p:sha(p) for p in input_hashes}==input_hashes
    from tools.e33tmoc_geometry_audit import tarfile
    bundle=OUT/('carrier_'+label+'_review.tar.gz')
    with tarfile.open(bundle,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and (p.suffix=='.json' or 'visual_audit' in p.parts):tar.add(p,arcname=str(p.relative_to(OUT)))
    print('[OC carriers complete]',d,'bundle SHA256',sha(bundle),flush=True)

if __name__=='__main__':main()
