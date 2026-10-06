"""同split同输入的载体构建、方向和真实LPIPS评测。"""
import argparse
import os
import tarfile
import cv2
import numpy as np
import torch
from PIL import Image
from garment_mask_utils import estimate_cloth_foreground_mask
from data.e32_target_pseudogt import image_at,masks,upsample_geometry
from data.e32_field_dataset import cache_path
from data.e33rf_real_rotation_dataset import rotate
from models.e33tm_generation_wrapper import spatial_carrier
from tools.e33rf_common import E32
from tools.e33tmif_metrics import pair,arm_geometry,summarize,bootstrap
from tools.e33tmoc_geometry_audit import stripe,extract,montage
from tools.e33tmoc_appearance_eval import patches,measure,frozen_lpips,model_hash,APPEARANCE_PROTOCOL
from tools.e33tmms_protocol import *

def folder(name,split):
    return OUT/('reproduce' if name.startswith('C') else name)/split/name

def build(name,ref,orientation,confidence,mask,sid):
    if name=='C0':
        s2,field=spatial_carrier(ref,orientation,mask)
        s1=Image.fromarray(cv2.remap(np.asarray(ref),field.uv[...,0],field.uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101))
        return s1,s2,dict(uv=field.uv),dict(method=name),[]
    if name=='C1':
        dense=cv2.resize(np.moveaxis(orientation,0,-1),mask.size,interpolation=cv2.INTER_LINEAR)
        s1=stripe(np.arctan2(dense[...,1],dense[...,0])/2,size=mask.size)
        rgb=np.asarray(s1).copy();rgb[np.asarray(mask)==0]=255
        return s1,Image.fromarray(rgb),{},dict(method=name,diagnostic_only=True),[]
    if name=='C2':
        from models.e33tmoc_carrier import construct
        return construct(ref,orientation,confidence,mask,sid,'C2_appearance')
    if name=='M1':
        from models.e33tmms_m1 import construct
        return construct(ref,orientation,confidence,mask,sid)
    raise ValueError(name)

def freeze_visual(rows,ids):
    path=OUT/'splits/visual_sets.json'
    if path.exists():return read(path)
    scores=[]
    for r in rows:
        if r['id'] not in ids:continue
        ref=image_at(DATASET/r['reference']);rgb=np.asarray(ref)
        fg=np.asarray(estimate_cloth_foreground_mask(ref,*ref.size)[0])>127
        values=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[fg].astype(float)/255
        target=masks(image_at(DATASET/r['sketch']))[0].astype(np.uint8)
        eroded=cv2.erode(target,np.ones((17,17),np.uint8))
        scores.append(dict(id=r['id'],texture=float(values.var()) if len(values) else 0.,
            blank=float(np.all(rgb>=245,axis=-1).mean()),boundary=1-float(eroded.sum()/max(1,target.sum()))))
    sort=lambda key,rev:[r['id'] for r in sorted(scores,key=lambda r:((-1 if rev else 1)*r[key],r['id']))[:16]]
    value=dict(sets=dict(field_success16=read(OUT/'splits/field_success_fixed_hash16.json'),
        high_texture16=sort('texture',True),low_texture16=sort('texture',False),
        blank_reference16=sort('blank',True),boundary16=sort('boundary',True)),
        input_scores=scores,selection='input-only source foreground gray variance, bright-background fraction, mask boundary ratio; before M1')
    write(path,value);return value

def evaluate(names,split,shard=None):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    # 分片只改变身份任务分配；每例随机数仍只由sid决定，结果统一在汇总作业判定。
    cohort=prepare() if shard is None else read(TM/'manifests/cohorts.json')['primary']
    ids=read(OUT/'splits'/(split+'.json'));rows=[r for r in cohort if r['id'] in ids]
    fingerprints=freeze_inputs(cohort,ids,split) if shard is None else read(OUT/'protocol'/('inputs_%s_N%d.json'%(split,len(ids))))
    if split=='diagnostic64' and shard is None:freeze_visual(cohort,ids)
    if shard is not None:rows=[r for i,r in enumerate(rows) if i%shard[1]==shard[0]]
    metric,init=frozen_lpips();metric_sha=model_hash(metric)
    if shard is None:write(OUT/'protocol/appearance_metrics.json',dict(APPEARANCE_PROTOCOL,**init,lpips_state_sha256=metric_sha))
    for name in names:
        direction={s:[] for s in ('S1','S2')};appearance={s:[] for s in direction}
        dest=folder(name,split)
        for row in rows:
            sid=row['id'];case_dir=dest/'seed42'/sid
            existing=case_dir/'case.json'
            if existing.exists():
                record=read(existing)
                assert record['input_sha256']=={n:fingerprints[n] for n in record['input_sha256']}
                for s in direction:direction[s].append(record['direction'][s]);appearance[s].append(record['appearance'][s])
                continue
            ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch'])
            mask=Image.fromarray(masks(sketch)[0].astype(np.uint8)*255)
            inner=cv2.erode((np.asarray(mask)>0).astype(np.uint8),np.ones((17,17),np.uint8))>0
            with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:
                orientations=z['orientation'][:3].copy();confidences=z['confidence'][:3].copy()
            with np.load(cache_path(E32,sid)) as z:
                gt=z['supervision_geometry'].copy();support=(z['supervision_interior']>=.95)&(gt[3]>=.25)
            refs=dict(zip(('R0','R90','R180'),(ref,rotate(ref,90),rotate(ref,180))))
            target_boxes=patches(ref,inner,sid,'shared','target');geometry={s:{} for s in direction}
            arms={s:{} for s in direction};app_arms={s:[] for s in direction}
            for ai,(arm,reference) in enumerate(refs.items()):
                s1,s2,state,metadata,bank=build(name,reference,orientations[ai],confidences[ai],mask,sid)
                case_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(case_dir/(arm+'_construction.npz'),**state)
                np.savez_compressed(case_dir/(arm+'_inputs.npz'),orientation=orientations[ai],confidence=confidences[ai],
                    target_support=np.asarray(mask)>0,gt_support=support)
                write(case_dir/(arm+'_construction.json'),metadata)
                if bank:
                    # 保存真实source bank与canonical版本，selected索引可恢复所有选择。
                    key='rgb' if name=='M1' else 'pixels'
                    data=dict(rgb=np.stack([p[key] for p in bank]))
                    if name=='M1':data.update(canonical=np.stack([p['canonical'] for p in bank]),canonical_support=np.stack([p['canonical_support'] for p in bank]),
                        Lab=np.stack([p['Lab'] for p in bank]),gradient=np.stack([p['gradient'] for p in bank]))
                    np.savez_compressed(case_dir/(arm+'_source_bank.npz'),**data)
                source_mask=np.asarray(estimate_cloth_foreground_mask(reference,*reference.size)[0])>127
                source_boxes=patches(reference,source_mask,sid,arm,'source')
                write(case_dir/(arm+'_appearance_patches.json'),dict(source=source_boxes,target=target_boxes))
                for stage,image in [('S1',s1),('S2',s2)]:
                    path=case_dir/stage/(arm+'.png');path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                    g=upsample_geometry(extract(image)).transpose(2,0,1);geometry[stage][arm]=g
                    arms[stage][arm]=arm_geometry(g,support,gt[3])
                    np.savez_compressed(path.with_name(arm+'_orientation.npz'),geometry=g,
                        orientation_angle=np.degrees(np.arctan2(g[1],g[0]))/2%180,
                        orientation_confidence=g[3],support_mask=support&(g[3]>=.25))
                    app=measure(image,reference,inner,source_mask,target_boxes,source_boxes,metric)
                    app.update(id=sid,arm=arm,stage=stage,image_sha256=sha(path));app_arms[stage].append(app)
                    write(case_dir/stage/(arm+'_appearance.json'),app)
            record=dict(id=sid,direction={},appearance={},input_sha256={n:h for n,h in fingerprints.items() if
                sid in n or n in (str(DATASET/row['reference']),str(DATASET/row['sketch']))})
            for stage in direction:
                record['direction'][stage]=dict(id=sid,stage=stage,arms=arms[stage],**pair(geometry[stage],support,gt[3]))
                keys=['Lab_color_distance','color_histogram_similarity','texture_score','patch_lpips','mean_RGB_distance']
                record['appearance'][stage]=dict(id=sid,**{k:float(np.mean(values)) if values else None for k in keys
                    for values in [[r[k] for r in app_arms[stage] if r[k] is not None]]})
                direction[stage].append(record['direction'][stage]);appearance[stage].append(record['appearance'][stage])
            write(existing,record);print('[MS benchmark]',name,split,sid,flush=True)
        if shard is not None:
            write(OUT/'execution'/('M1_shard_%d.json'%shard[0]),dict(index=shard[0],count=shard[1],
                ids=[r['id'] for r in rows],git_commit=commit(),lpips_state_sha256=metric_sha,
                complete=True,algorithm_change=False))
            continue
        for stage in direction:
            write(dest/(stage+'_direction_rows.json'),direction[stage]);write(dest/(stage+'_appearance_cases.json'),appearance[stage])
        result=dict(direction={s:summarize(rs) for s,rs in direction.items()},
            appearance={s:dict(case_count=len(rs),statistics={k:bootstrap([r[k] for r in rs]) for k in rs[0] if k!='id'}) for s,rs in appearance.items()})
        write(dest/'summary.json',result)
    assert model_hash(metric)==metric_sha and {n:sha(n) for n in fingerprints}==fingerprints
    if shard is None:frozen_check()

def reproduce():
    evaluate(['C0','C1','C2'],'diagnostic64');checks={};old_names=['C0_current','C1_analytic','C2_appearance']
    old_dir=dict(read(OC/'result_table.json')['carrier_initial'],**read(OC/'result_table.json')['carrier_appearance'])
    old_app=read(OC/'appearance_metrics/summary.json')
    for name,old in zip(['C0','C1','C2'],old_names):
        result=read(folder(name,'diagnostic64')/'summary.json')
        for stage in ['S1','S2']:
            rate=result['direction'][stage]['statistics']['r90_success']['mean']
            texture=result['appearance'][stage]['statistics']['texture_score']['mean']
            prior=old_app[old][stage]['statistics']['texture_score']['mean']
            checks[name+'_'+stage]=dict(r90_delta=rate-old_dir[old][stage]['statistics']['r90_success']['mean'],
                texture_relative_delta=(texture-prior)/prior)
    passed=all(abs(v['r90_delta'])<=.02 and abs(v['texture_relative_delta'])<=.05 for v in checks.values())
    write(OUT/'reproduce/gate.json',dict(checks=checks,**{'pass':passed}))
    decision(reproduction_pass=passed,next_route='M1' if passed else 'representation_search_reproduction_mismatch')
    print('[MS0 Gate]',passed,checks,flush=True)

def carrier_gate(name,split):
    result=read(folder(name,split)/'summary.json');base=read(folder('C0',split)/'summary.json')
    s1=result['direction']['S1']['statistics'];s2=result['direction']['S2']['statistics']
    app=result['appearance']['S2']['statistics'];baseline=base['appearance']['S2']['statistics']
    old={r['id']:r for r in read(folder('C0',split)/'S2_direction_rows.json')}
    new=read(folder(name,split)/'S2_direction_rows.json')
    gain=bootstrap([r['r90_success']-old[r['id']]['r90_success'] for r in new])
    if split=='diagnostic64':
        checks=dict(S1_r90=s1['r90_success']['mean']>=.45,S2_r90=s2['r90_success']['mean']>=.40,
            gain=gain['mean']>=.15,gain_ci=gain['ci95'][0]>0,R180=s2['r180_success']['mean']>=.60,
            readable=s2['r90_readable']['mean']>=.55,texture=app['texture_score']['mean']>=.19,
            Lab=app['Lab_color_distance']['mean']<=baseline['Lab_color_distance']['mean']*1.2)
        hard_stop=s2['r90_success']['mean']<.35 or app['texture_score']['mean']<.16
        passed=all(checks.values());decision(**{name+'_run':True,name+'_carrier_pass':passed},
            next_route=name+'_confirmation' if passed else 'M2')
    else:
        checks=dict(S2_r90=s2['r90_success']['mean']>=.35,gain=gain['mean']>=.12,
            gain_ci=gain['ci95'][0]>0,texture=app['texture_score']['mean']>=.17)
        hard_stop=False;passed=all(checks.values());decision(**{name+'_confirmation_pass':passed},next_route=name+'_E5_mini' if passed else 'M2')
    gate=dict(checks=checks,**{'pass':passed},hard_stop=hard_stop,paired_gain=gain)
    write(folder(name,split)/'gate.json',gate)
    table=read(OUT/'result_table.json') if (OUT/'result_table.json').exists() else {}
    table[name+'_'+split]=result;table[name+'_'+split+'_gate']=gate;write(OUT/'result_table.json',table)
    print('[MS carrier Gate]',name,split,gate,flush=True)

def visuals(name='M1'):
    cohort=prepare();lookup={r['id']:r for r in cohort}
    from PIL import ImageDraw
    from tools.e33tmoc_geometry_audit import overlay
    for group,ids in read(OUT/'splits/visual_sets.json')['sets'].items():
        panels=[]
        for sid in ids:
            row=lookup[sid];ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch'])
            refs=[ref,rotate(ref,90),rotate(ref,180)]
            with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:fields=z['orientation'][:3].copy()
            panel=Image.new('RGB',(9*128,592),'white');draw=ImageDraw.Draw(panel);draw.text((3,2),sid+' | '+row['caption'][:150],fill='black')
            for ai,arm in enumerate(['R0','R90','R180']):
                items=[('Reference '+arm,refs[ai]),('RF tangent',overlay(sketch,fields[ai]))]
                for method in ['C0','C1','C2',name]:
                    f=folder(method,'diagnostic64')/'seed42'/sid/'S2'/(arm+'.png')
                    items.append((method+' S2',Image.open(f).copy()))
                dest=folder(name,'diagnostic64')/'seed42'/sid/'S2'
                with np.load(dest/(arm+'_orientation.npz')) as z:
                    g=z['geometry'];supp=z['support_mask'].copy()
                image=Image.open(dest/(arm+'.png'));items.append(('Readout tangent',overlay(image,g[:2])))
                items.append(('Readout support',Image.fromarray(np.uint8(supp)*255).resize(ref.size,Image.Resampling.NEAREST)))
                empty=Image.new('RGB',ref.size,(230,230,230));ImageDraw.Draw(empty).text((10,220),'not run',fill='black')
                items.append(('Final: not run',empty))
                for col,(label,img) in enumerate(items):
                    y=22+ai*190;draw.text((col*128+2,y),label,fill='black');img=img.convert('RGB');img.thumbnail((124,166));panel.paste(img,(col*128+2,y+18))
            path=OUT/'visual_audit'/name/group/(sid+'.png');path.parent.mkdir(parents=True,exist_ok=True);panel.save(path);panels.append(panel)
        for i in range(0,len(panels),4):
            page=Image.new('RGB',(panels[0].width,panels[0].height*len(panels[i:i+4])),'white')
            for j,img in enumerate(panels[i:i+4]):page.paste(img,(0,j*img.height))
            page.save(OUT/'visual_audit'/name/group/('page%d.png'%(i//4)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['C0','C1','C2',name];results=[read(folder(m,'diagnostic64')/'summary.json') for m in methods]
    for label,kind,key,scale in [('S1_R90','direction','r90_success',100),('S2_R90','direction','r90_success',100),
        ('S2_Texture','appearance','texture_score',1),('S2_Lab','appearance','Lab_color_distance',1)]:
        stage=label[:2];vals=[r[kind][stage]['statistics'][key] for r in results]
        means=np.array([v['mean'] for v in vals])*scale
        ci=np.array([v['ci95'] for v in vals])*scale
        plt.figure(figsize=(7,4));plt.bar(methods,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),capsize=5)
        plt.title(label+'; fixed64 identity bootstrap');plt.tight_layout();dest=OUT/'curves';dest.mkdir(exist_ok=True)
        plt.savefig(dest/(label+'.png'),dpi=130);plt.close()

def bundle(label):
    path=OUT/(label+'_review.tar.gz')
    with tarfile.open(path,'w:gz') as tar:
        for f in sorted(OUT.rglob('*')):
            if f.is_file() and (f.suffix=='.json' or 'visual_audit' in f.parts or 'curves' in f.parts):tar.add(f,arcname=str(f.relative_to(OUT)))
    print('[MS bundle]',path,'SHA256',sha(path),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['reproduce','M1','confirmation','visual'])
    parser.add_argument('--shard-count',type=int);a=parser.parse_args()
    shard=(int(os.environ['SLURM_ARRAY_TASK_ID']),a.shard_count) if a.shard_count else None
    if a.stage=='reproduce':reproduce();bundle('MS0')
    elif a.stage=='M1':
        assert read(OUT/'decision_summary.json')['reproduction_pass']
        evaluate(['M1'],'diagnostic64',shard)
        if shard is None:carrier_gate('M1','diagnostic64');visuals();bundle('M1')
    elif a.stage=='confirmation':
        assert read(OUT/'decision_summary.json')['M1_carrier_pass']
        evaluate(['C0','M1'],'confirmation64');carrier_gate('M1','confirmation64');bundle('M1_confirmation')
    else:visuals();bundle('M1')

if __name__=='__main__':main()
