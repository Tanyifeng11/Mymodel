"""M3 完整E5生成、固定64例Gate、同身份旧Full及匹配初始化对照。"""
import argparse
import numpy as np,torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from data.e32_target_pseudogt import image_at,masks
from data.e32_field_dataset import cache_path
from data.e33rf_real_rotation_dataset import rotate
from models.e33tm_generation_wrapper import load_e5,generate,spatial_carrier
from models.e33tmms_m3 import OrientedFeatureTransport,SingleSiteInjection
from tools.e33tmms_learned_losses import frozen_texture
from tools.e33tmms_image import image_summary
from tools.e33tmms_benchmark import bundle
from tools.e33tmms_protocol import *
from tools.e33rf_common import E32
from tools.e33tm_metrics import Evaluator
from tools.e33tmif_metrics import pair,arm_geometry,bootstrap
from tools.e33tmoc_geometry_audit import overlay
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes

def paired_delta(rows,baseline):
    lookup={r['id']:r for r in baseline};assert set(lookup)=={r['id'] for r in rows}
    return {k:bootstrap([r[k]-lookup[r['id']][k] for r in rows if r[k] is not None and lookup[r['id']][k] is not None])
        for k in ['r90_success','r180_success','texture_score','text_score','contour_f1']}

def gate(rows,old,matched,split):
    new=image_summary(rows);baseline=image_summary(old);control=image_summary(matched)
    delta=paired_delta(rows,old);matched_delta=paired_delta(rows,matched);s=new['statistics'];b=baseline['statistics']
    text_drop=-delta['text_score']['mean']/b['text_score']['mean']
    checks=dict(Image_R90=s['r90_success']['mean']>=.35,gain=delta['r90_success']['mean']>=.20,
        R180=s['r180_success']['mean']>=.65,Contour=delta['contour_f1']['mean']>=-.05,
        Text=text_drop<=.10,Texture=s['texture_score']['mean']>=b['texture_score']['mean']*.9)
    if split=='confirmation64':checks['paired_R90_CI_lower_positive']=delta['r90_success']['ci95'][0]>0
    method_pass=all(checks.values())
    global_checks=dict(Text_no_significant_decline=delta['text_score']['ci95'][1]>=0,
        Contour_no_significant_decline=delta['contour_f1']['ci95'][1]>=0)
    passed=method_pass and (split!='confirmation64' or all(global_checks.values()))
    hard_stop=s['r90_success']['mean']<.25 and s['texture_score']['mean']<=b['texture_score']['mean']
    result=dict(checks=checks,method_pass=method_pass,global_checks=global_checks,**{'pass':passed},hard_stop=hard_stop,
        paired_delta_vs_oldFull=delta,paired_delta_vs_matched_initialization=matched_delta,
        relative_text_drop=text_drop,case_count=64,baseline='same identities IF seed42 S9 oldFull',
        matched_control='same50-step original E5 pure-noise images freshly rescored, or oldFull for retained initialization')
    dest=OUT/'M3'/split;write(dest/'summary.json',dict(new=new,oldFull=baseline,matched_E5=control));write(dest/'gate.json',result)
    if split=='diagnostic64':
        decision(M3_feature_pass=passed,M3_e5_pass=None if passed else False,
            next_route='M3_confirmation64' if passed else 'representation_search_failed')
    else:
        updates=dict(M3_confirmation_pass=method_pass,M3_e5_pass=passed,next_route='M3_full_validation' if passed else 'representation_search_failed')
        if passed:updates.update(selected_method='M3',selected_space='feature transport',selected_image_r90=s['r90_success']['mean'],
            selected_texture_sim=s['texture_score']['mean'],selected_text_delta=delta['text_score']['mean'],selected_contour_delta=delta['contour_f1']['mean'])
        decision(**updates)
    table=read(OUT/'result_table.json');table['M3_'+split]=dict(new=new,oldFull=baseline,matched_E5=control,gate=result);write(OUT/'result_table.json',table)
    print('[MS M3 Gate]',split,result,flush=True)

def record(sid,geos,arms,support,weight):
    result=dict(id=sid,stage='S9',arms=arms,**pair(geos,support,weight))
    for key in ['texture_score','text_score','contour_f1']:
        values=[v[key] for v in arms.values() if v[key] is not None];result[key]=float(np.mean(values)) if values else None
    return result

@torch.inference_mode()
def run(split,initialization):
    cohort=prepare();d=read(OUT/'decision_summary.json');assert d['M3_run']
    if split=='confirmation64':assert d['M3_feature_pass']
    ids=read(OUT/'splits'/(split+'.json'));rows=[r for r in cohort if r['id'] in ids];assert len(rows)==64
    fingerprints=freeze_inputs(cohort,ids,split)
    checkpoint=OUT/'M3/real_mixed/checkpoint_final.pt';done=read(OUT/'M3/real_mixed/training_complete.json')
    assert done['complete'] and done['steps']==2000 and done['checkpoint_sha256']==sha(checkpoint)
    assert read(OUT/'M3/controlled/training_complete.json')['steps']==4000
    transport=OrientedFeatureTransport().cuda().eval().requires_grad_(False)
    state=torch.load(checkpoint,map_location='cpu');assert state['seed']==42;transport.load_state_dict(state['model'],strict=True)
    pipe,modules,size,ns=load_e5();assert size==(384,512)
    encoder=frozen_texture();evaluator=Evaluator();modules.update(feature_encoder=encoder,text_image_evaluator=evaluator.model)
    before=module_hashes(modules);active=effective_hashes(modules);injection=SingleSiteInjection(pipe,transport)
    dest=OUT/'M3'/split;dest.mkdir(parents=True,exist_ok=True)
    implementation=dict(initialization=initialization,steps=50 if initialization=='pure-noise' else 8,
        comparison_note='pure-noise50 differs from prescribed oldFull8; matched original E5 control is separately reported',
        checkpoint_sha256=sha(checkpoint),seed=42,site=transport.site,alpha=.1,trainable=7513,
        forward_inputs='reference RGB, frozenRF2, original sketch mask,caption; no target RGB/GT',
        sampling_weights=transport.weights.softmax(0).cpu().tolist(),git_commit=commit())
    existing=dest/'implementation.json'
    if existing.exists():assert read(existing)['initialization']==initialization and read(existing)['checkpoint_sha256']==sha(checkpoint)
    write(existing,implementation)
    manifest=read(IF/'artifact_manifest.json')['files'];new=[];old=[];matched=[]
    for row in rows:
        sid=row['id'];case=dest/'seed42'/sid;case.mkdir(parents=True,exist_ok=True)
        old_path=IF/'stage_survival/seed42'/sid/'S9/pair.json';assert sha(old_path)==manifest[str(old_path.relative_to(IF))]
        old_row=read(old_path);old.append(old_row)
        saved=case/'case.json'
        if saved.exists():
            value=read(saved);assert all(sha(case/n)==h for n,h in value['files'].items());new.append(value['metrics']);matched.append(value['matched_metrics']);continue
        ref=image_at(DATASET/row['reference']);refs=[ref,rotate(ref,90),rotate(ref,180)]
        sketch=image_at(DATASET/row['sketch']);mask=Image.fromarray(np.uint8(masks(sketch)[0])*255)
        with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:ori=z['orientation'][:3].copy();q=z['confidence'][:3].copy()
        geom=F.interpolate(torch.from_numpy(np.concatenate([ori,q],1)).cuda(),(512,384),mode='bilinear',align_corners=False)
        geom[:,:2]=F.normalize(geom[:,:2],dim=1);geom=torch.cat([geom,to_tensor(mask)[None].cuda().expand(3,-1,-1,-1)],1)
        features=encoder(torch.stack([to_tensor(r) for r in refs]).cuda()*2-1)
        with np.load(cache_path(E32,sid)) as z:gt=z['supervision_geometry'].copy();support=(z['supervision_interior']>=.95)&(gt[3]>=.25)
        geos={};arms={};control_geos={};control_arms={};schedules={}
        for i,arm in enumerate(['R0','R90','R180']):
            injection.set(features[i:i+1],geom[i:i+1]);sampled,_=transport.sample(features[i:i+1],geom[i:i+1])
            color=sampled[0,:3].cpu().numpy().transpose(1,2,0);color=(color-color.min())/max(float(color.max()-color.min()),1e-8)
            Image.fromarray(np.uint8(np.round(color*255))).resize((384,512)).save(case/(arm+'_feature.png'))
            np.savez_compressed(case/(arm+'_feature.npz'),F_ref=features[i].cpu().numpy().astype(np.float16),
                F_target=sampled[0].cpu().numpy().astype(np.float16),residual=injection.residual[0].cpu().numpy().astype(np.float16))
            scaffold=spatial_carrier(refs[i],ori[i],mask)[0] if initialization=='old-Full' else None
            count=injection.count;image,info=generate(pipe,size,ns,row['caption'],sketch,refs[i],mask,42,scaffold=scaffold)
            expected=50 if initialization=='pure-noise' else 8;assert injection.count-count==expected
            image.save(case/(arm+'.png'));scores,g=evaluator.evaluate(image,row['caption'],refs[i],mask,sketch)
            scores.update(arm_geometry(g,support,gt[3]));geos[arm]=g;arms[arm]=scores;schedules[arm]=info
            np.savez_compressed(case/(arm+'_orientation.npz'),geometry=g,support_mask=support&(g[3]>=.25))
            write(case/(arm+'.json'),dict(metrics=scores,generation=info,conditional_injections=expected,site_shape=injection.last_shape))
            if initialization=='pure-noise':
                path=TM/'baseline_e5'/sid/'d42'/(arm+'.png');original=read(path.with_suffix('.json'))
                assert sha(path)==original['output_sha256'] and original['caption']==row['caption'] and original['noise_sha256']==info['noise_sha256']
                cs,cg=evaluator.evaluate(Image.open(path).convert('RGB'),row['caption'],refs[i],mask,sketch)
                cs.update(arm_geometry(cg,support,gt[3]));control_geos[arm]=cg;control_arms[arm]=cs
                write(case/(arm+'_matched_control.json'),dict(image_sha256=sha(path),record_sha256=sha(path.with_suffix('.json')),metrics=cs))
        assert len({v['noise_sha256'] for v in schedules.values()})==1
        result=record(sid,geos,arms,support,gt[3]);control=record(sid,control_geos,control_arms,support,gt[3]) if initialization=='pure-noise' else old_row
        write(saved,dict(id=sid,metrics=result,matched_metrics=control,oldFull_sha256=sha(old_path),
            files={str(p.relative_to(case)):sha(p) for p in case.iterdir() if p.is_file() and p.name!='case.json'}))
        new.append(result);matched.append(control);print('[MS M3 image]',split,sid,flush=True)
    write(dest/'rows.json',new);write(dest/'oldFull_rows.json',old);write(dest/'matched_control_rows.json',matched)
    gate(new,old,matched,split);injection.close()
    after=module_hashes(modules);effective_after=effective_hashes(modules)
    assert before==after and active==effective_after and {p:sha(p) for p in fingerprints}==fingerprints
    write(dest/'frozen_modules.json',dict(before=before,after=after,effective_before=active,effective_after=effective_after,**{'pass':True}))
    frozen_check()
    if split=='diagnostic64':visuals(cohort)
    bundle('M3_'+split)

def visuals(cohort):
    lookup={r['id']:r for r in cohort};root=OUT/'M3/diagnostic64/seed42'
    for group,ids in read(OUT/'splits/visual_sets.json')['sets'].items():
        panels=[];dest=OUT/'visual_audit/M3'/group;dest.mkdir(parents=True,exist_ok=True)
        for sid in ids:
            row=lookup[sid];ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch']);refs=[ref,rotate(ref,90),rotate(ref,180)]
            with np.load(IF/'reproduction/seed42/fields'/(sid+'.npz')) as z:field=z['orientation'].copy()
            panel=Image.new('RGB',(896,592),'white');draw=ImageDraw.Draw(panel);draw.text((3,2),sid,fill='black')
            for i,arm in enumerate(['R0','R90','R180']):
                image=Image.open(root/sid/(arm+'.png')).copy()
                with np.load(root/sid/(arm+'_orientation.npz')) as z:g=z['geometry'].copy();support=z['support_mask'].copy()
                old=Image.open(IF/'stage_survival/seed42'/sid/'S9'/(arm+'.png')).copy()
                items=[('Reference '+arm,refs[i]),('RF2',overlay(sketch,field[i])),('F_target channels0:3',Image.open(root/sid/(arm+'_feature.png'))),
                    ('M3 final',image),('old Full',old),('E26 readout',overlay(image,g[:2])),('Readout support',Image.fromarray(np.uint8(support)*255))]
                for col,(label,img) in enumerate(items):
                    y=22+i*190;draw.text((col*128+2,y),label,fill='black');img=img.convert('RGB');img.thumbnail((124,166));panel.paste(img,(col*128+2,y+18))
            panel.save(dest/(sid+'.png'));panels.append(panel)
        for i in range(0,16,4):
            page=Image.new('RGB',(896,2368),'white')
            for j,p in enumerate(panels[i:i+4]):page.paste(p,(0,j*592))
            page.save(dest/('page%d.png'%(i//4)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['diagnostic64','confirmation64'])
    parser.add_argument('--initialization',choices=['pure-noise','old-Full'],default='pure-noise');args=parser.parse_args();run(args.split,args.initialization)
