"""M1/M2 confirmation64的原E5固定mini；与同身份旧Full配对。"""
import argparse
import cv2,numpy as np,torch
from PIL import Image
from models.e33tm_generation_wrapper import load_e5,generate
from data.e32_target_pseudogt import image_at,masks
from data.e32_field_dataset import cache_path
from data.e33rf_real_rotation_dataset import rotate
from tools.e33rf_common import E32
from tools.e33tm_protocol import E5
from tools.e33tm_metrics import Evaluator
from tools.e33tmif_metrics import pair,arm_geometry,bootstrap,summarize
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes
from tools.e33tmms_protocol import *
from tools.e33tmms_benchmark import folder,bundle

def image_summary(rows):
    result=summarize(rows)
    for key in ['texture_score','text_score','contour_f1']:
        result['statistics'][key]=bootstrap([r[key] for r in rows])
    return result

def image_gate(method,rows,baseline):
    new=image_summary(rows);old=image_summary(baseline)
    lookup={r['id']:r for r in baseline};assert set(lookup)=={r['id'] for r in rows}
    delta={k:bootstrap([r[k]-lookup[r['id']][k] for r in rows if r[k] is not None and lookup[r['id']][k] is not None])
        for k in ['r90_success','r180_success','texture_score','text_score','contour_f1']}
    s=new['statistics'];b=old['statistics']
    text_drop=-delta['text_score']['mean']/b['text_score']['mean']
    checks=dict(Image_R90=s['r90_success']['mean']>=.30,gain=delta['r90_success']['mean']>=.15,
        R180=s['r180_success']['mean']>=.60,Contour=delta['contour_f1']['mean']>=-.05,
        Text=text_drop<=.10,Texture=s['texture_score']['mean']>=b['texture_score']['mean']*.9,
        Text_no_significant_decline=delta['text_score']['ci95'][1]>=0,
        Contour_no_significant_decline=delta['contour_f1']['ci95'][1]>=0)
    passed=all(checks.values());gate=dict(checks=checks,**{'pass':passed},paired_delta=delta,
        relative_text_drop=text_drop,case_count=64,baseline='same confirmation64 identities, IF stage_survival seed42 S9 oldFull')
    path=OUT/method/'e5_mini';write(path/'summary.json',dict(new=new,oldFull=old));write(path/'gate.json',gate)
    updates={method+'_e5_pass':passed,'next_route':method+'_full_validation' if passed else ('M2' if method=='M1' else 'M3')}
    if passed:updates.update(selected_method=method,selected_space='RGB exemplar' if method=='M1' else 'learned RGB',
        selected_image_r90=s['r90_success']['mean'],selected_texture_sim=s['texture_score']['mean'],
        selected_text_delta=delta['text_score']['mean'],selected_contour_delta=delta['contour_f1']['mean'])
    decision(**updates)
    table=read(OUT/'result_table.json');table[method+'_e5_mini']=dict(new=new,oldFull=old,gate=gate);write(OUT/'result_table.json',table)
    print('[MS E5 mini Gate]',method,gate,flush=True)

@torch.inference_mode()
def run(method):
    cohort=prepare();assert read(OUT/'decision_summary.json')[method+'_confirmation_pass']
    ids=read(OUT/'splits/confirmation64.json');rows=[r for r in cohort if r['id'] in ids];assert len(rows)==64
    fingerprints=freeze_inputs(cohort,ids,'confirmation64')
    pipe,modules,size,ns=load_e5();assert size==(384,512)
    evaluator=Evaluator();modules['text_image_evaluator']=evaluator.model
    before=module_hashes(modules);active=effective_hashes(modules)
    destination=OUT/method/'e5_mini';destination.mkdir(parents=True,exist_ok=True)
    write(destination/'implementation.json',dict(git_commit=commit(),E5_sha256=sha(E5),effective_weights=active,
        original_E5_texture_path=True,RF2_frozen=True,size=size,seed=42,scheduler=type(pipe.scheduler).__name__,
        scheduler_config=dict(pipe.scheduler.config),**PROTOCOL['refinement']))
    manifest=read(IF/'artifact_manifest.json')['files'];new=[];baseline=[]
    for row in rows:
        sid=row['id'];dest=destination/'seed42'/sid;case_path=dest/'case.json'
        old_path=IF/'stage_survival/seed42'/sid/'S9/pair.json'
        assert sha(old_path)==manifest[str(old_path.relative_to(IF))]
        old=read(old_path);baseline.append(old)
        if case_path.exists():
            saved=read(case_path);assert all(sha(dest/n)==h for n,h in saved['files'].items());new.append(saved['metrics']);continue
        ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch'])
        mask=Image.fromarray(masks(sketch)[0].astype(np.uint8)*255)
        with np.load(cache_path(E32,sid)) as z:
            gt=z['supervision_geometry'].copy();support=(z['supervision_interior']>=.95)&(gt[3]>=.25)
        refs=[ref,rotate(ref,90),rotate(ref,180)];geos={};arms={};schedules={};carrier_hash={}
        dest.mkdir(parents=True,exist_ok=True)
        for i,arm in enumerate(['R0','R90','R180']):
            carrier_path=folder(method,'confirmation64')/'seed42'/sid/'S2'/(arm+'.png')
            carrier=Image.open(carrier_path).convert('RGB');carrier_hash[arm]=sha(carrier_path)
            image,info=generate(pipe,size,ns,row['caption'],sketch,refs[i],mask,42,scaffold=carrier)
            assert info['refinement_start']['remaining_steps']==8
            image.save(dest/(arm+'.png'));scores,g=evaluator.evaluate(image,row['caption'],refs[i],mask,sketch)
            scores.update(arm_geometry(g,support,gt[3]));geos[arm]=g;arms[arm]=scores;schedules[arm]=info
            np.savez_compressed(dest/(arm+'_orientation.npz'),geometry=g,orientation_angle=np.degrees(np.arctan2(g[1],g[0]))/2%180,
                orientation_confidence=g[3],support_mask=support&(g[3]>=.25))
            write(dest/(arm+'.json'),dict(metrics=scores,generation=info,carrier_sha256=carrier_hash[arm]))
        assert len({r['noise_sha256'] for r in schedules.values()})==1
        record=dict(id=sid,stage='S9',arms=arms,**pair(geos,support,gt[3]))
        for key in ['texture_score','text_score','contour_f1']:
            values=[r[key] for r in arms.values() if r[key] is not None];record[key]=float(np.mean(values)) if values else None
        write(case_path,dict(id=sid,metrics=record,schedules=schedules,carrier_sha256=carrier_hash,
            oldFull_pair_sha256=sha(old_path),files={str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='case.json'}))
        new.append(record);print('[MS E5 mini]',method,sid,flush=True)
    write(destination/'rows.json',new);write(destination/'oldFull_rows.json',baseline)
    image_gate(method,new,baseline)
    after=module_hashes(modules);effective_after=effective_hashes(modules)
    assert before==after and active==effective_after and {p:sha(p) for p in fingerprints}==fingerprints
    write(destination/'frozen_modules.json',dict(before=before,after=after,effective_before=active,effective_after=effective_after,**{'pass':True}))
    frozen_check();bundle(method+'_E5_mini')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['M1','M2']);run(parser.parse_args().method)
