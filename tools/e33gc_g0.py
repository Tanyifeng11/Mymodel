"""同噪声、同50步E5复现；no-op/replay exact，固定旧donor的reference specificity对照。"""
import argparse
import cv2,numpy as np,torch
from PIL import Image
from data.e32_target_pseudogt import image_at,masks
from data.e32_field_dataset import cache_path
from data.e33rf_real_rotation_dataset import rotate
from data.e33gc_renderer import support
from models.e33tm_generation_wrapper import load_e5,generate
from models.e33gc_adapter import CausalResidual,TextureInjection
from tools.e33gc_sampling import prefix,final8,tensor_sha
from tools.e33gc_protocol import *
from tools.e33gc_smoke import png
from tools.e33tm_metrics import Evaluator
from tools.e33tmif_metrics import pair,arm_geometry,summarize,bootstrap
from tools.e33tmoc_appearance_eval import patches,measure,frozen_lpips,model_hash
from tools.e33rf_common import E32
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes

def record(sid,arms,geometries,valid,weight):
    row=dict(id=sid,arms=arms,**pair(geometries,valid,weight))
    for k in ['text_score','contour_f1','sketch_similarity','texture_score','Lab_color_distance','mean_RGB_distance','color_histogram_similarity','patch_lpips']:
        values=[a[k] for a in arms.values() if a[k] is not None];row[k]=float(np.mean(values)) if values else None
    return row

def run(shards):
    assert read(OUT/'protocol/frozen_config.json')==CONFIG
    torch.set_num_threads(2);cv2.setNumThreads(1)
    import os
    shard=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    cohort=read(TM/'manifests/cohorts.json')['primary'];ids=read(OUT/'splits/diagnostic64.json')
    cohort=[r for r in cohort if r['id'] in ids];assert len(cohort)==64
    selected=[r for i,r in enumerate(cohort) if i%shards==shard]
    donors=read(TM/'manifests/intervention_data.json')
    pipe,modules,size,ns=load_e5();assert size==(384,512) and type(pipe.scheduler).__name__=='DDIMScheduler'
    evaluator=Evaluator();modules['evaluator']=evaluator.model
    before=module_hashes(modules);effective=effective_hashes(modules)
    lpips,init=frozen_lpips();lpips_sha=model_hash(lpips)
    adapter=CausalResidual().cuda().eval().requires_grad_(False);injection=TextureInjection(pipe,adapter)
    if_manifest=read(IF/'artifact_manifest.json')['files']
    rows=[]
    for row in selected:
        sid=row['id'];dest=OUT/'G0_reproduction'/sid;case=dest/'case.json'
        if case.exists():
            saved=read(case);assert all(sha(dest/n)==h for n,h in saved['files'].items());rows.append(saved['B0']);continue
        ref=image_at(DATASET/row['reference']);sketch=image_at(DATASET/row['sketch']);mask,inner,falloff=support(sketch)
        mask_image=Image.fromarray(mask*255);pipeline_mask=torch.from_numpy(mask.astype(np.float32))[None,None].cuda()
        field_path=IF/'reproduction/seed42/fields'/(sid+'.npz');assert sha(field_path)==if_manifest[str(field_path.relative_to(IF))]
        with np.load(field_path) as z:field=np.concatenate([z['orientation'][:3],z['confidence'][:3]],1)
        geometry=torch.cat([torch.from_numpy(field).cuda(),torch.from_numpy(np.broadcast_to(cv2.resize(mask,(48,64),interpolation=cv2.INTER_NEAREST),(3,1,64,48)).copy()).cuda()],1)
        with np.load(cache_path(E32,sid)) as z:gt=z['supervision_geometry'].copy();valid=(z['supervision_interior']>=.95)&(gt[3]>=.25)
        refs=[ref,rotate(ref,90),rotate(ref,180)];dest.mkdir(parents=True,exist_ok=True)
        target_boxes=patches(ref,inner,sid,'shared','target');geometries={};arms={};noops=[];inputs={}
        for i,arm in enumerate(['R0','R90','R180']):
            injection.geometry=geometry[i:i+1];injection.falloff=torch.from_numpy(falloff)[None,None].cuda()
            cached,image=prefix(pipe,ns,row['caption'],sketch,refs[i],mask_image,injection)
            torch.save(cached,dest/(arm+'_prefix.pt'));image.save(dest/(arm+'_B0.png'))
            with torch.no_grad():rgb,_=final8(pipe,ns,row['caption'],sketch,refs[i],pipeline_mask,cached,injection,checkpointed=False)
            noop=png(rgb);noop.save(dest/(arm+'_B1.png'));exact=np.array_equal(np.asarray(image),np.asarray(noop));assert exact
            scores,g=evaluator.evaluate(image,row['caption'],refs[i],mask_image,sketch);scores.update(arm_geometry(g,valid,gt[3]))
            from garment_mask_utils import estimate_cloth_foreground_mask
            sm=np.asarray(estimate_cloth_foreground_mask(refs[i],384,512)[0])>127
            app=measure(image,refs[i],inner,sm,target_boxes,patches(refs[i],sm,sid,arm,'source'),lpips)
            scores.update({k:app[k] for k in ['Lab_color_distance','mean_RGB_distance','color_histogram_similarity','patch_lpips']})
            arms[arm]=scores;geometries[arm]=g
            np.savez_compressed(dest/(arm+'_orientation.npz'),geometry=g,support=valid&(g[3]>=.25))
            noops.append(dict(arm=arm,exact=exact,noise_sha256=cached['generation']['noise_sha256'],
                initial_latent_sha256=cached['generation']['initial_latent_sha256'],timesteps=cached['timesteps'].tolist(),
                conditional_embeddings_sha256=cached['conditional'],unconditional_embeddings_sha256=cached['unconditional'],
                reference_sha256=tensor_sha(torch.from_numpy(np.asarray(refs[i]).copy())),
                prefix_latent_sha256=tensor_sha(cached['latent']),scheduler_config=cached['scheduler_config']))
        assert len({r['noise_sha256'] for r in noops})==1
        baseline=record(sid,arms,geometries,valid,gt[3]);contrast={}
        source={'wrong':donors['donors'].get(sid,{}),'near':donors['text_compatible_near'].get(sid,{})}
        donor_ids={'wrong':source['wrong']['wrong_texture'] if source['wrong'] else None,'near':source['near'].get('donor')}
        injection.enabled=False
        for setting,donor_id in donor_ids.items():
            if not donor_id:continue
            donor=donors['donor_rows'][donor_id];donor_ref=image_at(DATASET/donor['reference']);donor_refs=[donor_ref,rotate(donor_ref,90),rotate(donor_ref,180)]
            results={};geos={}
            for i,arm in enumerate(['R0','R90','R180']):
                image,info=generate(pipe,size,ns,row['caption'],sketch,donor_refs[i],mask_image,42)
                assert info['noise_sha256']==noops[i]['noise_sha256']
                image.save(dest/(arm+'_B2_'+setting+'.png'))
                score,g=evaluator.evaluate(image,row['caption'],refs[i],mask_image,sketch);score.update(arm_geometry(g,valid,gt[3]))
                # 所有wrong/near图的外观分数仍相对于原matched reference，非相对于错误donor。
                sm=np.asarray(estimate_cloth_foreground_mask(refs[i],384,512)[0])>127
                app=measure(image,refs[i],inner,sm,target_boxes,patches(refs[i],sm,sid,arm,'source'),lpips)
                score.update({k:app[k] for k in ['Lab_color_distance','mean_RGB_distance','color_histogram_similarity','patch_lpips']});results[arm]=score;geos[arm]=g
            contrast[setting]=dict(donor=donor_id,donor_reference_sha256=sha(DATASET/donor['reference']),metrics=record(sid,results,geos,valid,gt[3]))
        inputs.update({str(DATASET/row[k]):sha(DATASET/row[k]) for k in ['reference','sketch']})
        saved=dict(id=sid,B0=baseline,B2=contrast,noop=noops,inputs=inputs,caption=row['caption'],
            negative_prompt=CONFIG['negative_prompt'],cfg=7.,steps=50,start='pure_noise',mask_sha256=tensor_sha(torch.from_numpy(mask)),
            files={str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='case.json'})
        write(case,saved);rows.append(baseline);print('[GC G0]',sid,baseline['r90_success'],flush=True)
    assert before==module_hashes(modules) and effective==effective_hashes(modules) and model_hash(lpips)==lpips_sha
    injection.close();write(OUT/'G0_reproduction'/('shard%d.json'%shard),dict(ids=[r['id'] for r in selected],complete=True,
        code_commit=commit(),frozen_before=before,frozen_after=module_hashes(modules),effective_before=effective,
        effective_after=effective_hashes(modules),lpips_state_sha256=lpips_sha,lpips_initialization=init))
    frozen=read(OUT/'frozen_check.json')['before']
    assert all(sha(name)==digest for name,digest in frozen.items())

def aggregate():
    prepare()
    for i in range(4):assert read(OUT/'G0_reproduction'/('shard%d.json'%i))['complete']
    cases=[read(OUT/'G0_reproduction'/sid/'case.json') for sid in read(OUT/'splits/diagnostic64.json')]
    rows=[c['B0'] for c in cases];summary=summarize(rows);rate=summary['statistics']['r90_success']['mean']
    exact=all(a['exact'] for c in cases for a in c['noop']);passed=abs(rate-0)<=1/64 and exact
    specificity={}
    for setting in ['wrong','near']:
        eligible=[c for c in cases if setting in c['B2']]
        specificity[setting]=dict(n=len(eligible),paired_matched_advantage={k:bootstrap([c['B0'][k]-c['B2'][setting]['metrics'][k]
            for c in eligible if c['B0'][k] is not None and c['B2'][setting]['metrics'][k] is not None]) for k in
            ['texture_score','text_score','contour_f1','sketch_similarity','Lab_color_distance','patch_lpips']},
            note='positive distance advantage means matched has greater distance, therefore worse; validN only')
    write(OUT/'G0_reproduction/rows.json',rows);write(OUT/'G0_reproduction/summary.json',summary)
    write(OUT/'G0_reproduction/specificity.json',specificity)
    write(OUT/'G0_reproduction/gate.json',dict(reproduction_pass=passed,baseline_R90=rate,historical_R90=0.,tolerance=1/64,
        adapter_noop_exact=exact,case_count=64,input_intervention_validity='pending independent input audit; this gate covers reproduction/no-op only',conditions='same pure-noise50,CFG7,sketch.6,texture1,noise42 and original captions'))
    decision(reproduction_pass=passed,adapter_noop_exact=exact,next_route='await_G1_G2_and_split_resolution' if passed else 'reproduction_or_protocol_invalid')
    frozen_check();bundle('G0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','aggregate']);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    run(a.shards) if a.action=='run' else aggregate()
