"""固定干预列表；每个case四arm噪声相同，可按逐图记录续跑。"""
import argparse
import gc
import cv2
import numpy as np
import torch
from PIL import Image
from data.e32_field_dataset import cache_path
from data.e32_target_pseudogt import image_at, masks
from data.e33rf_real_rotation_dataset import rotate, reference_group
from data.e33rc_real_pair_dataset import cached_real
from tools.e33rf_common import build, E32
from models.e33tm_generation_wrapper import load_e5, spatial_carrier, generate, TriModalField
from tools.e33tm_metrics import Evaluator, pair_metrics, summarize, compare, mean_available
from tools.e33tm_protocol import *
from tools.e33tm_caption_audit import prepare
from tools.e33tm_interventions import prepare as prepare_interventions
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes, validate_proof

def folder(setting,seed=42):
    if setting=='baseline': return OUT/'baseline_e5'
    if setting=='full': return OUT/('rf2_seed%d'%seed)
    if setting=='robust_baseline': return OUT/'robustness/baseline_e5'
    if setting=='robust_full': return OUT/'robustness'/('rf2_seed%d'%seed)
    if setting=='conflict_baseline': return OUT/'conflict_test/baseline_e5'
    if setting.startswith('conflict_'): return OUT/'conflict_test'/setting.replace('conflict_','')
    if setting=='near': return OUT/'text_compatible_near'
    return OUT/'ablations'/setting

class Experiment:
    def __init__(self,seed,run_tag=None):
        self.implementation_commit = commit()
        assert read(OUT/'field_precheck/summary.json')['pass']
        self.cohorts = prepare()
        interventions = prepare_interventions(self.cohorts)
        self.cohorts.update({k:interventions[k] for k in ('donors','donor_rows','text_compatible_near')})
        self.lookup = dict(self.cohorts['donor_rows'])
        self.lookup.update({r['id']:r for r in self.cohorts['dev']})
        self.seed = seed
        self.audit = OUT/'shards'/run_tag if run_tag else None
        self.pipe,self.modules,self.size,self.ns = load_e5()
        self.eval = Evaluator()
        self.modules['text_image_evaluator'] = self.eval.model
        self.before = module_hashes(self.modules)
        self.effective_before = effective_hashes(self.modules)
        write((self.audit or OUT)/'generation_implementation.json',dict(git_commit=self.implementation_commit,
            source_files={p:sha(p) for p in ('tools/e33tm_generate.py','tools/e33tm_metrics.py',
                'models/e33tm_generation_wrapper.py','tools/e33tm_protocol.py','tools/e33tm_interventions.py')},
            size=self.size,texture_num_tokens=self.pipe.effective_texture_num_tokens,
            evaluator_config_sha256=sha('models/clip/models/image_encoder/config.json'),
            evaluator_weights_sha256=sha('models/clip/pytorch_model.bin'),
            original_e5_sha256=sha(E5),modules_before=self.before,orientation_readable_threshold=.25,
            empty_texture_interior='null for auxiliary Texture Sim; report available-input N; retain identity in every primary Gate',
            image_case_success='nonempty readable paired support and axial response error<=15deg; no case exclusion'))
        self.model = None

    def ensure_model(self):
        if self.model is None:
            model,_ = build(self.seed,checkpoint=RF/('seed%d'%self.seed)/'RF2/checkpoint_final.pt')
            self.model = TriModalField(model).eval()

    @torch.inference_mode()
    def orientation(self,row,setting,donor):
        if setting in ('full','robust_full','no_text','shuffled_text') or setting.startswith('conflict_'):
            path = OUT/'field_precheck'/('seed%d'%self.seed)/'real/fields'/(row['id']+'.npz')
            with np.load(path) as z: return z['orientation'][:3].copy()
        self.ensure_model()
        source = donor if setting in ('wrong_texture','near') else row
        case = reference_group(source)
        if setting=='wrong_sketch': structure = cached_real(donor)['structure']
        elif setting=='no_sketch': structure = np.zeros_like(case['structure'].numpy())
        else: structure = cached_real(row)['structure']
        ref = case['reference'].cuda()
        st = torch.from_numpy(structure)[None].cuda().expand(3,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.model(ref,st)['orientation'].float().cpu().numpy()

    @torch.inference_mode()
    def run(self,records,setting,diffusion_seeds=(42,)):
        dest = folder(setting,self.seed)
        donor_key = {'shuffled_text':'shuffled_text','wrong_sketch':'wrong_sketch','wrong_texture':'wrong_texture'}
        baseline = setting in ('baseline','robust_baseline','conflict_baseline')
        for index,row in enumerate(records,1):
            donor = None
            if setting in donor_key:
                if self.cohorts['donors'][row['id']] is None: continue
                donor = self.lookup[self.cohorts['donors'][row['id']][donor_key[setting]]]
            if setting=='near':
                sid = self.cohorts['text_compatible_near'][row['id']]['donor']
                if not sid: continue
                donor = self.lookup[sid]
            original_sketch = image_at(DATASET/row['sketch'])
            original_mask,_ = masks(original_sketch)
            eval_mask = Image.fromarray(original_mask.astype(np.uint8)*255).resize(self.size,Image.Resampling.NEAREST)
            eval_sketch = original_sketch.resize(self.size,Image.Resampling.BICUBIC)
            sketch = image_at(DATASET/donor['sketch']) if setting=='wrong_sketch' else original_sketch
            if setting=='no_sketch':
                sketch = Image.new('RGB',(384,512),'white')
                input_mask = Image.new('L',self.size,255)
            else:
                input_mask = Image.fromarray(masks(sketch)[0].astype(np.uint8)*255).resize(self.size,Image.Resampling.NEAREST)
            sketch = sketch.resize(self.size,Image.Resampling.BICUBIC)
            text = row['caption']
            if setting=='no_text': text = ''
            if setting=='shuffled_text': text = donor['caption']
            if setting=='conflict_C1': text = neutral(text)
            if setting=='conflict_C2': text = counterfactual(text)
            texture_source = donor if setting in ('wrong_texture','near') else row
            rgb = image_at(DATASET/texture_source['reference'])
            references = [rgb,rotate(rgb,90),rotate(rgb,180)]
            if not baseline and setting!='no_texture':
                orientations = self.orientation(row,setting,donor)
            else: orientations = None
            with np.load(cache_path(E32,row['id'])) as z:
                gt = z['supervision_geometry'].copy()
                support = (z['supervision_interior']>=.95)&(gt[3]>=.25)
            arms = ARMS if setting in ('full','robust_full') else ARMS[:3]
            for diffusion_seed in diffusion_seeds:
                case_dest = dest/row['id']/('d%d'%diffusion_seed)
                scores,geometries,metadata = {},{},{}
                for ai,arm in enumerate(arms):
                    path,record_path = case_dest/(arm+'.png'),case_dest/(arm+'.json')
                    if path.exists() and record_path.exists():
                        record = read(record_path)
                        scores[arm] = record['metrics']
                        geometries[arm] = np.load(case_dest/(arm+'_orientation.npz'))['geometry']
                        metadata[arm] = record
                        continue
                    case_dest.mkdir(parents=True,exist_ok=True)
                    texture_on = arm!='Rzero' and setting!='no_texture'
                    ref = references[min(ai,2)] if texture_on else None
                    scaffold = None
                    if not baseline:
                        if texture_on:
                            scaffold,coordinate = spatial_carrier(ref,orientations[ai],input_mask)
                            np.savez_compressed(case_dest/(arm+'_field.npz'),orientation=orientations[ai],
                                                uv=coordinate.uv,confidence=coordinate.confidence)
                        else:
                            # 只留由sketch提供的白底轮廓，不保留纹理像素或RF方向场。
                            gray = np.where((np.asarray(input_mask)>0)[...,None],220,255)
                            scaffold = Image.fromarray(np.repeat(gray,3,axis=-1).astype(np.uint8))
                        scaffold.save(case_dest/(arm+'_scaffold.png'))
                    image,info = generate(self.pipe,self.size,self.ns,text,sketch,ref,input_mask,diffusion_seed,
                                        scaffold,setting=='no_sketch')
                    image.save(path)
                    metric_reference = references[min(ai,2)] if setting not in ('wrong_texture','near') else image_at(DATASET/row['reference'])
                    conflict=setting in ('conflict_C0','conflict_C1','conflict_C2')
                    metrics,geometry = self.eval.evaluate(image,row['caption'],metric_reference,eval_mask,eval_sketch,
                                                         prompt_caption=text if conflict else None)
                    record = dict(id=row['id'],setting=setting,rf_seed=self.seed,diffusion_seed=diffusion_seed,
                        arm=arm,caption=row['caption'],donor=None if donor is None else donor['id'],
                        path=str(path.relative_to(OUT)),metrics=metrics,output_sha256=sha(path),git_commit=self.implementation_commit,
                        source_sketch=str(DATASET/(donor['sketch'] if setting=='wrong_sketch' else row['sketch'])),
                        source_reference=str(DATASET/texture_source['reference']),**info)
                    np.savez_compressed(case_dest/(arm+'_orientation.npz'),geometry=geometry)
                    write(record_path,record)
                    scores[arm],geometries[arm],metadata[arm] = metrics,geometry,record
                assert len({r['noise_sha256'] for r in metadata.values()})==1, 'paired noise differs'
                row_metrics = pair_metrics(geometries,support,gt[3])
                if setting in ('conflict_C0','conflict_C1','conflict_C2'):
                    # C2的R0文本已变成horizontal，不能用它检验R90上vertical→horizontal的冲突解除。
                    anchor=folder('conflict_C0')/row['id']/('d%d'%diffusion_seed)/'R0_orientation.npz'
                    with np.load(anchor) as z: common=z['geometry'].copy()
                    anchored=pair_metrics(dict(geometries,R0=common),support,gt[3])
                    for key in ('success','error','coverage'):
                        row_metrics['common_anchor_r90_'+key]=anchored['r90_'+key]
                    if setting=='conflict_C0': assert anchored['r90_success']==row_metrics['r90_success']
                for key in scores['R0']:
                    row_metrics[key] = mean_available([scores[a][key] for a in ARMS[:3]])
                # 语义旋转稳定性同样以每case记录，辅助检查R90/R180相对R0。
                row_metrics['rotation_text_drop'] = max(0.,max((scores['R0']['text_score']-scores[a]['text_score'])/
                    max(scores['R0']['text_score'],1e-8) for a in ('R90','R180')))
                valid = support&(geometries['R0'][3]>=.25)
                row_metrics['matched_target_error'] = float(np.average(axial_local(geometries['R0'][:2],gt[:2])[valid],
                                                            weights=gt[3][valid])) if valid.any() else None
                result = dict(id=row['id'],setting=setting,rf_seed=self.seed,diffusion_seed=diffusion_seed,
                              arm_metrics=scores,**row_metrics)
                write(case_dest/'pair.json',result)
            print('[E33TM generate]',setting,self.seed,index,'/',len(records),row['id'],flush=True)
        rows = self.rows(records,setting,diffusion_seeds)
        audit_dest = self.audit/setting if self.audit else dest
        write(audit_dest/'rows.json',rows)
        write(audit_dest/'summary.json',summarize(rows))
        assert module_hashes(self.modules)==self.before
        effective_after = effective_hashes(self.modules)
        assert effective_after == self.effective_before
        write(audit_dest/'frozen_modules.json',dict(before=self.before,after=module_hashes(self.modules),
            effective_before=self.effective_before,effective_after=effective_after,**{'pass':True}))
        freeze_check()
        return rows

    def rows(self,records,setting,diffusion_seeds=(42,)):
        return collect_rows(self.cohorts,records,setting,self.seed,diffusion_seeds)

def collect_rows(cohort,records,setting,rf_seed,diffusion_seeds):
    result=[]
    for row in records:
        if setting in ('shuffled_text','wrong_sketch','wrong_texture') and cohort['donors'][row['id']] is None: continue
        if setting=='near' and not cohort['text_compatible_near'][row['id']]['available']: continue
        for seed in diffusion_seeds:
            result.append(read(folder(setting,rf_seed)/row['id']/('d%d'%seed)/'pair.json'))
    return result

def stage_tasks(cohort,work,index=0,count=1):
    records=cohort['primary'][index::count]
    if work=='remaining': return [(records,'full',(42,))]
    if work=='robustness':
        records=[r for r in cohort['primary'] if r['id'] in cohort['robust64']][index::count]
        return [(records,setting,(42,43,44,45)) for setting in ('robust_baseline','robust_full')]
    if work=='ablations':
        return [(records,setting,(42,)) for setting in
                ('no_text','shuffled_text','no_sketch','wrong_sketch','no_texture','wrong_texture')]
    if work=='diagnostics':
        conflict=cohort['conflict'][index::count]
        return [(records,'near',(42,))]+[(conflict,setting,(42,)) for setting in
               ('conflict_baseline','conflict_C0','conflict_C1','conflict_C2')]
    raise ValueError(work)

def shard_tag(work,seed,index,count):
    return '%s_s%d_%d_of_%d'%(work,seed,index,count)

def aggregate(cohort,work,seed,count):
    # 只读逐case结果；不重新加载E5，也不把分片当作独立统计单位。
    interventions=prepare_interventions(cohort)
    cohort.update({k:interventions[k] for k in ('donors','donor_rows','text_compatible_near')})
    for records,setting,diffusion_seeds in stage_tasks(cohort,work):
        proofs=[]; effective=[]
        for index in range(count):
            audit=OUT/'shards'/shard_tag(work,seed,index,count)
            assert read(audit/'manifest.json')['cohort_sha256']==sha(OUT/'manifests/cohorts.json')
            proof=read(audit/setting/'frozen_modules.json')
            assert proof['pass'] and proof['before']==proof['after']
            proofs.append(proof)
            effective.append(validate_proof(proof,audit/'generation_implementation.json'))
        assert all(p==effective[0] for p in effective)
        rows=collect_rows(cohort,records,setting,seed,diffusion_seeds)
        dest=folder(setting,seed)
        write(dest/'rows.json',rows);write(dest/'summary.json',summarize(rows))
        write(dest/'frozen_modules.json',dict(before=proofs[0]['before'],after=proofs[0]['after'],
            effective_before=effective[0],effective_after=effective[0],full_shard_proofs=proofs,
            weight_audit_sha256=sha(OUT/'weight_audit.json'),verified_shards=count,**{'pass':True}))
    freeze_check()
    if work=='remaining': gate(seed,cohort['primary'])
    print('[E33TM aggregate]',work,seed,'complete',flush=True)

def axial_local(a,b):
    from tools.e33r_evaluate import axial
    return axial(a,b)

def gate(seed,records):
    full = [read(folder('full',seed)/r['id']/'d42/pair.json') for r in records]
    base = [read(folder('baseline')/r['id']/'d42/pair.json') for r in records]
    result = compare(full,base)
    write(folder('full',seed)/'comparison.json',result)
    d = dict(**{'seed%d_image_r90'%seed:float(np.mean([r['r90_success'] for r in full])),
                'seed%d_image_r180'%seed:float(np.mean([r['r180_success'] for r in full]))})
    if seed==42:
        d.update(baseline_e5_image_r90=float(np.mean([r['r90_success'] for r in base])),
                 r90_gain_vs_e5=result['r90_success']['mean'],contour_f1_delta=result['contour_f1']['mean'],
                 text_alignment_delta=result['text_score']['mean'],texture_similarity_delta=result['texture_score']['mean'])
        field = read(OUT/'field_precheck/seed42/summary.json')['real_r90']
        if result['contour_f1']['mean']<-.05:
            d.update(next_route='structure_safe_injection_revision',hard_stop=True)
        elif result['r90_success']['mean']<=.05+1e-12 and field>=.9:
            d.update(next_route='field_to_generation_interface_bottleneck',hard_stop=True)
        else: d.update(next_route='seed43_seed44',hard_stop=False)
        d['seed42_hard_stop_checks'] = dict(field_r90=field,structure_drop=-result['contour_f1']['mean'],
                                           r90_gain=result['r90_success']['mean'])
    elif result['contour_f1']['mean']<-.05:
        d.update(next_route='structure_safe_injection_revision',hard_stop=True,
                 hard_stop_rf_seed=seed,hard_stop_contour_drop=-result['contour_f1']['mean'])
    decision(**d)
    print('[E33TM image Gate]',seed,result,flush=True)
    return not read(OUT/'decision_summary.json').get('hard_stop',False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','seed42','seed42_shard','remaining','robustness','ablations','diagnostics','worker','aggregate'),required=True)
    parser.add_argument('--work',choices=('remaining','robustness','ablations','diagnostics'))
    parser.add_argument('--rf-seed',type=int,choices=SEEDS,default=42)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=4)
    args = parser.parse_args()
    cohort = prepare()
    if read(OUT/'decision_summary.json').get('hard_stop'):
        print('[E33TM stopped]',read(OUT/'decision_summary.json')['next_route']);return
    if args.stage=='smoke':
        experiment = Experiment(42)
        records = [r for r in cohort['primary'] if r['id'] in cohort['smoke16']]
        experiment.run(records,'baseline');experiment.run(records,'full')
        write(OUT/'smoke_audit/numeric_check.json',dict(cases=len(records),all_arms_complete=True,
            same_noise_verified=True,frozen_modules_verified=True,visual_review_required=True))
        decision(next_route='smoke_visual_review')
        from tools.e33tm_visual_audit import smoke
        smoke(cohort)
        return
    assert read(OUT/'smoke_audit/visual_review.json')['pass'], 'review fixed16 panels before full dev'
    if args.stage in ('worker','aggregate'):
        assert args.work and 0<=args.shard_index<args.shard_count
        if args.stage=='aggregate':
            aggregate(cohort,args.work,args.rf_seed,args.shard_count)
            return
        tag=shard_tag(args.work,args.rf_seed,args.shard_index,args.shard_count)
        tasks=stage_tasks(cohort,args.work,args.shard_index,args.shard_count)
        write(OUT/'shards'/tag/'manifest.json',dict(cohort_sha256=sha(OUT/'manifests/cohorts.json'),
            work=args.work,rf_seed=args.rf_seed,shard_index=args.shard_index,shard_count=args.shard_count,
            settings={setting:dict(ids=[r['id'] for r in records],diffusion_seeds=diffusion_seeds)
                      for records,setting,diffusion_seeds in tasks}))
        experiment=Experiment(args.rf_seed,tag)
        for records,setting,diffusion_seeds in tasks: experiment.run(records,setting,diffusion_seeds)
    elif args.stage=='seed42_shard':
        assert 0<=args.shard_index<args.shard_count
        records=cohort['primary'][args.shard_index::args.shard_count]
        tag='seed42_%d_of_%d'%(args.shard_index,args.shard_count)
        write(OUT/'shards'/tag/'manifest.json',dict(ids=[r['id'] for r in records],
            cohort_sha256=sha(OUT/'manifests/cohorts.json'),shard_index=args.shard_index,
            shard_count=args.shard_count,diffusion_seed=42,rf_seed=42))
        experiment=Experiment(42,tag)
        experiment.run(records,'baseline');experiment.run(records,'full')
    elif args.stage=='seed42':
        experiment = Experiment(42)
        experiment.run(cohort['primary'],'baseline');experiment.run(cohort['primary'],'full')
        gate(42,cohort['primary'])
    elif args.stage=='remaining':
        for seed in (43,44):
            experiment = Experiment(seed);experiment.run(cohort['primary'],'full')
            proceed = gate(seed,cohort['primary'])
            del experiment;gc.collect();torch.cuda.empty_cache()
            if not proceed: break
    elif args.stage=='robustness':
        records = [r for r in cohort['primary'] if r['id'] in cohort['robust64']]
        experiment = Experiment(42)
        experiment.run(records,'robust_baseline',(42,43,44,45))
        experiment.run(records,'robust_full',(42,43,44,45))
    elif args.stage=='ablations':
        experiment = Experiment(42)
        for setting in ('no_text','shuffled_text','no_sketch','wrong_sketch','no_texture','wrong_texture'):
            experiment.run(cohort['primary'],setting)
    elif args.stage=='diagnostics':
        experiment = Experiment(42)
        experiment.run(cohort['primary'],'near')
        if cohort['conflict']:
            experiment.run(cohort['conflict'],'conflict_baseline')
            for setting in ('conflict_C0','conflict_C1','conflict_C2'):
                experiment.run(cohort['conflict'],setting)
        else: write(OUT/'conflict_test/not_applicable.json',dict(reason='no orientation-specific dev captions'))

if __name__=='__main__': main()
