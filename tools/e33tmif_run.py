"""冻结模型的阶段转储与固定条件实验；逐case完整记录后才允许续跑。"""
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from models.e33tm_generation_wrapper import load_e5,spatial_carrier,generate,TriModalField
from models.e33tmif_stage_logger import refine,decode
from tools.e33tm_metrics import Evaluator
from tools.e33tmif_metrics import pair,arm_geometry,frequency_diagnostics
from tools.e33tmif_protocol import *
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes
from tools.e33rf_common import build,E32
from data.e33rf_real_rotation_dataset import reference_group,rotate
from data.e32_field_dataset import cache_path
from data.e32_target_pseudogt import image_at,masks

def field_path(seed): return OUT/'reproduction'/('seed%d'%seed)/'fields'
def case_path(group,seed,sid): return OUT/group/('seed%d'%seed)/sid

def complete(path):
    if not path.exists(): return False
    row=read(path)
    return all(p.exists() and sha(p)==digest for name,digest in row.get('files',{}).items()
               for p in [path.parent/name])

class Experiment:
    def __init__(self,seed,tag):
        self.seed=seed;self.tag=tag;self.commit=commit()
        self.cohort,self.splits=prepare()
        self.pipe,self.modules,self.size,self.ns=load_e5()
        self.eval=Evaluator();self.modules['text_image_evaluator']=self.eval.model
        self.model=None
        self.before=module_hashes(self.modules);self.active=effective_hashes(self.modules)
        self.audit=OUT/'jobs'/tag
        write(self.audit/'implementation.json',dict(git_commit=self.commit,
            source_files={str(p):sha(p) for p in sorted(Path('tools').glob('e33tmif_*.py'))+
                sorted(Path('models').glob('e33tmif_*.py'))},
            original_e5_sha256=sha(E5),effective_before=self.active,full_before=self.before,
            scheduler_class=type(self.pipe.scheduler).__name__,scheduler_config=dict(self.pipe.scheduler.config),size=self.size))
        assert type(self.pipe.scheduler).__name__=='DDIMScheduler'

    def finish(self):
        after=module_hashes(self.modules);active=effective_hashes(self.modules)
        assert after==self.before and active==self.active
        write(self.audit/'frozen_modules.json',dict(before=self.before,after=after,
            effective_before=self.active,effective_after=active,**{'pass':True}))
        freeze_check()

    @torch.inference_mode()
    def fields(self,records):
        if self.model is None:
            rf,_=build(self.seed,checkpoint=RF/('seed%d'%self.seed)/'RF2/checkpoint_final.pt')
            self.model=TriModalField(rf).eval().requires_grad_(False)
            self.modules['rf2']=self.model
            self.before=module_hashes(self.modules);self.active=effective_hashes(self.modules)
        for row in records:
            path=field_path(self.seed)/(row['id']+'.npz')
            if path.exists(): continue
            case=reference_group(row)
            refs=torch.cat([case['reference'],torch.zeros_like(case['reference'][:1])]).cuda()
            structure=case['structure'][None].cuda().expand(4,-1,-1,-1)
            with torch.autocast('cuda',dtype=torch.bfloat16): prediction=self.model(refs,structure)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,orientation=prediction['orientation'].float().cpu().numpy(),
                confidence=torch.sigmoid(prediction['confidence_logits']).float().cpu().numpy(),
                support=case['support'].numpy(),gt=case['gt'].numpy())
            print('[IF field]',self.seed,row['id'],flush=True)

    def inputs(self,row):
        original=image_at(DATASET/row['sketch'])
        mask=Image.fromarray(masks(original)[0].astype(np.uint8)*255).resize(self.size,Image.Resampling.NEAREST)
        sketch=original.resize(self.size,Image.Resampling.BICUBIC)
        rgb=image_at(DATASET/row['reference'])
        refs=dict(zip(ARMS,(rgb,rotate(rgb,90),rotate(rgb,180))))
        with np.load(cache_path(E32,row['id'])) as z:
            gt=z['supervision_geometry'].copy();support=(z['supervision_interior']>=.95)&(gt[3]>=.25)
        with np.load(field_path(self.seed)/(row['id']+'.npz')) as z:
            orientation=z['orientation'][:3].copy();confidence=z['confidence'][:3].copy()
        return sketch,mask,refs,gt,support,orientation,confidence

    def stage(self,row,stage,image,geometry,mask,sketch,ref,support,gt,dest,info=None,confidence=None):
        folder=dest/stage;folder.mkdir(parents=True,exist_ok=True)
        arm=self.arm
        if image is not None:
            image.save(folder/(arm+'.png'))
            scores,geometry=self.eval.evaluate(image,row['caption'],ref,mask,sketch)
            freq=frequency_diagnostics(image,mask)
            scores.update({k:v for k,v in freq.items() if k!='fft_radial'})
            write(folder/(arm+'_frequency.json'),freq)
        else: scores={}
        summary=arm_geometry(geometry,support,gt[3]);summary.update(scores)
        if confidence is not None:
            summary['rf_predicted_confidence_mean']=float(confidence[0][support].mean()) if support.any() else None
            summary['orientation_confidence']=summary['rf_predicted_confidence_mean']
        np.savez_compressed(folder/(arm+'_orientation.npz'),geometry=geometry,
            orientation_angle=(np.rad2deg(np.arctan2(geometry[1],geometry[0]))/2)%180,
            orientation_confidence=geometry[3],support_mask=support&(geometry[3]>=.25),
            rf_predicted_confidence=confidence if confidence is not None else np.array([]))
        write(folder/(arm+'.json'),dict(id=row['id'],stage=stage,arm=arm,metrics=summary,info=info))
        self.geometries.setdefault(stage,{})[arm]=geometry
        self.scores.setdefault(stage,{})[arm]=summary

    def finalize_case(self,row,dest,gt,support,schedules):
        stages={}
        for stage,geometries in self.geometries.items():
            assert set(geometries)==set(ARMS)
            record=dict(id=row['id'],stage=stage,arms=self.scores[stage],
                **pair(geometries,support,gt[3],field=stage=='S0'))
            for key in ('text_score','contour_f1','sketch_similarity','texture_score','high_frequency_energy','local_contrast'):
                values=[m.get(key) for m in self.scores[stage].values() if m.get(key) is not None]
                record[key]=float(np.mean(values)) if values else None
            stages[stage]=record
            write(dest/stage/'pair.json',record)
        assert len({s['noise_sha256'] for s in schedules.values()})==1
        assert len({tuple(s['actual_timesteps']) for s in schedules.values()})==1
        files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='case.json'}
        write(dest/'case.json',dict(id=row['id'],rf_seed=self.seed,git_commit=self.commit,
            schedules=schedules,stages=stages,files=files))

    @torch.inference_mode()
    def run_case(self,row,group='stage_survival',strength=.15,conditions=(True,True,True),
                 steps=False,static=True,purity=None,control=False):
        dest=case_path(group,self.seed,row['id'])
        if complete(dest/'case.json'): return
        sketch,mask,refs,gt,support,orientation,confidence=self.inputs(row)
        self.geometries={};self.scores={};schedules={}
        for ai,arm in enumerate(ARMS):
            self.arm=arm;ref=refs[arm]
            carrier,coordinate=spatial_carrier(ref,orientation[ai],mask)
            if purity: carrier=self.pure(carrier,ref,coordinate,orientation[ai],mask,purity)
            if static:
                geo=np.stack([orientation[ai,0],orientation[ai,1],np.zeros_like(gt[3]),(support.astype(float))])
                self.stage(row,'S0',None,geo,mask,sketch,ref,support,gt,dest,confidence=confidence[ai])
                mapped=cv2.remap(np.asarray(ref),coordinate.uv[...,0],coordinate.uv[...,1],
                                cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                self.stage(row,'S1',Image.fromarray(mapped),None,mask,sketch,ref,support,gt,dest)
                self.stage(row,'S2',carrier,None,mask,sketch,ref,support,gt,dest)
                np.savez_compressed(dest/'S2'/(arm+'_coordinates.npz'),uv=coordinate.uv,confidence=coordinate.confidence)
            def observe(stage,image,info):
                if stage in ('S4','S9') or steps:
                    self.stage(row,stage,image,None,mask,sketch,ref,support,gt,dest,info)
            tensor_path=dest/'tensors'/arm/'initial.npz'
            image,metadata=refine(self.pipe,self.size,self.ns,row['caption'],sketch,ref,mask,carrier,
                                 strength=strength,conditions=conditions,observer=observe,save_tensors=tensor_path,save_step_tensors=steps)
            schedules[arm]=metadata
            write(dest/(arm+'_scheduler.json'),metadata)
            for stage,key in [('S3','posterior_mean'),('S5','initial_latent')]:
                write(dest/stage/(arm+'.json'),dict(id=row['id'],stage=stage,arm=arm,
                    readable=None,support_fraction=None,orientation_angle=None,orientation_confidence=None,
                    latent=metadata[key],tensor_path=str(tensor_path.relative_to(dest)),
                    note='潜变量仅记录张量与scheduler摘要，不使用RGB方向评测；posterior_mean已乘VAE scaling_factor'))
            if control:
                direct,direct_info=generate(self.pipe,self.size,self.ns,row['caption'],sketch,ref,mask,42,carrier)
                assert np.array_equal(np.asarray(image),np.asarray(direct)), 'logger changed final image'
                assert metadata['noise_sha256']==direct_info['noise_sha256']
                write(dest/(arm+'_observer_check.json'),dict(pixels_equal=True,noise_equal=True))
        self.finalize_case(row,dest,gt,support,schedules)
        print('[IF case]',group,self.seed,row['id'],'steps',steps,flush=True)

    def pure(self,carrier,ref,coordinate,orientation,mask,kind):
        inside=np.asarray(mask)>0;rgb=np.asarray(carrier).copy()
        if kind=='P1':
            features=self.eval.extractor.extract(ref,perceptual=False)['geometry']
            confidence=cv2.resize(features[...,3],ref.size,interpolation=cv2.INTER_LINEAR)
            valid=(np.any(np.asarray(ref)<245,axis=-1)&(confidence>=.25)).astype(np.uint8)
            warped=cv2.remap(valid,coordinate.uv[...,0],coordinate.uv[...,1],cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
            # 回退边界沿用其自己的有效像素，而非取用内部的重映射掩码。
            fallback=np.asarray(Image.fromarray(valid*255).resize(mask.size,Image.Resampling.NEAREST))>0
            valid=np.where(coordinate.confidence>0,warped,fallback)&inside
            rgb[inside&~valid]=220
        elif kind=='P2':
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
            normalized=np.clip(128+2*(gray-cv2.GaussianBlur(gray,(0,0),8)),32,224).astype(np.uint8)
            rgb=np.repeat(normalized[...,None],3,-1)
        elif kind=='P3':
            dense=cv2.resize(np.moveaxis(orientation,0,-1),mask.size,interpolation=cv2.INTER_LINEAR)
            theta=np.arctan2(dense[...,1],dense[...,0])/2
            y,x=np.indices(inside.shape);x=x-(mask.width-1)/2;y=y-(mask.height-1)/2
            gray=(128+80*np.cos(2*np.pi*(-np.sin(theta)*x+np.cos(theta)*y)/16)).astype(np.uint8)
            rgb=np.repeat(gray[...,None],3,-1)
        elif kind!='P0': raise ValueError(kind)
        rgb[~inside]=255
        return Image.fromarray(rgb)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',choices=('smoke','fields','reproduce','conditions','strengths','confirmation','purity','seed_confirmation'),required=True)
    parser.add_argument('--rf-seed',type=int,default=42,choices=SEEDS)
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    assert 0<=args.shard_index<args.shard_count
    tag='%s_s%d_%d_of_%d'%(args.work,args.rf_seed,args.shard_index,args.shard_count)
    e=Experiment(args.rf_seed,tag)
    primary=e.cohort['primary'];ids=e.splits['diagnostic64']
    if args.work=='smoke':
        assert args.rf_seed==42
        rows=[r for r in primary if r['id'] in e.splits['field_success_fixed_hash16']]
        e.fields(rows)
        for row in rows[args.shard_index::args.shard_count]: e.run_case(row,'smoke',steps=True,control=True)
    else:
        assert read(OUT/'visual_audit/smoke_review.json')['pass'], '必须先人工审阅固定16例smoke'
        if args.work=='fields':
            e.fields(e.cohort['dev'][args.shard_index::args.shard_count])
        elif args.work=='reproduce':
            rows=primary[args.shard_index::args.shard_count]
            for row in rows: e.run_case(row,steps=args.rf_seed==42 and row['id'] in ids)
        else:
            assert read(OUT/'reproduction/summary.json')['pass'], '复现失败，不进入诊断'
            if args.work=='confirmation':
                candidate=read(OUT/'strength_grid/candidate.json')
                assert candidate['strength'] is not None
                ids=e.splits['confirmation64']
            rows=[r for r in primary if r['id'] in ids][args.shard_index::args.shard_count]
            if args.work=='conditions':
                for name,condition in CONDITIONS.items():
                    for row in rows: e.run_case(row,'condition_factorization/'+name,conditions=condition,static=False)
            elif args.work=='strengths':
                for strength in STRENGTHS:
                    for row in rows: e.run_case(row,'strength_grid/%.2f'%strength,strength=strength,static=False)
            elif args.work=='confirmation':
                for strength in sorted({.15,candidate['strength']}):
                    for row in rows: e.run_case(row,'strength_confirmation/%.2f'%strength,strength=strength,static=False)
            elif args.work=='purity':
                assert read(OUT/'stage_survival/localization.json')['field_to_rgb_bottleneck']
                for kind in ('P0','P1','P2','P3'):
                    for row in rows: e.run_case(row,'carrier_purity/'+kind,purity=kind,steps=True)
            elif args.work=='seed_confirmation':
                stages=read(OUT/'stage_survival/localization.json')['stage_pair']
                assert args.rf_seed in (43,44)
                for row in rows: e.run_case(row,'seed_confirmation',steps=True)
    e.finish()

if __name__=='__main__': main()
