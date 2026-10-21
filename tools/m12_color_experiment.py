"""dev32颜色诊断与三个单因素免训练对照；不写入任何历史实验目录。"""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image
from tools.m12_color_metrics import descriptor, compare, regions, outside_metrics, calibrate, statistics

PROJECT = Path('/share/home/u2515283058/Mymodel')
RUN = Path(os.environ.get('M12_COLOR_RUN', str(PROJECT/'output_eval/m12_color_dev32_20261021')))
OLD = Path(os.environ.get('M12_COLOR_OLD', str(PROJECT/'output_eval/m12_gain_dev32_20261020')))
ARMS = ['BASE', 'LAB_AB', 'INIT_ADAIN', 'CFG5']
HISTORY = ['E5', 'B1', 'M10', 'M10_GAIN', 'M12', 'M12_GAIN']
SETTINGS = dict(identities=32, seeds=[42], arms=ARMS, size=[384,512], steps=50,
    base='M12_GAIN', alpha_B=.25, alpha_M=.5, gain_step=2400, cfg=7., sketch=.6, texture=1., ipa=1.,
    training_updates=0, sample='all original dev32 identities in original order; exposed development set, not confirmation',
    LAB_AB=dict(channels=['a','b'],L='unchanged before RGB gamut clipping',strength=1.,std_gain_clip=[.5,2.],
        region='old sketch softmask, erosion9 feather8; invalid old inference gate => exact OFF',reference='all texture pixels, no GT'),
    INIT_ADAIN=dict(strength=1.,dimensions=[0,2,3],std_unbiased=True,reference_inversion_steps=50,
        reference_unet='frozen original SD1.5 Writer UNet',conditioning='empty text, no CFG',
        style='VAE posterior mean, resized reference384x512, DDIMInverseScheduler matched to generation scheduler',
        content='paired random generation noise; adaptation of StyleID initial AdaIN, not content-image inversion or full StyleID',
        region='global latent statistics; background risk evaluated, no new mask or attention changes'),
    CFG5=dict(cfg=5.,other_changes=0),
    evaluation=dict(garment='sketch_only fixed mask, eroded9; <50pixels => NA',reference='whole texture, no output-dependent selection',
        color_primary='ref RGBuv Hellinger and LAB ab Wasserstein; lower is closer, no scalar combined score',
        rgbuv=dict(bins=64,sigma=.02,range=[-3,3],kernel='inverse quadratic',maximum_valid_pixels=4096,sampling='uniform in flattened valid-pixel order',dtype='float64'),
        separate=['GT RGBuv/ab', 'L Wasserstein','C signed delta and absolute log ratio','C>5 hue Hellinger, colored fraction>=.1',
            'outside colored fraction: C>8 S>.15 V>.15 beyond9px dilation','outside RGB white distance','LPIPS GT','F1 IoU original Leak','HP correct-minus-fixed-wrong NCC'],
        inference='GT never used in intervention; reference and GT scores may conflict',
        statistics='paired identity bootstrap10000 seed32042, single seed exploratory; counts and NA retained',
        guardrails=dict(f1_ci_lower=-.02,iou_ci_lower=-.01,leak_ci_upper=.005,extreme_f1=-.15),
        decision='diagnostic development only; no GO/confirmation or parameter tuning; all arms retained regardless outcome'),
    papers=dict(histogan='https://github.com/mahmoudnafifi/HistoGAN',styleid='https://github.com/jiwoogit/StyleID',
        colorizediffusion='https://arxiv.org/html/2401.01456v3',stableviton='https://arxiv.org/abs/2312.01725'))
SOURCES = ['tools/m12_color_experiment.py','tools/m12_color_metrics.py','tests/test_m12_color_metrics.py',
    'submit/m12_color_gpu.sh','submit/m12_color_cpu.sh']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def rows():
    return read(RUN/'protocol/identities.json')


def input_path(row,key):
    return RUN/'inputs'/row['id']/(key+'.jpg')


def image(arm,identity):
    return RUN/'outputs'/arm/(identity+'.png')


def old_image(arm,identity):
    if arm=='M12_GAIN':return OLD/'outputs/M12_GAIN'/identity/'seed42.png'
    if (OLD/'reused'/arm/(identity+'.png')).exists():return OLD/'reused'/arm/(identity+'.png')
    return PROJECT/'output_eval/m12_fixed_dual_path_20261018/dev32/images'/('R025_050' if arm=='M12' else arm)/(identity+'.png')


def color_records(arms, source):
    from eval.eval_utils import prepare_evaluation_masks
    records=[]
    for row in rows():
        bundle=prepare_evaluation_masks((384,512),sketch_path=str(input_path(row,'sketch')),mask_policy='sketch_only')
        assert bundle['garment'] is not None and bundle['garment'].any()
        inside,outside=regions(bundle['garment'])
        ref=descriptor(Image.open(input_path(row,'reference')).convert('RGB'))
        gt=descriptor(Image.open(input_path(row,'gt')).convert('RGB').resize((384,512),Image.Resampling.BILINEAR),inside)
        for arm in arms:
            path=source(arm,row['id']);rgb=Image.open(path).convert('RGB');desc=descriptor(rgb,inside)
            rec=dict(id=row['id'],arm=arm,seed=42,mask_valid=read(image('BASE',row['id']).with_suffix('.json'))['mask']['valid'],
                inside_pixels=int(inside.sum()),reference_chroma=ref['chroma'],gt_chroma=gt['chroma'] if gt else None,
                output_chroma=desc['chroma'] if desc else None,png_sha256=sha(path),
                **compare(desc,ref,'ref_'),**compare(desc,gt,'gt_'),**outside_metrics(rgb,outside))
            records.append(rec)
        print('COLOR_METRICS',row['id'],len(records),flush=True)
    return records


def stage1():
    selected=read(OLD/'protocol/dev32.json');assert len(selected)==32
    write(RUN/'protocol/identities.json',selected);write(RUN/'protocol/settings.json',SETTINGS)
    manifest=[]
    for row in selected:
        for key in ['gt','sketch','reference']:
            source=OLD/'inputs'/row['id']/(key+'.jpg')
            if not source.exists():source=Path(row[key])
            dest=input_path(row,key);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==sha(source)
            else:shutil.copy2(source,dest)
            manifest.append(dict(id=row['id'],kind=key,sha256=sha(dest)))
        source=old_image('M12_GAIN',row['id']);dest=image('BASE',row['id']);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);shutil.copy2(source.with_suffix('.json'),dest.with_suffix('.json'))
    write(RUN/'protocol/input_manifest.json',manifest)
    records=color_records(HISTORY,old_image)
    write(RUN/'stage1/per_image.json',records);write(RUN/'stage1/statistics.json',statistics(records,'M12_GAIN'))
    write(RUN/'stage1/complete.json',dict(images=192,scientific_new_images=0,settings_sha256=sha(RUN/'protocol/settings.json')))


def prepare():
    if (RUN/'protocol/frozen.json').exists():require();return
    stage1()
    from tools.m12_gain_experiment import SOURCE_FILES,GAIN_CHECKPOINT
    from tools.m54_protocol import E5,E5_SHA,B1,B1_SHA
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    assert sha(GAIN_CHECKPOINT)=='a5c6912d81b1861e7e74e29142e13b036ca6d25a8ce1ab049e915b63b67f5464'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    write(RUN/'protocol/frozen.json',dict(commit=commit(),settings=SETTINGS,identities_sha256=sha(RUN/'protocol/identities.json'),
        input_manifest_sha256=sha(RUN/'protocol/input_manifest.json'),code_sha256={p:sha(p) for p in sorted(set(SOURCES+SOURCE_FILES))},
        checkpoints=dict(E5=E5_SHA,B1=B1_SHA,M10_GAIN=sha(GAIN_CHECKPOINT)),frozen_before_new_outputs=True))
    print('COLOR_PREPARED',commit(),flush=True)


def require():
    frozen=read(RUN/'protocol/frozen.json')
    assert frozen['settings']==SETTINGS and frozen['commit']==commit()
    assert sha(RUN/'protocol/identities.json')==frozen['identities_sha256']
    assert all(sha(p)==v for p,v in frozen['code_sha256'].items())
    for rec in read(RUN/'protocol/input_manifest.json'):
        assert sha(RUN/'inputs'/rec['id']/(rec['kind']+'.jpg'))==rec['sha256']


def runtime():
    from tools import m12_gain_confirm as module
    previous=module.RUN;module.RUN=RUN
    try:r=module.runtime('M12_GAIN','runtime')
    finally:module.RUN=previous
    for p in [r.processor,r.processor.native_m10]:
        p.record_enabled=False
        if hasattr(p,'diagnostic_observer'):p.diagnostic_observer=None
    return r


def inversion(r,row):
    import torch
    from diffusers import DDIMInverseScheduler
    p=r.pipe;path=RUN/'inversion'/(row['id']+'.pt')
    if path.exists():return torch.load(path,map_location=p.device,weights_only=False)['style_zT']
    torch.cuda.synchronize();started=time.perf_counter()
    ref=Image.open(input_path(row,'reference')).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(ref,np.float32).transpose(2,0,1)/127.5-1)[None].to(p.device,p.vae.dtype)
    z=p.vae.encode(x).latent_dist.mean*p.vae.config.scaling_factor
    schedule=DDIMInverseScheduler.from_config(p.scheduler.config);schedule.set_timesteps(50,device=p.device)
    p.scheduler.set_timesteps(50,device=p.device)
    assert torch.equal(schedule.timesteps.flip(0),p.scheduler.timesteps)
    for t in schedule.timesteps:
        eps=r.writer.unet(z,t,encoder_hidden_states=r.empty_text).sample
        z=schedule.step(eps,t,z).prev_sample
    assert torch.isfinite(z).all()
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(style_zT=z.cpu(),timesteps=schedule.timesteps.cpu(),scheduler=dict(schedule.config)),path)
    torch.cuda.synchronize()
    write(path.with_suffix('.json'),dict(seconds=time.perf_counter()-started,sha256=sha(path),
        first_timestep=int(schedule.timesteps[0]),last_timestep=int(schedule.timesteps[-1]),steps=50,
        clean_reference_used_as_zT=False,empty_text=True,cfg=False))
    return z


def generate_one(r,row,arm,path):
    import torch
    from torchvision.transforms.functional import to_tensor
    from tools.e44_runtime import sketch_masks,digest
    p=r.pipe;r.geometry(str(input_path(row,'sketch')),str(input_path(row,'reference')))
    sketch=Image.open(input_path(row,'sketch')).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
    ref=Image.open(input_path(row,'reference')).convert('RGB');raw,mask,info=sketch_masks(sketch,p.device)
    r.set_reference(to_tensor(ref.resize((384,512),Image.Resampling.BILINEAR))[None].to(p.device)*2-1,mask,info,allow=True)
    style=inversion(r,row) if arm=='INIT_ADAIN' else None
    original=p.prepare_latents;traces={}
    def prepare_latents(*a,**kw):
        noise=original(*a,**kw);traces['paired_noise']=noise.detach().cpu()
        z=noise
        if style is not None:
            nf=noise.float();sf=style.float();dims=[0,2,3]
            z=((nf-nf.mean(dims,keepdim=True))/nf.std(dims,keepdim=True).clamp_min(1e-6)*sf.std(dims,keepdim=True)+sf.mean(dims,keepdim=True)).to(noise.dtype)
        traces['initial']=z.detach().cpu();return z
    p.prepare_latents=prepare_latents
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        rgb=p(prompt=row['caption'],null_prompt='',negative_prompt=' worst quality, low quality',
            ref_image=to_tensor(sketch)[None]*2-1,texture_clip_image=ref,width=384,height=512,
            num_inference_steps=50,guidance_scale=5. if arm=='CFG5' else 7.,sketch_scale=.6,ipa_scale=1.,
            texture_mode=r.ns.texture_mode,texture_condition_mode='token',texture_preprocess_mode='plain_resize',
            texture_num_tokens=p.effective_texture_num_tokens,texture_scale=1.,
            spatial_mask=to_tensor(raw)[None].to(p.device,torch.float16),generator=torch.Generator(device=p.device).manual_seed(42))[0]
    finally:p.prepare_latents=original
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    path.parent.mkdir(parents=True,exist_ok=True);rgb.save(path);torch.save(traces,path.with_suffix('.pt'))
    write(path.with_suffix('.json'),dict(id=row['id'],arm=arm,seed=42,mask=info,seconds=seconds,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        timing_scope='pipeline; reference inversion separately timed and retained',
        png_sha256=sha(path),paired_noise_sha256=digest(traces['paired_noise']),initial_sha256=digest(traces['initial']),
        gt_used_in_inference=False,training_updates=0,commit=commit(),style_inversion_sha256=sha(RUN/'inversion'/(row['id']+'.pt')) if style is not None else None))
    assert r.path_hashes()==r.source_hashes


def generate():
    import torch
    from tools.m54_runtime import release
    from tools.e44_runtime import sketch_masks
    require()
    if (RUN/'generation/complete.json').exists():return
    r=runtime()
    try:
        with torch.inference_mode():
            # 首先对旧dev32三例精确重放；失败时不生成新对照。
            for row in rows()[:3]:
                path=RUN/'smoke'/(row['id']+'.png')
                if not path.exists():generate_one(r,row,'BASE',path)
                assert sha(path)==sha(image('BASE',row['id'])),('BASE_REPLAY_CHANGED',row['id'])
                assert read(path.with_suffix('.json'))['initial_sha256']==read(image('BASE',row['id']).with_suffix('.json'))['initial_sha256']
            write(RUN/'smoke/complete.json',dict(passed=True,exact_replay=3))
            for row in rows():
                sketch=Image.open(input_path(row,'sketch')).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
                raw,mask,info=sketch_masks(sketch,'cpu')
                dest=image('LAB_AB',row['id'])
                if not dest.exists():
                    started=time.perf_counter();rgb=Image.open(image('BASE',row['id'])).convert('RGB');ref=Image.open(input_path(row,'reference')).convert('RGB')
                    result=calibrate(rgb,ref,mask[0,0].numpy(),info['valid']);dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(result).save(dest)
                    assert np.array_equal(result[mask[0,0].numpy()==0],np.asarray(rgb)[mask[0,0].numpy()==0])
                    write(dest.with_suffix('.json'),dict(id=row['id'],arm='LAB_AB',seed=42,mask=info,png_sha256=sha(dest),
                        seconds=time.perf_counter()-started,gt_used_in_inference=False,postprocess=True,commit=commit()))
                for arm in ['INIT_ADAIN','CFG5']:
                    dest=image(arm,row['id'])
                    if dest.exists():assert read(dest.with_suffix('.json'))['png_sha256']==sha(dest)
                    else:generate_one(r,row,arm,dest)
                    meta=read(dest.with_suffix('.json'));base=read(image('BASE',row['id']).with_suffix('.json'))
                    assert meta['paired_noise_sha256']==base['initial_sha256']
                    if arm=='CFG5':assert meta['initial_sha256']==base['initial_sha256']
                print('COLOR_GENERATED',row['id'],flush=True)
    finally:release(r)
    write(RUN/'generation/complete.json',dict(identities=32,seed=42,arms=ARMS,new_diffusion_outputs=64,postprocessed=32,reused_baselines=32))


def evaluate():
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from tools.m54_eval import regional_metrics,crop,ncc
    from tools.e33tmoc_appearance_eval import frozen_lpips
    require();assert (RUN/'generation/complete.json').exists()
    records=color_records(ARMS,image);index={(r['arm'],r['id']):r for r in records}
    metric,info=frozen_lpips();metric=metric.cuda()
    tensor=lambda im:torch.from_numpy(np.asarray(im,np.float32).transpose(2,0,1)/127.5-1)[None].cuda()
    def hp(im):
        import cv2
        a=cv2.cvtColor(np.asarray(crop(im)),cv2.COLOR_RGB2GRAY).astype(np.float32)
        return a-cv2.GaussianBlur(a,(0,0),2.)
    ordered=sorted(rows(),key=lambda r:r['id']);wrong={r['id']:ordered[(i+1)%32] for i,r in enumerate(ordered)}
    refs={r['id']:hp(Image.open(input_path(r,'reference')).convert('RGB')) for r in rows()}
    with torch.inference_mode():
        for row in rows():
            gt=Image.open(input_path(row,'gt')).convert('RGB').resize((384,512),Image.Resampling.BILINEAR)
            bundle=prepare_evaluation_masks((384,512),sketch_path=str(input_path(row,'sketch')),mask_policy='sketch_only')
            for arm in ARMS:
                path=image(arm,row['id']);rgb=Image.open(path).convert('RGB');rec=index[arm,row['id']]
                rec.update(lpips_gt=float(metric(tensor(rgb),tensor(gt)).mean()),
                    **regional_metrics(rgb,path,str(input_path(row,'gt')),bundle,rec['mask_valid']))
                feature=hp(crop(rgb,bundle['garment']));rec['hp_margin']=ncc(feature,refs[row['id']])-ncc(feature,refs[wrong[row['id']]['id']])
            print('COLOR_EVAL',row['id'],flush=True)
    result=statistics(records)
    guardrails={}
    base={r['id']:r for r in records if r['arm']=='BASE'}
    for arm in ARMS[1:]:
        delta=result['paired'][arm]
        extreme=[r['id'] for r in records if r['arm']==arm and r['contour_f1']-base[r['id']]['contour_f1']<-.15]
        checks=dict(f1=delta['contour_f1']['ci95'][0]>=-.02,iou=delta['iou']['ci95'][0]>=-.01,
            leak=delta['leak']['ci95'][1]<=.005,extreme=not extreme)
        guardrails[arm]=dict(checks=checks,all_passed=all(checks.values()),extreme_ids=extreme,
            paired_color_95_signal={k:delta[k]['n']==32 and delta[k]['ci95'][1]<0 for k in ['ref_rgbuv_h','ref_ab_w1']},
            status='DEVELOPMENT_DIAGNOSTIC_ONLY')
    write(RUN/'metrics/per_image.json',records);write(RUN/'metrics/statistics.json',result)
    write(RUN/'metrics/guardrails.json',guardrails)
    write(RUN/'metrics/complete.json',dict(images=128,lpips=info,environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version()),
        wrong_reference={k:v['id'] for k,v in wrong.items()},no_blind_review=True,no_pattern_ground_truth=True))


def finalize():
    from tools.m54_finalize import package
    require();assert read(RUN/'metrics/complete.json')['images']==128
    files={str(p.relative_to(RUN)):p for p in RUN.rglob('*') if p.is_file() and p.suffix in ['.json','.png','.jpg','.csv'] and p.name!='run_state.json'}
    for p in SOURCES:files['code/'+p]=Path(p)
    package(RUN/'m12_color_review.tar.gz',files)
    print('COLOR_PACKAGED',sha(RUN/'m12_color_review.tar.gz'),flush=True)


def accounting():
    from tools.m54_finalize import package
    jobs=read(RUN/'resources/jobs.json');raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    ledger=list(csv.DictReader(raw.splitlines(),delimiter='|'));assert len(ledger)==len(jobs)
    assert all(r['State'] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] for r in ledger)
    write(RUN/'resources/accounting.json',ledger)
    package(RUN/'m12_color_ledger.tar.gz',{str(p.relative_to(RUN)):p for p in (RUN/'resources').glob('*') if p.is_file()})
    print('COLOR_ACCOUNTED',[(r['JobIDRaw'],r['State']) for r in ledger],flush=True)


def submit(action,dependency=None):
    path=RUN/'resources/jobs.json';jobs=read(path) if path.exists() else []
    assert not any(j['action']==action for j in jobs)
    script='submit/m12_color_'+('gpu' if action in ['generate','evaluate'] else 'cpu')+'.sh'
    cmd=['sbatch','--parsable','--job-name=color_'+action]+(['--dependency=afterok:'+str(dependency)] if dependency else [])+[script,action]
    job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);jobs.append(dict(id=job,action=action,command=cmd))
    write(path,jobs);print('COLOR_SUBMITTED',action,job,flush=True);return job


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage1','prepare','generate','evaluate','finalize','accounting','submit']);parser.add_argument('--stage')
    args=parser.parse_args();(RUN/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='submit':submit(args.stage)
    else:globals()[args.action]()
