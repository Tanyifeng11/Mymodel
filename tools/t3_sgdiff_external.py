"""SGDiff同批补充评价：不训练、不输入草图，旧结果和旧代码保持原样。"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from tools.t3_p0p1 import PROJECT, SOURCE, read, write, sha, csv_write

ROOT = PROJECT/'output_eval/sgdiff_external_20261031'
OLD = PROJECT/'output_eval/t3_p0_p1_20261028'
SG = PROJECT.parent/'sgdiff'
CHECKPOINT = SG/'work_dirs/sgdiff_bf_style_v2/iter_50000.pth'
CONFIG = SG/'configs/sgdiff/sgdiff-bf-style-64x64.py'
SCRIPT = SOURCE/'scripts/t3_sgdiff_external.sbatch'
SMOKE_IDS = ['153804021', '168463453', '34176180', '41267269']
CAP = 7200


def jobs():
    return read(ROOT/'audit/jobs.json') if (ROOT/'audit/jobs.json').exists() else []


def resources():
    ledger = jobs()
    if not ledger:
        return dict(actual_gpu_seconds=0, maximum_accounted_gpu_seconds=0, jobs=[])
    output = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['job']) for j in ledger),
        '-o','JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End,NodeList'], text=True)
    rows = {r[0]:r for line in output.splitlines() if len(r:=line.split('|'))>=7}
    records=[]
    for job in ledger:
        row=rows.get(str(job['job']))
        state=row[1] if row else 'ACCOUNTING_PENDING'
        elapsed=int(row[2]) if row else 0
        match=re.search(r'(?:^|,)gres/gpu=(\d+)',row[3]) if row else None
        allocated=int(match.group(1)) if match else 0
        actual=allocated*elapsed
        final=any(state.startswith(s) for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'])
        records.append(dict(job,state=state,elapsed_seconds=elapsed,allocated_gpus=allocated,
            gpu_seconds=actual,maximum_accounted_gpu_seconds=actual if final else job['gpus']*job['limit_seconds']))
    total=sum(r['gpu_seconds'] for r in records)
    result=dict(actual_gpu_seconds=total,gpu_hours=total/3600,hard_cap_seconds=CAP,
        maximum_accounted_gpu_seconds=sum(r['maximum_accounted_gpu_seconds'] for r in records),jobs=records)
    write(ROOT/'audit/resources.json',result)
    (ROOT/'audit/resources_sacct.psv').write_text(output,encoding='utf-8')
    return result


def submit(role,action,minutes,gpus=0,dependency=(),extra=()):
    existing=[j for j in jobs() if j['role']==role]
    if existing:
        print('ALREADY_SUBMITTED',role,existing[0]['job'],flush=True)
        return existing[0]['job']
    assert resources()['maximum_accounted_gpu_seconds']+minutes*60*gpus<=CAP
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    command=['sbatch','--parsable','-N1','--ntasks=1','--partition='+('gpu' if gpus else 'cpu'),
        '--cpus-per-task=4','--mem=48G','--time=00:%02d:00'%minutes,'--job-name=sgext_'+role,
        '--output='+str(ROOT/'logs'/(role+'_%j.out'))]
    if gpus: command+=['--gpus='+str(gpus)]
    if dependency: command+=['--dependency=afterok:'+':'.join(map(str,dependency))]
    command += [str(SCRIPT),action]+list(extra)
    job=int(subprocess.check_output(command,text=True).strip().split(';')[0])
    ledger=jobs();ledger.append(dict(job=job,role=role,action=action,gpus=gpus,limit_seconds=minutes*60,
        submitted_at=time.strftime('%Y-%m-%d %H:%M:%S'),command=command))
    write(ROOT/'audit/jobs.json',ledger)
    print('SUBMITTED',role,job,flush=True)
    return job


def prepare():
    for directory in ['audit','protocol','inputs','outputs/SGDiff','native/SGDiff','base64','code','metrics','figures']:
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    rows=read(OLD/'protocol/input_manifest.json')
    assert sum(r['condition']=='original' for r in rows)==96 and len(rows)==128
    frozen=[]
    for row in rows:
        row=dict(row)
        for key in ['sketch','reference','gt']:
            origin=Path(row[key]);assert sha(origin)==row[key+'_sha256']
            target=ROOT/'inputs'/origin.name
            if not target.exists(): shutil.copy2(origin,target)
            assert sha(target)==row[key+'_sha256']
            row[key]=str(target)
        frozen.append(row)
    protocol=dict(name='SGDiff-external-20261031',training_updates=0,original_n=96,swapped_n=32,
        inputs=['text','reference'],sketch_used=False,GT_used=False,native_size=[256,256],base_size=[64,64],
        evaluation_size=[384,512],resize='PIL BILINEAR direct, native retained',steps=100,up_steps=35,
        guidance_order=['style','txt'],style_guidance=1.2,text_guidance=1.0,precision='float32',
        seed='same historical identity seed; SGDiff CPU RNG, paired base/up noise within SGDiff, distinct from SD latent',
        metrics='unchanged t3_p0p1_evaluate CUDA float32 TF32 off; no swapped LPIPS/FID',
        display='avoid stripe/polka examples; fabric original + sketch + GT; presentation selection never alters full metric set',
        scope='previously viewed development identities; missing sketch makes structural scores task alignment proxies, not matched-modal causal ranking',
        patch_limit='64x64 sketch-interior patches do not establish complete motif fidelity; SGDiff garment need not align with sketch',
        smoke_ids=SMOKE_IDS,gpu_seconds_cap=CAP)
    for name,value in [('input_manifest.json',frozen),('controlled_protocol.json',protocol)]:
        p=ROOT/'protocol'/name
        if p.exists(): assert read(p)==value
        else: write(p,value)
    sys.path.insert(0,str(SG))
    from mmengine.config import Config
    config=Config.fromfile(str(CONFIG))
    assert config.model.modalities==['txt','style']
    assert config.model.unet.image_size==64 and config.model.unet_up.image_size==256
    (ROOT/'protocol/resolved_config.py').write_text(config.pretty_text,encoding='utf-8')
    assets={}
    source_files=['tools/sample_sgdiff.py','mmagic/models/editors/sgdiff/sgdiff.py',
        'mmagic/models/editors/glide/glide.py','mmagic/models/editors/glide/glide_ckpt.py',
        'configs/sgdiff/sgdiff-bf-style-64x64.py','configs/sgdiff/sgdiff-bf-glide-64x64.py']
    for relative in source_files:
        p=SG/relative;assets[relative]=sha(p)
        target=ROOT/'code/sgdiff'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for relative in ['tools/t3_sgdiff_external.py','scripts/t3_sgdiff_external.sbatch','tools/t3_p0p1_evaluate.py']:
        target=ROOT/'code/Mymodel'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/relative,target)
    current_training={p.stem for p in (PROJECT.parent/'datasets/BF/training/cloth').glob('*') if p.is_file()}
    overlap=sorted(current_training & {r['id'] for r in rows})
    assert not overlap,overlap
    write(ROOT/'audit/assets.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
        config=str(CONFIG),source_sha256=assets,sgdiff_commit=subprocess.check_output(['git','-C',str(SG),'rev-parse','HEAD'],text=True).strip(),
        runner_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        current_training_files=len(current_training),training_overlap_ids=overlap,
        exposure_limit='current training folder only; historical immutable train list and pretraining exposure unknown',
        old_input_manifest_sha256=sha(OLD/'protocol/input_manifest.json'),
        metric_protocol_sha256=sha(PROJECT/'output_eval/e2a_sup_comp_01_20261021/evaluation/metric_protocol.json')))
    print('PREPARED',len(frozen),'CHECKPOINT_SHA256',sha(CHECKPOINT),flush=True)


def generate(smoke=False,shard=0,shards=1):
    import numpy as np
    import torch
    from PIL import Image
    from mmengine.config import Config
    sys.path.insert(0,str(SG))
    spec=importlib.util.spec_from_file_location('sgext_sample',SG/'tools/sample_sgdiff.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    asset=read(ROOT/'audit/assets.json')
    assert sha(CHECKPOINT)==asset['checkpoint_sha256']
    assert all(sha(SG/p)==h for p,h in asset['source_sha256'].items())
    rows=read(ROOT/'protocol/input_manifest.json')
    rows=[r for r in rows if r['condition']=='original' and r['id'] in SMOKE_IDS] if smoke else rows[shard::shards]
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=helper.load_model(Config.fromfile(str(ROOT/'protocol/resolved_config.py')),str(CHECKPOINT),'cuda:0')
    assert model.modalities==['txt','style'] and model.unet_up is not None
    model.eval().requires_grad_(False)
    actual={}
    digest=lambda t:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    original_up=model.infer_up
    def audited_up(*args,**kwargs):
        # 在原函数原本取噪声的位置显式记录同一CPU RNG张量，不修改模型/采样方程。
        value=torch.randn((1,3,256,256))
        actual['up_initial_latent_sha256']=digest(value)
        kwargs['init_image']=value.to(model.device)
        return original_up(*args,**kwargs)
    model.infer_up=audited_up
    job=os.environ.get('SLURM_JOB_ID','local')
    write(ROOT/'audit'/('runtime_'+job+'.json'),dict(torch=torch.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),
        model_dtype=str(model.dtype),checkpoint_sha256=asset['checkpoint_sha256'],source_sha256=sha(__file__),
        up_output_nonzero=int(torch.count_nonzero(model.unet_up.out.conv.weight).item()),smoke=smoke,shard=shard,shards=shards))
    for index,row in enumerate(rows):
        output=ROOT/'outputs/SGDiff'/row['condition']/(row['id']+'.png')
        if output.exists():
            meta=read(output.with_suffix('.json'))
            assert sha(output)==meta['output_sha256'] and meta['input_reference_sha256']==row['reference_sha256']
            assert meta['seed']==row['seed'] and meta['checkpoint_sha256']==asset['checkpoint_sha256']
            continue
        random.seed(row['seed']);np.random.seed(row['seed']);torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        # 与现有infer_mm默认值相同的CPU噪声；相同身份旧/新参考共享它及后续超分噪声。
        latent=torch.randn((3,3,64,64));base_sha=digest(latent)
        with torch.no_grad():
            result=model.infer_mm(init_image=latent.to(model.device),style=helper.load_style(row['reference'],'cuda:0'),
                prompt=row['caption'],batch_size=1,num_inference_steps=100,up_inference_steps=35,
                modality_order_cfg=dict(style=1.2,txt=1.0),run_up=True,show_progress=False)
        torch.cuda.synchronize()
        native=ROOT/'native/SGDiff'/row['condition']/(row['id']+'.png')
        base=ROOT/'base64'/row['condition']/(row['id']+'.png')
        for path,key in [(native,'samples'),(base,'low_res_samples')]:
            path.parent.mkdir(parents=True,exist_ok=True);helper.save_sample(result[key],path)
        output.parent.mkdir(parents=True,exist_ok=True)
        Image.open(native).convert('RGB').resize((384,512),Image.Resampling.BILINEAR).save(output)
        meta=dict(id=row['id'],condition=row['condition'],seed=row['seed'],arm='SGDiff',caption=row['caption'],
            text_used=True,reference_texture_used=True,sketch_used=False,GT_used_in_inference=False,
            input_reference_sha256=row['reference_sha256'],input_sketch_sha256=row['sketch_sha256'],
            sketch_hash_scope='evaluation/display only; not an inference input',checkpoint_sha256=asset['checkpoint_sha256'],
            initial_latent_sha256=base_sha,**actual,native_sha256=sha(native),base_sha256=sha(base),output_sha256=sha(output),
            native_size=[256,256],base_size=[64,64],steps=100,up_steps=35,style_guidance=1.2,text_guidance=1.0,
            seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,
            code_commit=asset['runner_commit'])
        write(output.with_suffix('.json'),meta)
        print('OUTPUT',row['condition'],row['id'],index+1,'/',len(rows),'SECONDS',meta['seconds'],flush=True)
    write(ROOT/'audit'/('generation_'+job+'.json'),dict(complete=True,smoke=smoke,shard=shard,shards=shards,n=len(rows)))


def evaluate():
    from tools import t3_p0p1_evaluate as legacy
    rows=read(ROOT/'protocol/input_manifest.json')
    by_id={r['id']:r for r in rows if r['condition']=='original'}
    for row in rows:
        path=ROOT/'outputs/SGDiff'/row['condition']/(row['id']+'.json')
        meta=read(path);assert sha(path.with_suffix('.png'))==meta['output_sha256']
        if row['condition']=='swapped':
            old=read(ROOT/'outputs/SGDiff/original'/(row['id']+'.json'))
            assert meta['initial_latent_sha256']==old['initial_latent_sha256']
            assert meta['up_initial_latent_sha256']==old['up_initial_latent_sha256']
            assert row['seed']==by_id[row['id']]['seed']
    legacy.ROOT=ROOT
    legacy.evaluate('SGDiff')


def smoke_grid():
    from PIL import Image,ImageDraw,ImageOps
    rows=[r for r in read(ROOT/'protocol/input_manifest.json') if r['condition']=='original' and r['id'] in SMOKE_IDS]
    page=Image.new('RGB',(1220,80+len(rows)*270),'white');draw=ImageDraw.Draw(page)
    for i,label in enumerate(['Reference','Sketch (not used)','GT (eval only)','Base64','SGDiff256']):
        draw.text((20+i*240,20),label,fill='black')
    for j,row in enumerate(rows):
        draw.text((5,60+j*270),row['id'],fill='black')
        paths=[row['reference'],row['sketch'],row['gt'],ROOT/'base64/original'/(row['id']+'.png'),ROOT/'native/SGDiff/original'/(row['id']+'.png')]
        for i,path in enumerate(paths):
            im=ImageOps.contain(Image.open(path).convert('RGB'),(220,240))
            page.paste(im,(i*240+(240-im.width)//2,80+j*270+(240-im.height)//2))
    page.save(ROOT/'figures/smoke.png')


def package(name):
    resources()
    if name=='smoke_review':smoke_grid()
    files=sorted({p for folder in ['audit','protocol','inputs','outputs','native','base64','code','metrics','figures','logs']
        for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='bundle_sha256sums.txt' and not p.name.startswith('package_')})
    sums=ROOT/'audit/bundle_sha256sums.txt'
    sums.write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files),encoding='utf-8')
    target=ROOT/(name+'.tar.gz')
    with tarfile.open(target,'w:gz') as tar:
        for p in files+[sums]:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print('BUNDLE_BYTES',target.stat().st_size,'SHA256',sha(target),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','generate','evaluate','package','resources','submit-smoke','release','submit-formal'])
    p.add_argument('--smoke',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--name',default='review_results');args=p.parse_args()
    if args.action=='submit-smoke':
        a=submit('prepare','prepare',10);b=submit('smoke','generate',10,1,[a],['--smoke'])
        submit('package_smoke','package',10,0,[b],['--name','smoke_review'])
    elif args.action=='submit-formal':
        assert read(ROOT/'protocol/smoke_release.json')['actual_image_review']
        a=[submit('formal_'+str(i),'generate',40,1,extra=['--shard',str(i),'--shards','2']) for i in range(2)]
        b=submit('evaluate','evaluate',10,1,a);submit('package_final','package',10,0,[b])
    elif args.action=='release':
        write(ROOT/'protocol/smoke_release.json',dict(actual_image_review=True,reviewer='AI nonblind',human_reviewers=0,
            scope='strict loading, nonzero super-resolution and runnable garment generation; not a quality superiority gate',hyperparameter_changes=0))
        print('SMOKE_RELEASED',flush=True)
    elif args.action=='generate':generate(args.smoke,args.shard,args.shards)
    elif args.action=='package':package(args.name)
    elif args.action=='resources':print(json.dumps(resources()),flush=True)
    else:globals()[args.action]()


if __name__=='__main__':main()
