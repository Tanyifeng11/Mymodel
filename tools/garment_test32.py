"""纯服装测试子集：32原参考+16换参考，11模型，只推理。"""
import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from tools.t3_p0p1 import PROJECT, SOURCE, read, write, sha

ROOT = PROJECT/'output_eval/garment_test32_20261031'
OLD = PROJECT/'output_eval/t3_p0_p1_20261028'
SG_OLD = PROJECT/'output_eval/sgdiff_external_20261031'
SPEC = SOURCE/'configs/garment_test32_20261031.json'
SCRIPT = SOURCE/'scripts/garment_test32.sbatch'
MODELS = ['E0','E1','B','T1','T2','C','T3','BF-Fashion','ControlNet+IP-Adapter','LOTS','SGDiff']
CAP = 7200


def jobs():
    return read(ROOT/'audit/jobs.json') if (ROOT/'audit/jobs.json').exists() else []


def resources():
    ledger = jobs()
    if not ledger:
        return dict(actual_gpu_seconds=0, maximum_accounted_gpu_seconds=0, jobs=[])
    output = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['job']) for j in ledger),
        '-o','JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End,NodeList'], text=True)
    rows = {r[0]:r for line in output.splitlines() if len(r:=line.split('|')) >= 7}
    records = []
    for job in ledger:
        row = rows.get(str(job['job']))
        state = row[1] if row else 'ACCOUNTING_PENDING'
        elapsed = int(row[2]) if row else 0
        match = re.search(r'(?:^|,)gres/gpu=(\d+)',row[3]) if row else None
        allocated = int(match.group(1)) if match else 0
        actual = allocated*elapsed
        final = any(state.startswith(s) for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'])
        records.append(dict(job, state=state, elapsed_seconds=elapsed, allocated_gpus=allocated,
            gpu_seconds=actual, maximum_accounted_gpu_seconds=actual if final else job['gpus']*job['limit_seconds']))
    result = dict(actual_gpu_seconds=sum(r['gpu_seconds'] for r in records), hard_cap_seconds=CAP,
        maximum_accounted_gpu_seconds=sum(r['maximum_accounted_gpu_seconds'] for r in records), jobs=records)
    write(ROOT/'audit/resources.json',result)
    (ROOT/'audit/resources_sacct.psv').write_text(output,encoding='utf-8')
    assert result['maximum_accounted_gpu_seconds'] <= CAP, result
    return result


def submit(role, action, minutes, gpus=0, dependency=(), extra=()):
    existing = [j for j in jobs() if j['role']==role]
    if existing:
        print('ALREADY_SUBMITTED',role,existing[0]['job'],flush=True)
        return existing[0]['job']
    assert resources()['maximum_accounted_gpu_seconds']+minutes*60*gpus <= CAP
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    command = ['sbatch','--parsable','-N1','--ntasks=1','--partition='+('gpu' if gpus else 'cpu'),
        '--cpus-per-task=4','--mem=48G','--time=00:%02d:00'%minutes,'--job-name=gtest_'+role,
        '--output='+str(ROOT/'logs'/(role+'_%j.out'))]
    if gpus: command += ['--gpus='+str(gpus)]
    if dependency: command += ['--dependency=afterok:'+':'.join(map(str,dependency))]
    command += [str(SCRIPT),action]+list(extra)
    job = int(subprocess.check_output(command,text=True).strip().split(';')[0])
    ledger = jobs()
    ledger.append(dict(job=job,role=role,action=action,gpus=gpus,limit_seconds=minutes*60,
        submitted_at=time.strftime('%Y-%m-%d %H:%M:%S'),command=command))
    write(ROOT/'audit/jobs.json',ledger)
    print('SUBMITTED',role,job,flush=True)
    return job


def prepare():
    spec = read(SPEC)
    assert len(spec['cases']) == 32
    for folder in ['audit','protocol','inputs','outputs','native','code','metrics','generation']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    data = PROJECT.parent/'datasets/BF'
    training = {p.stem.split('_')[0] for p in (data/'training/cloth').iterdir() if p.is_file()}
    previous = {r['id'].split('_')[0] for r in read(OLD/'protocol/input_manifest.json')}
    originals = []
    for cat in spec['categories']:
        cases = [r for r in spec['cases'] if r['category']==cat]
        assert len(cases)==8
        for case in cases:
            ident = case['id']
            assert ident.split('_')[0] not in training | previous, ident
            row = dict(id=ident,category=cat,condition='original',
                seed=int(hashlib.sha256(('garment_test32_seed/'+ident).encode()).hexdigest()[:8],16)%(2**31),
                caption=(data/'test'/cat/'text'/(ident+'.txt')).read_text(encoding='utf-8-sig').strip(),
                GT_used_in_inference=False)
            assert row['caption']
            for key,folder in [('sketch','sketch'),('reference','texture'),('gt','gt')]:
                source = data/'test'/cat/folder/(ident+'.jpg')
                digest = sha(source)
                assert digest == case[folder+'_sha256'], source
                target = ROOT/'inputs'/(digest+'.jpg')
                if not target.exists(): shutil.copy2(source,target)
                row.update({key:str(target),key+'_sha256':digest,key+'_source':str(source)})
            originals.append(row)
    swaps = []
    for cat in spec['categories']:
        chosen = sorted([r for r in originals if r['category']==cat],
            key=lambda r:hashlib.sha256(('garment_test32_swap/'+r['id']).encode()).hexdigest())[:4]
        for row in chosen:
            donor = sorted([r for r in originals if r['id']!=row['id'] and r['reference_sha256']!=row['reference_sha256']],
                key=lambda r:hashlib.sha256(('garment_test32_donor/'+row['id']+'/'+r['id']).encode()).hexdigest())[0]
            swapped = dict(row,condition='swapped',reference=donor['reference'],reference_sha256=donor['reference_sha256'],
                reference_source=donor['reference_source'],reference_donor_id=donor['id'],caption_unchanged=True)
            swaps.append(swapped)
    rows = originals+swaps
    protocol = dict(spec,models=MODELS,n_outputs=512,split='BF/test; fixed class-balanced small subset',
        seed_policy='one deterministic seed per identity, shared original/swap; SD architecture also shares explicit initial latent',
        swap_policy='four hash-selected identities/category; hash-selected distinct reference among 32; text and sketch unchanged',
        exposure_limit='current training product-root IDs excluded; previous96 excluded; immutable historical train/pretraining exposure unknown; SGDiff already had full-test inference',
        reference_metrics='64x64 sketch-interior patch-set response proxy, not complete motif identity',
        comparison_limit='SGDiff text+reference, BF sketch+reference, LOTS text+sketch; native resolutions and sample budgets differ',
        display='avoid stripes/polka examples; metric set never filtered by generation quality')
    for filename,value in [('input_manifest.json',rows),('controlled_protocol.json',protocol)]:
        target = ROOT/'protocol'/filename
        if target.exists(): assert read(target)==value
        else: write(target,value)
    for relative in ['audit/server_inventory.json','audit/BF_provenance.json','protocol/cnipa_choice.json']:
        shutil.copy2(OLD/relative,ROOT/relative)
    sgasset = read(SG_OLD/'audit/assets.json')
    sgasset.update(runner_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        current_test_manifest_sha256=sha(ROOT/'protocol/input_manifest.json'))
    write(ROOT/'audit/assets.json',sgasset)
    shutil.copy2(SG_OLD/'protocol/resolved_config.py',ROOT/'protocol/resolved_config.py')
    shutil.copy2(PROJECT.parent/'LOTS/assets.json',ROOT/'audit/LOTS_assets.json')
    for relative in ['tools/garment_test32.py','scripts/garment_test32.sbatch','configs/garment_test32_20261031.json',
                     'tools/t3_p0p1_generate.py','tools/t3_p0p1_evaluate.py','tools/t3_sgdiff_external.py']:
        target = ROOT/'code'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/relative,target)
    write(ROOT/'audit/prepared.json',dict(original_n=32,swapped_n=16,n_outputs=512,
        product_root_training_overlap=[],previous96_product_overlap=[],input_visual_review='32 inputs and replacement checked locally before inference; server byte hashes matched',
        code_commit=sgasset['runner_commit'],manifest_sha256=sha(ROOT/'protocol/input_manifest.json')))
    print('PREPARED',len(rows),sha(ROOT/'protocol/input_manifest.json'),flush=True)


def lots_runner():
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from transformers import AutoModel
    project = PROJECT.parent/'LOTS'
    sys.path.insert(0,str(project/'src'))
    from lots.lots_pipeline import LOTSPipeline
    asset = read(ROOT/'audit/LOTS_assets.json')
    adapter = Path(asset['adapter']['path'])/'ckpts/lots/lots.bin'
    assert sha(adapter)==asset['adapter']['files']['ckpts/lots/lots.bin']['sha256']
    base = StableDiffusionXLPipeline.from_pretrained(asset['sdxl']['path'],torch_dtype=torch.float16,
        variant='fp16',use_safetensors=True,add_watermarker=False,local_files_only=True)
    encoder = AutoModel.from_pretrained(asset['dino']['path'],local_files_only=True)
    pipe = LOTSPipeline(base,image_encoder=encoder,model_type='vits14',lots_ckpt=str(adapter),device='cuda',num_tokens=32)
    for module in [pipe.pipe.unet,pipe.pipe.vae,pipe.pipe.text_encoder,pipe.pipe.text_encoder_2,
                   pipe.image_encoder,pipe.image_proj_model,pipe.text_proj_model,pipe.pair_former]:
        module.eval().requires_grad_(False)
    actual = {}
    original = pipe.pipe.prepare_latents
    def capture(*args,**kwargs):
        value = original(*args,**kwargs)
        actual['initial_latent_sha256'] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        return value
    pipe.pipe.prepare_latents = capture
    def run(row):
        image = pipe.generate(prompt='High quality photo of a model, artistic, 4k',
            pil_images=[Image.open(row['sketch']).convert('RGB')],descriptions=[row['caption']],
            num_samples=1,num_inference_steps=30,resolution=512,scale=.8,seed=row['seed'])[0]
        native = ROOT/'native/LOTS/original'/(row['id']+'.png')
        native.parent.mkdir(parents=True,exist_ok=True)
        image.save(native)
        return image.resize((384,512),Image.Resampling.BILINEAR),dict(actual,text_used=True,reference_texture_used=False,
            steps=30,scale=.8,cfg=float(pipe.pipe.guidance_scale),scheduler=pipe.pipe.scheduler.__class__.__name__,
            native_sha256=sha(native),native_size=[512,512],checkpoint_sha256=sha(adapter))
    return run


def generate(arms):
    if arms==['SGDiff']:
        from tools import t3_sgdiff_external as sg
        sg.ROOT = ROOT
        return sg.generate()
    import numpy as np
    import torch
    from tools import t3_p0p1_generate as legacy
    legacy.ROOT = ROOT
    rows = read(ROOT/'protocol/input_manifest.json')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for arm in arms:
        assert arm in MODELS
        selected = [r for r in rows if arm!='LOTS' or r['condition']=='original']
        with torch.no_grad():
            runner = lots_runner() if arm=='LOTS' else legacy.bf() if arm=='BF-Fashion' else legacy.cnipa() if arm=='ControlNet+IP-Adapter' else legacy.internal(arm)
            for index,saved in enumerate(selected):
                row = dict(saved)
                path = ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
                if path.exists():
                    meta = read(path.with_suffix('.json'))
                    assert sha(path)==meta['output_sha256'] and meta['seed']==row['seed']
                    assert meta['input_reference_sha256']==(None if arm=='LOTS' else row['reference_sha256'])
                    continue
                latent = torch.randn((1,4,64,48),generator=torch.Generator(device='cuda').manual_seed(row['seed']),device='cuda',dtype=torch.float16)
                row['historical_initial_latent_sha256'] = legacy.digest(latent)
                torch.manual_seed(row['seed']);np.random.seed(row['seed'])
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin = time.perf_counter()
                image,meta = runner(row)
                torch.cuda.synchronize()
                assert image.size==(384,512)
                path.parent.mkdir(parents=True,exist_ok=True)
                image.save(path)
                meta.update(id=row['id'],arm=arm,condition=row['condition'],seed=row['seed'],input_caption=row['caption'],
                    sketch_used=True,reference_texture_used=arm!='LOTS',GT_used_in_inference=False,
                    input_sketch_sha256=row['sketch_sha256'],input_reference_sha256=None if arm=='LOTS' else row['reference_sha256'],
                    output_sha256=sha(path),seconds=time.perf_counter()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    code_commit=read(ROOT/'audit/prepared.json')['code_commit'],noise_scope='new fixed identity seed; not historical latent reuse')
                write(path.with_suffix('.json'),meta)
                print('OUTPUT',arm,row['condition'],row['id'],index+1,'/',len(selected),flush=True)
            del runner
        gc.collect();torch.cuda.empty_cache()
        write(ROOT/'generation'/(arm+'_complete.json'),dict(arm=arm,n=len(selected),complete=True))


def verify():
    rows = read(ROOT/'protocol/input_manifest.json')
    sd_hashes = {}
    for arm in MODELS:
        originals = {}
        for row in rows:
            if arm=='LOTS' and row['condition']=='swapped': continue
            p = ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
            meta = read(p.with_suffix('.json'))
            assert sha(p)==meta['output_sha256'] and meta['seed']==row['seed']
            assert meta['input_sketch_sha256']==row['sketch_sha256'] and not meta['GT_used_in_inference']
            assert meta['input_reference_sha256']==(None if arm=='LOTS' else row['reference_sha256'])
            if arm in ['E0','E1','B','T1','T2','C','T3','ControlNet+IP-Adapter']:
                key = (row['condition'],row['id'])
                sd_hashes.setdefault(key,meta['initial_latent_sha256'])
                assert sd_hashes[key]==meta['initial_latent_sha256']
            if row['condition']=='original': originals[row['id']] = meta
            else:
                assert originals[row['id']]['initial_latent_sha256']==meta['initial_latent_sha256']
                if arm=='SGDiff': assert originals[row['id']]['up_initial_latent_sha256']==meta['up_initial_latent_sha256']
    write(ROOT/'audit/input_noise_checks.json',dict(passed=True,models=11,n_outputs=512,
        input_sha256s=True,paired_identity_seeds=True,paired_within_model_latents=True,shared_SD_architecture_latents=True,GT_used=False))
    print('VERIFIED_512',flush=True)


def evaluate():
    verify()
    from tools import t3_p0p1_evaluate as legacy
    legacy.ROOT = ROOT
    legacy.output_path = lambda arm,row: ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
    for arm in MODELS: legacy.evaluate(arm,include_swap=True)


def package():
    resources()
    verify()
    for arm in MODELS:
        completed = read(ROOT/'metrics'/arm/'complete.json')
        assert completed['complete'] and completed['missing_generation']==0
        assert completed['n']==(32 if arm=='LOTS' else 48)
    files = sorted({p for folder in ['audit','protocol','inputs','outputs','native','base64','code','metrics','generation','logs']
        for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='bundle_sha256sums.txt' and not p.name.startswith('package_')})
    sums = ROOT/'audit/bundle_sha256sums.txt'
    sums.write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files),encoding='utf-8')
    target = ROOT/'review_results.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for p in files+[sums]: tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print('BUNDLE_BYTES',target.stat().st_size,'SHA256',sha(target),flush=True)


def submit_all():
    a = submit('prepare','prepare',10)
    generators = [submit('internal_a','generate',20,1,[a],['E0','E1','B','T1']),
        submit('internal_b','generate',17,1,[a],['T2','C','T3']),
        submit('bf','generate',13,1,[a],['BF-Fashion']),submit('cnipa','generate',7,1,[a],['ControlNet+IP-Adapter']),
        submit('sgdiff','generate',25,1,[a],['SGDiff']),submit('lots','generate',8,1,[a],['LOTS'])]
    b = submit('evaluate','evaluate',20,1,generators)
    submit('package_final','package',10,0,[b])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','generate','verify','evaluate','package','resources','submit-all'])
    parser.add_argument('arms',nargs='*')
    args = parser.parse_args()
    if args.action=='generate': generate(args.arms)
    elif args.action=='submit-all': submit_all()
    elif args.action=='resources': print(json.dumps(resources()),flush=True)
    else: globals()[args.action]()
