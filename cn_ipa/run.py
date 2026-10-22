import argparse
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from cn_ipa.common import *
from cn_ipa import runtime as rt
from cn_ipa.scoring import Scorer,means,selection_key,csv_write,statistics


def stage(name,**info):
    write(ROOT/'status.json',dict(stage=name,time=time.time(),commit=commit(),**info))
    print('CNIPA_STAGE',name,info,flush=True)


def validate_assets():
    assets=read(ROOT/'pretrained/assets.json')
    for path,info in assets['files'].items():
        assert sha(ROOT/'pretrained'/path)==info['sha256'],path
    official={'image_encoder/model.safetensors':'6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030',
        'ip_adapter/ip-adapter_sd15.bin':'68e1df30d760f280e578c302f1e73b37ea08654eff16a31153588047affe0058'}
    for path,value in official.items():assert assets['files'][path]['sha256']==value
    versions={name:__import__(name).__version__ for name in ['torch','diffusers','transformers','accelerate','datasets','numpy','PIL','lpips'] if name!='lpips'}
    versions.update(python=platform.python_version(),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    write(ROOT/'audit/versions.json',versions)
    paths=[p for p in SD.rglob('*') if p.is_file() and '.git' not in p.parts
        and p.suffix in ['.json','.bin','.safetensors','.txt'] and '.part' not in p.name]
    base_hashes={str(p):sha(p) for p in paths}
    write(ROOT/'audit/base_model_file_sha256.json',base_hashes)
    source=read('cn_ipa/sd15_official_files.json');checked={}
    for component,options in dict(unet=['diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'],
            vae=['diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'],
            text_encoder=['model.safetensors','pytorch_model.bin']).items():
        relative=next(f'{component}/{name}' for name in options if (SD/component/name).exists())
        checked[relative]=dict(actual=base_hashes[str(SD/relative)],official=source['files'][relative])
    write(ROOT/'audit/sd15_official_validation.json',dict(source=source,files=checked,
        passed=all(v['actual']==v['official'] for v in checked.values())))
    assert all(v['actual']==v['official'] for v in checked.values()),('SD15_WEIGHT_PROVENANCE_MISMATCH',checked)
    write(ROOT/'audit/official_weight_validation.json',dict(passed=True,official_lfs_sha256=official,assets=assets))


def selected_cn():
    return Path(read(ROOT/'selection/cn_checkpoint.json')['path'])


def cpu_preflight():
    stage('cpu_functional_preflight');torch.set_num_threads(4)
    pipe=rt.load(device='cpu');rows=read(ROOT/'manifests/dev.json')[:2]
    sketch,ref=rt.inputs(rows[0]);other_sketch,other_ref=rt.inputs(rows[1])
    pipe.set_ip_adapter_scale(.65)
    with torch.inference_mode():
        result=pipe(prompt=rows[0]['caption'],negative_prompt=rt.NEGATIVE,image=sketch,ip_adapter_image=ref,
            width=384,height=512,num_inference_steps=2,guidance_scale=7.,controlnet_conditioning_scale=1.,
            generator=torch.Generator(device='cpu').manual_seed(42)).images[0]
        assert result.size==SIZE
        folder=ROOT/'smoke/cpu_preflight';folder.mkdir(parents=True,exist_ok=True);result.save(folder/'two_step_path_check.png')
        text=pipe.text_encoder(pipe.tokenizer(rows[0]['caption'],padding='max_length',max_length=77,truncation=True,
            return_tensors='pt').input_ids)[0]
        encode=lambda image:pipe.image_encoder(pipe.feature_extractor(images=image,return_tensors='pt').pixel_values).image_embeds[:,None]
        condition=lambda image:pipe.control_image_processor.preprocess(image,height=512,width=384)
        a,b=encode(ref),encode(other_ref);s,u=condition(sketch),condition(other_sketch)
        z=torch.randn((1,4,64,48),generator=torch.Generator().manual_seed(42));t=torch.tensor([500])
        base=rt.forward(pipe,z,t,text,a,s)
        changed_ref=rt.forward(pipe,z,t,text,b,s);changed_sketch=rt.forward(pipe,z,t,text,a,u)
        off_a=rt.forward(pipe,z,t,text,a,s,0.);off_b=rt.forward(pipe,z,t,text,a,u,0.)
        pipe.set_ip_adapter_scale(0.)
        ip0a=rt.forward(pipe,z,t,text,a,s);ip0b=rt.forward(pipe,z,t,text,b,s)
    checks=dict(finite=bool(torch.isfinite(base).all()),reference_responds=not torch.equal(base,changed_ref),
        sketch_responds=not torch.equal(base,changed_sketch),cn_off=torch.equal(off_a,off_b),ip_off=torch.equal(ip0a,ip0b))
    write(ROOT/'audit/cpu_preflight.json',dict(passed=all(checks.values()),checks=checks,code_commit=commit(),
        size=list(SIZE),pipeline_steps=2,dtype='float32',device='CPU',
        scientific_result=False,scope='仅验证加载、尺寸和分支接口；正式GPU的DDIM50 P0仍为必需'))
    assert all(checks.values()),checks
    print('CNIPA_CPU_PREFLIGHT_PASSED',checks,flush=True)


def selected_ip():
    return Path(read(ROOT/'selection/ip_checkpoint.json')['path'])


def model_paths(arm):
    return (CN if arm=='C0' else selected_cn(),selected_ip() if arm=='C2' else None)


def smoke(arm='C0'):
    report=ROOT/'audit'/f'smoke_{arm}.json'
    if report.exists():assert read(report)['passed'];return
    stage('smoke_'+arm)
    pipe=rt.load(*model_paths(arm));rows=read(ROOT/'manifests/dev.json')[:4]
    folder=ROOT/'smoke'/arm
    records=[]
    for i,row in enumerate(rows):
        records.append(rt.generate(pipe,row,folder/f'normal_{i}.png'))
    first=rows[0];sketch,reference=rt.inputs(first);sketch2,reference2=rt.inputs(rows[1])
    def run(name,**kw):
        rt.generate(pipe,first,folder/(name+'.png'),**kw)
        return np.asarray(Image.open(folder/(name+'.png')).convert('RGB'))
    normal=np.asarray(Image.open(folder/'normal_0.png'))
    fake_gt=dict(first,gt='/THIS_GT_MUST_NEVER_BE_READ.png')
    rt.generate(pipe,fake_gt,folder/'repeat_with_unreadable_GT.png')
    repeat=np.asarray(Image.open(folder/'repeat_with_unreadable_GT.png'))
    reference_changed=run('different_reference',reference=reference2)
    sketch_changed=run('different_sketch',sketch=sketch2)
    text_changed=run('different_text',prompt='a red sleeveless short dress')
    ip0a=run('ip_zero_ref_a',ip_scale=0.)
    ip0b=run('ip_zero_ref_b',ip_scale=0.,reference=reference2)
    cn0a=run('cn_zero_sketch_a',cn_scale=0.)
    cn0b=run('cn_zero_sketch_b',cn_scale=0.,sketch=sketch2)
    delta=lambda a,b:float(np.abs(a.astype(float)-b.astype(float)).mean())
    checks=dict(repeat_deterministic=np.array_equal(normal,repeat),gt_not_read=np.array_equal(normal,repeat),
        reference_responds=delta(normal,reference_changed)>0,sketch_responds=delta(normal,sketch_changed)>0,
        text_responds=delta(normal,text_changed)>0,ip_off_reference_invariant=np.array_equal(ip0a,ip0b),
        cn_off_sketch_invariant=np.array_equal(cn0a,cn0b),size=normal.shape==(512,384,3))
    # 源权重的全部键、shape和转换后值均严格检查；不是仅按文件名推断格式。
    source=torch.load(IPA/'ip-adapter_sd15.bin',map_location='cpu',weights_only=False)
    loaded=rt.adapter_state(pipe.unet)
    if arm!='C2':
        for group in source:
            assert set(source[group])==set(loaded[group]),group
            for key,value in source[group].items():assert torch.equal(value.to(loaded[group][key].dtype),loaded[group][key]),key
    parameter_names=rt.trainable_parameters(pipe.unet)
    counts=dict(ipa=sum(p.numel() for p in parameter_names.values()),
        unet_base=sum(p.numel() for p in pipe.unet.parameters())-sum(p.numel() for p in parameter_names.values()),
        controlnet=sum(p.numel() for p in pipe.controlnet.parameters()),vae=sum(p.numel() for p in pipe.vae.parameters()),
        text_encoder=sum(p.numel() for p in pipe.text_encoder.parameters()),
        image_encoder=sum(p.numel() for p in pipe.image_encoder.parameters()))
    counts['total']=sum(counts.values())
    write(report,dict(passed=all(checks.values()),checks=checks,parameters=counts,
        deltas=dict(reference=delta(normal,reference_changed),sketch=delta(normal,sketch_changed),text=delta(normal,text_changed)),
        image_processor=pipe.feature_extractor.to_dict(),records=records,code_commit=commit(),
        lineart_direction='保持原始黑线/白底；不反相；开发集端点和定性拼图可查'))
    canvas=Image.new('RGB',(4*192,4*280),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        for j,(label,path) in enumerate([('Sketch',row['sketch']),('Reference',row['reference']),('GT evaluation only',row['gt']),
                (arm,str(folder/f'normal_{i}.png'))]):
            draw.text((j*192+3,i*280+3),label+' '+row['id'],fill='black')
            canvas.paste(Image.open(path).convert('RGB').resize((192,256)),(j*192,i*280+22))
    canvas.save(folder/'panel.png')
    assert all(checks.values()),checks
    rt.release(pipe)


def dev_run(pipe,folder,arm,cn_scale=1.,ip_scale=.65):
    scorer=Scorer();records=[]
    for row in read(ROOT/'manifests/dev.json'):
        path=folder/(row['id']+'.png')
        rt.generate(pipe,{k:v for k,v in row.items() if k!='gt'},path,cn_scale=cn_scale,ip_scale=ip_scale)
        records.append(scorer.score(row,path,arm))
        write(folder/'per_image.json',records)
        print('CNIPA_DEV',arm,len(records),row['id'],flush=True)
    write(folder/'means.json',means(records));del scorer;torch.cuda.empty_cache()
    return means(records)


def cn_select():
    if (ROOT/'selection/cn_checkpoint.json').exists():return
    stage('select_cn_checkpoint');scores=[]
    for step in CONFIG['cn_checkpoints']:
        path=ROOT/'checkpoints/cn_fashion'/f'checkpoint-{step}'/'controlnet'
        pipe=rt.load(path);folder=ROOT/'dev/cn_checkpoints'/str(step)
        score=dev_run(pipe,folder,f'CN_{step}',ip_scale=0.)
        scores.append(dict(step=step,path=str(path),**score,weights_sha256=sha(path/'diffusion_pytorch_model.safetensors')))
        rt.release(pipe)
    choice=min(scores,key=lambda r:(-r['contour_f1'],-r['iou'],r['leak'],r['lpips_gt'],r['step']))
    write(ROOT/'selection/cn_checkpoint_scores.json',scores);csv_write(ROOT/'selection/cn_checkpoint_scores.csv',scores)
    write(ROOT/'selection/cn_checkpoint.json',choice)


def grid(arm):
    result=ROOT/'selection'/f'scales_{arm}.json'
    if result.exists():return
    stage('scale_grid_'+arm);pipe=rt.load(*model_paths(arm))
    baseline=dev_run(pipe,ROOT/'dev'/arm/'cn_only',arm+'_CN_ONLY',ip_scale=0.)
    scores=[]
    for control in CONFIG['cn_scales']:
        for ip in CONFIG['ip_scales']:
            folder=ROOT/'dev'/arm/f'cn{control:.2f}_ip{ip:.2f}'
            score=dev_run(pipe,folder,arm,control,ip)
            key=selection_key(score,baseline)
            scores.append(dict(arm=arm,cn_scale=control,ip_scale=ip,eligible=key[0]==0,violation=key[1],**score))
            write(ROOT/'selection'/f'scale_scores_{arm}.json',scores)
    choice=min(scores,key=lambda r:(not r['eligible'],r['violation'],r['lpips_gt'],r['cn_scale'],r['ip_scale']))
    write(result,dict(choice=choice,cn_only=baseline,all=scores,rule=CONFIG['scale_selection']))
    csv_write(ROOT/'selection'/f'scale_scores_{arm}.csv',scores);rt.release(pipe)


def parity_and_gradients():
    from cn_ipa.train_ip import smoke_training
    smoke_training()


def cn_smoke():
    path=ROOT/'audit/cn_resume_smoke.json'
    if path.exists():assert read(path)['passed'];return
    stage('cn_resume_smoke')
    folder=ROOT/'smoke/CN_training'
    base=['bash','cn_ipa/job.sh','train_cn','--output_dir',str(folder),'--checkpointing_steps','1']
    if not (folder/'checkpoint-1').exists():
        subprocess.run(base+['--max_train_steps','1'],check=True)
    subprocess.run(base+['--max_train_steps','2','--resume_from_checkpoint','latest'],check=True)
    done=read(folder/'complete.json');assert done['updates']==2
    cursor=read(folder/'checkpoint-2/data_cursor.json')
    prior=read(folder/'checkpoint-1/data_cursor.json')
    assert cursor['samples_consumed']>=prior['samples_consumed']+4
    write(path,dict(passed=True,updates=2,resumed_from=1,next_batch=cursor['next_batch'],
        optimizer_RNG_scaler_saved=True,samples_consumed=cursor['samples_consumed'],
        prior_samples_consumed=prior['samples_consumed'],details=done,code_commit=commit(),scientific_training=False))


def train_cn():
    if (ROOT/'checkpoints/cn_fashion/complete.json').exists():return
    assert read(ROOT/'audit/cn_resume_smoke.json')['passed']
    stage('train_cn')
    command=['bash','cn_ipa/job.sh','train_cn']
    if list((ROOT/'checkpoints/cn_fashion').glob('checkpoint-*')):command+=['--resume_from_checkpoint','latest']
    subprocess.run(command,check=True)


def all_stages():
    from cn_ipa import train_ip
    if not (ROOT/'audit/sd15_official_validation.json').exists():validate_assets()
    versions=read(ROOT/'audit/versions.json');versions['gpu']=torch.cuda.get_device_name()
    write(ROOT/'audit/versions.json',versions)
    smoke('C0');cn_smoke();grid('C0')
    train_cn();cn_select();smoke('C1');parity_and_gradients()
    stage('train_ip');train_ip.train();stage('select_ip_checkpoint');train_ip.select()
    smoke('C2');grid('C1');grid('C2')
    freeze();test();evaluate()
    for arm in ['C0','C1','C2','M12','M12_GAIN']:
        subprocess.run([sys.executable,'-m','cn_ipa.run','profile','--arm',arm],check=True)
    finalize()


def profile(arm):
    result=ROOT/'profiles'/f'{arm}.json'
    if result.exists():return
    stage('profile_'+arm);examples=read(CONFIRM/'protocol/smoke_identities.json')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if arm in ['C0','C1','C2']:
        choice=read(ROOT/'selection/frozen_final_choice.json')['models'][arm]
        pipe=rt.load(choice['cn_path'],choice['ip_path']);records=[]
        rt.generate(pipe,examples[0],ROOT/'profiles'/arm/'warmup.png',42,choice['cn_scale'],choice['ip_scale'])
        for row in examples:
            records.append(rt.generate(pipe,row,ROOT/'profiles'/arm/(row['id']+'.png'),42,choice['cn_scale'],choice['ip_scale']))
        write(result,dict(arm=arm,n=4,records=records,parameters=read(ROOT/'audit'/f'smoke_{arm}.json')['parameters'],
            scope='native pipeline call, synchronized CUDA, excludes input loading/reference image setup/initial noise/PNG; includes native image encoder',
            gpu=torch.cuda.get_device_name(),dtype='float16',warmup=1,fresh_process=True))
        rt.release(pipe)
    else:
        # 历史模块从当前已锁定源码读取旧权重，所有运行输出重定向到本次ROOT。
        for name,target in [('output',HISTORY/'output'),('output_eval',HISTORY/'output_eval'),('data',HISTORY/'data'),('weight',HISTORY/'weight'),
                ('models/stable-diffusion-v1-5',SD),('models/clip',HISTORY/'models/clip')]:
            link=Path(name)
            if not link.exists():link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target,target_is_directory=True)
        from tools import m12_gain_confirm as legacy
        legacy.RUN=ROOT/'profiles/legacy'
        write(legacy.RUN/'protocol/smoke_identities.json',examples)
        runtime=legacy.runtime(arm,'runtime/'+arm)
        modules=list(runtime.modules.values())+([runtime.writer.unet] if runtime.writer else [])
        parameters={id(p):p for module in modules for p in module.parameters()}
        total=sum(p.numel() for p in parameters.values())
        try:legacy.profile(runtime,arm)
        finally:legacy.dispose(runtime)
        original=read(legacy.RUN/'production_profile'/f'{arm}_profile.json')
        original.update(gpu=torch.cuda.get_device_name(),dtype='float16',fresh_process=True,
            total_loaded_parameters=total,historical_training='E5/B1/Writer等历史成本另列，不能把新增gain1280参数当总成本')
        write(result,original)


def finalize():
    import csv
    import re
    import tarfile
    stage('finalize')
    records=read(ROOT/'metrics/per_image.json');ids=read(ROOT/'manifests/qualitative_ids.json')
    selected={r['id']:r for r in read(ROOT/'manifests/test_A.json')}
    groups=[('fixed',identity,42) for identity in ids]
    for arm in ['C0','C1','C2']:
        averaged=[r for r in read(ROOT/'metrics/per_identity.json') if r['variant']==arm and r['contour_f1'] is not None]
        worst=sorted(averaged,key=lambda r:(r['contour_f1'],r['id']))[:10]
        write(ROOT/'metrics'/f'worst10_{arm}.json',worst)
        for row in worst:
            seed=min((r for r in records if r['variant']==arm and r['id']==row['id'] and r.get('contour_f1') is not None),
                key=lambda r:r['contour_f1'])['seed']
            groups.append(('worst_'+arm,row['id'],int(seed)))
    for kind,identity,seed in groups:
        row=selected[identity];canvas=Image.new('RGB',(8*192,324),'white');draw=ImageDraw.Draw(canvas)
        pictures=[('Sketch',row['sketch']),('Reference',row['reference']),('GT',row['gt'])]
        pictures += [(arm,(ROOT if arm.startswith('C') else CONFIRM)/'outputs'/arm/identity/f'seed{seed}.png')
            for arm in ['C0','C1','C2','M12','M12_GAIN']]
        for index,(label,path) in enumerate(pictures):
            draw.text((index*192+4,3),label,fill='black')
            canvas.paste(Image.open(path).convert('RGB').resize((192,256)),(index*192,22))
        draw.text((4,282),f'{identity} seed {seed}: '+row['caption'],fill='black')
        path=ROOT/'qualitative'/kind/f'{identity}_seed{seed}.png';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    jobs=sorted({int(m[1]) for p in (ROOT/'resources').glob('*') if (m:=re.search(r'_(\d+)\.(?:log|err)$',p.name))})
    if jobs:
        raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(map(str,jobs)),
            '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
        (ROOT/'resources/accounting.psv').write_text(raw,encoding='utf-8')
        csv_write(ROOT/'resources/accounting.csv',list(csv.DictReader(raw.splitlines(),delimiter='|')))
    histories={}
    for name in ['cn_fashion','ip_fashion']:
        folder=ROOT/'checkpoints'/name
        histories[name]=dict(complete=read(folder/'complete.json'),
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines() if line])
    write(ROOT/'audit/training_runs.json',histories)
    budget={name:len(list((ROOT/name).rglob('*.png'))) for name in ['outputs','dev','smoke','profiles','qualitative']}
    budget.update(historical_test_images_reused=1728,
        dev_breakdown={p.name:len(list(p.rglob('*.png'))) for p in (ROOT/'dev').iterdir() if p.is_dir()},
        qualitative_is_panel_count=True)
    write(ROOT/'audit/image_budget.json',budget)
    historical=CONFIRM/'protocol/historical_cost_snapshot.json'
    if historical.exists():write(ROOT/'audit/historical_training_costs.json',read(historical))
    write(ROOT/'audit/finalized.json',dict(complete=True,code_commit=commit(),new_test_images=864,
        all_scored_images=2592,blind_human_review='NOT_RUN',new_test_B='NOT_RUN',
        training_diagnostics='CN断点恢复2次更新；IPA梯度烟测1次；均与正式预算分开'))
    archive=ROOT/'cnipa_results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for folder in ['audit','manifests','selection','metrics','outputs','smoke','profiles','qualitative','resources']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and path.suffix not in ['.bin','.pt','.safetensors']:
                    tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    stage('complete',archive=str(archive),sha256=sha(archive))


def freeze():
    path=ROOT/'selection/frozen_final_choice.json'
    if path.exists():return read(path)
    assert read(ROOT/'checkpoints/cn_fashion/complete.json')['updates']==5000
    assert read(ROOT/'checkpoints/ip_fashion/complete.json')['updates']==3000
    for arm in ['C0','C1','C2']:assert read(ROOT/'audit'/f'smoke_{arm}.json')['passed']
    result=dict(config=CONFIG,code_commit=commit(),models={},test_manifest_sha256=sha(ROOT/'manifests/test_A.json'))
    for arm in ['C0','C1','C2']:
        cn,ip=model_paths(arm)
        result['models'][arm]=dict(cn_path=str(cn),cn_sha256=sha(cn/'diffusion_pytorch_model.safetensors'),
            ip_path=str(ip or IPA/'ip-adapter_sd15.bin'),ip_sha256=sha(ip or IPA/'ip-adapter_sd15.bin'),
            **{k:read(ROOT/'selection'/f'scales_{arm}.json')['choice'][k] for k in ['cn_scale','ip_scale']})
    result['code_sha256']={str(p):sha(p) for p in Path('cn_ipa').glob('*') if p.is_file() and p.suffix in ['.py','.json','.sh']}
    write(path,result);return result


def test():
    frozen=read(ROOT/'selection/frozen_final_choice.json')
    assert commit()==frozen['code_commit']
    assert all(sha(p)==h for p,h in frozen['code_sha256'].items())
    assert sha(ROOT/'manifests/test_A.json')==frozen['test_manifest_sha256']
    for arm in ['C0','C1','C2']:
        stage('test_'+arm);choice=frozen['models'][arm]
        assert sha(Path(choice['cn_path'])/'diffusion_pytorch_model.safetensors')==choice['cn_sha256']
        assert sha(choice['ip_path'])==choice['ip_sha256']
        pipe=rt.load(choice['cn_path'],choice['ip_path'])
        for row in read(ROOT/'manifests/test_A.json'):
            for seed in CONFIG['seeds']:
                path=ROOT/'outputs'/arm/row['id']/f'seed{seed}.png'
                rt.generate(pipe,{k:v for k,v in row.items() if k!='gt'},path,seed,choice['cn_scale'],choice['ip_scale'])
            print('CNIPA_TEST',arm,row['id'],flush=True)
        rt.release(pipe)
    outputs={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'outputs').rglob('*.png')}
    assert len(outputs)==864
    write(ROOT/'audit/all_output_sha256.json',outputs)


def evaluate():
    assert len(read(ROOT/'audit/all_output_sha256.json'))==864
    stage('evaluate_test');scorer=Scorer();records=[];reuse=[]
    old=read(CONFIRM/'metrics/per_image.json')
    for row in read(ROOT/'manifests/test_A.json'):
        for seed in CONFIG['seeds']:
            for arm in ['C0','C1','C2']:
                path=ROOT/'outputs'/arm/row['id']/f'seed{seed}.png'
                assert sha(path)==read(path.with_suffix('.json'))['png_sha256']
                records.append(scorer.score(row,path,arm,seed))
            for arm in ['E5','B1','M10','M10_GAIN','M12','M12_GAIN']:
                path=CONFIRM/'outputs'/arm/row['id']/f'seed{seed}.png'
                metadata=read(path.with_suffix('.json'));assert sha(path)==metadata['png_sha256']
                measured=scorer.score(row,path,arm,seed)
                previous=next(r for r in old if r['variant']==arm and r['id']==row['id'] and int(r['seed'])==seed)
                deltas={k:abs(measured[k]-previous[k]) if measured.get(k) is not None and previous.get(k) is not None else None for k in KEYS}
                assert all(v is None or v<1e-6 for v in deltas.values()),('HISTORICAL_METRIC_MISMATCH',arm,row['id'],deltas)
                reuse.append(dict(id=row['id'],seed=seed,arm=arm,delta=deltas,png_sha256=sha(path)))
                records.append(measured)
        write(ROOT/'metrics/per_image.json',records)
        print('CNIPA_SCORE',row['id'],len(records),flush=True)
    assert len(records)==2592
    csv_write(ROOT/'metrics/per_image.csv',records)
    write(ROOT/'audit/historical_reuse.json',dict(passed=True,images=1728,checks=reuse,lpips=scorer.info,
        claim='相同原始输入、输出PNG校验和及四项评分复算；未把历史训练成本算作0'))
    statistics(records)
    stage('scientific_results_complete',new_images=864,all_scored_images=2592)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode');parser.add_argument('--arm',default='C0')
    args=parser.parse_args();require_audit()
    if args.mode=='assets':validate_assets()
    elif args.mode=='smoke':smoke(args.arm)
    elif args.mode=='grid':grid(args.arm)
    elif args.mode=='cn_select':cn_select()
    elif args.mode=='ip_smoke':parity_and_gradients()
    elif args.mode=='cn_smoke':cn_smoke()
    elif args.mode=='cpu_preflight':cpu_preflight()
    elif args.mode=='train_cn':train_cn()
    elif args.mode=='profile':profile(args.arm)
    elif args.mode=='finalize':finalize()
    elif args.mode=='all':all_stages()
    elif args.mode=='freeze':freeze()
    elif args.mode=='test':test()
    elif args.mode=='evaluate':evaluate()
    else:raise ValueError(args.mode)


if __name__=='__main__':main()
