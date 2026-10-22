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
    versions.update(python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    write(ROOT/'audit/versions.json',versions)
    paths=[p for p in SD.rglob('*') if p.is_file() and '.git' not in p.parts
        and p.suffix in ['.json','.bin','.safetensors','.txt'] and '.part' not in p.name]
    write(ROOT/'audit/base_model_file_sha256.json',{str(p):sha(p) for p in paths})
    write(ROOT/'audit/official_weight_validation.json',dict(passed=True,official_lfs_sha256=official,assets=assets))


def selected_cn():
    return Path(read(ROOT/'selection/cn_checkpoint.json')['path'])


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
    cursor=read(folder/'checkpoint-2/data_cursor.json');assert cursor['next_batch']==8
    write(path,dict(passed=True,updates=2,resumed_from=1,next_batch=cursor['next_batch'],
        optimizer_RNG_scaler_saved=True,details=done,code_commit=commit(),scientific_training=False))


def train_cn():
    if (ROOT/'checkpoints/cn_fashion/complete.json').exists():return
    assert read(ROOT/'audit/cn_resume_smoke.json')['passed']
    stage('train_cn')
    command=['bash','cn_ipa/job.sh','train_cn']
    if list((ROOT/'checkpoints/cn_fashion').glob('checkpoint-*')):command+=['--resume_from_checkpoint','latest']
    subprocess.run(command,check=True)


def all_stages():
    from cn_ipa import train_ip
    if not (ROOT/'audit/official_weight_validation.json').exists():validate_assets()
    smoke('C0');cn_smoke();grid('C0')
    train_cn();cn_select();smoke('C1');parity_and_gradients()
    stage('train_ip');train_ip.train();stage('select_ip_checkpoint');train_ip.select()
    smoke('C2');grid('C1');grid('C2')
    freeze();test();evaluate()


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
    elif args.mode=='train_cn':train_cn()
    elif args.mode=='all':all_stages()
    elif args.mode=='freeze':freeze()
    elif args.mode=='test':test()
    elif args.mode=='evaluate':evaluate()
    else:raise ValueError(args.mode)


if __name__=='__main__':main()
