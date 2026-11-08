"""B0/B1/B2/B4/B5同预算筛选；每臂单卡，可使用多张空闲卡并行不同臂。"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.e2_compat import compute_losses, e2_objective, load_trainable_strict
from models.e2_local_fft import LocalFFTLoss
from models.e2_local_texture import DescriptorLocalLoss, GaborDescriptor, ReferenceGramDescriptor
from tools.e2_fft_experiment import SOURCE, read, write, sha, setup, load, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash, optimizer_hash, frozen_hash, parameter_changes

ARMS = ['B0', 'B1', 'B2', 'B4', 'B5']


def audit_training(args, final=False):
    core,out=setup(args)
    source=core.ROOT/'training_manifest.json'
    rows=read(source)
    data=Path(args.project).parent/'datasets/BF/training'
    names=sorted({r[k] for r in rows for k in ['cloth','sketch','texture']})
    def inspect(name):
        p=data/name;s=p.stat()
        return name,dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
    with ThreadPoolExecutor(max_workers=4) as pool:
        files=dict(pool.map(inspect,names))
    value=dict(source_manifest_sha256=sha(source),training_rows=len(rows),files=files)
    path=out/'audit/training_inputs.json'
    if final:
        previous=read(path)
        assert value==previous,'训练输入版本改变，禁止按同预算结果比较'
        write(out/'audit/training_inputs_final.json',dict(passed=True,files=len(files),initial_sha256=sha(path)))
    elif path.exists():
        assert read(path)==value,'不覆盖初始训练输入版本'
    else:
        write(path,value)
    split=Path(args.parent)/'split.json'
    dest=out/'split.json'
    if dest.exists():assert dest.read_bytes()==split.read_bytes()
    else:dest.write_bytes(split.read_bytes())
    print('B_TRAIN_INPUTS',len(rows),len(files),sha(path),'FINAL',final,flush=True)


def assert_snapshot(args):
    out=Path(args.out);snapshot=read(out/'audit/training_inputs.json')
    root=Path(args.project).parent/'datasets/BF/training'
    for name,row in snapshot['files'].items():
        s=(root/name).stat()
        assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']), '训练文件版本改变: '+name
    return sha(out/'audit/training_inputs.json')


def configurations(args, updates=2000, diagnostics=False):
    out=Path(args.out);calibration=read(out/'calibration/result.json')
    assert calibration['passed']
    old=read(Path(args.parent)/'comparison/E2.json')
    fft_weight=read(Path(args.parent)/'comparison/E2_FFT.json')['fft_weight']
    weights=calibration['weights']
    result={}
    for arm in ARMS:
        config=dict(old,arm=arm,code_commit=environment()['commit'],updates=updates,workers=0 if diagnostics else 2,
            diagnostics=diagnostics,warmup=0 if diagnostics else 100,
            split=str(out/'split.json'),split_sha256=sha(out/'split.json'),
            protocol='matched_B_screen_v3',world_size=1,
            training_input_map_sha256=sha(out/'audit/training_inputs.json'),
            fft=arm=='B2',fft_weight=fft_weight if arm=='B2' else 1. if arm in ['B1','B4','B5'] else 0.)
        if arm=='B1':config['auxiliary_weights']={'gram':weights['gram']}
        if arm=='B4':config['auxiliary_weights']={'gabor':weights['gabor']}
        if arm=='B5':config['auxiliary_weights']={'fft':fft_weight,'gabor':weights['gabor']}
        result[arm]=config
    return result


def calibrate(args):
    from train_GAM_texture_joint import VGGGramStyleLoss
    core,out=setup(args)
    assert read(out/'regression/forward_result.json')['passed']
    assert read(out/'regression/image_result.json')['passed']
    assert_snapshot(args)
    pipe,network,_=load(args,core)
    network.train()
    data=dataset(pipe,core)
    style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    old_weight=read(Path(args.parent)/'comparison/E2_FFT.json')['fft_weight']
    auxiliary={'fft':LocalFFTLoss(),'gabor':DescriptorLocalLoss(GaborDescriptor()),
               'gram':DescriptorLocalLoss(ReferenceGramDescriptor(style))}
    rng=torch.Generator(device=args.device).manual_seed(42000)
    ratios={k:[] for k in auxiliary};records=[]
    for choice in read(out/'split.json')['smoke']:
        batch={k:v[None].to(args.device) for k,v in data[choice['index']].items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,raw,token,captured,_=compute_losses(pipe,network,batch,rng,style,core,audit_mode=True)
        targets=[captured['epsilon'],captured['vae_input'],network.adapter.projections[0].weight,
                 network.adapter.fusion.downconv1[0].weight,next(p for p in network.unet.parameters() if p.requires_grad)]
        names=['epsilon','vae_input','projection0','fusion_down1','GAM']
        losses={'old':e2_objective(raw),'old_gt_gram':raw['gram']};stats={}
        for name,term in auxiliary.items():
            losses[name],stats[name]=term(captured['decoded'],batch['texture_image'],batch['garment_mask'],
                captured['timesteps'],batch['dataset_index'],reference_keep=True)
        norms={};epsilon={}
        for name,loss in losses.items():
            gradients=torch.autograd.grad(loss,targets,retain_graph=True,allow_unused=True)
            norms[name]={k:float(v.detach().float().norm()) if v is not None else 0. for k,v in zip(names,gradients)}
            assert all(np.isfinite(v) for v in norms[name].values())
            epsilon[name]=gradients[0].detach().float().flatten()
        cosines={}
        for name in auxiliary:
            if stats[name]['valid_n']:
                assert all(v>0 for v in norms[name].values()), '新增有效损失没有完整梯度路径: '+name
                ratios[name].append(norms['old']['epsilon']/norms[name]['epsilon'])
                cosines[name]={target:float(F.cosine_similarity(epsilon[name][None],epsilon[target][None]))
                               for target in ['old','old_gt_gram']}
        records.append(dict(index=choice['index'],token=token,gradients=norms,gradient_cosines=cosines,
            losses={k:float(v.detach()) for k,v in losses.items()},patch_stats=stats))
        network.adapter.clear()
        write(out/'calibration/progress.json',dict(records=records))
        del captured,raw,losses,epsilon,gradients,batch
        print('B_CALIBRATE',choice['index'],norms,flush=True)
    assert all(len(v)>=2 for v in ratios.values()), '新增损失至少需要两个有效诊断参考'
    weights={name:.1*float(np.median(value)) for name,value in ratios.items()}
    weights['fft']=old_weight
    write(out/'calibration/result.json',dict(passed=True,weights=weights,records=records,
        target_new_term_epsilon_gradient_ratio=.1,fft_weight_preserved=True,
        gram_preprocess='RGB linear[-1,1] to[0,1], ImageNet channel normalization, native patch size; old GT Gram unchanged',
        gabor='8 carrier directions; wavelength4/8 forq32 plus16 forq64; sigma=.5lambda; radius3sigma; joint complex L2; valid convolution',
        environment=environment(),optimizer_updates=0,peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('B_WEIGHTS',weights,flush=True)


def smoke(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core,out=setup(args);assert_snapshot(args)
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8)
    assert acc.num_processes==1
    args.device=str(acc.device)
    pipe,network,_=load(args,core)
    before={k:v.clone() for k,v in network.trainable_state().items()}
    frozen_before=frozen_hash(pipe,network)
    data=dataset(pipe,core);style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    indices=[r['index'] for r in read(out/'split.json')['smoke']]*2
    results={}
    for arm,config in configurations(args,updates=4,diagnostics=False).items():
        config.update(workers=0,warmup=0)
        load_trainable_strict(network,before)
        torch.cuda.reset_peak_memory_stats()
        run=TrainingRun(args,core,pipe,network,data,style,config,indices,acc)
        folder=out/'smoke'/arm;folder.mkdir(parents=True,exist_ok=True)
        logs=[run.update(),run.update()]
        run.save(folder/'resume_test.pt')
        weights2=tensor_hash(network.trainable_state().items());opt2=optimizer_hash(run.optimizer)
        scheduler2=run.scheduler.state_dict()
        expected=run.update();weights3=network.trainable_state();hash3=tensor_hash(weights3.items())
        run.restore(folder/'resume_test.pt')
        assert tensor_hash(network.trainable_state().items())==weights2
        assert optimizer_hash(run.optimizer)==opt2 and run.scheduler.state_dict()==scheduler2
        replay=run.update();actual=network.trainable_state();actual_hash=tensor_hash(actual.items())
        errors={k:float((v-actual[k]).abs().max()) for k,v in weights3.items()}
        failed=[k for k,v in weights3.items() if not torch.allclose(v,actual[k],atol=1e-7,rtol=1e-5)]
        write(folder/'resume_comparison.json',dict(passed=not failed,bitwise_equal=hash3==actual_hash,
            max_abs=max(errors.values()),failed=failed,expected=expected,replay=replay))
        assert not failed and expected['losses']==replay['losses'] and expected['tokens']==replay['tokens']
        logs += [replay,run.update()]
        changes=parameter_changes(before,network)
        assert all(v>0 and np.isfinite(v) for v in changes.values())
        assert frozen_hash(pipe,network)==frozen_before
        if arm!='B0':assert sum(r['patch_counts']['valid_n'] for r in logs)>0
        run.save(folder/'latest.pt')
        results[arm]=dict(passed=True,updates=run.step,effective_batch=8,executed_update_calls=5,
            logs=logs,changes_l2=changes,resume_exact=hash3==actual_hash,frozen_unchanged=True,
            checkpoint_sha256=sha(folder/'latest.pt'))
        write(folder/'result.json',results[arm])
        network=acc.unwrap_model(run.network,keep_fp32_wrapper=False)
        del run,weights3,actual
        acc.free_memory()
        print('B_SMOKE_PASS',arm,flush=True)
    assert all([r['tokens'] for r in result['logs']]==[r['tokens'] for r in results['B0']['logs']] for result in results.values())
    write(out/'smoke/result.json',dict(passed=True,same_legacy_stream=True,arms=results,environment=environment()))
    for arm,config in configurations(args).items():write(out/'comparison'/(arm+'.json'),config)


def train(args):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from train_GAM_texture_joint import VGGGramStyleLoss
    core,out=setup(args)
    assert read(out/'smoke/result.json')['passed']
    assert read(out/'strict_pairs.json')['sufficient_for_planned_screening']
    config=read(out/'comparison'/(args.arm+'.json'))
    assert config['code_commit']==environment()['commit']
    assert assert_snapshot(args)==config['training_input_map_sha256']
    assert sha(config['split'])==config['split_sha256']
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    assert acc.num_processes==config['world_size']==1
    args.device=str(acc.device)
    pipe,network,_=load(args,core);data=dataset(pipe,core)
    style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    folder=out/'training'/args.arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        complete=read(folder/'complete.json')
        assert complete['config']==config and sha(folder/'latest.pt')==complete['checkpoint_sha256']
        print('B_ALREADY_COMPLETED',args.arm,flush=True);return
    run=TrainingRun(args,core,pipe,network,data,style,config,read(config['split'])['train_indices'],acc)
    if (folder/'latest.pt').exists():run.restore(folder/'latest.pt')
    log=folder/'training.jsonl'
    if log.exists():
        # 未落盘的更新在恢复时重放，剔除checkpoint之后的旧日志，避免重复计数。
        rows=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
        rows=[r for r in rows if r['step']<=run.step]
        log.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    while run.step<config['updates']:
        row=run.update()
        with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if run.step<4 or run.step%25==0:print('B_TRAIN',args.arm,run.step,row['losses'],flush=True)
        if run.step%100==0 or run.step==config['updates']:run.save(folder/'latest.pt')
    write(folder/'complete.json',dict(status='completed',updates=run.step,config=config,
        environment=environment(),checkpoint_sha256=sha(folder/'latest.pt')))
