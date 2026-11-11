"""M0/M1 同父权重、同 E2 损失、同随机流的 2000 更新筛选。"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from models.e2_compat import compute_losses, load_trainable_strict
from models.m1_patch_retrieval import attach_m1
from tools.e2_fft_experiment import SOURCE, setup, load, read, write, sha, environment
from tools.e2_fft_training import TrainingRun, dataset, tensor_hash, optimizer_hash, frozen_hash

ARMS = ('M0','M1')
PAIRS_SHA = 'c36330795b0702ea2153647e427986f1a32627757198e2168e4eb5593d00d2a0'


def m_environment():
    value=environment()
    for name in ['models/m1_patch_retrieval.py','tools/m1_experiment.py','tools/m1_evaluation.py',
                 'tests/test_m1_patch_retrieval.py','scripts/m1_experiment.sbatch']:
        value['source'][name]=sha(SOURCE/name)
    return value


def configuration(args, arm, smoke=False):
    out=Path(args.out)
    old=read(Path(args.baseline)/'comparison/B0.json')
    value=dict(old,arm=arm,code_commit=environment()['commit'],protocol='M1_S2T_PR_v1',
        fft=False,fft_weight=0.,updates=4 if smoke else 2000,effective_batch=8,
        world_size=1,workers=0 if smoke else 2,diagnostics=False,warmup=0 if smoke else 100,
        split=str(out/'split.json'),split_sha256=sha(out/'split.json'),
        training_input_map_sha256=sha(out/'audit/training_inputs.json'),
        m1=arm in ('M1','M1R'),reverse_film='pre_fusion_structure',attn_dim=128,num_heads=4,
        patch_stride=[2,2,1,1],attention_dropout=0.)
    value.pop('auxiliary_weights',None)
    if arm=='M1R':
        value.update(protocol='M1_residual_followup_v1',preserve_forward_film=True,residual_mask=False)
    return value


def prepare(args):
    from tools.e2_b_screen_training import audit_training
    from tools.e2_a_minimal import mask_snapshot
    from tools.e2_fft_experiment import prepare as regression_prepare
    core,out=setup(args)
    baseline=Path(args.baseline)
    assert read(baseline/'audit/training_inputs_final.json')['passed']
    audit_training(args)
    assert sha(out/'audit/training_inputs.json')==sha(baseline/'audit/training_inputs.json')
    mask_snapshot(args,core)
    regression_prepare(args)
    for name in ['strict_pairs.json','inputs/registry.json','audit/input_version_comparison.json']:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(baseline/name,dest)
    assert sha(out/'strict_pairs.json')==PAIRS_SHA
    pairs=read(out/'strict_pairs.json')['pairs']
    assert len(pairs)==len({p['identity_group'] for p in pairs})==32
    for pair in pairs:
        for key in ['sketch','gt','original_reference','swapped_reference']:
            source=Path(args.data)/pair[key]
            assert sha(source)==pair['hashes'][key]
            target=out/'inputs/frozen'/pair[key];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    for arm in ARMS:write(out/'comparison'/(arm+'.json'),configuration(args,arm))
    write(out/'protocol.json',dict(arms=ARMS,source=m_environment(),seed=42,
        training='both fresh from the identical mature E2 checkpoint; 2000 updates; batch8',
        only_intervention='texture-to-sketch FiLM replaced by patch cross attention',
        reverse_film='reads pre-fusion structure, identical dependency to legacy',
        distillation=False,primary='M1_minus_M0 gray_strict_response_change',
        primary_rule='paired identity 95% CI lower>0 and two-sided signflip p<.05',
        secondary_holm_family=3,protection_holm_family=13,
        inherited_exploratory_noninferiority=dict(edge_f1=.020,iou=.010,leak=.005,edge_l1=.002,lpips=.010,tcf_lab=1.),
        boundary_guard='radius2, oriented CI lower>=0; exploratory strict protection',
        missing='no imputation; paired complete cases, planned denominator32, failures explicit',
        failure_panels='all32 plus four worst oriented primary differences, no output-based sample selection',
        attention='fixed training probes and nine fixed queries; analysis masks only',
        scope='single training seed; previously inspected32, 23stripes/3checks/6dots; no confirmatory novelty claim'))
    print('M_PREPARE_PASS',flush=True)


def load_m(args,core,arm):
    pipe,network,checked=load(args,core)
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(args.device).clone()
    counts=attach_m1(network,enabled=arm in ('M1','M1R'),preserve_film=arm=='M1R')
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert torch.equal(cuda_rng,torch.cuda.get_rng_state(args.device))
    return pipe,network,dict(parent=checked,parameters=counts,attachment_rng_unchanged=True)


def m_frozen_hash(pipe,network):
    return dict(backbones=frozen_hash(pipe,network),adapter=tensor_hash(
        (n,p) for n,p in network.adapter.named_parameters() if not p.requires_grad))


class MonitoredRun(TrainingRun):
    def update(self):
        watched={};handles=[]
        if self.step<3 or (self.step+1)%100==0:
            def hook(name):
                def record(gradient):
                    term=gradient.detach().float().square().sum()
                    watched[name]=watched.get(name,0)+term
                return record
            for name,p in self.acc.unwrap_model(self.network).named_parameters():
                if '.s2t_patch_attn' in name and name.endswith('.weight') and any(
                    x in name for x in ['q_proj','patch_embed','kv_proj','out_proj']):
                    handles.append(p.register_hook(hook(name)))
        row=super().update()
        if watched:
            keys=list(watched)
            values=torch.stack([watched[n].sqrt() for n in keys]).cpu().tolist()
            row['m1_gradient_norms']=dict(zip(keys,values))
        for h in handles:h.remove()
        return row


def feature_calibration(args,core,pipe,network,data):
    out=Path(args.out)
    index=read(out/'split.json')['smoke'][0]['index']
    item=data[index]
    s=item['vae_sketch'][None].to(args.device);t=item['texture_image'][None].to(args.device)
    gray=(s*s.new_tensor([.2989,.5870,.1140])[None,:,None,None]).sum(1,keepdim=True)
    network.eval()
    with torch.no_grad():old=[x.detach().clone() for x in network.adapter.fusion(gray,t)]
    counts=attach_m1(network,True)
    with torch.no_grad():new=network.adapter.fusion(gray,t)
    rows=[]
    for a,b in zip(old,new):
        def summary(x):
            return dict(mean=float(x.mean()),std=float(x.std()),rms=float(x.square().mean().sqrt()),l2=float(x.norm()))
        rows.append(dict(shape=list(a.shape),old=summary(a),new=summary(b),
            cosine=float(torch.nn.functional.cosine_similarity(a.flatten()[None],b.flatten()[None])),
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())))
    assert [r['shape'] for r in rows]==[[1,288,64,48],[1,576,32,24],[1,1152,16,12],[1,1152,8,6]]
    assert all(r['finite'] for r in rows)
    write(out/'calibration/features.json',dict(index=index,scales=rows,parameters=counts,
        distillation=False,note='按批准方案直接同父权重筛选；分布偏移作为显式诊断记录，不隐含预训练'))
    print('FEATURE_CALIBRATION',rows,flush=True)


def smoke(args, arms=ARMS):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    core,out=setup(args)
    assert read(out/'regression/image_result.json')['passed']
    results={}
    for arm in arms:
        acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8)
        assert acc.num_processes==1
        args.device=str(acc.device)
        pipe,network,checked=load_m(args,core,arm)
        data=dataset(pipe,core)
        if arm=='M0':
            feature_calibration(args,core,pipe,network,data)
            # 校准只观察；正式冒烟重新加载共同父权重。
            network.adapter.unbind();del pipe,network,data;gc.collect();torch.cuda.empty_cache()
            pipe,network,checked=load_m(args,core,arm);data=dataset(pipe,core)
        before=tensor_hash(network.trainable_state().items())
        frozen=m_frozen_hash(pipe,network)
        style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
        config=configuration(args,arm,smoke=True)
        indices=[r['index'] for r in read(out/'split.json')['smoke']]*2
        run=MonitoredRun(args,core,pipe,network,data,style,config,indices,acc)
        folder=out/'smoke'/arm;folder.mkdir(parents=True,exist_ok=True)
        logs=[run.update(),run.update()]
        state2=tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
        opt2=optimizer_hash(run.optimizer);sched2=run.scheduler.state_dict()
        run.save(folder/'resume_test.pt')
        expected=run.update()
        expected_state=acc.unwrap_model(run.network).trainable_state()
        expected_hash=tensor_hash(expected_state.items())
        run.restore(folder/'resume_test.pt')
        assert state2==tensor_hash(acc.unwrap_model(run.network).trainable_state().items())
        assert opt2==optimizer_hash(run.optimizer) and sched2==run.scheduler.state_dict()
        replay=run.update()
        actual=acc.unwrap_model(run.network).trainable_state()
        errors={n:float((v-actual[n]).abs().max()) for n,v in expected_state.items()}
        assert all(torch.allclose(v,actual[n],atol=1e-7,rtol=1e-5) for n,v in expected_state.items())
        assert expected['losses']==replay['losses'] and expected['tokens']==replay['tokens']
        logs.extend([replay,run.update()])
        bare=acc.unwrap_model(run.network)
        assert m_frozen_hash(pipe,bare)==frozen
        assert before!=tensor_hash(bare.trainable_state().items())
        assert bare.adapter.injection_counts==[3,2,2,3]
        if arm in ('M1','M1R'):
            assert all(v>0 and np.isfinite(v) for v in logs[1]['m1_gradient_norms'].values())
            assert len(logs[1]['m1_gradient_norms'])==16
        run.save(folder/'latest.pt')
        results[arm]=dict(passed=True,logs=logs,initialization=checked,frozen_unchanged=True,
            injection_counts=bare.adapter.injection_counts,resume_parameter_max_abs=max(errors.values()),
            resume_bitwise_equal=expected_hash==tensor_hash(actual.items()),optimizer_restored_exactly=True,
            scheduler_restored_exactly=True,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        write(folder/'result.json',results[arm])
        bare.adapter.unbind()
        del expected_state,actual,bare,run,network,pipe,data,style
        acc.free_memory();gc.collect();torch.cuda.empty_cache()
        print('M_SMOKE_PASS',arm,flush=True)
    historical=read(Path(args.previous)/'smoke/result.json')['arms']['M0'] if arms==('M1R',) else results['M0']
    for result in results.values():
        assert [r['tokens'] for r in result['logs']]==[r['tokens'] for r in historical['logs']]
    write(out/'smoke/result.json',dict(passed=True,same_random_stream=True,arms=results,environment=m_environment()))


def train(args):
    from accelerate import Accelerator
    from train_GAM_texture_joint import VGGGramStyleLoss
    from tools.e2_b_screen_training import assert_snapshot
    core,out=setup(args)
    assert read(out/'smoke/result.json')['passed']
    config=read(out/'comparison'/(args.arm+'.json'))
    assert config==configuration(args,args.arm)
    assert_snapshot(args)
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=8)
    assert acc.num_processes==1
    args.device=str(acc.device)
    pipe,network,checked=load_m(args,core,args.arm)
    frozen=m_frozen_hash(pipe,network)
    data=dataset(pipe,core);style=VGGGramStyleLoss().to(args.device).eval().requires_grad_(False)
    run=MonitoredRun(args,core,pipe,network,data,style,config,read(out/'split.json')['train_indices'],acc)
    folder=out/'training'/args.arm;folder.mkdir(parents=True,exist_ok=True)
    log=folder/'training.jsonl'
    if (folder/'latest.pt').exists():
        run.restore(folder/'latest.pt')
        # 作业中断后只保留已有 checkpoint 覆盖的更新，避免重复计数。
        rows=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
        log.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['step']<=run.step),encoding='utf-8')
    while run.step<config['updates']:
        row=run.update()
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if run.step<4 or run.step%25==0:print('M_TRAIN',args.arm,run.step,row['losses'],row['seconds'],flush=True)
        if run.step%100==0 or run.step==config['updates']:run.save(folder/'latest.pt')
    assert m_frozen_hash(pipe,acc.unwrap_model(run.network))==frozen
    write(folder/'complete.json',dict(status='completed',updates=run.step,config=config,initialization=checked,
        checkpoint_sha256=sha(folder/'latest.pt'),frozen_unchanged=True,environment=m_environment()))


def final_audit(args):
    from tools.e2_b_screen_training import audit_training
    from tools.e2_a_minimal import mask_snapshot
    core,out=setup(args)
    audit_training(args,final=True);mask_snapshot(args,core,final=True)
    from tools.m1_evaluation import training_audit
    training_audit(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','regression','generate-regression','smoke','train','audit-final',
        'generate','evaluate','diagnose','bundle','status'])
    p.add_argument('--project',default='/share/home/u2515283058/Mymodel')
    p.add_argument('--parent',default='/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106')
    p.add_argument('--baseline',default='/share/home/u2515283058/Mymodel/output_eval/e2_b_screen_20261108')
    p.add_argument('--out',default='/share/home/u2515283058/Mymodel/output_eval/m1_s2t_patch_20261111')
    p.add_argument('--data',default='/share/home/u2515283058/datasets/BF')
    p.add_argument('--candidates',default=str(SOURCE/'configs/e2_b_candidates.json'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    if args.action in ['regression','generate-regression']:
        from tools import e2_fft_experiment as legacy
        getattr(legacy,args.action.replace('-','_'))(args)
    elif args.action=='audit-final':final_audit(args)
    elif args.action in ['generate','evaluate','diagnose','bundle','status']:
        from tools import m1_evaluation
        getattr(m1_evaluation,args.action)(args)
    else:globals()[args.action](args)


if __name__=='__main__':main()
