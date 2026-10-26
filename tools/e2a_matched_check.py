"""CPU 工程回归；合成张量只验证实现，不能代替真实 GPU 校准结果。"""
import ast
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from tools.e2a_matched_common import SOURCE,helper_sources,model_path,sha
from tools.e2a_matched_train import MatchedSupervision,select_gamma


def check():
    ns=dict(torch=torch,F=F)
    for name,text in helper_sources().items():exec(text,ns)
    loss=MatchedSupervision('T3M',ns)
    p=torch.linspace(-.8,.9,3*64*48).reshape(1,3,64,48).requires_grad_()
    gt=torch.ones_like(p)*.3;ref=torch.ones_like(p)*-.6
    mask=torch.zeros(1,1,64,48);mask[:,:,8:56,10:38]=1
    body,boundary,outside=ns['build_region_masks'](mask,9)
    a=loss.raw(p,gt,ref,mask,torch.tensor([0]),torch.ones(1))
    b=loss.raw(p,gt,-ref,mask,torch.tensor([999]),torch.ones(1))
    assert torch.equal(a['body_rgb'],ns['texture_color_stat_loss'](p,ref,body,torch.ones(1)))
    assert not torch.equal(a['body_rgb'],b['body_rgb'])
    assert torch.equal(a['rest'],b['rest']) and torch.equal(a['c_region'],b['c_region'])
    assert torch.equal(1-body,boundary+outside)
    assert torch.equal(a['c_region'],.05*ns['masked_l1_loss'](p,gt,boundary)+.1*ns['masked_l1_loss'](p,gt,outside))
    rest_grad=torch.autograd.grad(a['rest'],p,retain_graph=True)[0]
    C_grad=torch.autograd.grad(a['c_region'],p,retain_graph=True)[0]
    # 分区域归一化使梯度幅度通常不同，不能把 gamma 硬写为 .05 或 .1。
    ratio=float(C_grad.square().mean().sqrt()/rest_grad.square().mean().sqrt())
    rows=[[dict(c_region_rms=ratio*(i+1),rest_rms=i+1) for i in range(256)] for _ in range(2)]
    gamma,batches=select_gamma(rows)
    assert abs(gamma-ratio)<1e-10 and len(batches)==64
    assert all(abs(r['matched_ratio']-1)<1e-12 and r['samples']==8 for r in batches)
    objective=.1*a['body_rgb']+gamma*a['rest']
    from tools.e2a_sup_losses import weighted
    assert torch.equal(objective,weighted(a,dict(body_rgb=.1,rest=gamma)))
    drop=loss.raw(p,gt,ref,mask,torch.tensor([0]),torch.zeros(1))
    assert drop['body_rgb'].item()==0 and torch.equal(drop['rest'],a['rest'])
    for fill in [0,1]:
        raw=loss.raw(p,gt,ref,torch.full_like(mask,fill),torch.tensor([0]),torch.ones(1))
        assert all(torch.isfinite(v).item() for v in raw.values())
    layer=torch.nn.Conv2d(3,3,1);optimizer=torch.optim.AdamW(layer.parameters())
    before=[v.detach().clone() for v in layer.parameters()]
    rng=torch.get_rng_state().clone()
    for _ in range(3):
        output=layer(p.detach());raw=loss.raw(output,gt,ref,mask,torch.tensor([0]),torch.ones(1))
        for value in raw.values():
            grad=torch.autograd.grad(value,[output,*layer.parameters()],retain_graph=True)
            assert all(torch.isfinite(g).all().item() for g in grad)
        output.square().mean().backward();optimizer.zero_grad(set_to_none=True)
    assert not optimizer.state and all(torch.equal(v,b) for v,b in zip(layer.parameters(),before))
    assert torch.equal(rng,torch.get_rng_state())
    # 复现单进程的 cuda:None 元数据；仅模拟设备索引，不生成虚构 GPU 通过证据。
    from types import SimpleNamespace
    from tools.e2a_matched_smoke import SingleGPUHook
    class FakeAccelerator:
        process_index=0;num_processes=1
        def __init__(self):
            self.state=SimpleNamespace(device=torch.device('cuda'))
            self.scaler=SimpleNamespace(state_dict=lambda:{'scale':128})
        @property
        def device(self):return self.state.device
    with tempfile.TemporaryDirectory() as temporary:
        run=Path(temporary);acc=FakeAccelerator();model=torch.nn.Linear(2,2)
        opt=torch.optim.AdamW(model.parameters())
        env=dict(accelerator=acc,runtime=object(),optimizer=SimpleNamespace(optimizer=opt),
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1),
            args=SimpleNamespace(gam_init_ckpt=str(model_path('E1')),
                start_global_step=0,resume_from_checkpoint='',seed=42,train_batch_size=1,gradient_accumulation_steps=4),
            noise_scheduler=SimpleNamespace(config=SimpleNamespace(prediction_type='epsilon')),
            vae=torch.nn.Linear(2,2).requires_grad_(False),trainable_params=list(model.parameters()))
        env.update({k:model for k in ['unet','ref_unet','bf','spatial_texture_encoder','spatial_injection']})
        with patch('torch.cuda.device_count',return_value=1),patch('torch.cuda.current_device',return_value=0):
            SingleGPUHook('T3M','calibrate',run).setup(env)
        assert acc.device==torch.device('cuda',0)
    # Windows 没有 fcntl，调度测试只替换此平台互斥接口，不模拟 GPU 结果。
    import sys,types
    if sys.platform=='win32':sys.modules.setdefault('fcntl',types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2))
    from tools import e2a_matched_dispatch as dispatch
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);(root/'logs').mkdir()
        with patch.object(dispatch,'ROOT',root),patch.object(dispatch.reused,'ROOT',root),patch.object(dispatch,'commit',return_value='test'),patch.object(dispatch.subprocess,'check_output',return_value='123;cluster') as submit:
            job=dispatch.submit('formal_T3M','train',['--arm','T3M','--phase','formal'],2)
            assert '--time=7-00:00:00' in job['command'] and '--gres=gpu:2' in job['command']
            assert '--cpus-per-task=8' in job['command']
            with patch.object(dispatch,'job_state',return_value='RUNNING'):
                assert dispatch.submit('formal_T3M','train',[],2)==job
            assert submit.call_count==1
            with patch.object(dispatch,'job_state',return_value='FAILED'):
                try:dispatch.submit('formal_T3M','train',[],2)
                except AssertionError:pass
                else:raise AssertionError('Unreviewed failed run was resubmitted')
            smoke=dispatch.submit('single_gpu_graph_test','smoke',gpus=1)
            assert '--gres=gpu:1' in smoke['command'] and '--cpus-per-task=4' in smoke['command']
            assert '--time=00:30:00' in smoke['command']
    # 校准→pilot→真实图审→正式→评价的逐阶段保护，不能跳过工程图审。
    from tools.e2a_matched_common import write
    expected=[('prepare','prepare'),('calibrate_T3M','train'),('pilot_T3M','train'),
        ('pilot_eval_T3M','eval'),(None,None),('formal_T3M','train'),
        ('P2a_T3M','eval'),('P2b_T3M','eval'),('reference_swap_T3M','eval')]
    for stage,(key,action) in enumerate(expected):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            if stage>=1:write(root/'audit/preparation_complete.json',dict(passed=True))
            if stage>=2:
                write(root/'training/calibrate/T3M/calibration_complete.json',dict(passed=True))
                write(root/'gamma_calibration.json',dict(used_P2_or_LOCK=False))
            if stage>=3:write(root/'training/pilot/T3M/checkpoint_verification.json',dict(passed=True))
            if stage>=4:write(root/'evaluation/pilot_P2a/T3M/evaluation_complete.json',dict(complete=True))
            if stage>=5:write(root/'audit/pilot_release_T3M.json',dict(passed=True))
            if stage>=6:write(root/'training/formal/T3M/checkpoint_verification.json',dict(passed=True))
            if stage>=7:write(root/'evaluation/P2a/T3M/evaluation_complete.json',dict(complete=True))
            if stage>=8:write(root/'evaluation/P2b/T3M/evaluation_complete.json',dict(complete=True))
            with patch.object(dispatch,'ROOT',root),patch.object(dispatch,'folder',side_effect=lambda arm,phase:root/'training'/phase/arm),patch.object(dispatch,'accounting'),patch.object(dispatch,'training_audit'),patch.object(dispatch,'watch'),patch.object(dispatch,'status') as state,patch.object(dispatch,'submit',return_value={'id':1}) as submit:
                dispatch.advance()
                if key is None:
                    submit.assert_not_called();assert state.call_args.kwargs['stage']=='PILOT_REVIEW_REQUIRED_T3M'
                else:assert submit.call_count==1 and submit.call_args.args[:2]==(key,action)
    for complete in [False,True]:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write(root/'audit/preparation_complete.json',dict(passed=True))
            write(root/'audit/single_gpu_test_request.json',dict(passed=True))
            if complete:write(root/'audit/single_gpu_test_complete.json',dict(passed=True,optimizer_updates=0))
            with patch.object(dispatch,'ROOT',root),patch.object(dispatch,'folder',side_effect=lambda arm,phase:root/'training'/phase/arm),patch.object(dispatch,'accounting'),patch.object(dispatch,'watch'),patch.object(dispatch,'status'),patch.object(dispatch,'submit',return_value={'id':1}) as submit:
                dispatch.advance()
                assert submit.call_count==1
                if complete:assert submit.call_args.args[:2]==('calibrate_T3M','train')
                else:assert submit.call_args.args[:2]==('single_gpu_graph_test','smoke') and submit.call_args.kwargs['gpus']==1
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);(root/'logs').mkdir()
        write(root/'jobs.json',[dict(id=123,key='calibrate_T3M')])
        release=root/'audit/resubmit_release_123.json'
        write(release,dict(passed=True,replaced_job=123,key='calibrate_T3M',source_commit='old'))
        write(root/'audit/single_gpu_test_request.json',dict(source_commit='old',replaced_calibration_job=123,
            scheduling_release_sha256=sha(release)))
        write(root/'audit/preparation_complete.json',dict(source_sha256={
            'tools/e2a_matched_train.py':sha(SOURCE/'tools/e2a_matched_train.py')}))
        with patch.object(dispatch,'ROOT',root),patch.object(dispatch.reused,'ROOT',root),patch.object(dispatch,'commit',return_value='fixed'),patch.object(dispatch,'job_state',return_value='CANCELLED'),patch.object(dispatch.subprocess,'check_output',return_value='456'):
            for accepted in [False,True]:
                write(root/'audit/single_gpu_test_complete.json',dict(passed=True,source_commit='fixed' if accepted else 'stale'))
                if accepted:assert dispatch.submit('calibrate_T3M','train',[],2)['id']==456
                else:
                    try:dispatch.submit('calibrate_T3M','train',[],2)
                    except AssertionError:pass
                    else:raise AssertionError('Stale single-GPU result released calibration')
    for path in list(SOURCE.glob('tools/e2a_matched*.py')):
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    return dict(passed=True,CPU_synthetic_only=True,body_exact_C=True,rest_union_exact=True,
        C_combined_gradient_compared_to_rest=True,calibration_pooling_64x8=True,
        no_C_region_double_count=True,no_parameter_update_or_main_RNG_change=True,
        single_7day_2GPU_allocation=True,active_job_reused=True,failed_job_review_required=True,
        sequential_dispatch_scenarios=len(expected),pilot_visual_review_gate_preserved=True,
        optional_single_GPU_test_before_paired_calibration=True,single_GPU_test_limit_minutes=30,
        single_process_cuda_index_None_regression=True,
        current_source_single_GPU_pass_required_after_repair=True,
        python38_syntax=True,actual_GPU_calibration_pending=True)


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False))
