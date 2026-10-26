"""CPU 工程回归；合成张量只验证实现，不能代替真实 GPU 校准结果。"""
import ast
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from tools.e2a_matched_common import SOURCE,helper_sources
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
    for path in list(SOURCE.glob('tools/e2a_matched*.py')):
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    return dict(passed=True,CPU_synthetic_only=True,body_exact_C=True,rest_union_exact=True,
        C_combined_gradient_compared_to_rest=True,calibration_pooling_64x8=True,
        no_C_region_double_count=True,no_parameter_update_or_main_RNG_change=True,
        single_7day_2GPU_allocation=True,active_job_reused=True,failed_job_review_required=True,
        sequential_dispatch_scenarios=len(expected),pilot_visual_review_gate_preserved=True,
        python38_syntax=True,actual_GPU_calibration_pending=True)


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False))
