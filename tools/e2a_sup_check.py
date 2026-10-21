"""轻量算子验收；真实 VAE/可训练参数梯度由 GPU 的 TRAIN-CAL 验收。"""
import ast
import json
import weakref
import torch
import torch.nn.functional as F
from tools.e2a_sup_common import SOURCE
from tools.e2a_sup_losses import Supervision


def check():
    names=['_weighted_mean_per_sample','_masked_channel_mean_std','texture_color_stat_loss','build_region_masks','masked_l1_loss']
    tree=ast.parse((SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8'))
    ns=dict(torch=torch,F=F)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),
        '<historical-helpers>','exec'),ns)
    pred=torch.linspace(-.8,.9,3*64*48).reshape(1,3,64,48).requires_grad_()
    gt=torch.ones_like(pred)*.3;ref=torch.ones_like(pred)*-.6
    mask=torch.zeros(1,1,64,48);mask[:,:,8:56,10:38]=1
    body,boundary,outside=ns['build_region_masks'](mask,9)
    assert (body+boundary+outside-1).abs().max()<1e-5
    sentinel=torch.ones(1,requires_grad=True)*2
    reference_to_batch=weakref.ref(sentinel)
    ns['current_batch_tensor']=sentinel
    loss=Supervision('T3',ns)
    del ns['current_batch_tensor'],sentinel
    assert reference_to_batch() is None, 'Supervision retained a training batch graph'
    raw=loss.raw(pred,gt,ref,mask,torch.tensor([0]),torch.tensor([1.]))
    changed=loss.raw(pred,gt,-ref,mask,torch.tensor([0]),torch.tensor([1.]))
    assert torch.equal(raw['rest'],changed['rest']) and not torch.equal(raw['body_rgb'],changed['body_rgb'])
    assert torch.equal(raw['body_rgb'],ns['texture_color_stat_loss'](pred,ref,body,torch.tensor([1.])))
    merged=(ns['masked_l1_loss'](pred,gt,boundary)*boundary.sum()+
        ns['masked_l1_loss'](pred,gt,outside)*outside.sum())/(boundary.sum()+outside.sum())
    assert torch.allclose(raw['rest'],merged)
    for value in raw.values():
        grad=torch.autograd.grad(value,pred,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
    for fill in [0,1]:
        raw=loss.raw(pred,gt,ref,torch.full_like(mask,fill),torch.tensor([999]),torch.tensor([0.]))
        assert all(torch.isfinite(v) for v in raw.values())
    # 模拟 retain_graph 测量后释放计算图；不改变参数、optimizer state 或 RNG。
    layer=torch.nn.Conv2d(3,3,1)
    optimizer=torch.optim.AdamW(layer.parameters())
    before=[p.detach().clone() for p in layer.parameters()]
    rng=torch.get_rng_state().clone()
    for _ in range(3):
        output=layer(pred.detach())
        raw=loss.raw(output,gt,ref,mask,torch.tensor([0]),torch.tensor([1.]))
        for value in raw.values():
            grads=torch.autograd.grad(value,[output,*layer.parameters()],retain_graph=True)
            assert all(torch.isfinite(g).all() for g in grads)
        output.square().mean().backward()
        optimizer.zero_grad(set_to_none=True)
    assert not optimizer.state and all(p.grad is None for p in layer.parameters())
    assert all(torch.equal(a,b) for a,b in zip(before,layer.parameters()))
    assert torch.equal(rng,torch.get_rng_state())
    return dict(passed=True,historical_T3_body_exact=True,merged_rest_normalization_correct=True,
        reference_swap_targets_correct=True,empty_masks_and_dropped_texture_finite=True,
        batch_graph_not_retained=True,calibration_graph_release_no_update_or_rng_change=True,
        true_training_graph_checked_here=False,torch=torch.__version__)


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False))
