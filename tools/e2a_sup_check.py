"""轻量算子验收；真实 VAE/可训练参数梯度由 GPU 的 TRAIN-CAL 验收。"""
import ast
import json
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
    loss=Supervision('T3',ns)
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
    return dict(passed=True,historical_T3_body_exact=True,merged_rest_normalization_correct=True,
        reference_swap_targets_correct=True,empty_masks_and_dropped_texture_finite=True,
        true_training_graph_checked_here=False,torch=torch.__version__)


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False))
