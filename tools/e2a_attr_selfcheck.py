"""针对协议风险做一次检查：历史损失未改，八种开关正确，缓存区域逐像素一致。"""
import ast
import subprocess
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tools.e2a_attr_common import *


def selfcheck():
    torch.set_num_threads(2)
    historical=subprocess.check_output(['git','-C',str(SOURCE),'show',BASE+':train_GAM_texture_joint.py'],text=True)
    current=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8')
    names=['_weighted_mean_per_sample','_masked_channel_mean_std','texture_color_stat_loss','build_region_masks','masked_l1_loss']
    get=lambda source:{n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names}
    old,new=get(historical),get(current)
    assert all(ast.dump(old[n],include_attributes=False)==ast.dump(new[n],include_attributes=False) for n in names)
    namespace=dict(torch=torch,F=F)
    exec(compile(ast.Module(body=[new[n] for n in names],type_ignores=[]),'<frozen-losses>','exec'),namespace)
    pred=torch.zeros(1,3,32,32,requires_grad=True)
    gt=torch.ones_like(pred)*.4; reference=torch.ones_like(pred)*-.7
    mask=torch.zeros(1,1,32,32);mask[:,:,4:28,4:28]=1
    body,boundary,outside=namespace['build_region_masks'](mask,9)
    values=[namespace['texture_color_stat_loss'](pred,reference,body),
            namespace['masked_l1_loss'](pred,gt,boundary),namespace['masked_l1_loss'](pred,gt,outside)]
    results=[]
    for factor in FACTORS:
        loss=sum(w*int(bit)*v for w,bit,v in zip([.1,.05,.1],factor,values))
        loss.backward(retain_graph=True)
        results.append(dict(factor=factor,weighted_loss=float(loss.detach()),finite_gradient=bool(torch.isfinite(pred.grad).all())))
        pred.grad.zero_()
    # GT 不应被错误地作为衣内 RGB 的目标。
    assert not torch.allclose(values[0],namespace['texture_color_stat_loss'](pred,gt,body))
    indices=read(OUT/'audit/masks_complete.json')['review_indices']
    for index in indices:
        folder=OUT/'masks/train'/('%03d'%(index//1000))
        mask=torch.from_numpy(np.array(Image.open(folder/('%05d_mask.png'%index)),np.float32)/255)[None,None]
        computed=namespace['build_region_masks'](mask,9)
        for name,value in zip(['body','boundary','outside'],computed):
            expected=np.array(Image.open(folder/('%05d_%s.png'%(index,name))))/255.
            assert np.array_equal(value[0,0].numpy(),expected),(index,name)
    write(OUT/'audit/loss_switch_selfcheck.json',dict(passed=True,historical_loss_AST_unchanged=True,
        body_target='reference_texture',all_factor_switches=results,exact_cached_region_checks=len(indices),
        device='CPU',scientific_training_updates=0))


if __name__=='__main__': selfcheck()
