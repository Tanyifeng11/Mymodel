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
    historical=subprocess.check_output(['git','-C',str(SOURCE),'show',BASE+':train_GAM_texture_joint.py'],text=True,encoding='utf-8')
    current=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8')
    names=['_weighted_mean_per_sample','_masked_channel_mean_std','texture_color_stat_loss','build_region_masks','masked_l1_loss']
    get=lambda source:{n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names}
    old,new=get(historical),get(current)
    assert all(ast.dump(old[n],include_attributes=False)==ast.dump(new[n],include_attributes=False) for n in names)
    namespace=dict(torch=torch,F=F)
    exec(compile(ast.Module(body=[new[n] for n in names],type_ignores=[]),'<frozen-losses>','exec'),namespace)
    pred=torch.zeros(1,3,64,64,requires_grad=True)
    gt=torch.ones_like(pred)*.4; reference=torch.ones_like(pred)*-.7
    mask=torch.zeros(1,1,64,64);mask[:,:,16:48,16:48]=1
    body,boundary,outside=namespace['build_region_masks'](mask,9)
    values=[namespace['texture_color_stat_loss'](pred,reference,body),
            namespace['masked_l1_loss'](pred,gt,boundary),namespace['masked_l1_loss'](pred,gt,outside)]
    assert all(v.item()>0 for v in values)
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
    from accelerate import Accelerator
    from torch.utils.data import DataLoader,TensorDataset
    from tools.e2a_attr_runtime import EpochSampler
    dataset=TensorDataset(torch.arange(32)); sampler=EpochSampler(dataset,42)
    accelerator=Accelerator(cpu=True,mixed_precision='no')
    loader=accelerator.prepare(DataLoader(dataset,batch_size=1,sampler=sampler))
    loader.set_epoch(1); full=[int(batch[0][0]) for batch in loader]
    loader.set_epoch(1); partial=accelerator.skip_first_batches(loader,13); partial.set_epoch(1)
    remaining=[int(batch[0][0]) for batch in partial]
    assert full[13:]==remaining,'恢复后的 epoch 采样顺序错误'
    write(OUT/'audit/loss_switch_selfcheck.json',dict(passed=True,historical_loss_AST_unchanged=True,
        body_target='reference_texture',all_factor_switches=results,exact_cached_region_checks=len(indices),
        device='CPU',scientific_training_updates=0,resume_epoch_order_checked=True))


if __name__=='__main__': selfcheck()
