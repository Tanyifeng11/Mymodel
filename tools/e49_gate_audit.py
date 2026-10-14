"""在真实 U-Net 上验收三个关闭条件，避免零 gate 掩盖隔离错误。"""
import copy
import torch


@torch.no_grad()
def off_isolation(r, args, kw):
    active,allow,valid=r.active,r.allow_reference,r.mask_info['valid']
    def forward(arguments):
        with torch.autocast('cuda',dtype=torch.float16): return r.pipe.unet(*args,**arguments).sample
    r.active=False; conditional_base=forward(kw)
    uncond=copy.deepcopy(kw); uncond['cross_attention_kwargs'].pop('sa_hidden_states')
    unconditional_base=forward(uncond)
    result={}
    try:
        r.active=True; r.allow_reference=False
        result['reference_dropout_exact']=bool(torch.equal(conditional_base,forward(kw)))
        r.allow_reference=True; r.mask_info['valid']=False
        result['invalid_mask_exact']=bool(torch.equal(conditional_base,forward(kw)))
        r.mask_info['valid']=True
        result['unconditional_exact']=bool(torch.equal(unconditional_base,forward(uncond)))
        assert all(result.values()),result
        return result
    finally:
        r.active,r.allow_reference,r.mask_info['valid']=active,allow,valid
