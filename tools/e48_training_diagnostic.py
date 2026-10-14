"""重放失败支路前五次更新；检查实际注意力反向，结果不参与模型选择。"""
import math
import time
import torch
from torch.nn import functional as F
from tools.e48_train import make_runtime,inputs,forward_loss
from tools.e48_protocol import OUT,write,commit
from tools.e34_sarr_protocol import seed_all
from tools.e44_train import gradients
from tools.e47_runtime import close


def inspect(cache):
    if not cache or 'gradient' not in cache:return dict(enabled=False)
    q,k,v,g=[cache[x] for x in ['q','k','v','gradient']]
    result=dict(enabled=True,incoming_gradient_norm=float(g.float().norm()))
    with torch.no_grad():
        logits=q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])
        probabilities=logits.softmax(-1)
        result.update(q_abs_max=float(q.abs().max()),k_abs_max=float(k.abs().max()),
            logit_abs_max=float(logits.abs().max()),mean_max_probability=float(probabilities.max(-1).values.mean()),
            entropy=float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()))
    for dtype,label in [(torch.float16,'fp16'),(torch.float32,'fp32')]:
        with torch.autocast('cuda',enabled=False),torch.enable_grad():
            kk=k.to(dtype).detach().requires_grad_(True);vv=v.to(dtype).detach().requires_grad_(True)
            output=F.scaled_dot_product_attention(q.to(dtype),kk,vv)
            dk,dv=torch.autograd.grad(output,(kk,vv),g.to(dtype))
        result[label]=dict(key_grad_norm=float(dk.float().norm()),value_grad_norm=float(dv.float().norm()),
            key_nonzero=int(dk.count_nonzero()),value_nonzero=int(dv.count_nonzero()))
    return result


def main():
    torch.set_num_threads(2);seed_all(42)
    r=make_runtime('DECOUPLED');data=inputs(r);r.processor.capture_backward=True
    seed_all(42);optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');rows=[];sequence=[];started=time.perf_counter()
    path=OUT/'training/DECOUPLED_gradient_diagnostic.json'
    try:
        for step in range(1,6):
            optimizer.zero_grad(set_to_none=True);samples=[];gate=float(r.processor.gate)
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                r.processor.backward_cache=None
                loss,_=forward_loss(r,args,kw,noise,mask,'DECOUPLED',torch.zeros_like(mask))
                if loss.requires_grad:scaler.scale(loss/8).backward()
                samples.append(dict(record=record,attention=inspect(r.processor.backward_cache)))
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.))
            grad=gradients(r);old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            assert scaler.get_scale()>=old
            rows.append(dict(step=step,gate_before=gate,gate_after=float(r.processor.gate),
                grad_norm=norm,gradients=grad,samples=samples))
            write(path,dict(commit=commit(),complete=step==5,steps=rows,sequence=sequence,
                seconds=time.perf_counter()-started,purpose='engineering diagnosis only; no candidate checkpoint'))
            print('E48 GRADIENT DIAG',step,gate,grad,flush=True)
    finally:close(r)


if __name__=='__main__':main()
