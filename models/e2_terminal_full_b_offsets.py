"""冻结f68原B，末步用逐参数零偏移形成可学习的全层B函数。"""
import torch
from torch import nn
from torch.func import functional_call


PARAMETERS=274372


class TerminalFullBOffsets(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base.eval().requires_grad_(False)
        self.offsets=nn.ParameterList([nn.Parameter(torch.zeros_like(v)) for v in base.parameters()])
        assert sum(v.numel() for v in self.offsets)==PARAMETERS

    def increment_parameters(self):
        return list(self.offsets)

    def forward(self,zt,x0,**condition):
        # 原权重保持只读；不调用旧5388头，也不把其G移用到这个空间。
        values={name:weight+delta for (name,weight),delta in zip(self.base.named_parameters(),self.offsets)}
        return functional_call(self.base,values,(zt,x0),condition,strict=True)


def terminal_inputs(record,device):
    """按原SamplerB的FP32顺序恢复x0，保留CFG原epsilon；没有新UNet调用。"""
    alpha=torch.tensor(record['alpha'],device=device,dtype=torch.float32)
    zt,epsilon=[record[n].to(device).float() for n in ['zt','raw_epsilon']]
    x0=(zt-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
    condition={n:v.to(device) for n,v in record['target_record'][3].items()}
    condition['timestep']=torch.tensor([record['t']],device=device)
    return zt,x0,condition
