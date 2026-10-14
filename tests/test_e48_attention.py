"""CPU 核验：分解恒等式、零门控等价、独立支路梯度和 OFF 隔离。"""
import torch
from torch import nn
from types import SimpleNamespace
from models.e48_interventions import decomposition,split
from models.e48_reference_attention import IndependentReferenceProcessor
from torch.nn import functional as F


def test_decomposition():
    torch.manual_seed(48014);width=32;heads=4
    linear=nn.Linear(width,width,bias=False)
    with torch.no_grad():linear.weight.copy_(torch.eye(width))
    attn=SimpleNamespace(heads=heads,to_out=[linear,nn.Identity()],rescale_output_factor=1.)
    cache={k:torch.randn(1,n,width) for k,n in [('q',12),('k',12),('v',12),('kr',4),('vr',4)]}
    for gate in [.1,-.2]:
        parts,_=decomposition(attn,SimpleNamespace(gate=torch.tensor(gate)),cache)
        q,k,v,kr,vr=[split(cache[x],heads) for x in ['q','k','v','kr','vr']]
        own=F.scaled_dot_product_attention(q,k,v)
        joint=F.scaled_dot_product_attention(q,torch.cat([k,kr],2),torch.cat([v,vr],2))
        expected=(gate*(joint-own)).transpose(1,2).reshape(1,12,width)
        torch.testing.assert_close(parts['added']+parts['suppressed'],expected,atol=1e-6,rtol=1e-5)


def test_independent_reference():
    class Base(nn.Module):
        def forward(self,attn,hidden_states,**kwargs):return hidden_states*1.1+.2
    torch.manual_seed(48014);x=torch.randn(1,12,32)
    attn=SimpleNamespace(heads=4,to_q=nn.Linear(32,32),to_out=[nn.Linear(32,32),nn.Identity()],
        spatial_norm=None,group_norm=None,rescale_output_factor=1.)
    p=IndependentReferenceProcessor(Base(),'test',32,32,'B1');p.enabled=True;p.features=torch.randn(1,4,32)
    base=p.original(attn,x);zero=p(attn,x,sa_hidden_states={'test':x})
    assert torch.equal(base,zero)
    with torch.no_grad():p.gate.fill_(.1)
    output=p(attn,x,sa_hidden_states={'test':x});assert not torch.equal(output,base)
    output.square().mean().backward()
    assert all(torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in [p.gate,p.to_k_ref.weight,p.to_v_ref.weight])
    assert torch.equal(p(attn,x,sa_hidden_states=None),base)
    p.enabled=False;assert torch.equal(p(attn,x,sa_hidden_states={'test':x}),base)


if __name__=='__main__':
    test_decomposition();test_independent_reference();print('e48 CPU attention checks passed')
