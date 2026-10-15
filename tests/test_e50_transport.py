"""可证伪的对应/退回/结构算子检查，不加载大模型。"""
import torch
from torch import nn
from models.e50_structure_transport import correspondence,attend,project,ControlledAttention


def test_correspondence_identity_and_permutation():
    x=torch.eye(16)[None];mask=torch.ones(1,16,1)
    index,conf,_=correspondence(x,x,mask,(4,4))
    assert conf.all() and torch.equal(index,torch.arange(16)[None])
    perm=torch.tensor([3,4,5,1,0,2,8,9,10,11,12,13,14,15,6,7])
    j,c,_=correspondence(x,x[:,perm],mask,(4,4))
    assert c.all() and torch.equal(x[:,perm].gather(1,j[...,None].expand(-1,-1,16)),x)


def test_ambiguous_and_background_rejected():
    x=torch.ones(1,16,8);mask=torch.ones(1,16,1)
    _,conf,_=correspondence(x,x,mask,(4,4));assert not conf.any()
    y=torch.eye(16)[None];mask[:,:4]=0
    _,conf,_=correspondence(y,y,mask,(4,4));assert not conf[:,:4].any() and conf[:,4:].all()


def test_attention_delta_no_double_bias():
    torch.manual_seed(42);q,k,v=[torch.randn(1,16,8) for _ in range(3)]
    same=attend(q,k,v,2)-attend(q,k,v,2);assert same.count_nonzero()==0
    class A:
        to_out=[nn.Linear(8,8),nn.Identity()]
        rescale_output_factor=1.
    a=A();a.to_out[0].bias.data.fill_(9)
    assert project(a,same).count_nonzero()==0
    mask=torch.zeros(1,16,1);warp=torch.randn_like(v)
    altered=v+.1*mask*(warp-v)
    assert torch.equal(attend(q,k,altered,2),attend(q,k,v,2))


def test_controller_structure_and_off():
    from types import SimpleNamespace
    torch.manual_seed(50)
    a=SimpleNamespace(to_q=nn.Linear(8,8,bias=False),to_k=nn.Linear(8,8,bias=False),
        to_v=nn.Linear(8,8,bias=False),to_out=[nn.Linear(8,8),nn.Identity()],
        heads=2,rescale_output_factor=1.,spatial_norm=None,group_norm=None)
    class Original(nn.Module):
        def forward(self,a,x,**kwargs):
            return a.to_out[0](attend(a.to_q(x),a.to_k(x),a.to_v(x),a.heads))+.6
    original=Original();ctl=SimpleNamespace(phase='teacher',disabled=False,valid=True,cache={},kind='STRUCT',current={})
    wrapper=ControlledAttention(original,'test',ctl)
    teacher=torch.randn(1,16,8);student=torch.randn(1,16,8)
    assert torch.equal(wrapper(a,teacher),original(a,teacher))
    ctl.phase='candidate'
    expected=a.to_out[0](attend(a.to_q(teacher),a.to_k(teacher),a.to_v(student),2))+.6
    assert torch.allclose(wrapper(a,student),expected,atol=2e-7)
    ctl.disabled=True;assert torch.equal(wrapper(a,student),original(a,student))
    ctl.disabled=False;ctl.valid=False;assert torch.equal(wrapper(a,student),original(a,student))


if __name__=='__main__':
    test_correspondence_identity_and_permutation();test_ambiguous_and_background_rejected()
    test_attention_delta_no_double_bias();test_controller_structure_and_off();print('e50 local tests passed')
