import copy
import numpy as np
import pytest
import torch
from torch import nn

from models.e2_structure import StructureResidual
from models.e2_structure_ablation import SingleStructureResidual, UnlayeredStructureResidual, architecture, condition_maps


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def maps():
    x = torch.rand(1, 6, 64, 48); x[:, 5] = 1
    return x


def test_exact_parameter_counts_independent_rng_and_legacy_init():
    state = torch.get_rng_state().clone()
    spec = architecture(); assert spec['single']['extra_block']['parameters'] == 816
    assert len(set(spec['added_parameters'].values())) == 1
    assert torch.equal(state, torch.get_rng_state())
    a, b = SingleStructureResidual(), SingleStructureResidual()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    a, b = StructureResidual(mode='A3'), StructureResidual(mode='A5')
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_zero_heads_all_parameters_reached_and_restore():
    m = SingleStructureResidual(channels=(8, 8, 8, 8)); x = maps(); sizes=[(8,6),(4,3),(2,2),(1,1)]
    assert all(torch.count_nonzero(y) == 0 for y in m(x, sizes))
    opt = torch.optim.SGD(m.parameters(), lr=.1)
    for step in range(3):
        ys=m(x,sizes);sum((y-1).square().mean() for y in ys).backward()
        if step:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            assert all(p.grad.norm()>0 for p in m.parameters())
        opt.step();opt.zero_grad()
    saved=copy.deepcopy(m.state_dict());restored=SingleStructureResidual(channels=(8,8,8,8));restored.load_state_dict(saved,strict=True)
    assert all(torch.equal(a,b) for a,b in zip(m(x,sizes),restored(x,sizes)))


def test_single_branch_ignores_unconsumed_geometry_and_invalid_falls_back():
    m=SingleStructureResidual(channels=(8,8,8,8));x=maps();other=x.clone();other[:,1:5]=torch.randn_like(other[:,1:5])
    for h in m.heads:nn.init.normal_(h.weight)
    sizes=[(8,6),(4,3),(2,2),(1,1)]
    assert all(torch.equal(a,b) for a,b in zip(m(x,sizes),m(other,sizes)))
    x[:,5]=0;assert all(torch.count_nonzero(y)==0 for y in m(x,sizes))


def test_canny_only_replaces_condition_line_and_uses_fixed_thresholds():
    import cv2
    sketch=torch.ones(1,3,64,48);sketch[:,:,10:54,10:12]=-1;sketch[:,:,10:12,10:38]=-1
    a,b=condition_maps(sketch,'A1'),condition_maps(sketch,'A2')
    assert torch.equal(a[:,1:],b[:,1:])
    gray=((sketch[0,0]+1)*127.5).round().byte().numpy()
    expected=cv2.Canny(gray,40,120,apertureSize=3,L2gradient=False)/255.
    assert np.array_equal(b[0,0].numpy(),expected.astype(np.float32))


def test_single_convolution_geometry_and_no_independent_second_branch():
    m=SingleStructureResidual();convs=[x for x in m.encoder if isinstance(x,nn.Conv2d)]
    field,jump=1,1
    for c in convs:
        field+=(c.kernel_size[0]-1)*jump;jump*=c.stride[0]
    assert len(convs)==5 and field==19 and jump==8
    assert not hasattr(m,'outer') and not hasattr(m,'inner')


def test_A4_uses_gray_instead_of_dead_zero_column_and_has_no_semantic_split():
    m=UnlayeredStructureResidual(channels=(8,8,8,8));x=maps();sizes=[(8,6),(4,3),(2,2),(1,1)]
    for h in m.heads:nn.init.normal_(h.weight,std=.01)
    other=x.clone();other[:,[1,4]]=torch.rand_like(other[:,[1,4]])
    assert all(torch.equal(a,b) for a,b in zip(m(x,sizes),m(other,sizes)))
    sum(y.square().mean() for y in m(x,sizes)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters())
    assert torch.all(m.outer[0].weight.grad.abs().sum((0,2,3))>0)


def test_sixarm_structure_statistics_preserve_pairwise_values_and_expand_holm():
    from tools.e2_a_ablation_evaluation import structure_comparisons
    from tools.bf_t3_three_arm_report import paired
    arms=['A0','A1','A2','A3','A4','A5'];pairs=[dict(case_id=str(i)) for i in range(32)]
    rows=[dict(arm=a,case_id=str(i),condition='old',boundary_f1_r2=.2+j*(i%3-1)*.01,
        inner_support_proxy=.9+j*(i%3-1)*.001,edge_f1=.5+j*(i%3-1)*.02,iou=.8+j*(i%3-1)*.001)
        for j,a in enumerate(arms) for i in range(32)]
    stats,mechanism=structure_comparisons(rows,rows,pairs)
    assert len(stats)==20 and len(mechanism)==12
    assert sum(x['family']=='primary' for x in stats)==5
    for row in stats:
        key=row['metric'][4:];j=arms.index(row['arm'])
        expected=paired([next(x for x in rows if x['arm']==row['arm'] and x['case_id']==str(i))[key]-
                         next(x for x in rows if x['arm']=='A0' and x['case_id']==str(i))[key] for i in range(32)])
        assert all(row[k]==expected[k] for k in ['n','mean','ci95','p'])
        assert row['p']<=row['p_holm']<=1
    assert all('p' not in x and 'p_holm' not in x for x in mechanism)
