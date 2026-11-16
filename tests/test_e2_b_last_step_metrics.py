"""检查幅度与颜色分离，以及综合残留被通道抵消的情形。"""
import numpy as np
from tools.e2_b_last_step_metrics import score_pixels, residue_components, summarize


def fixture():
    y,x = np.indices((128,128))
    stripe = np.where(x%16<8,180,80).astype(np.uint8)
    old = np.where(y%16<8,180,80).astype(np.uint8)
    mask = np.zeros((128,128),bool);mask[5:-5,5:-5]=True
    return np.repeat(stripe[...,None],3,2), np.repeat(old[...,None],3,2), mask


def test_positive_has_unit_amplitude_and_zero_errors():
    target,anchor,mask=fixture()
    score=score_pixels(target,target,target,anchor,mask,[(32,32)])
    assert abs(score['l_amplitude']-1)<1e-6 and score['l_correlation']>.99999
    for k in ('palette_swd','low_l','edge','old_projection','old_l_projection'):
        assert score[k] < 1e-6


def test_weaker_texture_is_not_rejected_as_a_color_change():
    target,anchor,mask=fixture()
    weak=np.rint((target.astype(float)-130)*.5+130).astype(np.uint8)
    a=score_pixels(target,target,target,anchor,mask,[(32,32)])
    b=score_pixels(weak,target,target,anchor,mask,[(32,32)])
    assert b['l_amplitude']<.75 and b['l_correlation']>.99
    result=summarize([dict(case='one',condition='v16',group='neutral',S0=a,S1=b)])
    assert result['all']['metrics']['l_amplitude_error']['worse']==1


def test_channel_cancellation_does_not_hide_absolute_contributions():
    q=np.tile(np.array([[-2.,-2.,0.],[2.,2.,0.]]),(50,1))
    p=np.zeros_like(q)
    current=q.copy();current[:,1]*=-1
    score=residue_components(current,p,q)
    assert score['old_projection']<1e-6
    assert score['old_projection_channel_abs_sum']>.99
    assert score['old_l_projection']>.99
