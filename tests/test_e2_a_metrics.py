import numpy as np
import pytest
from PIL import Image
from tools.e2_a_evaluation import boundary_f1, local_response


def test_boundary_identity_translation_and_failed_generation():
    a=np.zeros((64,48),bool);a[10:54,10:38]=True
    b=np.roll(a,2,axis=1)
    assert boundary_f1(a,a,1)==1.
    assert boundary_f1(b,a,2)==1.
    assert boundary_f1(b,a,1)<1.
    assert boundary_f1(np.zeros_like(a),a,2)==0.


def test_local_response_flat_and_known_edge():
    a=np.ones((64,48,3),np.uint8)*128
    assert np.allclose(local_response(Image.fromarray(a)),0.)
    a[:,24:]=255
    response=local_response(Image.fromarray(a))
    assert np.isfinite(response).all() and response[:,22:26].mean()>.5


@pytest.mark.parametrize('arms', [['A0','A5'], ['B0','B1','B2','B4','B5']])
def test_texture_statistics_use_the_declared_baseline(arms):
    from tools.e2_b_screen_evaluation import response_comparisons
    from tools.bf_t3_three_arm_report import paired
    pairs=[dict(index=i) for i in range(32)]
    metrics=[m+'_'+s+'_response_change' for m in ['gray','rgb'] for s in ['strict','tolerant']]
    responses=[dict(arm=arm,index=i,**{m:float(i)+j*(i%3-1)*.1 for m in metrics})
               for j,arm in enumerate(arms) for i in range(32)]
    actual=response_comparisons(responses,pairs,arms)
    assert len(actual)==4*(len(arms)-1)
    for row in actual:
        assert row['comparison']==row['arm']+'_minus_'+arms[0]
        j=arms.index(row['arm'])
        expected=paired([(float(i)+j*(i%3-1)*.1)-float(i) for i in range(32)])
        assert all(row[k]==expected[k] for k in ['n','mean','ci95','p'])
        assert row['p']<=row['p_holm']<=1.
        if len(arms)==2:
            assert row['p_holm']==min(1.,expected['p']*(1 if row['family']=='primary' else 3))
