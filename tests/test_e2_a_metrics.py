import numpy as np
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
