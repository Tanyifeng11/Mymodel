import numpy as np
import pytest

from tools.e2_followup_data import spectral_proxy, color_summary, color_distance, neutral_prompt, qualifies
from PIL import Image


@pytest.mark.parametrize('period',[4,8,16])
def test_known_period_is_recovered(period):
    y,x=np.indices((64,64))
    value=.5+.4*np.sin(2*np.pi*x/period)
    result=spectral_proxy(value)
    assert result['valid'] and abs(result['period']-period)<1e-8
    assert abs(result['line_angle']-90)<1e-8


def test_orientation_and_solid_are_distinguished():
    y,x=np.indices((64,64))
    result=spectral_proxy(.5+.4*np.sin(2*np.pi*y/8))
    assert min(result['line_angle'],180-result['line_angle'])<1e-8
    assert not spectral_proxy(np.ones((64,64))*.5)['valid']


def test_same_color_does_not_qualify_dirty_reference():
    image=Image.new('RGB',(64,64),'gray')
    color=color_summary(image)
    a=dict(index=1,color=color,spectral_proxy=dict(valid=True,period=8.,line_angle=0.))
    b=dict(index=2,color=color,spectral_proxy=dict(valid=True,period=4.,line_angle=0.))
    review={'1':dict(pattern='stripes',clean=True),'2':dict(pattern='stripes',clean=False)}
    assert qualifies(a,b,review) is None
    review['2']['clean']=True
    assert qualifies(a,b,review)['period_ratio']==2.
    assert color_distance(color,color)==(0.,0.)


def test_appearance_neutral_prompt_keeps_only_garment_type():
    prompt=neutral_prompt('a black and white striped denim dress with floral sleeves')
    assert 'dress' in prompt
    assert not any(x in prompt for x in ['black','white','striped','denim','floral'])
