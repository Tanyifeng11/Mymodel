"""核验解析干预方向、尺度符号及支持区域，防止错误合成监督。"""
import unittest
import numpy as np
from PIL import Image
from data.e33_interventions import transform,audit_intervention,central_valid_box
from data.e33_self_reference import select_reference


def stripe(n=96):
    x=np.arange(n)[None,:]
    values=np.broadcast_to(127+90*np.sin(2*np.pi*x/8),(n,n)).astype(np.uint8)
    return np.repeat(values[...,None],3,axis=2)


class InterventionIntegrity(unittest.TestCase):
    def test_real_frequency_scale_sign_and_rotation(self):
        rgb=stripe()
        for arm,sign in [('rot90',0),('scale_up',-1),('scale_down',1)]:
            r=audit_intervention(rgb,arm)
            self.assertTrue(r['valid'],r)
            if sign:self.assertGreater(sign*r['period_response_log2'],.2)
            self.assertLess(r['orientation_response_error_deg'],1)

    def test_padding_excluded(self):
        rgb=stripe();changed,valid=transform(rgb,'scale_down')
        self.assertEqual(changed.shape,rgb.shape)
        self.assertFalse(valid[0].any())
        lo,hi=central_valid_box(valid)
        self.assertTrue(valid[lo:hi,lo:hi].all())
        self.assertLess(hi-lo,len(rgb))

    def test_legal_candidate_and_no_confidence_candidate(self):
        gray=np.broadcast_to(127+90*np.sin(2*np.pi*np.arange(384)[None,:]/8),(512,384)).astype(np.uint8)
        target=Image.fromarray(np.repeat(gray[...,None],3,2));sketch=Image.new('RGB',(384,512),'white')
        foreground=np.zeros((512,384),bool);foreground[32:-32,32:-32]=True
        g=np.zeros((4,64,48),np.float32);g[0]=-1;g[2]=np.log(.125);g[3]=.9
        chosen=select_reference(target,sketch,foreground,g)
        self.assertTrue(chosen['controlled_available'])
        x,y,x1,y1=chosen['box'];self.assertTrue(foreground[y:y1,x:x1].all())
        g[3]=0
        self.assertFalse(select_reference(target,sketch,foreground,g)['controlled_available'])


if __name__=='__main__':unittest.main()
