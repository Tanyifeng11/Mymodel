import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from models.sptg import SPTG, combine_guidance, SETTINGS
from tools.report_sptg import report


class SPTGTests(unittest.TestCase):
    def test_report_pairing_and_baseline_reproduction(self):
        with tempfile.TemporaryDirectory() as folder:
            root,source=Path(folder)/'run',Path(folder)/'source'
            for sid in (2,5,14,18,22,23):
                name=f'sample_{sid:06d}'
                for branch in ('on','off'):
                    p=source/name/branch/'e5/token/sample'
                    p.mkdir(parents=True)
                    if branch=='on':(p/'probe.json').write_text(json.dumps({'metadata':{}}))
                    else:Image.new('RGB',(8,8),'white').save(p/'generated_x.png')
                for mode,(weight,rho) in SETTINGS.items():
                    p=root/name/mode/'e5/token/sample'
                    p.mkdir(parents=True)
                    Image.new('RGB',(8,8),'white').save(p/'generated_x.png')
                    row=dict(sample_id=sid,generation_seed=42+sid,prompt='p',sketch_path='s',texture_path='t',clip_i_texture=.5)
                    (p.parent.parent/'metrics_per_sample.json').write_text(json.dumps([row]))
                    trace=dict(mode=mode,texture_weight=weight,rho=rho,metadata={},records=[
                        dict(step_index=i,applied=False,actual_delta_rms=0.,actual_projection_delta_rms=0.) for i in range(50)])
                    (p/'sptg.json').write_text(json.dumps(trace))
            self.assertEqual(report(root,source)['status'],'pass')
            Image.new('RGB',(8,8),'black').save(root/'sample_000002/baseline/e5/token/sample/generated_x.png')
            with self.assertRaisesRegex(ValueError,'未复现'):
                report(root,source)

    def test_baseline_identity_and_texture_gain(self):
        pred=torch.randn(1,4,8,8).half()
        a,b=torch.ones_like(pred).float(),torch.ones_like(pred).float()*2
        floor=torch.zeros_like(a)
        self.assertIs(combine_guidance(pred,a,b,floor,1.,0.)[0],pred)
        strong=combine_guidance(pred,a,b,floor,1.2,0.)[0]
        self.assertTrue(torch.equal(strong,(pred.float()+.2*b).half()))
        self.assertTrue(torch.equal(combine_guidance(pred,a,b,floor,1.2,1.)[0],strong))

    def test_negative_component_only(self):
        a=torch.zeros(1,2,8,8);a[:,0]=1
        b=torch.zeros_like(a);b[:,0]=-2;b[:,1]=3
        pred=torch.zeros_like(a)
        result,d=combine_guidance(pred,a,b,pred,1.2,1.)
        self.assertTrue(torch.allclose(d['correction'][:,0],torch.full((1,8,8),2.)))
        self.assertTrue(torch.equal(d['correction'][:,1],torch.zeros(1,8,8)))
        self.assertTrue(torch.allclose(result,.2*b+1.2*d['correction']))
        half=combine_guidance(pred,a,b,pred,1.2,.5)[1]['correction']
        self.assertTrue(torch.allclose(half,d['correction']*.5))
        zero=combine_guidance(pred,a*0,b,pred,1.2,1.)[1]['correction']
        self.assertTrue(torch.equal(zero,pred))

    def test_forward_restoration_and_log(self):
        for mode in ('baseline','strong_texture','sptg_rho05','sptg_rho10'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                s,t=SimpleNamespace(scale=.6),SimpleNamespace(scale=1.,attn_map='saved')
                probe=SPTG([s],[t],torch.ones(1,1,8,8),folder,{},mode)
                def forward(sketch_on,scrub):
                    torch.rand(1)
                    t.attn_map='changed'
                    return torch.full((1,4,4,4),2+(.6 if sketch_on else 0)-t.scale*(0 if scrub else 1))
                pred=torch.full((1,4,4,4),1.6)
                rng=torch.get_rng_state().clone()
                output=probe.observe(pred,forward,0,999)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual((t.scale,t.attn_map),(1.,'saved'))
                self.assertEqual(len(probe.trace['records']),1)
                if mode=='baseline':self.assertIs(output,pred)


if __name__=='__main__':
    unittest.main()
