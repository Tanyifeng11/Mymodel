import tempfile
import json
import shutil
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.full_condition_probe import FullConditionProbe, local_geometry
from tools.report_full_conditions import report


class FullProbeTests(unittest.TestCase):
    def test_report_full_run_and_pixel_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            s,t=SimpleNamespace(scale=.6),SimpleNamespace(scale=1.)
            probe=FullConditionProbe([s],[t],torch.ones(1,1,8,8),root/'fixture',{})
            def forward(sketch_on,scrub):
                return torch.ones(1,4,4,4)*(2+(.6 if sketch_on else 0)-t.scale*(0 if scrub else 1))
            probe.observe(torch.full((1,4,4,4),1.6),forward,0,999)
            data=probe.report
            data['records']=[dict(data['records'][0],step_index=i,timestep=999-i) for i in range(50)]
            for sid in (2,5,14,18,22,23):
                for branch in ('off','on'):
                    run=root/f'sample_{sid:06d}'/branch/'e5'
                    token=run/'token'/'sample'
                    token.mkdir(parents=True)
                    Image.new('RGB',(8,8),'white').save(token/'generated_x.png')
                    (run/'metrics_per_sample.json').write_text(json.dumps([dict(sample_id=sid,
                        generation_seed=42+sid,prompt='p',sketch_path='s',texture_path='t')]))
                    if branch=='on':
                        target=token/'full_condition_probe'
                        target.mkdir()
                        (target/'probe.json').write_text(json.dumps(data))
                        for step in range(50):
                            shutil.copyfile(root/'fixture/step_00.npz',target/f'step_{step:02d}.npz')
            self.assertEqual(report(root)['status'],'pass')
            Image.new('RGB',(8,8),'black').save(root/'sample_000002/on/e5/token/sample/generated_x.png')
            with self.assertRaisesRegex(ValueError,'未复现'):
                report(root)

    def test_decomposition_shutdown_and_trajectory_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            sketch=SimpleNamespace(scale=.6)
            texture=SimpleNamespace(scale=1.,last_gate=7,attn_map='saved',balanced_gate_trace_enabled=True)
            probe=FullConditionProbe([sketch],[texture],torch.ones(1,1,8,8),folder,{})
            def forward(sketch_on,scrub):
                torch.rand(1)
                texture.last_gate=99
                texture.attn_map='changed'
                return torch.ones(1,4,4,4)*(2+(.6 if sketch_on else 0)-texture.scale*(0 if scrub else 1))
            prediction=torch.full((1,4,4,4),1.6)
            rng=torch.get_rng_state().clone()
            for step in (0,1,25,49):
                self.assertIs(probe.observe(prediction,forward,step,999-step),prediction)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual((texture.scale,texture.last_gate,texture.attn_map),(1.,7,'saved'))
            self.assertTrue(texture.balanced_gate_trace_enabled)
            with np.load(Path(folder)/'step_00.npz') as z:
                np.testing.assert_allclose(z['sketch_guidance'],.6,atol=1e-6)
                np.testing.assert_allclose(z['texture_guidance'],-1,atol=1e-6)
            self.assertEqual(probe.report['records'][0]['shutdown_audits']['texture_off_sketch_on_max_abs'],0.)
            self.assertEqual(probe.report['records'][0]['regions']['global']['local_negative_fraction'],1.)

    def test_shutdown_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            s,t=SimpleNamespace(scale=.6),SimpleNamespace(scale=1.)
            probe=FullConditionProbe([s],[t],torch.ones(1,1,8,8),folder,{})
            def leak(sketch_on,scrub):
                return torch.ones(1,4,4,4)*(1 if scrub else 2)
            with self.assertRaisesRegex(ValueError,'关闭审计失败'):
                probe.observe(torch.ones(1,4,4,4)*2,leak,0,999)
            self.assertEqual(t.scale,1.)

    def test_local_opposition_and_zero_directions(self):
        a=torch.ones(1,4,5,5)
        cosine,valid=local_geometry(a,-a,torch.zeros_like(a))
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(cosine,-torch.ones_like(cosine)))
        self.assertFalse(local_geometry(a,a*0,a*0)[1].any())
        self.assertFalse(local_geometry(a,-a,a)[1].any())


if __name__=='__main__':
    unittest.main()
