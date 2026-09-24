import unittest
import torch
from tools.e14_causal_suite import RegionGate,configure_gate
from tools.e14_denoising_complete import aggregate
from types import SimpleNamespace

class CausalTests(unittest.TestCase):
    def test_gate_resolutions_and_identity(self):
        for h,w in [(64,48),(32,24),(16,12),(8,6)]:
            x=torch.randn(1,h*w,8)
            torch.testing.assert_close(RegionGate(torch.ones(1,1,512,384))(x),x,rtol=0,atol=0)
            self.assertEqual(RegionGate(torch.zeros(1,1,512,384))(x).abs().sum(),0)
        with self.assertRaises(ValueError):RegionGate(torch.ones(1,1,512,384))(torch.ones(1,7,8))
    def test_time_window(self):
        proc=SimpleNamespace(to_k_ip=True);net=SimpleNamespace(attn_processors={'p':proc})
        for t,active in [(101,False),(141,True),(181,True),(261,True),(481,False)]:
            gate=configure_gate(net,torch.ones(1,1,64,48),'region_window',t)
            self.assertEqual(gate is not None,active)
        configure_gate(net,None,'matched',181)
        self.assertIsNone(proc.texture_probe_transform)
    def test_verified_subset(self):
        rows=[dict(sample_index=1,losses=dict(matched=1,wrong_1=2,wrong_2=2,verified_pattern=3)),dict(sample_index=2,losses=dict(matched=1,wrong_1=2,wrong_2=2))]
        self.assertEqual(aggregate(rows)['verified_pattern']['sample_count'],1)

if __name__=='__main__':unittest.main()
