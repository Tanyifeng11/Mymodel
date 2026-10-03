"""覆盖容易改变科学结论的通道、成组ranking与Gate边界。"""
import unittest
import numpy as np
import torch
from data.e33rc_real_pair_dataset import orientation_reference
from tools.e33rc_losses import real_losses,retention_loss
from tools.e33rc_common import retention_gate,real_gate,variant_model
from data.e33rc_wrong_reference import train_wrong_references
from data.e32_wrong_reference_sampler import wrong_references

class CurriculumTests(unittest.TestCase):
    def test_no_period_and_no_channel_shift(self):
        source=np.arange(395)[None,None,:]
        value=orientation_reference(source)
        self.assertEqual(value.shape[-1],394)
        self.assertEqual(value[0,0,386],387)
        self.assertNotIn(386,value)
    def test_ranking_uses_paired_target_and_mask(self):
        gt=torch.zeros(2,4,2,3);gt[:,0]=1;gt[:,3]=1
        angles=torch.tensor([[0.,30.,40.],[0.,30.,40.]],requires_grad=True)
        radians=torch.deg2rad(2*angles)
        ori=torch.stack([radians.cos(),radians.sin()],2)[:,:,:,None,None].expand(-1,-1,-1,2,3)
        pred={'orientation':ori};mask=torch.ones(2,2,3,dtype=torch.bool)
        recon,rank=real_losses(pred,gt,mask)
        self.assertLess(float(rank),1e-5)
        self.assertLess(float(recon),1e-5)
        (recon+rank).backward();self.assertTrue(torch.isfinite(angles.grad).all())
        mask[:]=False
        self.assertEqual(float(real_losses(pred,gt,mask)[1]),0)
        weight=torch.ones(2,2,3)
        self.assertLess(float(retention_loss(ori,ori.detach(),weight)),1e-5)
    def test_gate_not_old_r0_or_prior_gate(self):
        stat=lambda x:dict(mean=x,ci95=[x,x],n=128)
        summary={k:stat(v) for k,v in [('clean_r90_success',.90),('noisy_r90_both_success',.85),
            ('r180_identity_success',.90),('zero_ratio',.10),('sensitivity_ratio',2.)]}
        summary['finite_prediction_rate']=1
        self.assertTrue(retention_gate(summary)['pass'])
        summary['clean_r90_success']=stat(.849)
        self.assertTrue(retention_gate(summary)['collapse'])
        real=dict(near_advantage=stat(5),errors={'matched':stat(7)},rot90_success=stat(.5),finite_prediction_rate=1)
        self.assertTrue(real_gate(real)['pass'])
        real['near_advantage']['ci95']=[0,10]
        self.assertFalse(real_gate(real)['pass'])
        self.assertEqual(variant_model('E_geometry_only'),'G_geometry_only')
    def test_training_nearest_same_as_exhaustive_e32(self):
        records=[{'id':str(i)} for i in range(80)]
        h=np.random.default_rng(42).normal(size=(80,24)).astype(np.float32)
        h[2]=h[1];h[3]=h[1]
        hashes={r['id']:dict(reference=r['id'],target=r['id']) for r in records}
        hashes['2']['reference']='1'
        features={r['id']:dict(histogram=h[i]) for i,r in enumerate(records)}
        exhaustive=wrong_references(records,features,hashes)
        fast=train_wrong_references(records,h,hashes)
        self.assertEqual([fast[r['id']]['color_near'] for r in records],[exhaustive[r['id']]['color_near'] for r in records])

if __name__=='__main__':unittest.main()
