"""只检查有科学意义的消融契约；真实回放另外由GPU门禁执行。"""
import unittest
import torch
from models.m54_candidates import m10_statistic_residual


class StatisticAblations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x=torch.randn(1,768,640)
        self.v=torch.randn(1,256,640)*1.7+.3

    def full(self):
        x,v=self.x,self.v
        z=(x-x.mean(1,keepdim=True))/x.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
        return z*v.std(1,keepdim=True,unbiased=False)+v.mean(1,keepdim=True)-x

    def test_mean_and_std_decompose_original_full_residual(self):
        mu=m10_statistic_residual(self.x,self.v,'M10_MEAN')
        sd=m10_statistic_residual(self.x,self.v,'M10_STD')
        torch.testing.assert_close(mu+sd,self.full(),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(mu,self.v.mean(1,keepdim=True).expand_as(self.x)-self.x.mean(1,keepdim=True))
        torch.testing.assert_close(sd.mean(1),torch.zeros(1,640),atol=2e-6,rtol=0)

    def test_mask_one_exactly_replays_full_and_zero_isolates(self):
        one=torch.ones(1,1,32,24)
        self.assertTrue(torch.equal(m10_statistic_residual(self.x,self.v,'M10_MASK',one),self.full()))
        self.assertEqual(m10_statistic_residual(self.x,self.v,'M10_MASK',one*0).count_nonzero(),0)
        mask=torch.rand_like(one)
        self.assertTrue(torch.equal(m10_statistic_residual(self.x,self.v,'M10_MASK',mask),self.full()*mask.flatten(2).transpose(1,2)))

    def test_constant_features_stay_finite(self):
        for mode in ['M10_MEAN','M10_STD','M10_MASK']:
            result=m10_statistic_residual(self.x*0+2,self.v*0+3,mode,torch.ones(1,1,32,24))
            self.assertTrue(torch.isfinite(result).all())


if __name__=='__main__':unittest.main()
