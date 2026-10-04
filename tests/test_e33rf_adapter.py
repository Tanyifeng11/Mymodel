"""验证冻结模型内反传、消融输入保留，以及轴向旋转损失的物理含义。"""
import unittest
import torch
from models.e33r_rotation_control import RotationControl,SketchPrior
from models.e33rf_real_adapter import FrozenCausalAdapter,RealAdapter
from tools.e33rf_losses import losses
from tools.e33rf_common import real_gate,controlled_gate,pilot_gate

class AdapterTests(unittest.TestCase):
    def test_zero_init_and_gradient_through_real_frozen_backbone(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        base=RotationControl(SketchPrior());original=sum(p.numel() for p in base.parameters() if p.requires_grad)
        model=FrozenCausalAdapter(base).train();reference=torch.randn(1,3,4,394);structure=torch.randn(1,7,8,8)
        before={k:v.clone() for k,v in base.state_dict().items()}
        with torch.no_grad():expected=base(reference,structure)['orientation']
        actual=model(reference,structure)['orientation'];self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(base.training);self.assertEqual(original,745475)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),26430)
        # 对真实网络输出施加非对称目标，梯度须穿过冻结attention/乘法方向头。
        actual[:,1].mean().backward()
        self.assertGreater(model.adapter.net[-1].weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        torch.optim.AdamW(model.adapter.parameters(),lr=1e-3).step()
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in base.state_dict().items()))
    def test_partial_adapters_preserve_other_channels(self):
        x=torch.randn(2,3,4,394)
        for variant,start,stop in [('E_geometry_only',384,387),('F_dino_only',0,384)]:
            adapter=RealAdapter(32,variant)
            with torch.no_grad():adapter.net[-1].bias.fill_(1.)
            y,delta=adapter(x);self.assertTrue(torch.equal(y[...,:start],x[...,:start]))
            self.assertTrue(torch.equal(y[...,stop:],x[...,stop:]))
            self.assertTrue(torch.allclose(y[...,start:stop]-x[...,start:stop],torch.full_like(y[...,start:stop],.1)))
            self.assertTrue(torch.equal(delta[...,start:stop],torch.ones_like(delta[...,start:stop])))
    def prediction(self,ori):return dict(orientation=ori,adapter_delta=torch.zeros(ori.shape[0],ori.shape[1],1,1,394))
    def test_axial_physical_90_and_180_and_ablation(self):
        gt=torch.tensor([[[[1.]],[[0.]],[[0.]],[[1.]]]]);support=torch.ones(1,1,1,dtype=torch.bool)
        correct=torch.tensor([[[[[1.]],[[0.]]],[[[-1.]],[[0.]]],[[[1.]],[[0.]]]]])
        good,_=losses(self.prediction(correct),gt,support,'RF2');self.assertAlmostEqual(good.item(),0.)
        no_response=correct.clone();no_response[:,1]=correct[:,0]
        bad,_=losses(self.prediction(no_response),gt,support,'RF2');self.assertAlmostEqual(bad.item(),2.)
        removed,_=losses(self.prediction(no_response),gt,support,'RF2','A_no_equivariance');self.assertAlmostEqual(removed.item(),0.)
        wrong180=correct.clone();wrong180[:,2]=-correct[:,0]
        bad180,_=losses(self.prediction(wrong180),gt,support,'RF2');self.assertAlmostEqual(bad180.item(),1.)
    def test_per_case_ranking_not_hinge_of_mean(self):
        # 两病例：near一例过margin，一例完全无优势。先平均再hinge会错误给0。
        gt=torch.zeros(2,4,1,1);gt[:,0]=1.;gt[:,3]=1.;support=torch.ones(2,1,1,dtype=torch.bool)
        angles=torch.tensor([[0.,90.,0.,30.,40.],[0.,90.,0.,0.,0.]])
        rad=torch.deg2rad(2*angles);ori=torch.stack([rad.cos(),rad.sin()],2)[...,None,None]
        _,parts=losses(self.prediction(ori),gt,support,'RF3')
        self.assertAlmostEqual(parts['rank'],(.5*5+.25*7.5)/2,places=4)
    def test_complete_gate_and_strict_ci(self):
        stat=lambda x:dict(mean=x,ci95=[1.,2.])
        real=dict(errors=dict(matched=stat(7)),near_advantage=stat(5),rot90_success=stat(.5),finite_prediction_rate=1)
        cf=dict(clean_r90_success=stat(.95),noisy_r90_both_success=stat(.9),r180_identity_success=stat(.95),finite_prediction_rate=1)
        self.assertTrue(real_gate(real)['pass']);self.assertTrue(controlled_gate(cf)['pass']);self.assertTrue(pilot_gate(cf,real)['pass'])
        real['near_advantage']['ci95'][0]=0;self.assertFalse(real_gate(real)['pass'])
        cf['clean_r90_success']['mean']=.9499;self.assertFalse(pilot_gate(cf,real)['pass'])
if __name__=='__main__':unittest.main()
