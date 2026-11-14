"""检查真正会误判门槛的情况：抹平纹样和Adam动量反向。"""
import unittest
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import torch

from tools import e2_b_endpoint_losses as loss
from tools import e2_b_balanced_losses as old_loss


class EndpointTests(unittest.TestCase):
    @staticmethod
    def function(name):
        path=Path(__file__).resolve().parents[1]/'tools/e2_state_b_endpoint.py'
        tree=ast.parse(path.read_text(encoding='utf8'))
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        scope=dict(torch=torch,copy=copy,loss=loss,old_loss=old_loss)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
        return scope[name]

    def test_full25_matches_closed_form_and_carries_all_b_gradients(self):
        theta=torch.nn.Parameter(torch.tensor(.1))
        scheduler=SimpleNamespace(timesteps=torch.arange(981,0,-20),alphas_cumprod=torch.full((1000,),.5),
                                  step=lambda eps,t,z,**kw:(z-.1*eps,))
        state=dict(zt=torch.ones(1),epsilon=torch.full((1,),.2),index=25,t=481)
        entry=dict(pair={'case_id':'train'},condition='ref',states=[state])
        obj=SimpleNamespace(args=SimpleNamespace(device='cpu'),pipe=SimpleNamespace(scheduler=scheduler),
                            b=lambda z,x0,**kw:theta*z,records={('train','ref'):(None,None,None,{})},
                            predictor=SimpleNamespace(epsilon=lambda z,t,e:.2*z))
        result,_=self.function('rollout')(obj,entry)
        derivative=torch.autograd.grad(result.sum(),theta)[0]
        factor=1-.1*.2+.1*theta.detach()
        self.assertTrue(torch.allclose(result,factor.pow(25),atol=1e-6))
        self.assertTrue(torch.allclose(derivative,2.5*factor.pow(24),atol=1e-5))

    def test_rejected_update_restores_parameters_and_optimizer(self):
        b=torch.nn.Linear(2,1,bias=False);opt=torch.optim.AdamW(b.parameters(),lr=.01)
        before=loss.flatten(list(b.parameters()))
        gradients=torch.ones(1,len(loss.TASKS),2)
        prior=[dict.fromkeys(loss.TASKS,1.)]
        report=self.function('guarded_step')(b,opt,gradients,torch.ones(len(loss.TASKS)),
                                              lambda:[dict.fromkeys(loss.TASKS,2.)],prior)
        self.assertFalse(report['accepted'])
        self.assertTrue(torch.equal(before,loss.flatten(list(b.parameters()))))
        self.assertEqual(opt.state_dict()['state'],{})

    def test_amplitude_catches_fading_when_correlation_stays_high(self):
        torch.set_num_threads(2)
        x=torch.arange(128)[None,:].expand(128,-1)
        target=(.4+.12*(x%16<8).float())[None,None].expand(1,3,-1,-1).clone()
        faded=.46+.35*(target-.46)
        weight=torch.ones(1,1,128,128)
        exact,_=loss.luminance(target,target,weight,[(32,32)])
        self.assertLess(float(exact['amplitude']),1e-7)
        self.assertLess(float(exact['correlation']),1e-6)
        faded.requires_grad_()
        values,metrics=loss.luminance(faded,target,weight,[(32,32)])
        self.assertGreater(metrics['l_correlation'],.99)
        self.assertLess(metrics['l_amplitude'],.5)
        self.assertGreater(float(values['amplitude'].detach()),.3)
        self.assertGreater(float(torch.autograd.grad(values['amplitude'],faded)[0].norm()),0)

    def test_projection_protects_actual_adam_displacement(self):
        p=torch.nn.Parameter(torch.tensor([1.,1.]))
        opt=torch.optim.AdamW([p],lr=.1,weight_decay=.01)
        p.grad=torch.tensor([-10.,0.]);opt.step()
        before=p.detach().clone();p.grad=torch.tensor([.01,1.]);opt.step()
        raw=before-p.detach()
        constraints=torch.tensor([[1.,0.],[0.,1.]])
        self.assertLess(float(raw[0]),0)
        repaired,info=loss.project_update(raw,constraints)
        self.assertTrue(info['feasible'])
        self.assertTrue((constraints@repaired>=-1e-7).all())
        self.assertLessEqual(float(repaired.norm()),float(raw.norm())+1e-6)

    def test_projection_keeps_each_reference_not_only_pair_mean(self):
        rows=torch.tensor([[1.,0.],[-1.,1.],[0.,1.]])
        projected,info=loss.project_update(torch.tensor([1.,-.5]),rows)
        self.assertTrue(info['feasible'])
        self.assertTrue((rows@projected>=-1e-6).all())


if __name__=='__main__':unittest.main()
