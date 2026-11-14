"""昂贵GPU小试前检查预算隔离、梯度冲突处理及可达目标。"""
import json
import ast
from collections import Counter
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from tools.e2_b_balanced_data import schedule
from tools import e2_b_balanced_losses as losses


class BalancedPilotTests(unittest.TestCase):
    def test_sampling_restores_vae_precision_on_success_and_error(self):
        # 独立执行真实采样包装器，避免测试依赖GPU模型文件。
        source=Path(__file__).resolve().parents[1]/'tools/e2_state_b_balanced.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample']
        class VAE(torch.nn.Conv2d):
            @property
            def dtype(self):return self.weight.dtype
        for dtype in (torch.float16,torch.float32):
            for fail in (False,True):
                vae=VAE(1,1,1).to(dtype=dtype);b=torch.nn.Linear(1,1);b.train(False)
                pipe=SimpleNamespace(vae=vae)
                def sampling(*args):
                    self.assertEqual(vae.dtype,torch.float16)
                    if fail:raise RuntimeError('sampler stopped')
                    return 7
                scope={'torch':torch,'joint':SimpleNamespace(sample=sampling)}
                exec(compile(tree,str(source),'exec'),scope)
                if fail:
                    with self.assertRaises(RuntimeError):scope['sample'](None,None,pipe,None,None,None,b)
                else:self.assertEqual(scope['sample'](None,None,pipe,None,None,None,b),7)
                self.assertEqual(vae.dtype,dtype)
                self.assertFalse(b.training)

    def test_balanced_budget_and_identity_isolation(self):
        spec=json.loads((Path(__file__).resolve().parents[1]/'configs/e2_b_joint_refs.json').read_text(encoding='utf8'))
        spec['train']=[{'case_id':f'validation_{i}'} for i in (621,1641,870,1526)]
        rows=schedule(spec)
        self.assertEqual(Counter(r['case'] for r in rows),dict.fromkeys((p['case_id'] for p in spec['train']),100))
        self.assertEqual(Counter(r['group'] for r in rows),dict.fromkeys(('neutral','pastel','multi','synthetic'),100))
        for group in ('neutral','pastel','multi','synthetic'):
            self.assertEqual(sum(r['online'] for r in rows if r['group']==group),50)
            self.assertEqual(sum(r['hops']==4 for r in rows if r['group']==group),25)
        self.assertTrue(all(len(r['conditions'])==2 for r in rows))
        self.assertEqual(rows,schedule(spec))

    def test_pcgrad_removes_known_two_task_interference(self):
        gradients=torch.tensor([[1.,0.],[-1.,1.],[0.,0.]])
        merged=losses.merge(gradients,torch.ones(3),True,7)
        self.assertTrue((gradients@merged>=-1e-6).all())
        self.assertTrue(torch.allclose(losses.merge(gradients,torch.ones(3),False,7),gradients.sum(0)))

    def test_reachable_positive_zero_loss_and_residual_gradient(self):
        torch.set_num_threads(2)
        y,x=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij')
        stripe=(x%16<8).float()
        goal=torch.stack((.3+.4*stripe,.35+.3*stripe,.4+.2*stripe))[None]
        original=(goal*.95+.01).clamp(0,1)
        weight=torch.ones(1,1,128,128)
        old=(goal+.03*(y%8<4)[None,None]).clamp(0,1)
        template=losses.templates(old,goal,[(32,32)])
        zero=losses.terms(goal,goal,original,goal,weight,template,[(32,32)])
        self.assertLess(max(float(v) for v in zero.values()),1e-7)
        current=(goal+.02*(y%8<4)[None,None]).clone().requires_grad_()
        values=losses.terms(current,goal,original,goal,weight,template,[(32,32)])
        self.assertGreater(float(values['projection'].detach()),0)
        self.assertGreater(float(values['edge'].detach()),0)
        values['latent']=(current-goal).square().mean()
        groups=losses.groups(values)
        grads=losses.task_gradients(groups,[current])
        self.assertTrue(torch.isfinite(grads).all())
        self.assertTrue((grads.norm(dim=1)>0).all())


if __name__=='__main__':unittest.main()
