"""不加载权重，验证真实 attention processor 的观察回调和零条件对照。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
import torch.nn.functional as F


class NoiseResponseTests(unittest.TestCase):
    def test_observer_and_zero_control(self):
        path = Path(__file__).resolve().parents[1]/'adapter/attention_processor.py'
        nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.ClassDef) and n.name == 'IPAttnProcessor2_0']
        env = dict(torch=torch, nn=nn, F=F)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
        proc = env['IPAttnProcessor2_0'](hidden_size=8, cross_attention_dim=8, num_tokens=2)
        attn = SimpleNamespace(spatial_norm=None, group_norm=None, norm_cross=False,
            heads=2, to_q=nn.Linear(8,8), to_k=nn.Linear(8,8), to_v=nn.Linear(8,8),
            to_out=[nn.Linear(8,8),nn.Identity()], residual_connection=False, rescale_output_factor=1)
        x, text, tokens = torch.randn(1,4,8), torch.randn(1,5,8), torch.randn(1,2,8)
        context = torch.cat([text,tokens],1)
        original = proc(attn,x,encoder_hidden_states=context)
        seen = []
        proc.texture_probe_observer = lambda v: seen.append(v.detach().clone())
        observed = proc(attn,x,encoder_hidden_states=context)
        torch.testing.assert_close(original,observed,rtol=0,atol=0)
        self.assertEqual(len(seen),1)
        self.assertGreater(seen[-1].norm().item(),0)
        zero = proc(attn,x,encoder_hidden_states=torch.cat([text,torch.zeros_like(tokens)],1))
        self.assertEqual(seen[-1].norm().item(),0)
        proc.scale = 0
        disabled = proc(attn,x,encoder_hidden_states=context)
        torch.testing.assert_close(zero,disabled,rtol=0,atol=0)
        # 真正processor路径：全零区域干预等价于关闭纹理分支。
        proc.scale = 1
        proc.texture_probe_transform = lambda residual: residual * 0
        masked = proc(attn,x,encoder_hidden_states=context)
        torch.testing.assert_close(masked,disabled,rtol=0,atol=0)
        proc.texture_probe_transform = lambda residual: residual
        torch.testing.assert_close(proc(attn,x,encoder_hidden_states=context),original,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
