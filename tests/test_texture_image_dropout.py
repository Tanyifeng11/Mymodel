"""执行真实TextureAdapter类和BF前向；无需下载模型。"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from models.bf_texture_module import BFTextureConditioner


def adapter_class():
    path = Path(__file__).resolve().parents[1] / 'train_texture_adapter.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TextureAdapter']
    env = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['TextureAdapter']


class RecordingUNet(torch.nn.Module):
    def forward(self, latents, steps, context):
        self.context = context
        return SimpleNamespace(sample=context.sum((1, 2)))


class ImageDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bf = BFTextureConditioner(clip_embeddings_dim=16, cross_attention_dim=8,
                                      stage_channels=(8, 16, 32, 64))
        self.unet = RecordingUNet()
        self.model = adapter_class()(self.unet, torch.nn.ModuleList(), self.bf)
        self.args = dict(noisy_latents=torch.zeros(2, 4, 4, 4), timesteps=torch.zeros(2),
                         encoder_hidden_states=torch.randn(2, 5, 8),
                         clip_outputs=SimpleNamespace(image_embeds=torch.randn(2, 16),
                             hidden_states=[torch.randn(2, 5, 16)]),
                         texture_images=torch.randn(2, 3, 32, 32))

    def test_old_pooled_mask_is_ineffective(self):
        before = self.model(**self.args)[1]
        self.args['clip_outputs'].image_embeds.zero_()
        after = self.model(**self.args)[1]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        legacy = self.model(**self.args, drop_image_embeds=[1,1], image_dropout_mode='legacy_pooled')[1]
        torch.testing.assert_close(before, legacy, rtol=0, atol=0)
        self.assertGreater(after.abs().sum().item(), 0)

    def test_off_and_mixed_batch(self):
        before = self.model(**self.args)[1]
        off = self.model(**self.args, drop_image_embeds=[0, 0])[1]
        torch.testing.assert_close(before, off, rtol=0, atol=0)
        mixed = self.model(**self.args, drop_image_embeds=[1, 0])[1]
        self.assertEqual(torch.count_nonzero(mixed[0]).item(), 0)
        torch.testing.assert_close(before[1], mixed[1], rtol=0, atol=0)
        torch.testing.assert_close(self.unet.context[:, :5], self.args['encoder_hidden_states'])

    def test_dropped_reference_independence_and_gradient(self):
        self.args['texture_images'].requires_grad_()
        self.args['clip_outputs'].hidden_states[0].requires_grad_()
        pred, tokens = self.model(**self.args, drop_image_embeds=[1, 1])
        pred.sum().backward()
        self.assertEqual(torch.count_nonzero(self.args['texture_images'].grad).item(), 0)
        self.assertEqual(torch.count_nonzero(self.args['clip_outputs'].hidden_states[0].grad).item(), 0)
        self.args['texture_images'] = torch.randn_like(self.args['texture_images'])
        self.args['clip_outputs'].hidden_states = [torch.randn(2, 5, 16)]
        other, _ = self.model(**self.args, drop_image_embeds=[1, 1])
        torch.testing.assert_close(pred, other, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
