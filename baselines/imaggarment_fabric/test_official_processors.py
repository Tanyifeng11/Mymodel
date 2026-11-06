"""不加载大权重，在 CPU 核验官方算子、兼容包装和梯度通路。"""
import unittest
import torch
from diffusers.models.attention_processor import Attention
from .network import ImageProjModel, FabricAttnProcessor, IPAttnProcessor2_0
from .network import LogoCacheSAttnProcessor2_0, LogoRefSAttnProcessor2_0


class OfficialProcessorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_pooled_projection_has_four_tokens(self):
        projection = ImageProjModel()
        first = projection(torch.zeros(1, 1024))
        second = projection(torch.ones(1, 1024))
        self.assertEqual(tuple(first.shape), (1, 4, 768))
        self.assertFalse(torch.equal(first, second))
        second.square().mean().backward()
        self.assertGreater(float(projection.proj.weight.grad.abs().sum()), 0)

    def test_compatibility_wrapper_preserves_official_attention(self):
        attention = Attention(query_dim=32, cross_attention_dim=768, heads=4, dim_head=8)
        official = IPAttnProcessor2_0(32, 768, num_tokens=4)
        wrapper = FabricAttnProcessor(32, 768, num_tokens=4)
        wrapper.load_state_dict(official.state_dict(), strict=True)
        hidden = torch.randn(1, 8, 32)
        context = torch.randn(1, 81, 768)
        first = official(attention, hidden, encoder_hidden_states=context)
        second = wrapper(attention, hidden, encoder_hidden_states=context, sa_hidden_states={'unused': None})
        self.assertTrue(torch.equal(first, second))
        second.square().mean().backward()
        self.assertGreater(float(wrapper.to_k_ip.weight.grad.abs().sum()), 0)

    def test_sketch_cache_reaches_reference_and_adapter(self):
        attention = Attention(query_dim=32, heads=4, dim_head=8)
        cache = LogoCacheSAttnProcessor2_0('layer', 32)
        reference = torch.randn(1, 8, 32, requires_grad=True)
        cache(attention, reference)
        reader = LogoRefSAttnProcessor2_0('layer', 32)
        result = reader(attention, torch.randn(1, 8, 32),
                        sa_hidden_states={'layer': cache.cache['hidden_states']})
        result.square().mean().backward()
        self.assertGreater(float(reference.grad.abs().sum()), 0)
        self.assertGreater(float(reader.to_k_ref.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
