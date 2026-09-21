import unittest
import torch
from tools.e14_checkpoint_history import CHECKPOINTS, compare, compact, fingerprint


class HistoryTests(unittest.TestCase):
    def test_exact_paths(self):
        self.assertIn('checkpoint-28365/', CHECKPOINTS['e0'])
        self.assertTrue(CHECKPOINTS['texture'].endswith('texture_adapter.bin'))

    def test_module_differences_and_hashes(self):
        a = {'stage1.weight': torch.ones(2, 3), 'resampler_queries': torch.zeros(1, 16, 4)}
        b = {k: v.clone() for k, v in a.items()}
        self.assertEqual(fingerprint(a), fingerprint(b))
        b['stage1.weight'][0, 0] = 2
        self.assertNotEqual(fingerprint(a)[0], fingerprint(b)[0])
        report = compare(a, b)
        self.assertTrue(report['resampler_queries']['equal_values'])
        self.assertEqual(report['stage1']['changed_elements'], 1)
        b['stage1.weight'] = torch.ones(3, 3)
        self.assertFalse(compare(a, b)['stage1']['compatible'])

    def test_early_checkpoint_has_no_synthetic_tcpm(self):
        small = compact({'bf_texture_conditioner': {}, 'metadata': {'texture_mode': 'patch_resampled'}})
        self.assertNotIn('tcpm_lite', small)
        self.assertEqual(small['meta']['texture_mode'], 'patch_resampled')
        with self.assertRaises(ValueError):
            compact({'bf_texture_conditioner': {}, 'meta': {'stage_token_hw': [4, 4]}})

    def test_bf_only_build_and_capture(self):
        from models.bf_texture_module import BFTextureConditioner
        from tools.e8_token_probe import build_conditioner
        from tools.e14_pattern_probe import capture
        original = BFTextureConditioner(clip_embeddings_dim=16, cross_attention_dim=8,
                                       stage_channels=(8, 16, 32, 64))
        bf, tcpm = build_conditioner({'bf_texture_conditioner': original.state_dict()}, bf_only=True)
        self.assertIsNone(tcpm)
        values = capture(bf, tcpm, dict(clip_vision_tokens=torch.randn(1, 4, 16),
                         texture_images=torch.randn(1, 3, 32, 32)), None, None)
        self.assertEqual(tuple(values['tokens_post_ln'].shape), (1, 16, 8))
        self.assertFalse(any(k.startswith('tcpm') for k in values))


if __name__ == '__main__':
    unittest.main()
