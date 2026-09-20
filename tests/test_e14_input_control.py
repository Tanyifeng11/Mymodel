import unittest
import numpy as np
import torch
from tools.e14_input_control import rank_binary, channel_signature


class InputControlTests(unittest.TestCase):
    def test_exact_distribution_after_fp16_normalization(self):
        x = torch.rand(1, 3, 32, 24)
        a, _ = rank_binary(x)
        b, _ = rank_binary(1-x)
        for t in (a, b):
            self.assertEqual(int((t[0,0] < .5).sum()), 192)
        mean = torch.tensor([.48,.46,.41]).view(1,3,1,1)
        std = torch.tensor([.26,.27,.28]).view(1,3,1,1)
        self.assertEqual(channel_signature(((a-mean)/std).half()), channel_signature(((b-mean)/std).half()))

    def test_idempotent_for_already_controlled_image(self):
        x = torch.full((1,3,16,16), 224/255)
        x[:,:,:4] = 32/255
        y, _ = rank_binary(x)
        torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_ties_are_deterministic_and_shape_preserved(self):
        x = torch.ones(1,3,16,16)
        a, stats = rank_binary(x)
        b, _ = rank_binary(x)
        self.assertEqual(a.shape,x.shape)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(stats[0]['cutoff_tied_pixels'],256)


if __name__ == '__main__':
    unittest.main()
