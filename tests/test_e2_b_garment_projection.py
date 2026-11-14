"""旧纹样和新纹样同时存在时，只惩罚旧锚图独有的成分。"""
import unittest

import numpy as np
import torch

from tools.e2_b_garment_projection import GarmentProjection


class GarmentProjectionTest(unittest.TestCase):
    def test_positive_and_same_old_component_match_acceptance(self):
        y, x = np.indices((96, 96))
        new_pattern = 24 * (x % 12 < 6)
        old_pattern = 36 * (y % 16 < 8)
        rgb = lambda a: np.repeat(a[..., None], 3, axis=2).astype(np.uint8)
        positive = rgb(104 + new_pattern)
        anchor = rgb(104 + new_pattern + old_pattern)
        mask = np.zeros((96, 96), bool)
        mask[4:-4, 4:-4] = True
        metric = GarmentProjection(positive, anchor, mask, 'cpu')
        self.assertEqual(metric.exact(positive), 0.)
        self.assertAlmostEqual(metric.exact(anchor), 1., places=5)
        current = rgb(104 + new_pattern + old_pattern // 2)
        value = torch.from_numpy(current.copy()).permute(2, 0, 1)[None].float().div(255).requires_grad_()
        loss = metric.surrogate(value)
        self.assertAlmostEqual(float(loss.detach()), metric.exact(current), delta=.01)
        gradient, = torch.autograd.grad(loss, value)
        with torch.no_grad():
            changed = (value - gradient).clamp(0, 1)
        self.assertLess(float(metric.surrogate(changed)), float(loss.detach()))


if __name__ == '__main__':
    unittest.main()
