"""测试评价的必要性质，不用效果结果来选择协议。"""
import unittest
import numpy as np
from tools.m12_color_metrics import descriptor, compare, calibrate


class ColorMetrics(unittest.TestCase):
    def test_background_invariance_and_palette(self):
        a = np.full((16, 16, 3), [30, 90, 180], np.uint8)
        mask = np.zeros((16, 16), bool);mask[4:12, 4:12] = True
        b = a.copy();b[~mask] = [255, 0, 0]
        same = compare(descriptor(a, mask), descriptor(b, mask), '')
        self.assertEqual(same['rgbuv_h'], 0.)
        changed = np.full_like(a, [180, 90, 30])
        self.assertGreater(compare(descriptor(a), descriptor(changed), '')['rgbuv_h'], .5)

    def test_luminance_separation_and_neutral_na(self):
        a = descriptor(np.full((16, 16, 3), 80, np.uint8))
        b = descriptor(np.full((16, 16, 3), 180, np.uint8))
        delta = compare(a, b, '')
        self.assertGreater(delta['l_w1'], 30.)
        self.assertLess(delta['ab_w1'], .1)
        self.assertIsNone(delta['hue_h'])
        self.assertIsNone(delta['chroma_log_ratio'])

    def test_off_and_outside_are_exact(self):
        a = np.full((20, 20, 3), [20, 90, 190], np.uint8)
        ref = np.full_like(a, [130, 80, 50]);mask = np.zeros((20, 20), np.float32);mask[4:16, 4:16] = 1
        self.assertTrue(np.array_equal(calibrate(a, ref, mask, False), a))
        self.assertTrue(np.array_equal(calibrate(a, ref, mask, True)[mask == 0], a[mask == 0]))


if __name__ == '__main__':
    unittest.main()
