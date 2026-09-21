import unittest
import numpy as np
from tools.e14_real_orientation import make_splits, probe


class OrientationTest(unittest.TestCase):
    def test_grouped_probe(self):
        rows = [dict(sample_id='%s_%s' % (g, y), source_group=str(g), orientation=y)
                for g in range(5) for y in ('horizontal', 'vertical')]
        for group, train, test in make_splits(rows):
            self.assertFalse({rows[i]['source_group'] for i in train} &
                             {rows[i]['source_group'] for i in test})
        x = np.array([[-1.], [1.]] * 5)
        self.assertEqual(probe(x, rows)['balanced_accuracy'], 1.)
        # 同一参考的颜色完全相同：不能借来源标签预测方向。
        colors = np.repeat(np.arange(5.)[:, None], 2, axis=0)
        self.assertEqual(probe(colors, rows)['balanced_accuracy'], .5)
        self.assertEqual(probe(np.ones((10, 3)), rows)['balanced_accuracy'], .5)


if __name__ == '__main__':
    unittest.main()
