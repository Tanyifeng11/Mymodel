import unittest
import numpy as np
from tools.e14_real_orientation import make_splits, probe, exact_linear_coordinates, linear_model


class OrientationTest(unittest.TestCase):
    def test_exact_coordinates_preserve_linear_model(self):
        rng = np.random.RandomState(42)
        a, b = rng.randn(12, 100), rng.randn(4, 100)
        u, v = exact_linear_coordinates(a, b)
        np.testing.assert_allclose(u @ u.T, a @ a.T, atol=1e-10)
        np.testing.assert_allclose(v @ u.T, b @ a.T, atol=1e-10)
        y = np.array([0, 1] * 6)
        original, equivalent = linear_model(.1), linear_model(.1)
        original.fit(a, y)
        equivalent.fit(u, y)
        np.testing.assert_allclose(original.decision_function(b), equivalent.decision_function(v), atol=1e-6)

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
        self.assertEqual(probe(x, rows, no_pca=True)['balanced_accuracy'], 1.)
        self.assertEqual(probe(colors, rows, no_pca=True)['balanced_accuracy'], .5)
        result = probe(np.ones((10, 3)), rows, no_pca=True)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertTrue(all(f['selected_C'] == .0001 for f in result['folds']))


if __name__ == '__main__':
    unittest.main()
