import unittest
import numpy as np
from tools.e14_grouped_linear import make_splits, probe, transform


class GroupedLinearTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=f'{p}_f{f}_a0_p{phase:02d}', pattern=p, source_group=f'{p}_f{f}')
                     for p in ('dots', 'plaid', 'stripe') for f in (3, 5, 8, 12) for phase in (0, 27)]

    def test_frequency_and_source_isolation(self):
        folds = make_splits(self.rows)
        tests = []
        for held, train, test, inner in folds:
            tests.extend(test.tolist())
            for a, b in [(train, test)] + inner:
                self.assertFalse({self.rows[i]['source_group'] for i in a} &
                                 {self.rows[i]['source_group'] for i in b})
        self.assertEqual(sorted(tests), list(range(len(self.rows))))

    def test_preprocessing_does_not_fit_test_data(self):
        x = np.array([[1., 2.], [2., 4.], [3., 7.], [4., 8.], [5., 10.]])
        a, b = transform(x, np.array([0, 1, 2, 3]), np.array([4]), 1)
        x[4] = 1e6
        c, d = transform(x, np.array([0, 1, 2, 3]), np.array([4]), 1)
        np.testing.assert_allclose(a, c)
        self.assertGreater(np.linalg.norm(d), np.linalg.norm(b))

    def test_class_signal_and_constant_baseline(self):
        classes = ['dots', 'plaid', 'stripe']
        x = np.array([np.eye(3)[classes.index(r['pattern'])] for r in self.rows])
        folds = make_splits(self.rows)
        self.assertEqual(probe(x, self.rows, folds)['balanced_accuracy'], 1)
        self.assertAlmostEqual(probe(np.ones_like(x), self.rows, folds)['balanced_accuracy'], 1/3)


if __name__ == '__main__':
    unittest.main()
