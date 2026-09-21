import unittest
import numpy as np
from tools.e14_rotation_features import analyze
from tools.e14_pattern_probe import similarity


class RotationFeatureTests(unittest.TestCase):
    def test_shared_orientation_and_invariant_baseline(self):
        rows=[dict(sample_id=str(i),source_group=str(i//2),orientation=['horizontal','vertical'][i%2]) for i in range(4)]
        x=np.array([[1.,0],[0,1],[1,0],[0,1]])
        r=analyze(similarity(x),rows)
        self.assertEqual(len(r['rotation_pairs']),2)
        self.assertEqual(len(r['cross_pattern_pairs']),4)
        self.assertTrue(all(v['same_minus_opposite_cosine']>0 for v in r['cross_pattern_orientation_preferences']))
        r=analyze(np.ones((4,4)),rows)
        self.assertEqual(r['mean_rotation_distance'],0)
        self.assertIsNone(r['rotation_to_cross_ratio'])


if __name__=='__main__':unittest.main()
