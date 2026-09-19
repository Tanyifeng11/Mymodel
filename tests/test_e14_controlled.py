import unittest
from tools.e14_controlled_patterns import pattern_mask


class ControlledPatternsTests(unittest.TestCase):
    def test_exact_occupancy_and_distinct_patterns(self):
        hashes = set()
        for kind in ('stripe', 'plaid', 'dots'):
            for cycles in (3, 5, 8, 12):
                for angle in (0, 30):
                    for phase in (0., .27):
                        mask = pattern_mask(kind, 256, cycles, angle, phase)
                        self.assertEqual(int(mask.sum()), 16384)
                        hashes.add(mask.tobytes())
        self.assertEqual(len(hashes), 48)


if __name__ == '__main__':
    unittest.main()
