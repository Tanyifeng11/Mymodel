import unittest
import numpy as np
from tools.e14_color_matched import source_units, source_scores, match_triplets, evaluate_scores


class ColorMatchingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=str(i), source_group=g, pattern=p, color_group=c)
                     for i, (g, p, c) in enumerate([
                         ('a', 'stripe', 'dark'), ('b', 'stripe', 'dark'),
                         ('c', 'dots', 'dark'), ('d', 'dots', 'light')])]
        self.units = source_units(self.rows)
        self.color = np.array([[1, .8, .81, .8], [.8, 1, .3, .8], [.81, .3, 1, .8], [.8, .8, .8, 1]])

    def test_matching_color_and_source_constraints(self):
        t = match_triplets(self.color, self.units, .02, .5)
        self.assertEqual([(x['anchor'], x['positive'], x['negative']) for x in t], [(0, 1, 2)])
        self.assertEqual(match_triplets(self.color, self.units, .001, .5), [])
        self.assertEqual(match_triplets(self.color, self.units, .02, .9), [])

    def test_ties_and_no_data(self):
        t = match_triplets(self.color, self.units, .02, .5)
        result = evaluate_scores(np.ones((4, 4)), t, self.units)
        self.assertEqual(result['accuracy'], .5)
        self.assertIsNone(evaluate_scores(self.color, [], self.units)['accuracy'])

    def test_duplicate_crops_do_not_add_sources(self):
        rows = self.rows + [{**self.rows[0], 'sample_id': 'copy'}]
        units = source_units(rows)
        matrix = np.arange(25).reshape(5, 5)
        scores = source_scores(matrix, units)
        self.assertEqual(len(units), 4)
        self.assertEqual(scores[0, 1], (matrix[0, 1] + matrix[4, 1]) / 2)

    def test_source_label_conflicts_rejected(self):
        with self.assertRaises(ValueError):
            source_units(self.rows + [{**self.rows[0], 'pattern': 'dots'}])

    def test_anchor_macro_average_not_triplet_average(self):
        triples = [dict(anchor=0, positive=1, negative=2)] * 3
        triples += [dict(anchor=1, positive=0, negative=2)]
        scores = np.array([[1, .9, .1, 0], [.9, 1, .95, 0], [.1, .95, 1, 0], [0, 0, 0, 1]])
        result = evaluate_scores(scores, triples, self.units)
        self.assertEqual(result['accuracy'], .5)


if __name__ == '__main__':
    unittest.main()
