import unittest
import torch
from tools.e14_training_supervision_audit import select_rows, training_classes


class SupervisionAuditTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_unique(self):
        rows = [dict(caption='striped' if i < 10 else 'plain') for i in range(30)]
        indices = select_rows(rows, 12, 42)
        self.assertEqual(indices, select_rows(rows, 12, 42))
        self.assertEqual(len(set(indices)), 12)
        self.assertTrue(all(i < 10 for i in indices[:6]))

    def test_actual_gram_permutation_invariance(self):
        _, loss = training_classes()
        x = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        y = x.flatten(2).flip(-1).reshape_as(x)
        torch.testing.assert_close(loss._gram(x), loss._gram(y))


if __name__ == '__main__':
    unittest.main()
