import unittest

import torch

from tools.e23_m2 import bank_for, token_interventions, token_rms


class E23M2InterventionTests(unittest.TestCase):
    def test_token_controls_hold_the_intended_quantities(self):
        generator = torch.Generator().manual_seed(7)
        mean = torch.randn(1, 1, 32, generator=generator)
        e5 = mean + .5 * torch.randn(1, 16, 32, generator=generator)
        e17 = 1.3 * mean + .01 * torch.randn(1, 16, 32, generator=generator)
        changed, _ = token_interventions((e5, e5), (e17, e17))
        collapse = changed["E5_collapse"][0]
        norm = changed["E17_norm"][0]
        decollapse = changed["E17_self_decollapse"][0]
        self.assertTrue(torch.allclose(collapse, collapse[:, :1].expand_as(collapse)))
        self.assertAlmostEqual(float(token_rms(collapse)), float(token_rms(e5)), places=5)
        self.assertAlmostEqual(float(token_rms(norm)), float(token_rms(e5)), places=5)
        self.assertAlmostEqual(float(token_rms(decollapse)), float(token_rms(e17)), places=5)
        self.assertAlmostEqual(float(token_rms(decollapse - decollapse.mean(1, keepdim=True))),
                               float(token_rms(e5 - e5.mean(1, keepdim=True))), places=5)
        self.assertTrue(torch.equal(changed["E17_self_decollapse"][1], e17))

    def test_cfg_swap_changes_one_branch_at_a_time(self):
        e5 = (torch.ones(1, 16, 3), torch.full((1, 16, 3), 2.))
        e17 = (torch.full((1, 16, 3), 3.), torch.full((1, 16, 3), 4.))
        banks = {"E5": {(0, "original"): e5, (0, "rot90"): e5},
                 "E17_direct": {(0, "original"): e17, (0, "rot90"): e17}}
        expected = {"G0": (1, 2), "G1": (3, 2), "G2": (1, 4), "G3": (3, 4)}
        for group, (positive, negative) in expected.items():
            pair = bank_for(banks, group, 1)[0, "original"]
            self.assertEqual(float(pair[0][0, 0, 0]), positive)
            self.assertEqual(float(pair[1][0, 0, 0]), negative)


if __name__ == "__main__":
    unittest.main()
