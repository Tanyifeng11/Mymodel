import unittest

import numpy as np
import torch
from PIL import Image

from models.pattern_scaffold import apply_garment_mask, render_stripe_field
from tools.e18_1_clean_patterns import PALETTES, make_image
from tools.e25_spatial_diagnosis import (STRENGTHS, period_readout, spec_for,
                                         templates_for, timestep_for)


class E25OracleTests(unittest.TestCase):
    def test_renderer_matches_both_reference_rotations(self):
        palette = PALETTES[0]
        for index in (0, 1):
            ref = {"id": index, "frequency": 9, "phase": .23, "palette": 0}
            source = make_image(9, .23, palette)
            if index % 2:
                source = source.transpose(Image.Transpose.ROTATE_90)
            for variant in ("original", "rot90"):
                turns = index % 2 + int(variant == "rot90")
                expected = source if variant == "original" else source.transpose(Image.Transpose.ROTATE_90)
                meta = {"variant": variant, "theta": float((90 + 90 * turns) % 180)}
                actual = render_stripe_field(spec_for(ref, meta))
                self.assertTrue(np.array_equal(np.asarray(actual), np.asarray(expected)))

    def test_clean_period_calibration_and_white_background(self):
        mask = Image.new("L", (256, 256), 0)
        pixels = np.asarray(mask).copy()
        pixels[24:232, 24:232] = 255
        mask = Image.fromarray(pixels)
        frequencies = (6, 7, 9, 11, 13, 15)
        for theta in (0, 90):
            ref = {"id": int(theta == 0), "frequency": 9, "phase": .23, "palette": 0}
            variant = {"variant": "original", "theta": theta}
            roi = (32, 32, 192, 192)
            bank = templates_for(ref, variant, mask, mask.size, roi, frequencies)
            for frequency in frequencies:
                candidate = dict(ref, frequency=frequency)
                image = apply_garment_mask(render_stripe_field(spec_for(candidate, variant)), mask, mask.size)
                self.assertTrue(np.all(np.asarray(image)[pixels == 0] == 255))
                self.assertTrue(period_readout(image, roi, theta, bank, frequency)["correct"])

    def test_ddim_strength_mapping(self):
        class Scheduler:
            alphas_cumprod = torch.linspace(1., .001, 1000)

            def set_timesteps(self, count):
                self.timesteps = torch.arange(980, -1, -20)

        steps = [timestep_for(Scheduler(), s) for s in STRENGTHS.values()]
        self.assertEqual([r["remaining_steps"] for r in steps], [8, 18, 28])
        self.assertEqual([r["timestep"] for r in steps], [140, 340, 540])


if __name__ == "__main__":
    unittest.main()
