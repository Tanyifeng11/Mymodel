import copy
import json
import unittest
from unittest import mock

import torch

from models.text_guided_queries import TextGuidedQueries
from tools.check_e8c_frozen import REQUIRED_COMPONENTS, audit_e8c, main


def checkpoints():
    reference = {name: {"weight": torch.tensor([0.12345, 0.75])} for name in REQUIRED_COMPONENTS}
    reference["bf_texture_conditioner"] = {
        "resampler_queries": torch.ones(1, 2, 8), "resampler.weight": torch.ones(8, 8),
        "token_norm.weight": torch.ones(8),
    }
    reference["meta"] = {"use_aa_tcr_fuse": 0}
    candidate = copy.deepcopy(reference)
    candidate["bf_texture_conditioner"].update({
        "text_guidance." + key: value for key, value in TextGuidedQueries(8, 4, 2).state_dict().items()
    })
    candidate["meta"].update(text_guidance_dim=4, text_guidance_heads=2,
                             text_guidance_max_ratio=0.3, resampler_training="text_only")
    return reference, candidate


class FrozenE8cTests(unittest.TestCase):
    def test_only_text_addition_passes(self):
        reference, candidate = checkpoints()
        report, rows = audit_e8c(reference, candidate)
        self.assertTrue(report["frozen_passed"])
        self.assertEqual(report["allowed_text_tensors"], 9)
        self.assertFalse(any(row["unexpected"] for row in rows))
        json.dumps(report, allow_nan=False)

    def test_query_and_visual_resampler_drift_fail(self):
        for key in ("resampler_queries", "resampler.weight"):
            with self.subTest(key=key):
                reference, candidate = checkpoints()
                candidate["bf_texture_conditioner"][key].add_(0.01)
                report, rows = audit_e8c(reference, candidate)
                self.assertFalse(report["frozen_passed"])
                row = next(row for row in rows if row["key"] == key)
                self.assertTrue(row["unexpected"])
                self.assertFalse(row["expected_trainable"])

    def test_other_frozen_drift_and_quantization_fail(self):
        for quantize in (False, True):
            with self.subTest(quantize=quantize):
                reference, candidate = checkpoints()
                value = candidate["unet"]["weight"]
                candidate["unet"]["weight"] = value.half().float() if quantize else value + 0.01
                report, _ = audit_e8c(reference, candidate)
                self.assertFalse(report["frozen_passed"])

    def test_missing_text_schema_fails(self):
        reference, candidate = checkpoints()
        del candidate["bf_texture_conditioner"]["text_guidance.gate"]
        report, _ = audit_e8c(reference, candidate)
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any("invalid_text_schema" in row["status"] for row in report["violations"]))

    def test_only_disabled_missing_to_empty_aa_is_compatible(self):
        reference, candidate = checkpoints()
        candidate["aa_tcr_fuser"] = {}
        report, _ = audit_e8c(reference, candidate)
        self.assertTrue(report["frozen_passed"])
        self.assertEqual(report["compatible_format_differences"], ["disabled_aa_tcr_missing_to_empty"])
        candidate["meta"]["use_aa_tcr_fuse"] = 1
        self.assertFalse(audit_e8c(reference, candidate)[0]["frozen_passed"])

    def test_nonfinite_shape_and_missing_component_fail(self):
        for problem in ("nonfinite", "shape", "missing"):
            with self.subTest(problem=problem):
                reference, candidate = checkpoints()
                if problem == "nonfinite":
                    candidate["unet"]["weight"][0] = float("nan")
                elif problem == "shape":
                    candidate["unet"]["weight"] = torch.ones(3)
                else:
                    del candidate["unet"]
                report, _ = audit_e8c(reference, candidate)
                self.assertFalse(report["frozen_passed"])
                json.dumps(report, allow_nan=False)

    def test_e5_text_and_wrong_training_mode_fail(self):
        reference, candidate = checkpoints()
        candidate["meta"]["resampler_training"] = "text"
        self.assertFalse(audit_e8c(reference, candidate)[0]["frozen_passed"])
        reference, candidate = checkpoints()
        reference["meta"]["text_guidance_dim"] = 4
        self.assertFalse(audit_e8c(reference, candidate)[0]["frozen_passed"])
        for mode in ("visual", "text", "text_only"):
            with self.subTest(initialization_mode=mode):
                reference, candidate = checkpoints()
                reference["meta"]["resampler_training"] = mode
                report, _ = audit_e8c(reference, candidate)
                self.assertFalse(report["frozen_passed"])
                self.assertTrue(any(error["error"] == "initialization_training_mode_must_be_off"
                                    for error in report["errors"]))

    @mock.patch("tools.check_e8c_frozen.write_outputs")
    @mock.patch("tools.check_e8c_frozen.load_checkpoint")
    def test_cli_return_codes_and_failed_report(self, load, write):
        args = ["--e5-ckpt", "e5.pt", "--text-ckpt", "e8c.pt", "--output-dir", "check"]
        load.side_effect = checkpoints()
        self.assertEqual(main(args), 0)
        reference, candidate = checkpoints()
        candidate["bf_texture_conditioner"]["resampler_queries"].add_(1)
        load.side_effect = [reference, candidate]
        self.assertEqual(main(args), 1)
        load.side_effect = RuntimeError("synthetic load error")
        self.assertEqual(main(args), 1)
        report = write.call_args.args[1]
        self.assertTrue(report["execution_failed"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["errors"][0]["message"], "synthetic load error")


if __name__ == "__main__":
    unittest.main()
