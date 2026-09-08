"""核验 E9 冻结检查：只放行完整的局部旁路，并要求 A/B 除来源外配置一致。"""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.local_detail_adapter import DEFAULT_LOCAL_DETAIL_LAYER, LocalDetailAdapter
from tools.check_e9_frozen import (
    ADAPTER_TENSORS, REQUIRED_COMPONENTS, audit_parity, audit_variant, main,
)


LAYER = DEFAULT_LOCAL_DETAIL_LAYER
PROCESSOR_INDEX = "37"


def base_meta():
    return {
        "use_aa_tcr_fuse": 0, "resampler_training": "off", "text_guidance_dim": 0,
        "texture_num_tokens": 16, "texture_mode": "patch_resampled",
        "texture_condition_mode": "token", "texture_preprocess_mode": "plain_resize",
        "clip_hidden_layer": -1, "width": 384, "height": 512, "layer_group_enabled": 1,
        "use_texture_gate": 1, "use_tcpm_lite": 1, "tcpm_mask_inner_only": 1,
        "region_kernel_size": 9,
    }


def branch_meta(source):
    return {
        "local_detail_source": source, "local_detail_grid": 16, "local_detail_layer": LAYER,
        "local_detail_dim": 8, "local_detail_heads": 2, "local_detail_lr": 5e-5,
        "train_batch_size": 1, "gradient_accumulation_steps": 8, "max_train_steps": 1000,
        "num_warmup_steps": 50, "max_grad_norm": 1.0, "mixed_precision": "fp16",
        "dataset_json_path": "data/train_bf_texture.json", "data_root_path": "/datasets/BF/training",
        "training_data_sha256": "a" * 64, "seed": 42, "train_global_step": 1000,
    }


def adapter_state():
    torch.manual_seed(0)
    state = LocalDetailAdapter(16, 768, 8, 2).state_dict()
    assert set(state) == set(ADAPTER_TENSORS), sorted(state)
    return state


def checkpoints(source="local"):
    reference = {name: {"weight": torch.tensor([0.12345, 0.75])} for name in REQUIRED_COMPONENTS}
    reference["bf_texture_conditioner"] = {
        "resampler_queries": torch.ones(1, 2, 16), "resampler.weight": torch.ones(16, 16),
        "token_norm.weight": torch.ones(16),
    }
    reference["meta"] = base_meta()
    candidate = copy.deepcopy(reference)
    state = adapter_state()
    for component, prefix in (("unet", LAYER + "."), ("texture_adapter", PROCESSOR_INDEX + ".")):
        candidate[component].update({prefix + "local_detail_adapter." + key: value.clone()
                                     for key, value in state.items()})
    candidate["meta"].update(branch_meta(source))
    return reference, candidate


class FrozenE9Tests(unittest.TestCase):
    def test_only_local_detail_addition_passes(self):
        reference, candidate = checkpoints()
        report, rows = audit_variant(reference, candidate, "e9_b", "local")
        self.assertTrue(report["frozen_passed"], report["errors"] + report["violations"])
        self.assertEqual(report["allowed_local_detail_tensors"], 2 * len(ADAPTER_TENSORS))
        self.assertFalse(any(row["unexpected"] for row in rows))
        json.dumps(report, allow_nan=False)

    def test_query_and_visual_resampler_drift_fail(self):
        # 默认白名单把这两项当可训练；E9 必须覆盖为冻结。
        for key in ("resampler_queries", "resampler.weight"):
            with self.subTest(key=key):
                reference, candidate = checkpoints()
                candidate["bf_texture_conditioner"][key] = (
                    candidate["bf_texture_conditioner"][key] + 0.01)
                report, rows = audit_variant(reference, candidate, "e9_b", "local")
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
                report, _ = audit_variant(reference, candidate, "e9_b", "local")
                self.assertFalse(report["frozen_passed"])

    def test_incomplete_branch_weights_fail(self):
        reference, candidate = checkpoints()
        del candidate["unet"][LAYER + ".local_detail_adapter.alpha"]
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any(item.get("error") == "local_detail_weights_incomplete"
                            for item in report["errors"]))

    def test_branch_on_unexpected_layer_fails(self):
        reference, candidate = checkpoints()
        candidate["unet"]["down_blocks.0.attn2.processor.local_detail_adapter.alpha"] = torch.zeros(())
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any(item.get("error") == "local_detail_weights_on_unexpected_layer"
                            for item in report["errors"]))

    def test_disagreeing_exports_fail(self):
        reference, candidate = checkpoints()
        key = PROCESSOR_INDEX + ".local_detail_adapter.to_q.weight"
        candidate["texture_adapter"][key] = candidate["texture_adapter"][key] + 0.01
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any(item.get("error") == "local_detail_exports_disagree"
                            for item in report["errors"]))

    def test_source_mismatch_and_disabled_source_fail(self):
        for expected, actual in (("local", "resampled"), ("local", "off")):
            with self.subTest(expected=expected, actual=actual):
                reference, candidate = checkpoints(actual)
                report, _ = audit_variant(reference, candidate, "e9_b", expected)
                self.assertFalse(report["frozen_passed"])
                self.assertTrue(any(item.get("error") == "local_detail_source_mismatch"
                                    for item in report["errors"]))

    def test_initialization_with_existing_new_modules_fails(self):
        cases = (
            ("meta", "local_detail_source", "local"),
            ("meta", "resampler_training", "text_only"),
            ("meta", "text_guidance_dim", 256),
        )
        for component, key, value in cases:
            with self.subTest(key=key):
                reference, candidate = checkpoints()
                reference[component][key] = value
                report, _ = audit_variant(reference, candidate, "e9_b", "local")
                self.assertFalse(report["frozen_passed"])

    def test_initialization_carrying_branch_weights_fails(self):
        reference, candidate = checkpoints()
        reference["unet"][LAYER + ".local_detail_adapter.alpha"] = torch.zeros(())
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any(item.get("error") == "initialization_contains_new_module_weights"
                            for item in report["errors"]))

    def test_base_config_drift_from_e5_fails(self):
        reference, candidate = checkpoints()
        candidate["meta"]["width"] = 512
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(any(item.get("error") == "base_config_differs_from_e5"
                            for item in report["errors"]))

    def test_missing_required_component_fails(self):
        reference, candidate = checkpoints()
        del candidate["tcpm_lite"]
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertFalse(report["frozen_passed"])

    def test_legacy_missing_aa_tcr_is_tolerated(self):
        reference, candidate = checkpoints()
        candidate["aa_tcr_fuser"] = {}
        report, _ = audit_variant(reference, candidate, "e9_b", "local")
        self.assertTrue(report["frozen_passed"], report["errors"] + report["violations"])
        self.assertIn("disabled_aa_tcr_missing_to_empty", report["compatible_format_differences"])


class ParityTests(unittest.TestCase):
    def test_matching_training_config_passes(self):
        report = audit_parity({"e9_a": branch_meta("resampled"), "e9_b": branch_meta("local")})
        self.assertEqual(report["status"], "passed", report["differences"])
        self.assertEqual(report["missing_fields"], [])

    def test_training_amount_difference_fails(self):
        second = branch_meta("local")
        second["max_train_steps"] = 2000
        report = audit_parity({"e9_a": branch_meta("resampled"), "e9_b": second})
        self.assertEqual(report["status"], "failed")
        self.assertEqual([item["field"] for item in report["differences"]], ["max_train_steps"])

    def test_identical_source_fails(self):
        report = audit_parity({"e9_a": branch_meta("local"), "e9_b": branch_meta("local")})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(item["field"] == "local_detail_source" for item in report["differences"]))

    def test_single_variant_is_skipped(self):
        report = audit_parity({"e9_b": branch_meta("local")})
        self.assertEqual(report["status"], "skipped")


class MainTests(unittest.TestCase):
    def paths(self, tmp, reference, candidates):
        root = Path(tmp)
        torch.save(reference, root / "e5.pt")
        for label, candidate in candidates.items():
            torch.save(candidate, root / (label + ".pt"))
        return root

    def run_main(self, candidates_sources):
        reference, _ = checkpoints()
        candidates = {label: checkpoints(source)[1] for label, source in candidates_sources.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = self.paths(tmp, reference, candidates)
            argv = ["--e5-ckpt", str(root / "e5.pt"), "--output-dir", str(root / "out")]
            for label in candidates:
                argv += ["--e9-%s-ckpt" % label.rsplit("_", 1)[-1], str(root / (label + ".pt"))]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(argv)
            report = json.loads((root / "out/frozen_check.json").read_text(encoding="utf-8"))
            csv_lines = (root / "out/frozen_differences.csv").read_text(encoding="utf-8-sig").splitlines()
        return code, report, csv_lines

    def test_both_variants_pass_and_report_parity(self):
        code, report, csv_lines = self.run_main({"e9_a": "resampled", "e9_b": "local"})
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(sorted(report["variants"]), ["e9_a", "e9_b"])
        self.assertEqual(report["parity"]["status"], "passed")
        # 新增张量在两组、两处导出各 9 个，加表头共 37 行。
        self.assertEqual(len(csv_lines), 1 + 2 * 2 * len(ADAPTER_TENSORS))

    def test_single_variant_skips_parity(self):
        code, report, _ = self.run_main({"e9_b": "local"})
        self.assertEqual(code, 0)
        self.assertEqual(report["parity"]["status"], "skipped")

    def test_missing_candidate_argument_fails(self):
        reference, _ = checkpoints()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(reference, root / "e5.pt")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--e5-ckpt", str(root / "e5.pt"), "--output-dir", str(root / "out")])
            report = json.loads((root / "out/frozen_check.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertFalse(report["frozen_passed"])
        self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
