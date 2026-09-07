"""用小型真实 BF 模块验证诊断能发现故障，且不改变原始 checkpoint。"""

import contextlib
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.bf_texture_module import BFTextureConditioner
from models.tcpm_lite import TCPMLite
from tools.diagnose_e8 import main
from tools.e8_checkpoint_audit import compare_checkpoints, metadata_comparison
from tools.e8_token_probe import build_conditioner, make_zero_init, probe_sample


def checkpoints():
    bf = BFTextureConditioner(clip_embeddings_dim=24, cross_attention_dim=32,
                             stage_channels=(8, 8, 8, 8))
    base = {
        "checkpoint_format": "gam_texture_joint_v3",
        "unet": {"weight": torch.randn(8)}, "ref_unet": {"weight": torch.randn(8)},
        "texture_adapter": {"weight": torch.randn(8)},
        "bf_texture_conditioner": copy.deepcopy(bf.state_dict()),
        "tcpm_lite": TCPMLite(32).state_dict(),
        "meta": {"texture_num_tokens": 16, "use_tcpm_lite": 1, "text_guidance_dim": 0},
    }
    visual = copy.deepcopy(base)
    visual["bf_texture_conditioner"]["resampler_queries"].add_(0.01)
    bf.load_state_dict(visual["bf_texture_conditioner"], strict=True)
    bf.configure_text_guidance(16, 4, 0.3)
    bf.text_guidance.gate.data.fill_(0.1)
    text = copy.deepcopy(visual)
    text["bf_texture_conditioner"] = copy.deepcopy(bf.state_dict())
    text["meta"].update(bf.text_guidance_config())
    return {"e5": base, "e8a": visual, "e8b": text}


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.states = checkpoints()

    def test_expected_changes_and_frozen_corruption(self):
        for candidate in ("e8a", "e8b"):
            rows = compare_checkpoints(self.states["e5"], self.states[candidate], candidate,
                                       candidate_has_text=candidate == "e8b")
            self.assertFalse(any(row["unexpected"] for row in rows))
            self.assertTrue(any(row["status"] == "changed_trainable" for row in rows))
        self.states["e8b"]["unet"]["weight"][0] += 1
        rows = compare_checkpoints(self.states["e5"], self.states["e8b"], "test", True)
        self.assertTrue(any(row["component"] == "unet" and row["unexpected"] for row in rows))

    def test_missing_weights_and_nonfinite_are_not_accepted(self):
        del self.states["e8b"]["bf_texture_conditioner"]["text_guidance.gate"]
        self.states["e8b"]["tcpm_lite"]["residual_scale"].fill_(float("nan"))
        del self.states["e8b"]["texture_adapter"]
        rows = compare_checkpoints(self.states["e5"], self.states["e8b"], "test", True)
        self.assertTrue(any("invalid_text_schema" in row["status"] for row in rows))
        self.assertTrue(any(row["candidate_nonfinite"] for row in rows))
        self.assertTrue(any(row["status"] == "missing_component" for row in rows))
        with self.assertRaises((ValueError, RuntimeError)):
            build_conditioner(self.states["e8b"])

    def test_precision_rounding_and_metadata_are_reported(self):
        self.states["e8a"]["unet"]["weight"] = self.states["e5"]["unet"]["weight"].half().float()
        rows = compare_checkpoints(self.states["e5"], self.states["e8a"], "test")
        row = next(row for row in rows if row["component"] == "unet")
        self.assertTrue(row["precision_cast_only"])
        self.assertTrue(row["unexpected"])
        self.assertGreater(row["max_abs_delta"], 0)
        self.states["e8a"]["meta"]["width"] = 384
        changes = metadata_comparison(self.states["e5"], self.states["e8a"])
        self.assertTrue(any(row["key"] == "width" and row["status"] == "missing_reference"
                            for row in changes["inference"]))

    def test_token_probe_isolates_shared_drift_and_text_switch(self):
        models = {name: build_conditioner(state) for name, state in self.states.items()}
        config = models["e8b"][0].text_guidance_config()
        models["e5_zero_init"] = make_zero_init(models["e5"], config)
        positive = {
            "clip_image_embeds": torch.randn(1, 24), "clip_vision_tokens": torch.randn(1, 5, 24),
            "texture_images": torch.randn(1, 3, 32, 32), "text_embeds": torch.randn(1, 7, 32),
            "text_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 0]]).bool(),
        }
        negative = {key: torch.zeros_like(value) for key, value in positive.items()
                    if key not in ("text_embeds", "text_mask")}
        negative.update(text_embeds=torch.randn(1, 7, 32), text_mask=positive["text_mask"].clone())
        rows, stats = probe_sample(models, positive, negative, "000000")
        self.assertTrue(all(row["exact_equal"] for row in rows if row["comparison"] == "e5_zero_init_vs_e5"))
        self.assertTrue(all(row["exact_equal"] for row in rows if row["stage"] == "patch_tokens"))
        self.assertTrue(any(row["delta_rms"] > 0 for row in rows
                            if row["comparison"] == "e8a_vs_e5" and row["stage"] == "bf_tokens"))
        self.assertTrue(any(row["delta_rms"] > 0 for row in rows
                            if row["comparison"] == "e8b_on_vs_e8b_off" and row["branch"] == "unconditional"))
        self.assertEqual({row["branch"] for row in stats}, {"conditional", "unconditional"})

    def test_weights_only_cli_preserves_inputs_and_writes_results(self):
        with tempfile.TemporaryDirectory(prefix="e8_diagnosis_test_", dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            argv, before = [], {}
            for name, state in self.states.items():
                path = root / (name + ".pt")
                torch.save(state, path)
                before[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                argv.extend(["--" + name + "-ckpt", str(path)])
            output = root / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(argv + ["--weights-only", "--output-dir", str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["token_probe"]["status"], "skipped")
            self.assertTrue((output / "weight_differences.csv").is_file())
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_cli_findings_do_not_mark_execution_failed(self):
        self.states["e8a"]["unet"]["weight"][0] += 1
        with tempfile.TemporaryDirectory(prefix="e8_diagnosis_test_", dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            argv = []
            for name, state in self.states.items():
                path = root / (name + ".pt")
                torch.save(state, path)
                argv.extend(["--" + name + "-ckpt", str(path)])
            output = root / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(argv + ["--weights-only", "--output-dir", str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "attention_required")
            self.assertGreater(report["weights"]["unexpected_entries"], 0)
            with (output / "weight_differences.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(row for row in csv.DictReader(handle)
                           if row["comparison"] == "e8a_vs_e5" and row["component"] == "unet")
            self.assertEqual(row["status"], "unexpected_change")
            self.assertEqual(row["unexpected"], "True")
            self.assertEqual(row["changed_numel"], "1")

    def test_cli_load_failure_returns_nonzero_and_keeps_error(self):
        with tempfile.TemporaryDirectory(prefix="e8_diagnosis_test_", dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            argv = []
            for name in self.states:
                argv.extend(["--" + name + "-ckpt", str(root / (name + ".pt"))])
            output = root / "report"
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = main(argv + ["--weights-only", "--output-dir", str(output)])
            self.assertEqual(code, 1)
            report = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("FileNotFoundError" in error for error in report["errors"]))
            self.assertIn("[ERROR]", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
