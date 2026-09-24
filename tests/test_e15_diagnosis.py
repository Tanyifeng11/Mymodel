import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.e15_common import (LayerCapture, compact_table, flatten_representation,
                              format_heatmap, gram_statistics, normalized_response,
                              relative_change, summary_stats)
from tools.e15_stages import d3_timestep_region


class MockConditioner(torch.nn.Module):
    """只复刻 BF 的子模块调用顺序，用于验证 hook 捕获。"""

    def __init__(self):
        super().__init__()
        self.stage_token_hw = (2, 2)
        self.stage1 = torch.nn.Linear(4, 4)
        self.stage2 = torch.nn.Linear(4, 4)
        self.stage3 = torch.nn.Linear(4, 4)
        self.stage4 = torch.nn.Linear(4, 4)
        self.resampler = torch.nn.MultiheadAttention(4, 1, batch_first=True)
        self.token_mlp = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.token_norm = torch.nn.LayerNorm(4)

    def run_probe(self):
        x = torch.randn(1, 3, 4)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        query = torch.randn(1, 2, 4)
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        out, _ = self.resampler(query, fused, fused)
        out = out + self.token_mlp(out)
        return self.token_norm(out)


class MetricTests(unittest.TestCase):
    def test_normalized_response(self):
        reference = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(normalized_response(reference, np.zeros(2, dtype=np.float32),
                                                  np.zeros(2, dtype=np.float32)), 1.0, places=6)
        self.assertIsNone(normalized_response(reference, reference, reference))
        self.assertAlmostEqual(relative_change(reference, np.zeros(2, dtype=np.float32)), 1.0, places=6)

    def test_gram_statistics_limits(self):
        identical = np.ones((4, 3), dtype=np.float32)
        result = gram_statistics(identical)
        self.assertEqual(result["effective_rank"], 0.0)
        random_block = np.random.RandomState(0).randn(6, 5).astype(np.float32)
        ranked = gram_statistics(random_block)
        self.assertGreater(ranked["effective_rank"], 1.0)
        self.assertLessEqual(ranked["effective_rank"], 5.0001)
        self.assertAlmostEqual(sum(ranked["pc_variance"]), 1.0, places=4)

    def test_summary_stats_skips_none(self):
        result = summary_stats([1.0, None, float("nan"), 3.0])
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mean"], 2.0, places=6)


class CaptureTests(unittest.TestCase):
    def test_layer_capture_records_every_layer(self):
        conditioner = MockConditioner()
        capture = LayerCapture(conditioner)
        try:
            conditioner.run_probe()
            for name in ["cnn1", "cnn2", "cnn3", "cnn4", "fused",
                         "resampler_out", "mlp_out", "pre_ln", "final"]:
                self.assertIn(name, capture.current, name)
            self.assertEqual(capture.query.shape[0], 1)
            vector = flatten_representation(capture.current["final"])
            self.assertEqual(vector.shape[0], 2 * 4)
            self.assertEqual(vector.dtype, np.float32)
        finally:
            capture.remove()
        self.assertEqual(capture.handles, [])


class D3Tests(unittest.TestCase):
    def _write_report(self, root):
        values = {"matched": 1.0, "wrong_1": 2.0, "color_nearest": 1.5,
                  "rot90": 1.2, "zero_tokens": 3.0}
        records = []
        for index in (425, 1639):
            for timestep in (1, 181):
                records.append({
                    "sample_index": index, "seed": 42, "timestep": timestep,
                    "losses": dict(values),
                    "region_losses": {"interior": dict(values),
                                      "boundary": {k: v * 2 for k, v in values.items()},
                                      "background": {k: v * 3 for k, v in values.items()}},
                })
        payload = {"complete": True, "records": records, "samples": [{"sample_index": 425},
                                                                    {"sample_index": 1639}]}
        path = Path(root) / "denoising_report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_delta_matches_manual_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_report(tmp)
            ctx = {"args": {"previous_report": str(report)}, "output": tmp}
            result = d3_timestep_region(ctx)
            self.assertEqual(result["timesteps"], [1, 181])
            self.assertAlmostEqual(result["delta"]["random"]["interior"][0], 1.0, places=6)
            self.assertAlmostEqual(result["delta"]["random"]["boundary"][0], 2.0, places=6)
            self.assertAlmostEqual(result["delta"]["rot90"]["background"][0], 0.6, places=6)
            self.assertEqual(result["interior_positive_samples"]["random"], [2, 2])
            self.assertTrue((Path(tmp) / "d3_timestep_region" / "report.json").is_file())


class FormatTests(unittest.TestCase):
    def test_table_helpers_stay_within_terminal_width(self):
        table = compact_table(["layer", "value"], [["cnn1", 0.123456], ["final", None]])
        for line in table.splitlines():
            self.assertLessEqual(len(line), 88)
        heat = format_heatmap(["interior", "boundary"], [1, 181], [[0.1, -0.2], [None, 0.3]])
        self.assertEqual(len(heat.splitlines()), 3)



class D5Tests(unittest.TestCase):
    """D5 只做真实生成，这里只校验层配置与参数契约，不加载模型。"""

    def test_configs_match_group_definition(self):
        from tools.e15_d5_generation import CONFIGS, REGIONS
        from tools.e15_common import GROUP_RANGES

        def spec(text):
            return {int(item) for item in text.split(",") if item}

        self.assertEqual(sorted(CONFIGS),
                         ["baseline", "no_g3", "no_g4", "no_texture", "only_g3"])
        self.assertEqual(CONFIGS["baseline"], "")
        self.assertEqual(spec(CONFIGS["no_g4"]), set(GROUP_RANGES["G4"]))
        self.assertEqual(spec(CONFIGS["no_g3"]), set(GROUP_RANGES["G3"]))
        self.assertEqual(spec(CONFIGS["only_g3"]), set(range(16)) - set(GROUP_RANGES["G3"]))
        self.assertEqual(spec(CONFIGS["no_texture"]), set(range(16)))
        self.assertEqual(REGIONS, ["interior", "boundary", "background"])

    def test_namespace_covers_every_referenced_arg(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "inference_IMAGGarment-1.py").read_text(encoding="utf-8")
        used = set()
        for function in ["generate_one", "prepare", "resolve_image_encoder_path",
                         "restore_bf_conditioner_for_inference"]:
            match = re.search(r"def " + function + r"\(.*?\n(?=\ndef |\nclass |\Z)",
                              source, re.S)
            self.assertIsNotNone(match, function)
            used |= set(re.findall(r"args\.([A-Za-z_0-9]+)", match.group(0)))
            used |= set(re.findall(r"getattr\(args, .([A-Za-z_0-9]+).", match.group(0)))
        namespace = (root / "tools/e15_d5_generation.py").read_text(encoding="utf-8")
        block = namespace[namespace.index("def build_inference_args"):
                          namespace.index("def build_samples")]
        provided = set(re.findall(r"([A-Za-z_0-9]+)=", block))
        self.assertEqual(sorted(used - provided), [])

if __name__ == "__main__":
    unittest.main()