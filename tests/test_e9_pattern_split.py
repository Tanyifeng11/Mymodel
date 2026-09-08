"""核验 E9 图案子集挑选：判据只看参考图，排序稳定，输出仍是可用的划分格式。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_e9_pattern_split import (
    band_energy, color_spread, edge_density, keyword_hit, main, select, to_gray,
)


def stripes(size=64, period=8):
    columns = np.arange(size)
    pattern = np.where((columns // (period // 2)) % 2 == 0, 30, 225).astype(np.uint8)
    return np.repeat(pattern[None, :], size, axis=0)


def checks(size=64, period=8):
    rows, columns = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pattern = ((rows // (period // 2) + columns // (period // 2)) % 2 == 0)
    return np.where(pattern, 30, 225).astype(np.uint8)


def flat(size=64, value=128):
    return np.full((size, size), value, dtype=np.uint8)


def gradient(size=64):
    ramp = np.linspace(40, 210, size, dtype=np.float32)
    return np.repeat(ramp[None, :], size, axis=0).astype(np.uint8)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e9-pattern-metrics-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, name, array, mode="L"):
        path = self.root / name
        if mode == "RGB" and array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        Image.fromarray(array).save(path)
        return path

    def test_patterns_score_above_flat_and_gradient(self):
        scores = {}
        for name, array in (("stripes", stripes()), ("checks", checks()),
                            ("flat", flat()), ("gradient", gradient())):
            gray = to_gray(self.save(name + ".png", array), 128)
            scores[name] = (band_energy(gray), edge_density(gray))
        for pattern in ("stripes", "checks"):
            for plain in ("flat", "gradient"):
                self.assertGreater(scores[pattern][0], scores[plain][0], (pattern, plain))
                self.assertGreater(scores[pattern][1], scores[plain][1], (pattern, plain))

    def test_flat_image_has_no_edges_and_no_spread(self):
        path = self.save("flat.png", flat(), mode="RGB")
        self.assertEqual(edge_density(to_gray(path, 128)), 0.0)
        self.assertAlmostEqual(color_spread(path, 128), 0.0, places=6)

    def test_multicolour_reference_has_larger_spread(self):
        colourful = np.zeros((64, 64, 3), dtype=np.uint8)
        colourful[:, :32, 0] = 240
        colourful[:, 32:, 2] = 240
        plain = self.save("plain.png", flat(), mode="RGB")
        self.assertGreater(color_spread(self.save("colourful.png", colourful, "RGB"), 128),
                           color_spread(plain, 128))

    def test_keyword_hit_matches_whole_words_only(self):
        self.assertTrue(keyword_hit("a striped cotton shirt"))
        self.assertTrue(keyword_hit("navy plaid-check dress"))
        self.assertFalse(keyword_hit("a plain navy shirt"))
        self.assertFalse(keyword_hit(None))


class SelectTests(unittest.TestCase):
    def samples(self):
        return [
            dict(sample_id="000000", score=0.9, band_energy=0.30, edge_density=0.20),
            dict(sample_id="000001", score=0.5, band_energy=0.20, edge_density=0.10),
            dict(sample_id="000002", score=0.2, band_energy=0.02, edge_density=0.01),
            dict(sample_id="000003", score=0.5, band_energy=0.15, edge_density=0.08),
        ]

    def test_threshold_filters_and_orders_by_score(self):
        selected, eligible = select(self.samples(), 3, 0.10, 0.05)
        self.assertEqual(eligible, 3)
        self.assertEqual([item["sample_id"] for item in selected], ["000000", "000001", "000003"])

    def test_ties_break_on_sample_id_for_determinism(self):
        first, _ = select(self.samples(), 2, 0.10, 0.05)
        second, _ = select(list(reversed(self.samples())), 2, 0.10, 0.05)
        self.assertEqual([item["sample_id"] for item in first],
                         [item["sample_id"] for item in second])

    def test_request_beyond_eligible_returns_what_exists(self):
        selected, eligible = select(self.samples(), 10, 0.10, 0.05)
        self.assertEqual((len(selected), eligible), (3, 3))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e9-pattern-main-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_root = self.root / "data"
        (self.data_root / "texture").mkdir(parents=True)
        arrays = [stripes(), checks(), flat(), gradient(), stripes(period=16), checks(period=16)]
        self.split = []
        for index, array in enumerate(arrays):
            name = "texture/%06d.png" % index
            Image.fromarray(np.stack([array] * 3, axis=-1)).save(self.data_root / name)
            self.split.append(dict(sample_id="%06d" % index, idx=index,
                                   prompt="striped dress" if index % 2 == 0 else "plain dress",
                                   sketch="sketch/%06d.png" % index, texture=name,
                                   target="gt/%06d.png" % index, mask=None,
                                   category="top", filename="%06d.png" % index))
        self.split_path = self.root / "full_split.json"
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")

    def run_main(self, extra=()):
        argv = ["--split_path", str(self.split_path), "--data_root", str(self.data_root),
                "--output_split", str(self.root / "pattern.json"),
                "--summary_json", str(self.root / "summary.json"),
                "--swap_json", str(self.root / "swap.json"), "--count", "4", *extra]
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = main(argv)
        return code, out.getvalue()

    def test_selects_patterned_references_and_keeps_split_format(self):
        code, _ = self.run_main()
        self.assertEqual(code, 0)
        selected = json.loads((self.root / "pattern.json").read_text(encoding="utf-8"))
        self.assertEqual(len(selected), 4)
        # 纯色和渐变参考不应入选。
        chosen = {item["source_sample_id"] for item in selected}
        self.assertNotIn("000002", chosen)
        self.assertNotIn("000003", chosen)
        # sample_id 连续重编号，其余字段沿用原划分，供 run_fixed_benchmark 直接使用。
        self.assertEqual([item["sample_id"] for item in selected],
                         ["000000", "000001", "000002", "000003"])
        for item in selected:
            for field in ("prompt", "sketch", "texture", "target", "category", "filename"):
                self.assertIn(field, item)

    def test_summary_records_scope_and_thresholds(self):
        self.run_main()
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("不是质量指标", summary["scope"])
        self.assertEqual(summary["total_candidates"], len(self.split))
        self.assertEqual(summary["selected_count"], 4)
        self.assertEqual(set(summary["thresholds"]), {"min_band_energy", "min_edge_density"})

    def test_swap_pairs_are_disjoint_and_deterministic(self):
        self.run_main()
        first = json.loads((self.root / "swap.json").read_text(encoding="utf-8"))["pairs"]
        self.run_main()
        second = json.loads((self.root / "swap.json").read_text(encoding="utf-8"))["pairs"]
        self.assertEqual(first, second)
        for pair in first:
            self.assertNotEqual(pair["sample_id"], pair["swap_sample_id"])
            self.assertNotEqual(pair["texture"], pair["swap_texture"])

    def test_output_is_deterministic_across_runs(self):
        self.run_main()
        first = (self.root / "pattern.json").read_text(encoding="utf-8")
        self.run_main()
        self.assertEqual(first, (self.root / "pattern.json").read_text(encoding="utf-8"))

    def test_insufficient_eligible_warns_but_still_writes(self):
        code, _ = self.run_main(["--count", "6", "--min_band_energy", "0.2",
                                 "--min_edge_density", "0.15"])
        self.assertEqual(code, 0)
        selected = json.loads((self.root / "pattern.json").read_text(encoding="utf-8"))
        self.assertLess(len(selected), 6)
        self.assertGreater(len(selected), 0)

    def test_no_eligible_sample_returns_failure(self):
        code, _ = self.run_main(["--min_band_energy", "0.99", "--min_edge_density", "0.99"])
        self.assertEqual(code, 1)

    def test_missing_texture_is_reported(self):
        (self.data_root / self.split[0]["texture"]).unlink()
        with self.assertRaises(FileNotFoundError):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(["--split_path", str(self.split_path), "--data_root", str(self.data_root),
                      "--output_split", str(self.root / "pattern.json"), "--count", "2"])


if __name__ == "__main__":
    unittest.main()
