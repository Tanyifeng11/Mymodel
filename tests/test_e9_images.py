"""用原尺度小图核验 E9 三组对照的配对、像素统计与旁路生效判定。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.check_e9_images import EXPERIMENTS, main


class E9ImageTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="e9-images-", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = {}
        for experiment in EXPERIMENTS:
            run = self.root / experiment
            (run / "generated").mkdir(parents=True)
            self.rows[experiment] = []
            for index in range(2):
                sample_id = "%06d" % index
                filename = "token_%s.png" % sample_id
                Image.fromarray(np.full((3, 4, 3), 40 + index, dtype=np.uint8)).save(
                    run / "generated" / filename)
                self.rows[experiment].append(dict(
                    sample_id=sample_id, generation_seed=42 + index, prompt="striped dress",
                    texture_path="/dataset/texture/%s.jpg" % sample_id,
                    sketch_path="/dataset/sketch/%s.jpg" % sample_id,
                    source_target_path="/dataset/gt/%s.jpg" % sample_id,
                    target_path="/server/%s/real/%s" % (experiment, filename),
                    gen_path="/server/%s/generated/%s" % (experiment, filename)))
        self.save_rows()

    def save_rows(self):
        for experiment, rows in self.rows.items():
            (self.root / experiment / "metrics_per_sample.json").write_text(
                json.dumps(rows), encoding="utf-8")

    def write_image(self, experiment, sample_index, value):
        Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8)).save(
            self.root / experiment / "generated/token_%06d.png" % sample_index)

    def check(self, expected_count=2):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--experiments-dir", str(self.root), "--expected-count", str(expected_count),
                         "--output-dir", str(self.root / "image_check")])
        report = json.loads((self.root / "image_check/image_check.json").read_text(encoding="utf-8"))
        return code, report

    def test_identical_outputs_flag_inactive_bypass(self):
        # 与 E8c 相反：E9 训练后逐像素等同 E5 说明旁路没有生效。
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "attention_required")
        self.assertEqual(report["inactive_bypass"], ["e9_a", "e9_b"])
        self.assertTrue(report["identical_variants"])

    def test_active_bypass_with_distinct_variants_completes(self):
        for experiment, offsets in (("e9_a", (6, 6)), ("e9_b", (12, 12))):
            for index in range(2):
                self.write_image(experiment, index, 40 + index + offsets[index])
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["inactive_bypass"], [])
        self.assertFalse(report["identical_variants"])
        summaries = {row["comparison"]: row for row in report["comparisons"]}
        self.assertEqual(summaries["e9_a_vs_e5"]["max_abs_delta"], 6)
        self.assertEqual(summaries["e9_b_vs_e5"]["max_abs_delta"], 12)
        self.assertEqual(summaries["e9_b_vs_e9_a"]["max_abs_delta"], 6)
        self.assertEqual(len((self.root / "image_check/image_differences.csv"
                              ).read_text().splitlines()), 7)

    def test_one_inactive_variant_is_named(self):
        for index in range(2):
            self.write_image("e9_b", index, 60 + index)
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "attention_required")
        self.assertEqual(report["inactive_bypass"], ["e9_a"])
        self.assertFalse(report["identical_variants"])

    def test_two_way_e5_b_check_is_supported(self):
        for index in range(2):
            self.write_image("e9_b", index, 60 + index)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--experiments-dir", str(self.root), "--experiment-names", "e5,e9_b",
                         "--expected-count", "2", "--output-dir", str(self.root / "two_way_check")])
        report = json.loads((self.root / "two_way_check/image_check.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual([row["comparison"] for row in report["comparisons"]], ["e9_b_vs_e5"])

    def test_two_way_e5_c_check_is_supported(self):
        run = self.root / "e9_c"
        (run / "generated").mkdir(parents=True)
        self.rows["e9_c"] = []
        for index, row in enumerate(self.rows["e5"]):
            filename = "token_%06d.png" % index
            self.rows["e9_c"].append({**row,
                                       "target_path": "/server/e9_c/real/%s" % filename,
                                       "gen_path": "/server/e9_c/generated/%s" % filename})
        self.save_rows()
        for index in range(2):
            self.write_image("e9_c", index, 60 + index)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--experiments-dir", str(self.root), "--experiment-names", "e5,e9_c",
                         "--expected-count", "2", "--output-dir", str(self.root / "two_way_c_check")])
        report = json.loads((self.root / "two_way_c_check/image_check.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual([row["comparison"] for row in report["comparisons"]], ["e9_c_vs_e5"])

    def test_downloaded_path_fallback_is_used(self):
        # metrics_per_sample.json 里保留的服务器绝对路径在本地不可用。
        self.assertTrue(all(not Path(row["gen_path"]).is_file()
                            for rows in self.rows.values() for row in rows))
        code, _ = self.check()
        self.assertEqual(code, 0)

    def test_input_mismatch_fails(self):
        for field, value in (("prompt", "plain shirt"), ("texture_path", "/other.jpg"),
                             ("generation_seed", 7), ("source_target_path", "/other/gt.jpg")):
            with self.subTest(field=field):
                original = self.rows["e9_b"][0][field]
                self.rows["e9_b"][0][field] = value
                self.save_rows()
                code, report = self.check()
                self.rows["e9_b"][0][field] = original
                self.save_rows()
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")

    def test_sample_count_mismatch_fails(self):
        code, report = self.check(expected_count=3)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")

    def test_size_mismatch_fails(self):
        Image.fromarray(np.full((5, 4, 3), 40, dtype=np.uint8)).save(
            self.root / "e9_b/generated/token_000000.png")
        code, report = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")

    def test_missing_generated_image_fails(self):
        (self.root / "e9_a/generated/token_000001.png").unlink()
        code, report = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
