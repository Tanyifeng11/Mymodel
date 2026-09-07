"""用原尺度小图核验 E8c 对照的配对、像素统计与退出状态。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.check_e8c_images import EXPERIMENTS, main


class E8cImageTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="e8c-images-", dir=temporary_root)
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
                Image.fromarray(np.full((3, 4, 3), 40 + index, dtype=np.uint8)).save(run / "generated" / filename)
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
            (self.root / experiment / "metrics_per_sample.json").write_text(json.dumps(rows), encoding="utf-8")

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--experiments-dir", str(self.root), "--expected-count", "2",
                         "--output-dir", str(self.root / "image_check")])
        report = json.loads((self.root / "image_check/image_check.json").read_text(encoding="utf-8"))
        return code, report

    def test_equal_images_and_downloaded_path_fallback(self):
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(all(row["exact_equal_count"] == 2 and row["pixel_mae_255"] == 0
                            and row["max_abs_delta"] == 0 for row in report["comparisons"]))

    def test_pixel_differences_remain_visible_without_execution_failure(self):
        for name, value in (("text_only_off", 41), ("text_only_on", 49)):
            Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8)).save(
                self.root / name / "generated/token_000000.png")
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "attention_required")
        summaries = report["comparisons"]
        self.assertEqual([row["exact_equal_count"] for row in summaries], [1, 1, 1])
        self.assertEqual([row["pixel_mae_255"] for row in summaries], [0.5, 4.5, 4.0])
        self.assertEqual([row["max_abs_delta"] for row in summaries], [1, 9, 8])
        self.assertEqual(len((self.root / "image_check/image_differences.csv").read_text().splitlines()), 7)

    def test_mismatched_inputs_duplicates_and_missing_images_fail(self):
        row = self.rows["text_only_on"][0]
        for field, changed in (("prompt", "different"), ("generation_seed", 999),
                               ("texture_path", "/other.jpg"), ("sketch_path", "/other.jpg"),
                               ("source_target_path", "/other.jpg"), ("sample_id", "000001")):
            with self.subTest(field=field):
                original = row[field]
                row[field] = changed
                self.save_rows()
                code, report = self.check()
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                row[field] = original
        self.save_rows()
        (self.root / "text_only_on/generated/token_000000.png").unlink()
        code, report = self.check()
        self.assertEqual(code, 1)
        self.assertIn("FileNotFoundError", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
