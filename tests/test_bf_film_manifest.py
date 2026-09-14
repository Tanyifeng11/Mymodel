"""核验 FiLM 清单同步：使用当前文本、路径限定 BF 根目录、选样可复现。"""

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from uuid import uuid4

from tools.prepare_bf_film_manifest import main, prepare_manifest, rooted_path


class BFFilmManifestTests(unittest.TestCase):
    def setUp(self):
        # 普通 mkdir 继承权限，兼容 Windows 沙箱中 tempfile 的私有 ACL 限制。
        temporary_root = Path(tempfile.gettempdir()).resolve()
        self.base = temporary_root / ("bf-film-manifest-" + uuid4().hex)
        self.base.mkdir()
        self.assertEqual(self.base.resolve().parent, temporary_root)
        self.addCleanup(shutil.rmtree, self.base)
        self.root = self.base / "BF"
        self.source = self.base / "source.json"

    def row(self, sample_id, caption="当前条纹服装描述", mask=True):
        sample = Path(sample_id)
        prefix = sample.parent
        target_dir = "cloth" if sample.parts[0] == "training" else "gt"
        item = {"sample_id": sample_id, "caption": "旧描述", "warnings": ["保留原字段"]}
        for field, folder, extension in (
            ("cloth", target_dir, ".jpg"), ("sketch", "sketch", ".jpg"),
            ("texture", "texture", ".jpg"), ("mask", "mask", ".png"),
        ):
            if field == "mask" and not mask:
                continue
            relative = prefix / folder / (sample.name + extension)
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture: existence checks only")
            item[field] = relative.as_posix()
        text = self.root / prefix / "text" / (sample.name + ".txt")
        text.parent.mkdir(parents=True, exist_ok=True)
        text.write_text(caption, encoding="utf-8-sig")
        return item

    def save(self, rows):
        self.source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def test_refreshes_current_text_and_preserves_source_and_fields(self):
        original = self.row("training/100", "  当前黑白条纹连衣裙  ")
        self.save([original])
        before = self.source.read_bytes()
        text_path = self.root / "training/text/100.txt"
        text_before = text_path.read_bytes()
        output = self.base / "new" / "film.json"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = main(["--bf-root", str(self.root), "--source-manifest", str(self.source),
                            "--output", str(output)])
        rows = json.loads(output.read_text(encoding="utf-8"))
        expected = dict(original, caption="当前黑白条纹连衣裙")
        self.assertEqual(rows, [expected])
        self.assertEqual(summary["caption_changed_eligible_count"], 1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(text_path.read_bytes(), text_before)

    def test_supports_categorized_test_and_optional_mask(self):
        item = self.row("test/dress/200", "a striped dress", mask=False)
        self.save([item])
        rows, stats = prepare_manifest(self.root, self.source)
        self.assertEqual(rows[0]["caption"], "a striped dress")
        self.assertNotIn("mask", rows[0])
        self.assertEqual(stats["excluded_count"], 0)

    def test_root_contract_and_traversal(self):
        item = self.row("training/100")
        self.save([item])
        with self.assertRaisesRegex(ValueError, "BF 根目录"):
            prepare_manifest(self.root / "training", self.source)
        for path in (".", "../outside.jpg", "training/../../outside.jpg", "cloth/100.jpg",
                     "C:/outside.jpg", "/outside.jpg", "\\\\host\\share\\file.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                rooted_path(self.root.resolve(), path)
        self.assertEqual(rooted_path(self.root.resolve(), "training\\cloth\\100.jpg"),
                         (self.root / "training/cloth/100.jpg").resolve())

    def test_excludes_missing_inputs_and_invalid_text_without_caption_fallback(self):
        rows = [self.row("training/100")]
        for index, field in enumerate(("cloth", "sketch", "texture", "mask"), 101):
            item = self.row("training/" + str(index))
            (self.root / item[field]).unlink()
            rows.append(item)
        missing_text = self.row("training/105")
        (self.root / "training/text/105.txt").unlink()
        rows.extend([missing_text, self.row("training/106", " \n ")])
        invalid_path = copy.deepcopy(rows[0])
        invalid_path["texture"] = "training/../../outside.jpg"
        rows.append(invalid_path)
        mismatched = copy.deepcopy(rows[0])
        mismatched["sample_id"] = "training/wrong"
        rows.append(mismatched)
        self.save(rows)
        selected, stats = prepare_manifest(self.root, self.source)
        self.assertEqual([item["sample_id"] for item in selected], ["training/100"])
        self.assertEqual(stats["excluded_count"], 8)
        self.assertEqual(stats["excluded_reasons"], {
            "missing_cloth": 1, "missing_sketch": 1, "missing_texture": 1,
            "missing_mask": 1, "missing_text": 1, "empty_text": 1,
            "invalid_path": 1, "sample_id_mismatch": 1,
        })

    def test_sampling_is_reproducible_and_does_not_reorder_selected_rows(self):
        self.save([self.row("training/" + str(index)) for index in range(20)])
        first, stats = prepare_manifest(self.root, self.source, max_samples=6, seed=42)
        second, _ = prepare_manifest(self.root, self.source, max_samples=6, seed=42)
        third, _ = prepare_manifest(self.root, self.source, max_samples=6, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(stats["eligible_count"], 20)
        self.assertEqual(stats["selected_count"], 6)
        indices = [int(row["sample_id"].split("/")[-1]) for row in first]
        self.assertEqual(indices, sorted(indices))

    def test_refuses_source_or_implicit_output_overwrite(self):
        self.save([self.row("training/100")])
        before = self.source.read_bytes()
        arguments = ["--bf-root", str(self.root), "--source-manifest", str(self.source)]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(arguments + ["--output", str(self.source), "--overwrite"])
        self.assertEqual(self.source.read_bytes(), before)
        output = self.base / "existing.json"
        output.write_text("do not replace", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(arguments + ["--output", str(output)])
        self.assertEqual(output.read_text(encoding="utf-8"), "do not replace")
        with contextlib.redirect_stdout(io.StringIO()):
            main(arguments + ["--output", str(output), "--overwrite"])
        self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))), 1)


if __name__ == "__main__":
    unittest.main()
