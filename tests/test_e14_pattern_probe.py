import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.bf_texture_module import BFTextureConditioner
from models.tcpm_lite import TCPMLite
from tools.e14_pattern_probe import (FIELDS, aggregate, capture, evaluate, prepare,
                                     read_labels, readouts, retrieval, similarity)


class E14Tests(unittest.TestCase):
    def test_hooks_preserve_forward_and_capture_boundaries(self):
        torch.manual_seed(4)
        bf = BFTextureConditioner(clip_embeddings_dim=16, cross_attention_dim=16,
                                  stage_channels=(8, 8, 16, 16))
        tcpm = TCPMLite(16)
        inputs = dict(texture_images=torch.randn(1, 3, 64, 64),
                      clip_vision_tokens=torch.randn(1, 16, 16))
        text = torch.randn(1, 8, 16)
        with torch.inference_mode():
            expected = bf(**inputs)[0]
            values = capture(bf, tcpm, inputs, text, text)
            torch.testing.assert_close(values["tokens_pre_ln"],
                                       values["resampler_raw"] + bf.token_mlp(values["resampler_raw"]))
        torch.testing.assert_close(expected, values["tokens_post_ln"], rtol=0, atol=0)
        self.assertEqual(values["cnn3_pool"].shape, (1, 64, 16))
        features, shapes = readouts(values)
        self.assertEqual(features["fused__set16_s0"].shape, (16, 16))
        self.assertFalse(bf.resampler._forward_hooks)

    def test_retrieval_excludes_source_and_color_and_handles_ties(self):
        rows = [dict(sample_id=str(i), source_group=g, color_group=c, pattern=p)
                for i, (g, c, p) in enumerate([
                    ("a", "red", "stripe"), ("a", "red", "stripe"),
                    ("b", "red", "stripe"), ("c", "red", "plaid"),
                    ("d", "blue", "stripe")])]
        records = retrieval(np.ones((5, 5)), rows)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["positive_sources"], 1)
        self.assertEqual(records[0]["r1"], 0.5)
        self.assertEqual(records[0]["triplet"], 0.5)
        self.assertEqual(aggregate(records)["eligible_sources"], 2)

    def test_matching_is_permutation_invariant(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(16, 8)).astype(np.float32)
        result = similarity(np.stack([x, x[::-1]]))
        self.assertAlmostEqual(float(result[0, 1]), 1, places=6)

    def test_prepare_requires_manual_labels_and_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (32, 32), "red").save(root / "texture.png")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([dict(sample_id="validation/a", texture="texture.png",
                                                  caption="a red striped garment")]), encoding="utf-8")
            prepare(Namespace(manifest=str(manifest), data_root=str(root), output=str(root / "candidates"),
                              seed=42, per_class=40))
            labels = root / "candidates" / "labels.csv"
            with self.assertRaises(ValueError):
                read_labels(labels)
            rows = []
            for i in range(4):
                r = dict.fromkeys(FIELDS, "")
                r.update(sample_id=str(i), pattern="stripe" if i < 2 else "plaid", color_group="red",
                         source_group=str(i), confirmed="1", pattern_visible="1")
                rows.append(r)
            with labels.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            self.assertEqual(len(read_labels(labels)), 4)
            for i, row in enumerate(rows):
                row["feature_file"] = f"{i}.npz"
                np.savez(root / row["feature_file"], tiny__meanstd=np.array([i < 2, i >= 2], dtype=np.float32))
            (root / "index.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            evaluate(Namespace(features=str(root), linear=False, seed=42))
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["retrieval"]["tiny__meanstd"]["r1"], 1)


if __name__ == "__main__":
    unittest.main()
