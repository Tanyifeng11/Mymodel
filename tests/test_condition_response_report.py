"""验证服务器核验器能发现图像不一致和探针数据缺失。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from models.condition_response_probe import ConditionResponseProbe
from tools.report_condition_response_probe import report


class ReportTests(unittest.TestCase):
    def test_valid_then_pixel_mismatch_then_missing_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('off', 'on'):
                folder = root / arm / 'token' / '000000_test'
                folder.mkdir(parents=True)
                path = folder / 'generated_000000.png'
                Image.new('RGB', (8, 8), 'white').save(path)
                row = dict(sample_id='000000', generation_seed=42, prompt='dress',
                           sketch_path='sketch.png', texture_path='texture.png', source_gen_path=str(path))
                (root / arm / 'metrics_per_sample.json').write_text(json.dumps([row]), encoding='utf-8')
            folder = root / 'on/token/000000_test/condition_response_probe'
            s, t = SimpleNamespace(scale=0.6), SimpleNamespace(scale=1.)
            probe = ConditionResponseProbe([s], [t], torch.ones(1, 1, 8, 8), folder)
            forward = lambda: torch.ones(1, 4, 2, 2) * (s.scale - t.scale)
            for step in [0, 5, 15, 25, 49]:
                probe.observe(forward(), forward, step, 999-step)
            result = report(root/'off', root/'on', 1, root/'report')
            self.assertEqual(result['status'], 'pass')
            Image.new('RGB', (8, 8), 'black').save(root/'on/token/000000_test/generated_000000.png')
            with self.assertRaisesRegex(ValueError, '像素不同'):
                report(root/'off', root/'on', 1, root/'report')
            self.assertEqual(json.loads((root/'report/verification.json').read_text(encoding='utf-8'))['status'], 'fail')
            (folder/'step_49.npz').unlink()
            with self.assertRaises(FileNotFoundError):
                report(root/'off', root/'on', 1, root/'report')


if __name__ == '__main__':
    unittest.main()
