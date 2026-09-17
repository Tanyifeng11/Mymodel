import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.report_condition_interventions import MODES, summarize


class ReportTests(unittest.TestCase):
    def test_pairing_summary_and_unchanged_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root, source = Path(folder)/'run', Path(folder)/'source'
            for sid in (2, 5, 14, 18, 22, 23):
                name = f'sample_{sid:06d}'
                original = source/name/'on'
                original.mkdir(parents=True)
                Image.new('RGB', (12, 16), 'white').save(original/'generated_x.png')
                for mode in MODES:
                    target = root/name/mode/'e5'
                    token = target/'token'/'sample'
                    token.mkdir(parents=True)
                    Image.new('RGB', (12, 16), 'white').save(token/'generated_x.png')
                    row = dict(sample_id=str(sid), generation_seed=42+sid, prompt='p',
                               sketch_path='s', texture_path='t', struct_edge_f1=.5)
                    (target/'metrics_per_sample.json').write_text(json.dumps([row]))
                    trace = dict(mode=mode, metadata={}, trigger_steps=[], records=[
                        dict(step_index=i, applied=False, correction_rms=0.) for i in range(50)])
                    (token/'intervention.json').write_text(json.dumps(trace))
            result = summarize(root, source)
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(len(result['samples']), 6)
            self.assertEqual(result['samples']['sample_000018']['boundary']['metric_delta']['struct_edge_f1'], 0.)
            self.assertTrue((root/'report/sample_000023.png').exists())
            Image.new('RGB', (12, 16), 'black').save(root/'sample_000002/boundary/e5/token/sample/generated_x.png')
            with self.assertRaisesRegex(ValueError, '未施加干预'):
                summarize(root, source)


if __name__ == '__main__':
    unittest.main()
