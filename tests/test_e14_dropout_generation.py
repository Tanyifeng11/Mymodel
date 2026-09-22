import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.e14_dropout_generation import report


class GenerationReportTests(unittest.TestCase):
    def test_grid_and_noise_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[]
            Image.new('RGB',(384,512),'white').save(root/'fixture.png')
            for stage in ['baseline','legacy_pooled','zero_final_tokens']:
                for seed in [42,142]:
                    for ref in [13,30]:
                        for variant in ['original','rot90']:
                            records.append(dict(stage=stage,seed=seed,sample_id='ref_%04d_%s'%(ref,variant),
                                                noise_sha256=str(seed),image='fixture.png'))
            path=root/'manifest.json'
            path.write_text(json.dumps(dict(complete=True,records=records)))
            report(root)
            self.assertEqual(len(list(root.glob('comparison_*.png'))),4)
            self.assertEqual(len(json.loads((root/'generation_review.json').read_text())),24)
            records[0]['noise_sha256']='different'
            path.write_text(json.dumps(dict(complete=True,records=records)))
            with self.assertRaises(ValueError):report(root)


if __name__=='__main__':unittest.main()
