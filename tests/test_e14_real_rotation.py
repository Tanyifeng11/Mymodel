import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e14_real_rotation import rgb_counts,paired_direction,report
from tools.e14_generation_response import spectrum


class RealRotationTests(unittest.TestCase):
    def test_rotation_preserves_exact_rgb_counts_and_flips_axis(self):
        y,x=np.mgrid[:128,:128]
        image=Image.fromarray((128+80*np.cos(x*2*np.pi/16)).astype('uint8')).convert('RGB')
        rotated=image.transpose(Image.Transpose.ROTATE_90)
        for a,b in zip(rgb_counts(image),rgb_counts(rotated)):np.testing.assert_array_equal(a,b)
        a=spectrum(image,[(0,0,128,128)])['vertical_score']
        b=spectrum(rotated,[(0,0,128,128)])['vertical_score']
        self.assertGreater(paired_direction(a,b,a,b),1.8)
        self.assertGreater(paired_direction(b,a,b,a),1.8)
        self.assertLess(paired_direction(a,b,b,a),0)
        self.assertIsNone(paired_direction(a,b,None,a))

    def test_report_pairing_and_missing_texture_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'inputs').mkdir()
            records=[];variants={}
            for name,score in [('original',1.),('rot90',-1.)]:
                Image.new('RGB',(256,256),'gray').save(root/'inputs'/f'{name}.png')
                variants[name]=dict(file=f'{name}.png',reference_metrics=dict(vertical_score=score))
                (root/name).mkdir()
                Image.new('RGB',(384,512),'gray').save(root/name/'generated.png')
                Image.new('L',(384,512),255).save(root/name/'000005_mask.png')
                records.append(dict(sample_id='000005',pair_id='r',seed=42,variant=name,directory=name))
            (root/'inputs/inputs.json').write_text(json.dumps(dict(pairs=[dict(pair_id='r',variants=variants)])),encoding='utf-8')
            (root/'manifest.json').write_text(json.dumps(dict(records=records)),encoding='utf-8')
            report(root)
            result=json.loads((root/'rotation_report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['groups']),1)
            self.assertIsNone(result['groups'][0]['signed_direction_following'])
            self.assertTrue((root/'comparison_000005_r_seed42.png').is_file())


if __name__=='__main__':unittest.main()
