import unittest
import json
import tempfile
from pathlib import Path
from argparse import Namespace
import numpy as np
from PIL import Image
from tools.e14_generation_response import spectrum,tiles_inside,run,report,VARIANTS


class GenerationResponseTests(unittest.TestCase):
    def test_direction_frequency_and_flat(self):
        y,x=np.mgrid[:64,:64]
        def metric(a):return spectrum(Image.fromarray(a.astype('uint8')),[(0,0,64,64)])
        vertical=metric(128+80*np.cos(2*np.pi*x*4/64))
        horizontal=metric(128+80*np.cos(2*np.pi*y*4/64))
        fine=metric(128+80*np.cos(2*np.pi*x*10/64))
        self.assertGreater(vertical['vertical_score'],.9)
        self.assertLess(horizontal['vertical_score'],-.9)
        self.assertGreater(fine['frequency_cycles_per_pixel'],vertical['frequency_cycles_per_pixel'])
        self.assertIsNone(metric(np.full((64,64),176))['vertical_score'])

    def test_roi_never_contains_mask_boundary(self):
        mask=np.zeros((256,256),bool);mask[30:220,30:220]=True
        tiles=tiles_inside(mask)
        self.assertTrue(tiles)
        for x,y,w,h in tiles:self.assertTrue(mask[y-4:y+h+4,x-4:x+w+4].all())
        self.assertEqual(tiles_inside(np.zeros((64,64),bool)),[])

    def test_preparation_and_report_without_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            Image.new('RGB',(384,512),'white').save(p/'sketch.png')
            (p/'split.json').write_text(json.dumps([dict(sample_id='000002',sketch='sketch.png')]))
            args=Namespace(output=str(p/'run'),split=str(p/'split.json'),data_root=str(p),sample_ids=[2],
                           seeds=[42],checkpoint='e5',texture_checkpoint='tex',clip_model='clip',
                           prompt='a garment',device='cpu',prepare_only=True)
            run(args)
            root=p/'run';m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(m['records']),6)
            self.assertEqual(m['reference_hashes']['coarse_vertical'],m['reference_hashes']['repeat_coarse_vertical'])
            for row in m['records']:
                dest=root/row['directory'];dest.mkdir(parents=True)
                Image.new('RGB',(384,512),'gray').save(dest/'generated.png')
                Image.new('L',(384,512),255).save(dest/'000002_mask.png')
            report(root)
            r=json.loads((root/'response_report.json').read_text(encoding='utf-8'))
            self.assertTrue(r['groups'][0]['repeat_exact'])
            self.assertIsNone(r['groups'][0]['signed_following_deltas']['vertical_frequency'])


if __name__=='__main__':unittest.main()
