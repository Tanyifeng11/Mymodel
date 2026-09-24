import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from tools.e14_denoising_complete import aggregate,color_pairs
from tools.e14_denoising_regions import partition


class CompleteDenoisingTests(unittest.TestCase):
    def test_sample_weighting(self):
        records=[]
        for i,delta,count in [(1,2.,4),(2,-1.,1)]:
            for _ in range(count):records.append(dict(sample_index=i,losses=dict(matched=2.,wrong_1=2+delta,wrong_2=2+delta)))
        r=aggregate(records)['wrong_average']
        self.assertEqual(r['mean_delta'],.5)
        self.assertEqual(r['matched_better_samples'],1)

    def test_color_selection_excludes_self_and_duplicates(self):
        colors=[(255,0,0),(250,0,0),(0,0,255),(255,0,0)]
        rows=[dict(cloth=str(i),texture=str(i)+'.png') for i in range(4)]
        with patch('tools.e14_denoising_complete.Image.open',side_effect=lambda p:Image.new('RGB',(8,8),colors[int(Path(p).stem)])):
            pairs=color_pairs(rows,list(range(4)),{0:'red',1:'near',2:'blue',3:'red'},'.')
            self.assertEqual(pairs[0]['index'],1)

    def test_regions_reconstruct_global_error(self):
        mask=np.zeros((512,384),dtype='uint8');mask[100:400,100:300]=255
        parts=partition(Image.fromarray(mask),(64,48))
        self.assertTrue(np.all(sum(v.astype(int) for v in parts.values())==1))
        error=np.random.default_rng(42).random((64,48))
        self.assertAlmostEqual(sum(error[m].mean()*m.mean() for m in parts.values()),error.mean())


if __name__=='__main__':unittest.main()
