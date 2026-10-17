"""固定评分区域不能由推理增强门控决定；不执行模型或科研训练。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from eval.eval_utils import derive_region_masks
from tools.m54_eval import regional_metrics,assess
from tools.m54_protocol import SIZE


class EvaluationTests(unittest.TestCase):
    def test_inactive_inference_mask_still_scores_the_same_fixed_sketch_region(self):
        with tempfile.TemporaryDirectory() as directory:
            mask=np.zeros((SIZE[1],SIZE[0]),bool);mask[100:400,100:300]=True
            pixels=np.full((SIZE[1],SIZE[0],3),255,np.uint8);pixels[mask]=0
            image=Image.fromarray(pixels);path=Path(directory)/'fixed.png';image.save(path)
            outside,boundary=derive_region_masks(mask)
            bundle=dict(garment=mask,outside=outside,boundary=boundary,stats=dict(mask_source='fixed_sketch_fixture'))
            off=regional_metrics(image,path,str(path),bundle,False)
            on=regional_metrics(image,path,str(path),bundle,True)
            self.assertFalse(off['inference_mask_valid']);self.assertTrue(on['inference_mask_valid'])
            for key in ['contour_f1','iou','leak']:
                self.assertTrue(np.isfinite(off[key]));self.assertEqual(off[key],on[key])
            self.assertEqual(off['iou'],1.);self.assertEqual(off['evaluation_mask_pixels'],60000)

    def test_incomplete_core_metrics_cannot_be_marked_evaluation_complete(self):
        base=dict(id='fixture',variant='E5',lpips_gt=.3,contour_f1=.8,iou=.7,leak=.01)
        with self.assertRaisesRegex(AssertionError,'MISSING_OR_NONFINITE_CORE_METRIC'):
            assess([base,dict(base,variant='B1',contour_f1=None)],'fixture')


if __name__=='__main__':unittest.main()
