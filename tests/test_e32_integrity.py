"""E32 监督隔离和数值契约，不用随机网络测试实际模型效果。"""

import unittest

import numpy as np
from PIL import Image

from data.e32_target_pseudogt import masks, statistical_descriptor, structure_input
from data.e32_wrong_reference_sampler import wrong_references
from tools.e32_stage0_pair_audit import descriptor_similarity
from tools.e32_common import bootstrap


class E32Integrity(unittest.TestCase):
    def test_target_never_changes_structure_input(self):
        pixels = np.full((512,384,3),255,np.uint8)
        pixels[30:480,30:34]=0
        pixels[30:480,350:354]=0
        pixels[30:34,30:354]=0
        pixels[476:480,30:354]=0
        sketch = Image.fromarray(pixels)
        first, _, _ = masks(sketch, Image.new('RGB',sketch.size,'black'))
        second, _, _ = masks(sketch, Image.new('RGB',sketch.size,'red'))
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(structure_input(sketch,first),structure_input(sketch,second))

    def test_radial_descriptor_rotation(self):
        rng=np.random.default_rng(42)
        image=Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
        first=statistical_descriptor(image)
        second=statistical_descriptor(image.transpose(Image.Transpose.ROTATE_90))
        self.assertEqual(first.shape,(38,))
        np.testing.assert_allclose(first[:32],second[:32],atol=3e-5)

    def test_descriptor_self_and_no_color_only_gate(self):
        rng=np.random.default_rng(42)
        values=rng.normal(size=(16,64)).astype(np.float32)
        changed=values.copy();changed[:,32:38]+=100
        self.assertAlmostEqual(descriptor_similarity(values,values),1,places=5)
        self.assertAlmostEqual(descriptor_similarity(values,changed),1,places=5)
        self.assertLess(descriptor_similarity(values,-values),.9)

    def test_deterministic_distinct_negatives(self):
        rows=[dict(id=str(i)) for i in range(5)]
        features={str(i):dict(histogram=np.eye(5,dtype=np.float32)[i]) for i in range(5)}
        hashes={str(i):dict(reference=str(i),target=str(i)) for i in range(5)}
        hashes['1']['reference']='0'
        wrong=wrong_references(rows,features,hashes)
        self.assertEqual(wrong,wrong_references(rows,features,hashes))
        for row in rows:
            sid=row['id']
            for name in ('color_near','random'):
                donor=wrong[sid][name]
                self.assertNotEqual(sid,donor)
                self.assertNotEqual(hashes[sid]['reference'],hashes[donor]['reference'])

    def test_reference_unit_bootstrap(self):
        positive=bootstrap([.2]*16)
        self.assertGreater(positive['ci95'][0],0)
        self.assertEqual(positive['n'],16)
        self.assertIsNone(bootstrap([])['ci95'])


if __name__=='__main__':
    unittest.main()
