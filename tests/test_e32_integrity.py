"""E32 监督隔离和数值契约，不用随机网络测试实际模型效果。"""

import unittest

import numpy as np
import torch
from PIL import Image

from data.e32_target_pseudogt import masks, statistical_descriptor, structure_input
from data.e32_wrong_reference_sampler import wrong_references
from tools.e32_stage0_pair_audit import descriptor_similarity
from tools.e32_common import bootstrap
from tools.e32_a_geometry_train import geometry_loss
from models.e32_field import ExplicitPatternField


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

    def test_zero_has_no_geometry_reconstruction(self):
        gt=torch.zeros(1,4,64,48);gt[:,0]=1;gt[:,2]=-3;gt[:,3]=.7
        batch=dict(supervision_geometry=gt,supervision_interior=torch.ones(1,64,48),pair_weight=torch.ones(1))
        pred=dict(orientation=gt[:,:2].clone(),log_frequency=torch.full((1,1,64,48),-3.),
                  confidence=torch.full((1,3,64,48),.5))
        zero=torch.ones(1,dtype=torch.bool)
        first,parts=geometry_loss(pred,batch,zero)
        self.assertEqual(parts['orientation'],0)
        self.assertEqual(parts['period'],0)
        gt[:,0]=-1;gt[:,2]=-8
        second,_=geometry_loss(pred,batch,zero)
        torch.testing.assert_close(first,second)

    def test_field_forward_shapes_and_gradients(self):
        torch.set_num_threads(2)
        model=ExplicitPatternField()
        self.assertFalse(any(p.requires_grad for p in model.appearance.parameters()))
        out=model(torch.randn(1,16,12,395),torch.randn(1,7,128,96))
        self.assertEqual(out['orientation'].shape,(1,2,64,48))
        self.assertEqual(out['appearance'].shape,(1,64,64,48))
        self.assertEqual(out['confidence'].shape,(1,3,64,48))
        gt=torch.zeros(1,4,64,48);gt[:,0]=1;gt[:,2]=-3;gt[:,3]=.7
        loss,_=geometry_loss(out,dict(supervision_geometry=gt,supervision_interior=torch.ones(1,64,48),
                                    pair_weight=torch.ones(1)),torch.zeros(1,dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.reference_projection[0].weight.grad)
        self.assertTrue(torch.isfinite(model.reference_projection[0].weight.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA AMP check requires a GPU')
    def test_cuda_mixed_precision_loss_backward(self):
        model=ExplicitPatternField().cuda()
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        reference=torch.randn(1,16,12,395,device='cuda')
        structure=torch.randn(1,7,128,96,device='cuda')
        gt=torch.zeros(1,4,64,48,device='cuda');gt[:,0]=1;gt[:,2]=-3;gt[:,3]=.7
        with torch.autocast(device_type='cuda',dtype=dtype):
            out=model(reference,structure)
        loss,_=geometry_loss(out,dict(supervision_geometry=gt,supervision_interior=torch.ones(1,64,48,device='cuda'),
                                    pair_weight=torch.ones(1,device='cuda')),torch.zeros(1,dtype=torch.bool,device='cuda'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.reference_projection[0].weight.grad).all())
        print('E32 CUDA AMP passed:',torch.cuda.get_device_name(),dtype,flush=True)


if __name__=='__main__':
    unittest.main()
