"""CPU 接口检查：条件映射、GT隔离、写路径边界。"""
import unittest
from pathlib import Path
from .common import generation_row, training_row, output, ROOT


class ContractTests(unittest.TestCase):
    def test_fabric_mapping_preserves_caption(self):
        source = dict(cloth='cloth/42.jpg', sketch='sketch/42.jpg', texture='texture/42.jpg', caption='a striped shirt')
        result = training_row(source, Path('/data/train'), 'train')
        self.assertEqual(result['caption'], 'a striped shirt')
        self.assertEqual(result['fabric_path'], str(Path('/data/train/texture/42.jpg')))

    def test_generator_never_receives_gt(self):
        source = dict(id='42', category='top', condition='original', seed=42, caption='shirt', sketch='s', reference='f',
                      sketch_sha256='a', reference_sha256='b', split='heldout128', gt='secret_gt.png', target_path='secret')
        result = generation_row(source)
        self.assertNotIn('gt', result)
        self.assertNotIn('target_path', result)

    def test_output_rejects_existing_model_path(self):
        with self.assertRaises(ValueError):
            output(ROOT.parent.parent / 'weight/GAM.pt')
        with self.assertRaises(ValueError):
            output(ROOT / '..' / 'old_experiment/model.pt')


if __name__ == '__main__':
    unittest.main()
