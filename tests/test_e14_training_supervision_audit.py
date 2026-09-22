import unittest
import torch
from tools.e14_training_supervision_audit import select_rows, training_classes


class SupervisionAuditTests(unittest.TestCase):
    def test_probe_training_inputs_equal_dataset(self):
        import json
        import tempfile
        from pathlib import Path
        from PIL import Image
        import numpy as np
        from tools.e14_training_supervision_audit import DummyTokenizer
        from tools.e14_pattern_probe import training_preprocess
        Dataset, _ = training_classes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.fromarray(np.random.RandomState(3).randint(0, 256, (63, 71, 3), dtype=np.uint8))
            image.save(root/'ref.png')
            (root/'data.json').write_text(json.dumps([dict(texture='ref.png',cloth='ref.png',caption='')]))
            ds = Dataset(str(root/'data.json'),DummyTokenizer(),width=384,height=512,
                         image_root_path=str(root),texture_preprocess_mode='plain_resize')
            expected = ds[0]
            cnn, clip = training_preprocess(image,ds.clip_image_processor,384,512)
            self.assertTrue(torch.equal(cnn[0],expected['texture_image']))
            self.assertTrue(torch.equal(clip,expected['clip_texture_image']))
            self.assertTrue(torch.equal(cnn.half()[0],expected['texture_image'].half()))

    def test_sampling_is_deterministic_and_unique(self):
        rows = [dict(caption='striped' if i < 10 else 'plain') for i in range(30)]
        indices = select_rows(rows, 12, 42)
        self.assertEqual(indices, select_rows(rows, 12, 42))
        self.assertEqual(len(set(indices)), 12)
        self.assertTrue(all(i < 10 for i in indices[:6]))

    def test_actual_gram_permutation_invariance(self):
        _, loss = training_classes()
        x = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        y = x.flatten(2).flip(-1).reshape_as(x)
        torch.testing.assert_close(loss._gram(x), loss._gram(y))


if __name__ == '__main__':
    unittest.main()
