import unittest
import torch
from checkpoint_utils import load_texture_warmstart


class Processor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_palette_tokens = False
        self.to_k_ip = torch.nn.Linear(4, 4, bias=False)
        self.to_k_palette = torch.nn.Linear(4, 4, bias=False)
        self.to_v_palette = torch.nn.Linear(4, 4, bias=False)
        self.palette_branch_scale = torch.nn.Parameter(torch.ones(()))


class WarmstartTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.processor = Processor()
        self.model.adapter_modules = torch.nn.ModuleList([self.model.processor])
        self.old = {k: torch.full_like(v, .25) for k,v in self.model.state_dict().items() if 'palette' not in k}

    def test_disabled_shared_palette_only(self):
        filled = load_texture_warmstart(self.model, self.old)
        self.assertEqual(len(filled), 6)
        for key in filled:
            self.assertEqual(self.model.state_dict()[key].abs().sum().item(), 0)
            self.assertFalse(self.model.get_parameter(key).requires_grad)
        torch.testing.assert_close(self.model.processor.to_k_ip.weight, self.old['processor.to_k_ip.weight'])

    def test_rejects_other_incompatibilities(self):
        for change in ['enabled', 'missing', 'extra', 'shape']:
            with self.subTest(change=change):
                self.setUp()
                state = dict(self.old)
                if change == 'enabled': self.model.processor.use_palette_tokens = True
                if change == 'missing': del state['processor.to_k_ip.weight']
                if change == 'extra': state['unknown'] = torch.zeros(1)
                if change == 'shape': state['processor.to_k_ip.weight'] = torch.zeros(2)
                with self.assertRaises(RuntimeError): load_texture_warmstart(self.model,state)


if __name__ == '__main__':
    unittest.main()
