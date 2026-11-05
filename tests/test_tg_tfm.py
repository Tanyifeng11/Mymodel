"""验证真实 BF 的等价性、梯度通路、EOS 与旧 CFG，不加载扩散大模型。"""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from models.bf_original_fusion import bFMFusion
from models.tg_tfm import TGTFMFusion, TGTFMAdapter, TGTFMNetwork


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.mark.parametrize('variant', ['T0', 'T1', 'T2', 'T3'])
def test_zero_heads_reproduce_original_bf(variant):
    old = bFMFusion(1, 3, ngf=2, bottleneck_depth=3, num_resblocks=0)
    new = TGTFMFusion(variant, ngf=2, bottleneck_depth=3, text_dim=8, hidden_dim=4)
    result = new.load_state_dict(old.state_dict(), strict=False)
    assert all(k.startswith('text_modulators.') for k in result.missing_keys)
    s, r = torch.randn(2, 1, 128, 96), torch.randn(2, 3, 128, 96)
    text, eos = torch.randn(2, 7, 8), torch.tensor([2, 4])
    expected, actual = old(s, r), new(s, r, text, eos)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert [list(a.shape[1:]) for a in actual] == [[4,16,12],[8,8,6],[16,4,3],[16,2,1]]


def test_text_heads_learn_and_ignore_padding():
    model = TGTFMFusion('T3', ngf=2, bottleneck_depth=3, text_dim=8, hidden_dim=4)
    s, r = torch.randn(2, 1, 128, 128), torch.randn(2, 3, 128, 128)
    text, eos = torch.randn(2, 7, 8), torch.tensor([2, 4])
    output = model(s, r, text, eos)
    sum(x.square().mean() for x in output).backward()
    heads = [p for n, p in model.named_parameters() if n.endswith('head.3.weight')]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads)
    assert sum(float(p.grad.norm()) for p in heads) > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    optimizer.step()
    a = model(s, r, text, eos)
    changed = text.clone()
    for index, end in enumerate(eos):
        changed[index, end+1:] += 100
    b = model(s, r, changed, eos)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    c = model(s, r, text.flip(0), eos)
    assert any(not torch.equal(x, y) for x, y in zip(a, c))


class Skip(nn.Module):
    def forward(self, hidden, res_hidden_states_tuple):
        return sum(x.mean((1,2,3)) for x in res_hidden_states_tuple)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList(Skip() for _ in range(4))

    def forward(self, sample, timestep, encoder_hidden_states=None, cross_attention_kwargs=None):
        outputs = []
        for block, c, side, count in zip(self.up_blocks, (16,16,8,4), (2,4,8,16), (3,2,2,3)):
            skips = tuple(sample.new_zeros(len(sample),c,side,side) for _ in range(count))
            outputs.append(block(sample,res_hidden_states_tuple=skips))
        return SimpleNamespace(sample=torch.stack(outputs).sum(0))


def test_cfg_and_legacy_key_validation():
    adapter = TGTFMAdapter('T3',ngf=2,bottleneck_depth=3,text_dim=8,hidden_dim=4,
                           channels=(4,8,16,16),eos_token_id=9)
    net = TGTFMNetwork(UNet(), adapter)
    legacy = {k:v for k,v in net.trainable_state().items() if '.text_modulators.' not in k}
    assert net.load_checked(legacy,legacy=True)['new']
    with pytest.raises(ValueError):
        net.load_checked(dict(list(legacy.items())[1:]),legacy=True)
    for projection in adapter.projections:
        nn.init.normal_(projection.weight,std=.1)
    s, r = torch.randn(2,3,128,128), torch.randn(2,3,128,128)
    ids = torch.tensor([[1,3,9,9,9,9,9],[1,2,3,4,9,9,9]])
    sample, text = torch.zeros(2,1), torch.randn(2,7,8)
    out = net(sample,100,text,s,r,{},ids)
    assert adapter.injection_counts == [3,2,2,3]
    assert adapter.context[2].tolist() == [2,4]
    uncond = net.unet(sample,100,encoder_hidden_states=-text).sample
    assert torch.equal(uncond,torch.zeros_like(uncond))
    assert adapter.injection_counts == [0,0,0,0]
    assert adapter.unconditional_calls == 1
    assert not torch.equal(out,uncond)
