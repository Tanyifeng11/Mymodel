import torch
from models.t3_local_texture import LocalTextureBranch, reference_consistency


def test_zero_start_and_learning_chain():
    torch.manual_seed(1)
    branch = LocalTextureBranch(channels=(8, 8))
    image = torch.rand(1, 3, 64, 64)
    hidden = torch.randn(1, 8, 8, 8)
    mask = torch.ones(1, 1, 64, 64)
    branch.condition(image, mask)
    output = branch.blocks[0](hidden, branch.context, mask, torch.tensor([100]))
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert branch.blocks[0].output.weight.grad.abs().sum() > 0


def test_mask_and_unconditional_are_zero():
    branch = LocalTextureBranch(channels=(8, 8))
    branch.blocks[0].output.weight.data.fill_(.01)
    branch.condition(torch.rand(1, 3, 64, 64), torch.zeros(1, 1, 64, 64))
    output = branch.blocks[0](torch.randn(1, 8, 8, 8), branch.context, branch.mask, torch.tensor([100]))
    assert torch.count_nonzero(output) == 0


def test_reference_loss_detects_rotation_and_backpropagates():
    x = torch.arange(128).float()
    stripe = (.5 + .4 * torch.sin(x * (2 * torch.pi / 16)))[None, None, None].expand(1, 3, 128, 128).clone()
    rotated = stripe.transpose(-1, -2).contiguous()
    assert reference_consistency(stripe, stripe) < 1e-7
    assert reference_consistency(stripe, rotated) > .5
    predicted = (stripe * .8 + rotated * .2).requires_grad_(True)
    reference_consistency(predicted, stripe).backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0


def test_hooks_disable_unconditional_and_early_timesteps():
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])

        def forward(self, hidden, timestep, cross_attention_kwargs=None):
            return self.blocks[1](self.blocks[0](hidden))

    net = Stub()
    branch = LocalTextureBranch(channels=(8,8))
    branch.layers = ('blocks.0','blocks.1')
    branch.enabled = True
    for block in branch.blocks:
        block.output.bias.data.fill_(1)
    branch.condition(torch.rand(1,3,64,64),torch.ones(1,1,64,64))
    branch.bind(net)
    value = torch.zeros(1,8,8,8)
    assert torch.equal(net(value,torch.tensor([100])),value)
    assert torch.equal(net(value,torch.tensor([900]),cross_attention_kwargs={'sa_hidden_states':{}}),value)
    assert net(value,torch.tensor([100]),cross_attention_kwargs={'sa_hidden_states':{}}).abs().sum()>0
    branch.unbind()
