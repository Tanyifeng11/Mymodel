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


def test_synthetic_measure_has_valid_period_and_orientation():
    import numpy as np
    from tools.t3_local_pilot_report import pattern_measure
    yy,xx=np.mgrid[:128,:128]
    for angle in [0,90]:
        for period in [12,16,20,24]:
            gray=127+80*np.sin(2*np.pi*(xx if angle==90 else yy)/period)
            pixels=np.repeat(gray[...,None],3,-1)
            result=pattern_measure(pixels,dict(patches=[(0,0)],expected_angle=angle,expected_period=period))
            assert result['orientation_correct'] and result['period_relative_error']<.1


def test_optimizer_and_random_stream_resume_matches_next_update(tmp_path):
    from tools.t3_local_pilot import restore_training_state
    torch.manual_seed(12)
    model = torch.nn.Linear(3,2)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    rng = torch.Generator().manual_seed(7)
    def update(net,opt,generator):
        opt.zero_grad()
        x = torch.randn((4,3),generator=generator)
        target = torch.randn((4,2),generator=generator)
        (net(x)-target).square().mean().backward()
        opt.step()
    for _ in range(4):
        update(model,optimizer,rng)
    path = tmp_path/'resume.pt'
    torch.save(dict(branch=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),total_updates=800),path)
    resumed = torch.nn.Linear(3,2)
    resumed.load_state_dict(model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(),lr=.01)
    resumed_rng = torch.Generator().manual_seed(999)
    audit = restore_training_state(resumed,resumed_optimizer,resumed_rng,path,800)
    assert audit['optimizer_steps'] == [4]
    update(model,optimizer,rng)
    update(resumed,resumed_optimizer,resumed_rng)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),resumed.parameters()))
    assert torch.equal(rng.get_state(),resumed_rng.get_state())
