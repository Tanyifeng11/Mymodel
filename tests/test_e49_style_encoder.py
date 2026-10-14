"""无权重下载的小模型验收；真实 E5 回放另由 GPU audit 执行。"""
import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
from models.e49_style_encoder import FrozenStyleEncoder, StyleReferenceProcessor


def test_encoder_input_intervention_and_rng():
    config=CLIPVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,image_size=32,patch_size=8,projection_dim=16,attention_dropout=0.)
    model=CLIPVisionModelWithProjection(config).eval().requires_grad_(False)
    plain=FrozenStyleEncoder(model,'CLIP'); shuffled=FrozenStyleEncoder(model,'SSVM')
    pixels=torch.randn(3,3,32,32)
    assert torch.equal(model(pixels).last_hidden_state,plain.encode_pixels(pixels))
    permutation=torch.stack([torch.randperm(16) for _ in range(3)])
    one=shuffled.encode_pixels(pixels,permutation)
    two=shuffled.encode_pixels(pixels,torch.arange(16).repeat(3,1))
    assert one.shape==(3,16,32) and (one-two).abs().max()>1e-5
    raw=torch.randn(1,3,48,32).tanh(); state=torch.get_rng_state().clone()
    for training in [False,True]:
        plain.training=shuffled.training=training
        plain.reset(); shuffled.reset()
        for _ in range(3):
            plain(raw); shuffled(raw)
            assert plain.last_audit['crop_boxes']==shuffled.last_audit['crop_boxes']
            assert plain.last_audit['position_permutation_sha256']==shuffled.last_audit['position_permutation_sha256']
    assert torch.equal(state,torch.get_rng_state())


def test_style_aggregation_is_trainable_and_order_invariant():
    processor=StyleReferenceProcessor(torch.nn.Identity(),'test',640,32)
    features=torch.randn(1,48,32)
    processor.style_embed(features).square().mean().backward()
    assert processor.style_embed.proj_in.weight.grad.norm()>0
    with torch.no_grad():
        assert torch.allclose(processor.style_embed(features),
            processor.style_embed(features[:,torch.randperm(48)]),atol=2e-5,rtol=2e-5)
