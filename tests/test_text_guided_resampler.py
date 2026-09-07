"""CPU 检查：E5 等价初始化、条件对齐、梯度、冻结范围和权重往返。"""

import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import typing
import unittest

from PIL import Image
import torch

from models.bf_texture_module import BFTextureConditioner
from models.tcpm_lite import TCPMLite
from models.text_guided_queries import (
    TextGuidedQueries, guidance_config_from_checkpoint, text_content_mask,
)


ROOT = Path(__file__).resolve().parents[1]


def small_conditioner(guided=False):
    return BFTextureConditioner(
        clip_embeddings_dim=24, cross_attention_dim=32, num_tokens=16,
        stage_channels=(8, 8, 8, 8), stage_token_hw=(2, 2),
        text_guidance_dim=16 if guided else 0,
    )


def pipeline_method(name):
    # 本机无 diffusers；执行真实方法体，用小型编码器替身检查条件接线。
    tree = ast.parse((ROOT / "pipelines/IMAGGarment_pipeline.py").read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "IMAGGarment")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = {
        **vars(typing), "torch": torch, "Image": Image,
        "text_content_mask": text_content_mask,
        "TextualInversionLoaderMixin": type("TextualInversionLoaderMixin", (), {}),
        "LoraLoaderMixin": type("LoraLoaderMixin", (), {}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "pipelines/IMAGGarment_pipeline.py"), "exec"), namespace)
    return namespace[name]


def training_load_functions():
    # 隔离执行真实加载函数，避免 CPU 测试依赖 diffusers/accelerate。
    path = ROOT / "train_GAM_texture_joint.py"
    names = {"load_partial_state", "validate_text_only_source", "load_joint_checkpoint_into_models",
             "_is_palette_key", "_is_balanced_gate_key"}
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"nn": torch.nn, "guidance_config_from_checkpoint": guidance_config_from_checkpoint}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def small_joint_models(guided=False):
    unet = torch.nn.Sequential(torch.nn.Linear(4, 4))
    unet.attn_processors = {"test": unet[0]}
    return {
        "unet": unet, "ref_unet": torch.nn.Linear(4, 4), "bf": small_conditioner(guided),
        "spatial_texture_encoder": torch.nn.Linear(4, 4),
        "spatial_injection": torch.nn.Linear(4, 4),
        "palette_token_mlp": torch.nn.Linear(4, 4), "tcpm_lite": TCPMLite(32),
    }


def joint_state(models, mode="off"):
    state = {("bf_texture_conditioner" if name == "bf" else name): copy.deepcopy(model.state_dict())
             for name, model in models.items()}
    state["texture_adapter"] = copy.deepcopy(
        torch.nn.ModuleList(models["unet"].attn_processors.values()).state_dict())
    state["meta"] = {**models["bf"].text_guidance_config(), "resampler_training": mode}
    return state


class FakeTokenizer:
    model_max_length = 8
    eos_token_id = 2

    def __call__(self, captions, **kwargs):
        captions = [captions] if isinstance(captions, str) else captions
        ids = []
        for caption in captions:
            words = [3 + i for i, _ in enumerate(caption.split())]
            row = [0] + words[:6] + [2]
            ids.append(row + [2] * (8 - len(row)))
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(ids).ne(2))


class FakeTextEncoder:
    dtype = torch.float32
    config = SimpleNamespace(use_attention_mask=False)

    def __call__(self, input_ids, **kwargs):
        return (torch.nn.functional.one_hot(input_ids, num_classes=10).float(),)


class ResamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def inputs(self):
        return {
            "texture_images": torch.randn(2, 3, 32, 32),
            "clip_vision_tokens": torch.randn(2, 9, 24),
            "text_embeds": torch.randn(2, 7, 32),
            "text_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]).bool(),
        }

    def test_initial_output_matches_e5(self):
        base, candidate = small_conditioner(), small_conditioner(True)
        missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(k.startswith("text_guidance.") for k in missing))
        self.assertFalse(unexpected)
        data = self.inputs()
        a, _ = base(**data)
        c, _ = candidate(**data)
        self.assertEqual(c.shape, (2, 16, 32))
        torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_gradient_wakes_up_and_frozen_weights_stay_fixed(self):
        model = small_conditioner(True)
        model.train_resampler_only()
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        allowed = ("resampler_queries", "resampler.", "text_guidance.")
        self.assertTrue(all(k.startswith(allowed) for k, p in model.named_parameters() if p.requires_grad))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        data, target = self.inputs(), torch.randn(2, 16, 32)
        for step in range(3):
            optimizer.zero_grad()
            prediction, _ = model(**data)
            torch.nn.functional.mse_loss(prediction, target).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if step == 0:
                self.assertGreater(model.text_guidance.gate.grad.abs().item(), 0)
            if step == 1:
                self.assertGreater(model.text_guidance.to_out.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.resampler_queries, before["resampler_queries"]))

    def test_text_only_learns_through_frozen_bf_tcpm_and_downstream(self):
        base, candidate = small_conditioner(), small_conditioner(True)
        candidate.load_state_dict(base.state_dict(), strict=False)
        base.requires_grad_(False)
        candidate.train_text_guidance_only()
        tcpm = TCPMLite(32, residual_scale_init=0.2).requires_grad_(False)
        downstream = torch.nn.Sequential(torch.nn.Linear(32, 8), torch.nn.SiLU(),
                                         torch.nn.Linear(8, 3)).requires_grad_(False)
        modules = {"bf": candidate, "tcpm": tcpm, "downstream": downstream}
        before = {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}
        names = [name for name, p in candidate.named_parameters() if p.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("text_guidance.") for name in names))
        data = self.inputs()

        def predict(model):
            tokens = tcpm(model(**data)[0], data["text_embeds"])
            return downstream(tokens.mean(dim=1))

        baseline = predict(base).detach()
        torch.testing.assert_close(predict(candidate), baseline, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(candidate.text_guidance.parameters(), lr=1e-2)
        target = torch.randn_like(baseline)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(predict(candidate), target).backward()
            for name, parameter in candidate.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(candidate.text_guidance.gate.grad.abs().item(), 0)
            if step > 0:
                self.assertGreater(candidate.text_guidance.to_out.weight.grad.abs().sum().item(), 0)
            optimizer.step()

        changed_text = []
        for module_name, module in modules.items():
            for name, value in module.state_dict().items():
                if module_name == "bf" and name.startswith("text_guidance."):
                    if not torch.equal(value, before[module_name][name]):
                        changed_text.append(name)
                else:
                    torch.testing.assert_close(value, before[module_name][name], rtol=0, atol=0)
            if module_name != "bf":
                self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIn("text_guidance.gate", changed_text)
        self.assertIn("text_guidance.to_out.weight", changed_text)
        self.assertFalse(torch.equal(predict(candidate), baseline))
        candidate.text_guidance_enabled = False
        torch.testing.assert_close(predict(candidate), baseline, rtol=0, atol=0)

    def test_text_only_requires_text_module(self):
        with self.assertRaisesRegex(ValueError, "text_only"):
            small_conditioner().train_text_guidance_only()

    def test_strict_e5_load_only_allows_new_text_keys(self):
        functions = training_load_functions()
        state = joint_state(small_joint_models())
        candidate = small_joint_models(True)
        functions["validate_text_only_source"](state)
        with contextlib.redirect_stdout(io.StringIO()):
            functions["load_joint_checkpoint_into_models"](state, **candidate, strict_base=True)
        for name, value in state["bf_texture_conditioner"].items():
            torch.testing.assert_close(candidate["bf"].state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(candidate["bf"].text_guidance.gate.item(), 0)
        for component, missing_key in (("bf_texture_conditioner", "resampler_queries"),
                                       ("unet", "0.weight"), ("tcpm_lite", "residual_scale"),
                                       ("texture_adapter", "0.weight")):
            with self.subTest(component=component), contextlib.redirect_stdout(io.StringIO()):
                broken = copy.deepcopy(state)
                del broken[component][missing_key]
                with self.assertRaises((ValueError, RuntimeError)):
                    functions["load_joint_checkpoint_into_models"](
                        broken, **small_joint_models(True), strict_base=True)
        broken = copy.deepcopy(state)
        del broken["texture_adapter"]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "texture_adapter"):
            functions["load_joint_checkpoint_into_models"](
                broken, **small_joint_models(True), strict_base=True)

    def test_text_only_resume_rejects_other_modes_and_incomplete_text(self):
        functions = training_load_functions()
        e8c = joint_state(small_joint_models(True), "text_only")
        functions["validate_text_only_source"](e8c, resume=True)
        with contextlib.redirect_stdout(io.StringIO()):
            functions["load_joint_checkpoint_into_models"](
                e8c, **small_joint_models(True), strict_base=True)
        for mode, guided in (("off", False), ("visual", False), ("text", True)):
            with self.subTest(mode=mode):
                state = joint_state(small_joint_models(guided), mode)
                with self.assertRaisesRegex(ValueError, "text_only"):
                    functions["validate_text_only_source"](state, resume=True)
                if mode != "off":
                    with self.assertRaisesRegex(ValueError, "E5"):
                        functions["validate_text_only_source"](state)
        del e8c["bf_texture_conditioner"]["text_guidance.to_out.weight"]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "text_guidance"):
            functions["load_joint_checkpoint_into_models"](
                e8c, **small_joint_models(True), strict_base=True)

    def test_empty_single_word_padding_and_ratio_bound(self):
        module = TextGuidedQueries(32, 16, 4, 0.3)
        module.gate.data.fill_(5)
        ids = torch.tensor([[0, 2, 2, 2, 2], [0, 9, 2, 2, 2]])
        mask = text_content_mask(ids, 2)
        self.assertEqual(mask.sum(dim=-1).tolist(), [0, 1])
        q, text = torch.randn(2, 16, 32), torch.randn(2, 5, 32)
        actual = module(q, text, mask)
        torch.testing.assert_close(actual[0], q[0], rtol=0, atol=0)
        self.assertGreater((actual[1] - q[1]).abs().sum().item(), 0)
        self.assertLessEqual(module.last_stats["query_relative_rms_max"], 0.300001)
        changed_padding = text.clone()
        changed_padding[~mask] = 100 * torch.randn_like(changed_padding[~mask])
        torch.testing.assert_close(module(q, changed_padding, mask), actual, rtol=0, atol=0)

    def test_half_precision_empty_rows_have_finite_gradients(self):
        module = TextGuidedQueries(32, 16).half()
        module.gate.data.fill_(1)
        q = torch.randn(2, 16, 32).half()
        output = module(q, torch.zeros(2, 5, 32).half(), torch.zeros(2, 5).bool())
        output.float().square().mean().backward()
        torch.testing.assert_close(output, q, rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))

    def test_autocast_handles_masked_and_empty_text(self):
        # CPU 上也实际启用 FP16 autocast；有 GPU 时覆盖服务器的 CUDA 路径。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = TextGuidedQueries(32, 16).to(device)
        q = torch.randn(2, 16, 32, device=device)
        text = torch.randn(2, 5, 32, device=device)
        mask = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], device=device).bool()
        for gate_value in (0.0, 0.5):
            with self.subTest(gate=gate_value):
                module.zero_grad(set_to_none=True)
                module.gate.data.fill_(gate_value)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = module(q, text, mask)
                    changed_padding = text.clone()
                    changed_padding[~mask] = 100 * torch.randn_like(changed_padding[~mask])
                    padding_output = module(q, changed_padding, mask)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output[1], q[1], rtol=0, atol=0)
                torch.testing.assert_close(padding_output, output, rtol=0, atol=0)
                if gate_value == 0:
                    torch.testing.assert_close(output, q, rtol=0, atol=0)
                self.assertLessEqual(module.last_stats["query_relative_rms_max"], 0.300001)
                output.float().square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in module.parameters()))
                self.assertGreater(module.gate.grad.abs().item(), 0)
                if gate_value:
                    self.assertGreater(module.to_out.weight.grad.abs().sum().item(), 0)

    def test_conditioner_autocast_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = small_conditioner(True).to(device)
        model.train_resampler_only()
        model.text_guidance.gate.data.fill_(0.5)
        data = {name: value.to(device) for name, value in self.inputs().items()}
        data["text_mask"][1] = False
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output, _ = model(**data)
        self.assertTrue(torch.isfinite(output).all())
        torch.nn.functional.mse_loss(output.float(), torch.randn_like(output).float()).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_checkpoint_roundtrip_and_metadata_validation(self):
        model = small_conditioner(True).eval()
        model.text_guidance.gate.data.fill_(0.6)
        payload = {"bf_texture_conditioner": model.state_dict(), "meta": model.text_guidance_config()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        config = guidance_config_from_checkpoint(saved["bf_texture_conditioner"], saved["meta"])
        restored = small_conditioner().eval()
        restored.configure_text_guidance(**config)
        restored.load_state_dict(saved["bf_texture_conditioner"], strict=True)
        data = self.inputs()
        torch.testing.assert_close(model(**data)[0], restored(**data)[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            guidance_config_from_checkpoint(saved["bf_texture_conditioner"], {})
        with self.assertRaises(ValueError):
            guidance_config_from_checkpoint({}, saved["meta"])
        restored.text_guidance_enabled = False
        self.assertFalse(torch.allclose(model(**data)[0], restored(**data)[0]))

    def test_visual_control_freezes_text_and_feature_extractors(self):
        model = small_conditioner()
        model.train_resampler_only()
        self.assertIsNone(model.text_guidance)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all(n.startswith(("resampler_queries", "resampler.")) for n in names))

    def test_actual_encode_prompt_keeps_cfg_masks_separate(self):
        pipe = SimpleNamespace(tokenizer=FakeTokenizer(), text_encoder=FakeTextEncoder())
        encode = pipeline_method("encode_prompt")
        args = (pipe, ["red plaid", "blue"], torch.device("cpu"), 2, True)
        positive, negative, pos_mask, neg_mask = encode(
            *args, negative_prompt=["", "gray"], return_text_masks=True
        )
        self.assertEqual(positive.shape, (4, 8, 10))
        self.assertEqual(negative.shape, positive.shape)
        self.assertEqual(pos_mask.sum(dim=-1).tolist(), [2, 2, 1, 1])
        self.assertEqual(neg_mask.sum(dim=-1).tolist(), [0, 0, 1, 1])
        self.assertEqual(len(encode(*args, negative_prompt=["", "gray"])), 2)
        with self.assertRaises(ValueError):
            encode(pipe, None, torch.device("cpu"), 1, False,
                   prompt_embeds=positive[:1], return_text_masks=True)

    def test_actual_image_embedding_call_routes_both_conditions(self):
        calls = []
        def conditioner(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 16, 10), []
        pipe = SimpleNamespace(
            device="cpu", bf_texture_conditioner=conditioner, use_tcpm_lite=False,
            clip_image_processor=lambda **kwargs: SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8)),
            image_encoder=lambda *args, **kwargs: SimpleNamespace(
                image_embeds=torch.zeros(1, 10), hidden_states=[torch.zeros(1, 5, 10)]),
            cond_image_processor=SimpleNamespace(preprocess=lambda *args, **kwargs: torch.zeros(1, 3, 8, 8)),
            _apply_aa_tcr_fuse=lambda texture, text, caption: texture,
        )
        positive, negative = torch.randn(1, 8, 10), torch.randn(1, 8, 10)
        pos_mask, neg_mask = torch.ones(1, 8).bool(), torch.zeros(1, 8).bool()
        pipeline_method("get_image_embeds")(
            pipe, pil_image=Image.new("RGB", (8, 8)), width=8, height=8,
            text_embeds=positive, negative_text_embeds=negative,
            text_mask=pos_mask, negative_text_mask=neg_mask,
        )
        self.assertIs(calls[0]["text_embeds"], positive)
        self.assertIs(calls[0]["text_mask"], pos_mask)
        self.assertIs(calls[1]["text_embeds"], negative)
        self.assertIs(calls[1]["text_mask"], neg_mask)
        self.assertTrue(calls[1]["apply_text_guidance"])
        self.assertEqual(calls[1]["texture_images"].count_nonzero().item(), 0)
        calls.clear()
        pipeline_method("get_image_embeds")(
            pipe, pil_image=Image.new("RGB", (8, 8)), width=8, height=8,
            text_embeds=positive, text_mask=pos_mask,
        )
        self.assertFalse(calls[1]["apply_text_guidance"])
        self.assertIsNone(calls[1]["text_embeds"])


if __name__ == "__main__":
    unittest.main()
