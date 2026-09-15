"""E12 的 CPU 核验：冻结、权重恢复、文本隔离与像素验收。"""

import contextlib
import copy
import io
import json
import random
import shutil
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from models.bf_texture_module import BFTextureConditioner
from models.nexus_texture_adapter import NexusTextureAdapter, nexus_config_from_checkpoint, shuffled_adapter_prompts
from test_film_inference import actual_functions
from test_film_training import training_namespace, film_args
from test_text_guided_resampler import small_joint_models, joint_state
from tools.check_e12_off_pixels import check_pixels


@contextlib.contextmanager
def task_directory():
    root = (Path(__file__).resolve().parents[1] / "tmp").resolve()
    path = root / ("e12-test-" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.resolve().is_relative_to(root):
            shutil.rmtree(path)


def nexus_args(namespace, **overrides):
    args = film_args(namespace, film_hidden_dim=0, nexus_dim=16)
    for name in vars(args):
        if name.startswith("lambda_"):
            setattr(args, name, 0.0)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class E12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = training_namespace()

    def test_only_diffusion_loss_and_no_combined_experiments(self):
        self.ns["validate_nexus_training_args"](nexus_args(self.ns))
        for overrides in ({"lambda_style": 1}, {"lambda_edge": .05}, {"film_hidden_dim": 8},
                          {"film_text_mode": "shuffled"}, {"nexus_lr": 0},
                          {"local_detail_source": "local"}, {"use_tcpm_lite": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.ns["validate_nexus_training_args"](nexus_args(self.ns, **overrides))

    def test_attention_zero_init_text_response_and_gradient(self):
        module = NexusTextureAdapter(text_dim=32, channels=8, inner_dim=16)
        features, text = torch.randn(2, 8, 64, 48), torch.randn(2, 77, 32)
        self.assertTrue(torch.equal(module(features, text), features))
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
        target = torch.randn_like(features)
        for step in range(2):
            optimizer.zero_grad()
            loss = (module(features, text) - target).square().mean()
            loss.backward()
            self.assertGreater(module.output_proj.weight.grad.abs().sum().item(), 0)
            if step:
                self.assertGreater(module.local_conv[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(module.attention.k_proj_weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertFalse(torch.equal(module(features, text), module(features, text.flip(0))))
        self.assertEqual(module.last_stats["visual_tokens"], 3072)

    def test_native_stage3_full_path_gradients_and_off_equivalence(self):
        base = BFTextureConditioner(clip_embeddings_dim=24, cross_attention_dim=32,
                                   stage_channels=(8, 8, 8, 8)).eval()
        candidate = copy.deepcopy(base)
        candidate.configure_nexus(16)
        candidate.train_nexus_only()
        inputs = dict(texture_images=torch.randn(1, 3, 512, 384),
                      clip_vision_tokens=torch.randn(1, 5, 24), text_embeds=torch.randn(1, 77, 32))
        with torch.no_grad():
            expected = base(**inputs)[0]
            self.assertTrue(torch.equal(expected, candidate(**inputs)[0]))
        shapes = []
        hook = candidate.nexus.attention.register_forward_pre_hook(
            lambda module, args: shapes.append((args[0].shape, args[1].shape)))
        optimizer = torch.optim.AdamW(candidate.nexus.parameters(), lr=1e-2)
        target = torch.randn_like(expected)
        try:
            for step in range(2):
                optimizer.zero_grad()
                output = candidate(**inputs)[0]
                (output - target).square().mean().backward()
                self.assertGreater(candidate.nexus.output_proj.weight.grad.abs().sum().item(), 0)
                if step:
                    self.assertGreater(candidate.nexus.local_conv[0].weight.grad.abs().sum().item(), 0)
                optimizer.step()
        finally:
            hook.remove()
        self.assertEqual(shapes[0], (torch.Size([1, 3072, 16]), torch.Size([1, 77, 32])))
        for name, weight in base.state_dict().items():
            self.assertTrue(torch.equal(weight, candidate.state_dict()[name]))
        for name, parameter in candidate.named_parameters():
            if not name.startswith("nexus."):
                self.assertIsNone(parameter.grad)
        with torch.no_grad():
            self.assertFalse(torch.equal(expected, candidate(**inputs)[0]))
            self.assertTrue(torch.equal(expected, candidate(**inputs, apply_nexus=False)[0]))
            candidate.nexus_enabled = False
            inputs.pop("text_embeds")
            self.assertTrue(torch.equal(expected, candidate(**inputs)[0]))

    def test_native_grid_half_precision_and_autocast(self):
        module = NexusTextureAdapter(text_dim=32, channels=8, inner_dim=16)
        features, text = torch.randn(1, 8, 64, 48).half(), torch.randn(1, 77, 32).half()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(features, text)
            self.assertTrue(torch.equal(output, features))
            self.assertEqual(output.dtype, features.dtype)
            (output.float() - 1).square().mean().backward()
        self.assertTrue(torch.isfinite(module.output_proj.weight.grad).all())
        with torch.no_grad():
            self.assertTrue(torch.equal(module.half()(features, text), features))

    def test_inference_restores_adapter_and_off_switch(self):
        methods = actual_functions("pipelines/IMAGGarment_pipeline.py", {"load_bf_texture_conditioner"}, class_name="IMAGGarment")
        restore = actual_functions("inference_IMAGGarment-1.py", {"restore_bf_conditioner_for_inference"})[
            "restore_bf_conditioner_for_inference"]
        bf = BFTextureConditioner(clip_embeddings_dim=24, cross_attention_dim=32,
                                  stage_channels=(8, 8, 8, 8), nexus_dim=16)
        pipe = SimpleNamespace(device="cpu", num_tokens=16, bf_texture_conditioner=None,
            unet=SimpleNamespace(config=SimpleNamespace(cross_attention_dim=32)),
            image_encoder=SimpleNamespace(config=SimpleNamespace(hidden_size=24)))
        pipe.load_bf_texture_conditioner = lambda state, meta: methods["load_bf_texture_conditioner"](pipe, state, meta)
        restore(pipe, bf.state_dict(), bf.nexus_config(), SimpleNamespace(disable_nexus_adapter=True))
        self.assertIsNotNone(pipe.bf_texture_conditioner.nexus)
        self.assertFalse(pipe.bf_texture_conditioner.nexus_enabled)
        restore(pipe, None, {}, SimpleNamespace())
        self.assertTrue(pipe.bf_texture_conditioner.nexus_enabled)
        base = BFTextureConditioner(clip_embeddings_dim=24, cross_attention_dim=32, stage_channels=(8, 8, 8, 8))
        restore(pipe, base.state_dict(), {}, SimpleNamespace())
        self.assertIsNone(pipe.bf_texture_conditioner.nexus)

    def test_base_loading_freezing_and_checkpoint_roundtrip(self):
        args = nexus_args(self.ns)
        models = small_joint_models()
        reference = joint_state(models)
        models["bf"].configure_nexus(16)
        self.ns["validate_nexus_source"](reference, args)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["load_joint_checkpoint_into_models"](reference, **models, strict_base=True)
        self.ns["freeze_nexus_only"](models["bf"], [v for k, v in models.items() if k != "bf"])
        for key, module in models.items():
            for name, parameter in module.named_parameters():
                self.assertEqual(parameter.requires_grad, key == "bf" and name.startswith("nexus."))
        for name, weight in reference["bf_texture_conditioner"].items():
            self.assertTrue(torch.equal(weight, models["bf"].state_dict()[name]))
        with task_directory() as directory:
            self.ns["save_training_checkpoint"](
                accelerator=None, **models, output_dir=directory, global_step=25,
                args=args, resolved_image_encoder_path="clip",
            )
            saved = torch.load(Path(directory) / "checkpoint-25/joint_model.pt", weights_only=False)
        self.assertEqual(saved["meta"]["nexus_dim"], 16)
        self.assertEqual(self.ns["validate_nexus_source"](saved, args, resume=True), 25)
        restored = small_joint_models()
        restored["bf"].configure_nexus(**nexus_config_from_checkpoint(saved["bf_texture_conditioner"], saved["meta"]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["load_joint_checkpoint_into_models"](saved, **restored, strict_base=True)
        for name, weight in models["bf"].state_dict().items():
            self.assertTrue(torch.equal(weight, restored["bf"].state_dict()[name]))
        for key in ("nexus.output_proj.bias", "stage2.0.bias"):
            broken = copy.deepcopy(saved)
            del broken["bf_texture_conditioner"][key]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, RuntimeError)):
                self.ns["load_joint_checkpoint_into_models"](broken, **restored, strict_base=True)

    def test_shuffled_prompts_are_stable_and_only_change_adapter_text(self):
        samples = [{"sample_id": i, "prompt": f"text {i}"} for i in range(7)]
        before, rng = copy.deepcopy(samples), random.getstate()
        mapping = shuffled_adapter_prompts(samples)
        self.assertEqual(samples, before)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(mapping, shuffled_adapter_prompts(samples))
        self.assertEqual(set(mapping.values()), {row["prompt"] for row in samples})
        self.assertTrue(all(mapping[row["sample_id"]] != row["prompt"] for row in samples))
        with self.assertRaises(ValueError):
            shuffled_adapter_prompts(samples[:1])

    def test_cfg_and_tcpm_keep_original_text(self):
        methods = actual_functions("pipelines/IMAGGarment_pipeline.py", {"get_image_embeds"}, class_name="IMAGGarment")
        calls, tcpm_calls = [], []
        def conditioner(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 16, 32), []
        def tcpm(texture, text):
            tcpm_calls.append(text)
            return texture
        pipe = SimpleNamespace(
            device="cpu", bf_texture_conditioner=conditioner, use_tcpm_lite=True, tcpm_lite=tcpm,
            clip_image_processor=lambda **kwargs: SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8)),
            image_encoder=lambda *args, **kwargs: SimpleNamespace(
                image_embeds=torch.zeros(1, 24), hidden_states=[torch.zeros(1, 5, 24)]),
            cond_image_processor=SimpleNamespace(preprocess=lambda *args, **kwargs: torch.zeros(1, 3, 8, 8)),
            _apply_aa_tcr_fuse=lambda texture, text, caption: texture,
        )
        correct, shuffled, negative = [torch.randn(1, 5, 32) for _ in range(3)]
        methods["get_image_embeds"](
            pipe, pil_image=Image.new("RGB", (8, 8)), width=8, height=8,
            text_embeds=correct, nexus_text_embeds=shuffled, negative_text_embeds=negative,
        )
        self.assertIs(calls[0]["text_embeds"], correct)
        self.assertIs(calls[0]["nexus_text_embeds"], shuffled)
        self.assertIs(calls[1]["text_embeds"], negative)
        self.assertTrue(calls[1]["apply_nexus"])
        self.assertIs(tcpm_calls[0], correct)
        self.assertIs(tcpm_calls[1], negative)

    def test_pixels_compare_original_images_and_fail_on_difference(self):
        with task_directory() as directory:
            base = Path(directory)
            for group in ("e5", "off"):
                folder = base / group
                folder.mkdir()
                Image.new("RGB", (4, 4)).save(folder / "generated.png")
                (folder / "metrics_per_sample.json").write_text(json.dumps([
                    {"sample_id": 0, "source_gen_path": str(folder / "generated.png"), "gen_path": "unused"}
                ]), encoding="utf-8")
            check_pixels(base / "e5", base / "off", 1)
            Image.new("RGB", (4, 4), "red").save(base / "off/generated.png")
            with self.assertRaises(ValueError):
                check_pixels(base / "e5", base / "off", 1)


if __name__ == "__main__":
    unittest.main()
