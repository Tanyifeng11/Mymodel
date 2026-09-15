"""CPU 核验 FiLM 的 checkpoint 恢复、CFG 接线与固定评测开关。"""

import argparse
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from checkpoint_utils import extract_texture_metadata, infer_clip_embed_dim, infer_texture_num_tokens
from models.bf_texture_module import BFTextureConditioner
from models.text_guided_queries import guidance_config_from_checkpoint, text_content_mask
from models.text_texture_film import film_config_from_checkpoint
from models.nexus_texture_adapter import nexus_config_from_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def actual_functions(path, names, class_name=None, extra=None):
    """直接执行实际方法，隔离本机未安装的 diffusers。"""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("待核验函数不存在")
    namespace = {
        **vars(typing), "torch": torch, "Image": Image, "argparse": argparse, "os": os,
        "BFTextureConditioner": BFTextureConditioner,
        "extract_texture_metadata": extract_texture_metadata,
        "infer_clip_embed_dim": infer_clip_embed_dim, "infer_texture_num_tokens": infer_texture_num_tokens,
        "guidance_config_from_checkpoint": guidance_config_from_checkpoint,
        "film_config_from_checkpoint": film_config_from_checkpoint,
        "nexus_config_from_checkpoint": nexus_config_from_checkpoint, "text_content_mask": text_content_mask,
        "TextualInversionLoaderMixin": type("TextualInversionLoaderMixin", (), {}),
        "LoraLoaderMixin": type("LoraLoaderMixin", (), {}),
    }
    namespace.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def small_conditioner(film_dim=8):
    return BFTextureConditioner(clip_embeddings_dim=24, cross_attention_dim=32,
                                stage_channels=(8, 8, 8, 8), film_hidden_dim=film_dim)


class FiLMInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.methods = actual_functions(
            "pipelines/IMAGGarment_pipeline.py",
            {"load_bf_texture_conditioner", "load_texture_adapter", "encode_prompt", "get_image_embeds"},
            class_name="IMAGGarment",
        )
        self.pipe = SimpleNamespace(
            device="cpu", num_tokens=16, bf_texture_conditioner=None,
            unet=SimpleNamespace(config=SimpleNamespace(cross_attention_dim=32)),
            image_encoder=SimpleNamespace(config=SimpleNamespace(hidden_size=24)),
            use_tcpm_lite=True,
        )
        self.pipe.load_bf_texture_conditioner = lambda state, meta: self.methods["load_bf_texture_conditioner"](
            self.pipe, state, meta)
        self.restore = actual_functions("inference_IMAGGarment-1.py", {"restore_bf_conditioner_for_inference"})[
            "restore_bf_conditioner_for_inference"]

    def test_joint_checkpoint_rebuilds_missing_bf_and_restores_film(self):
        model = small_conditioner()
        with torch.no_grad():
            model.film.mlp[-1].bias.fill_(0.125)
        self.restore(self.pipe, model.state_dict(), model.film_config(), SimpleNamespace())
        restored = self.pipe.bf_texture_conditioner
        self.assertTrue(restored.film_enabled)
        self.assertEqual(restored.film_config(), model.film_config())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[key], value.half()), key)

    def test_explicit_off_only_disables_film_and_e5_replacement_removes_it(self):
        model = small_conditioner()
        self.restore(self.pipe, model.state_dict(), model.film_config(),
                     SimpleNamespace(disable_texture_film=True))
        self.assertIsNotNone(self.pipe.bf_texture_conditioner.film)
        self.assertFalse(self.pipe.bf_texture_conditioner.film_enabled)
        self.assertTrue(self.pipe.use_tcpm_lite)
        base = small_conditioner(0)
        self.restore(self.pipe, base.state_dict(), {}, SimpleNamespace())
        self.assertIsNone(self.pipe.bf_texture_conditioner.film)
        self.assertFalse(self.pipe.bf_texture_conditioner.film_enabled)

    def test_adapter_only_film_respects_inference_switch(self):
        self.pipe.bf_texture_conditioner = small_conditioner()
        self.restore(self.pipe, None, {}, SimpleNamespace(disable_texture_film=True))
        self.assertFalse(self.pipe.bf_texture_conditioner.film_enabled)
        self.restore(self.pipe, None, {}, SimpleNamespace())
        self.assertTrue(self.pipe.bf_texture_conditioner.film_enabled)

    def test_texture_adapter_entrypoint_preserves_film_when_image_proj_is_also_present(self):
        model = small_conditioner()
        processor = torch.nn.Linear(2, 2)
        self.pipe.unet.attn_processors = {"test": processor}
        self.pipe.texture_ckpt = "fixture.bin"
        self.pipe.use_texture_gate = False
        self.pipe._setup_layer_groups = lambda: None
        state = {
            "image_proj": {"unused": torch.zeros(1)},
            "bf_texture_conditioner": model.state_dict(), "meta": model.film_config(),
            "texture_adapter": torch.nn.ModuleList([processor]).state_dict(),
        }
        with patch("torch.load", return_value=state):
            self.methods["load_texture_adapter"](self.pipe)
        self.assertIsNotNone(self.pipe.bf_texture_conditioner.film)
        del state["bf_texture_conditioner"]
        with patch("torch.load", return_value=state), self.assertRaises(ValueError):
            self.methods["load_texture_adapter"](self.pipe)

    def test_incomplete_film_or_frozen_weights_fail(self):
        model = small_conditioner()
        for missing_key in ("film.mlp.3.bias", "stage2.0.bias"):
            state = copy.deepcopy(model.state_dict())
            del state[missing_key]
            with self.assertRaises((ValueError, RuntimeError)):
                self.restore(self.pipe, state, model.film_config(), SimpleNamespace())
        with self.assertRaises(ValueError):
            self.restore(self.pipe, None, model.film_config(), SimpleNamespace())

    def test_precoded_embeddings_require_matching_masks_before_expansion(self):
        pipe = SimpleNamespace(text_encoder=SimpleNamespace(dtype=torch.float32))
        positive, negative = torch.randn(1, 5, 32), torch.randn(1, 5, 32)
        encode = self.methods["encode_prompt"]
        args = (pipe, None, torch.device("cpu"), 2, True)
        kwargs = dict(prompt_embeds=positive, negative_prompt_embeds=negative, return_text_masks=True)
        with self.assertRaises(ValueError):
            encode(*args, **kwargs)
        with self.assertRaises(ValueError):
            encode(*args, **kwargs, prompt_text_mask=torch.ones(1, 4), negative_text_mask=torch.zeros(1, 5))
        with self.assertRaises(ValueError):
            encode(*args, **kwargs, prompt_text_mask=torch.ones(1, 5), negative_text_mask=torch.zeros(2, 5))
        result = encode(*args, **kwargs, prompt_text_mask=torch.ones(1, 5), negative_text_mask=torch.zeros(1, 5))
        self.assertEqual(result[2].shape, (2, 5))
        self.assertEqual(result[2].dtype, torch.bool)
        self.assertTrue(result[2].all())
        self.assertFalse(result[3].any())

    def test_cfg_routes_negative_film_text_and_disables_unused_negative_branch(self):
        calls = []
        def conditioner(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 16, 32), []
        pipe = SimpleNamespace(
            device="cpu", bf_texture_conditioner=conditioner, use_tcpm_lite=False,
            clip_image_processor=lambda **kwargs: SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8)),
            image_encoder=lambda *args, **kwargs: SimpleNamespace(
                image_embeds=torch.zeros(1, 24), hidden_states=[torch.zeros(1, 5, 24)]),
            cond_image_processor=SimpleNamespace(preprocess=lambda *args, **kwargs: torch.zeros(1, 3, 8, 8)),
            _apply_aa_tcr_fuse=lambda texture, text, caption: texture,
        )
        pos, neg = torch.randn(1, 5, 32), torch.randn(1, 5, 32)
        pos_mask, neg_mask = torch.ones(1, 5).bool(), torch.zeros(1, 5).bool()
        kwargs = dict(pil_image=Image.new("RGB", (8, 8)), width=8, height=8,
                      text_embeds=pos, text_mask=pos_mask)
        self.methods["get_image_embeds"](pipe, **kwargs, negative_text_embeds=neg, negative_text_mask=neg_mask)
        self.assertIs(calls[0]["text_mask"], pos_mask)
        self.assertIs(calls[1]["text_embeds"], neg)
        self.assertIs(calls[1]["text_mask"], neg_mask)
        self.assertTrue(calls[1]["apply_film"])
        calls.clear()
        self.methods["get_image_embeds"](pipe, **kwargs)
        self.assertFalse(calls[1]["apply_film"])

    def test_film_alone_requests_masks_and_old_conditioner_does_not(self):
        tree = ast.parse((ROOT / "pipelines/IMAGGarment_pipeline.py").read_text(encoding="utf-8"))
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "use_texture_text_masks"
                                  for target in node.targets))
        expression = compile(ast.Expression(assignment.value), "texture_mask_condition", "eval")
        for film, enabled, expected in ((object(), True, True), (object(), False, False), (None, True, False)):
            pipe = SimpleNamespace(bf_texture_conditioner=SimpleNamespace(film=film, film_enabled=enabled))
            self.assertEqual(eval(expression, {"self": pipe}), expected)
        self.assertFalse(eval(expression, {"self": SimpleNamespace(bf_texture_conditioner=SimpleNamespace())}))


class FiLMBenchmarkTests(unittest.TestCase):
    def test_fixed_benchmark_forwards_and_records_the_switch(self):
        commands = []
        namespace = actual_functions(
            "tools/run_fixed_benchmark.py",
            {"build_argparser", "experiment_to_flags", "mode_to_flags", "generation_seed_for_sample", "run_one_inference"},
            extra={
                "_existing_generation_candidates": lambda *args: ({
                    "sample_out": "output/sample", "generated": "output/generated.png", "comparison": "output/grid.png"}, []),
                "ensure_dir": lambda path: None,
                "existing_file": lambda path: path in {"input/sketch.png", "input/texture.png"},
                "subprocess": SimpleNamespace(run=lambda cmd, check: commands.append(cmd)),
            },
        )
        cli = ["--dataset_json", "data.json", "--data_root", "input", "--gam_ckpt", "film.pt",
               "--texture_ckpt", "base.pt", "--run_name", "film_on", "--use_tcpm_lite", "1"]
        for disabled in (False, True):
            args = namespace["build_argparser"]().parse_args(cli + (["--disable_texture_film"] if disabled else []))
            flags = namespace["experiment_to_flags"]("film_on", args)
            self.assertEqual(flags["disable_texture_film"], disabled)
            namespace["run_one_inference"](
                args, {"sample_id": "000001", "prompt": "blue plaid"}, "token", "output",
                {"sketch_path": "input/sketch.png", "texture_path": "input/texture.png"},
            )
            self.assertEqual("--disable_texture_film" in commands[-1], disabled)
            self.assertEqual(commands[-1][commands[-1].index("--use_tcpm_lite") + 1], "1")


if __name__ == "__main__":
    unittest.main()
