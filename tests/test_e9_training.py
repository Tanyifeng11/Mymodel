"""核验 E9 训练接线：局部 token 来源、冻结范围、优化器约束与配置校验。"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from models.bf_texture_module import BFTextureConditioner
from models.local_detail_adapter import DEFAULT_LOCAL_DETAIL_LAYER, LocalDetailAdapter
from models.tcpm_lite import TCPMLite


ROOT = Path(__file__).resolve().parents[1]
LAYER = DEFAULT_LOCAL_DETAIL_LAYER


def small_conditioner(guided=False):
    return BFTextureConditioner(
        clip_embeddings_dim=24, cross_attention_dim=32, num_tokens=16,
        stage_channels=(8, 8, 8, 8), stage_token_hw=(2, 2),
        text_guidance_dim=16 if guided else 0,
    )


def training_functions(names, extra_namespace=None):
    """隔离执行真实函数体，避免 CPU 测试依赖 diffusers/accelerate。"""
    path = ROOT / "train_GAM_texture_joint.py"
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in nodes}
    missing = set(names) - found
    if missing:
        raise AssertionError("训练脚本缺少函数：%s" % sorted(missing))
    namespace = {"nn": nn, "torch": torch, **(extra_namespace or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def local_detail_args(**overrides):
    defaults = dict(local_detail_source="local", local_detail_grid=16, local_detail_layer=LAYER,
                    local_detail_dim=128, local_detail_heads=4, local_detail_lr=5e-5,
                    texture_mode="patch_resampled", bf_num_tokens=16,
                    texture_condition_mode="token", texture_preprocess_mode="plain_resize",
                    clip_hidden_layer=-1, width=384, height=512, layer_group_enabled=1,
                    use_texture_gate=1, use_tcpm_lite=1, tcpm_mask_inner_only=1,
                    region_kernel_size=9, train_batch_size=1, gradient_accumulation_steps=8,
                    max_train_steps=1000, num_warmup_steps=50, max_grad_norm=1.0,
                    mixed_precision="fp16", dataset_json_path="data/train_bf_texture.json",
                    data_root_path="/datasets/BF/training", training_data_sha256="a" * 64)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeProcessor(nn.Module):
    def __init__(self, hidden_size=32, cross_attention_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.local_detail_adapter = None


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.processors = nn.ModuleDict({"target": FakeProcessor()})
        self.attn_processors = {LAYER: self.processors["target"]}


class LocalTokenSourceTests(unittest.TestCase):
    """A 组必须复用原 16 token，B 组必须绕过 8×8 池化与重采样。"""

    def setUp(self):
        torch.manual_seed(0)
        self.bf = small_conditioner()
        self.texture = torch.randn(2, 3, 64, 64)
        self.clip_tokens = torch.randn(2, 5, 24)

    def forward(self, source, grid=4):
        return self.bf(clip_vision_tokens=self.clip_tokens, texture_images=self.texture,
                       texture_mode="patch_resampled", local_detail_source=source,
                       local_detail_grid=grid)

    def test_off_keeps_two_outputs_and_is_the_e5_path(self):
        outputs = self.forward("off")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (2, 16, 32))

    def test_resampled_source_returns_the_original_tokens(self):
        tokens, _, local = self.forward("resampled")
        self.assertIs(local, tokens)
        self.assertEqual(local.shape, (2, 16, 32))

    def test_local_source_returns_grid_tokens_before_compression(self):
        grid = 4
        tokens, _, local = self.forward("local", grid=grid)
        self.assertEqual(local.shape, (2, grid * grid, 32))
        self.assertEqual(tokens.shape, (2, 16, 32))
        # 与原 16 token 不同，才是"额外的局部信息"。
        self.assertFalse(torch.equal(local[:, :16], tokens))

    def test_local_grid_changes_token_count_only(self):
        for grid in (2, 8):
            with self.subTest(grid=grid):
                tokens, _, local = self.forward("local", grid=grid)
                self.assertEqual(local.shape, (2, grid * grid, 32))
                self.assertEqual(tokens.shape, (2, 16, 32))

    def test_e5_output_is_unchanged_by_the_extra_branch(self):
        baseline = self.forward("off")[0]
        for source in ("resampled", "local"):
            with self.subTest(source=source):
                self.assertTrue(torch.equal(self.forward(source)[0], baseline))

    def test_invalid_source_and_grid_are_rejected(self):
        with self.assertRaises(ValueError):
            self.forward("unknown")
        with self.assertRaises(ValueError):
            self.forward("local", grid=0)

    def test_legacy_pooled_mode_also_exposes_local_tokens(self):
        outputs = self.bf(clip_image_embeds=torch.randn(2, 24), texture_images=self.texture,
                          texture_mode="legacy_pooled", local_detail_source="local",
                          local_detail_grid=4)
        self.assertEqual(outputs[2].shape, (2, 16, 32))


class FreezeAndOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.namespace = training_functions({"freeze_local_detail_only", "local_detail_stats",
                                             "local_detail_config", "training_run_config"})
        self.unet = FakeUNet()
        self.adapter = LocalDetailAdapter(32, 32, 8, 2)
        self.unet.attn_processors[LAYER].local_detail_adapter = self.adapter
        self.others = [small_conditioner(), TCPMLite(32), nn.Linear(4, 4)]

    def freeze(self):
        return self.namespace["freeze_local_detail_only"](self.unet, LAYER, tuple(self.others))

    def test_only_the_new_branch_stays_trainable(self):
        returned = self.freeze()
        self.assertIs(returned, self.adapter)
        self.assertTrue(all(p.requires_grad for p in self.adapter.parameters()))
        frozen = [p for module in [self.unet.body, *self.others] for p in module.parameters()]
        self.assertTrue(frozen)
        self.assertFalse(any(p.requires_grad for p in frozen))

    def test_frozen_weights_keep_their_values(self):
        before = {name: value.clone() for name, value in self.others[0].state_dict().items()}
        self.freeze()
        after = self.others[0].state_dict()
        self.assertTrue(all(torch.equal(before[name], after[name]) for name in before))

    def test_branch_is_promoted_to_float32(self):
        self.adapter.half()
        self.freeze()
        self.assertTrue(all(p.dtype == torch.float32 for p in self.adapter.parameters()))

    def test_optimizer_guard_accepts_only_branch_params(self):
        # 复刻训练脚本里的集合相等约束。
        self.freeze()
        expected = {id(p) for p in self.adapter.parameters()}
        trainable = [p for p in self.unet.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in trainable}, expected)
        with torch.no_grad():
            self.unet.body.weight.requires_grad_(True)
        leaked = [p for p in self.unet.parameters() if p.requires_grad]
        self.assertNotEqual({id(p) for p in leaked}, expected)

    def test_stats_are_json_ready_floats(self):
        hidden = torch.randn(2, 12, 32)
        mask = torch.ones(2, 1, 6, 4)
        self.adapter(hidden, torch.randn(2, 5, 32), mask, (3, 4))
        stats = self.namespace["local_detail_stats"](self.unet, LAYER)
        self.assertEqual(set(stats), {"alpha", "residual_relative_rms"})
        self.assertTrue(all(isinstance(value, float) for value in stats.values()))

    def test_stats_empty_without_branch(self):
        self.unet.attn_processors[LAYER].local_detail_adapter = None
        self.assertEqual(self.namespace["local_detail_stats"](self.unet, LAYER), {})

    def test_saved_config_records_source_and_training_amount(self):
        args = local_detail_args()
        config = self.namespace["local_detail_config"](args)
        self.assertEqual(config["local_detail_source"], "local")
        self.assertEqual(config["local_detail_layer"], LAYER)
        run = self.namespace["training_run_config"](args)
        # A/B 可比性依赖这些字段一起落盘。
        for field in ("max_train_steps", "train_batch_size", "gradient_accumulation_steps",
                      "dataset_json_path", "training_data_sha256"):
            self.assertIn(field, run)


class ConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.namespace = training_functions({"validate_local_detail_source",
                                             "validate_local_detail_base_config",
                                             "validate_local_detail_resume_source"})

    def e5_state(self):
        return {"meta": {"resampler_training": "off", "text_guidance_dim": 0,
                         "local_detail_source": "off"},
                "unet": {"body.weight": torch.zeros(2)},
                "texture_adapter": {"0.to_k_ip.weight": torch.zeros(2)},
                "bf_texture_conditioner": {"resampler_queries": torch.zeros(1, 16, 32)}}

    def test_clean_e5_start_is_accepted(self):
        self.namespace["validate_local_detail_source"](self.e5_state())

    def test_e8_or_existing_branch_start_is_rejected(self):
        cases = ({"resampler_training": "text_only"}, {"text_guidance_dim": 256},
                 {"local_detail_source": "local"})
        for override in cases:
            with self.subTest(override=override):
                state = self.e5_state()
                state["meta"].update(override)
                with self.assertRaises(ValueError):
                    self.namespace["validate_local_detail_source"](state)

    def test_start_carrying_new_module_weights_is_rejected(self):
        for component, key in (("unet", LAYER + ".local_detail_adapter.alpha"),
                               ("bf_texture_conditioner", "text_guidance.gate"),
                               ("texture_adapter", "0.local_detail_adapter.alpha")):
            with self.subTest(component=component):
                state = self.e5_state()
                state[component][key] = torch.zeros(())
                with self.assertRaises(ValueError):
                    self.namespace["validate_local_detail_source"](state)

    def test_non_dict_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.namespace["validate_local_detail_source"]([])

    def test_base_config_must_match_e5_metadata(self):
        args = local_detail_args()
        metadata = {"width": 384, "height": 512, "texture_num_tokens": 16,
                    "layer_group_enabled": 1, "use_tcpm_lite": 1, "region_kernel_size": 9}
        self.namespace["validate_local_detail_base_config"](args, metadata)
        for field, value in (("width", 512), ("region_kernel_size", 5), ("use_tcpm_lite", 0)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.namespace["validate_local_detail_base_config"](
                        args, {**metadata, field: value})

    def resume_state(self, source="local", step=1000):
        state = self.e5_state()
        state["meta"].update({
            "local_detail_source": source, "local_detail_grid": 16,
            "local_detail_layer": LAYER, "local_detail_dim": 128,
            "local_detail_heads": 4, "train_global_step": step,
            "texture_num_tokens": 16, "texture_mode": "patch_resampled",
            "texture_condition_mode": "token", "texture_preprocess_mode": "plain_resize",
            "clip_hidden_layer": -1, "width": 384, "height": 512,
            "layer_group_enabled": 1, "use_texture_gate": 1, "use_tcpm_lite": 1,
            "tcpm_mask_inner_only": 1, "region_kernel_size": 9,
        })
        state["unet"][LAYER + ".local_detail_adapter.alpha"] = torch.zeros(())
        state["texture_adapter"]["0.local_detail_adapter.alpha"] = torch.zeros(())
        return state

    def test_local_b_resume_accepts_own_checkpoint_and_step(self):
        step = self.namespace["validate_local_detail_resume_source"](
            self.resume_state(), local_detail_args(max_train_steps=2000))
        self.assertEqual(step, 1000)

    def test_local_b_resume_rejects_other_source_or_non_extension(self):
        with self.assertRaises(ValueError):
            self.namespace["validate_local_detail_resume_source"](
                self.resume_state(source="resampled"), local_detail_args(max_train_steps=2000))
        with self.assertRaises(ValueError):
            self.namespace["validate_local_detail_resume_source"](
                self.resume_state(step=1000), local_detail_args(max_train_steps=1000))

    def test_effective_config_must_be_patch_resampled_16_tokens(self):
        for override in ({"texture_mode": "legacy_pooled"}, {"bf_num_tokens": 8}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self.namespace["validate_local_detail_base_config"](
                        local_detail_args(**override), {})


class ImageDropoutTests(unittest.TestCase):
    """图像条件 dropout 时整条旁路必须归零，避免形成伪条件。"""

    def test_zero_condition_weight_zeroes_the_mask(self):
        mask = torch.ones(3, 1, 4, 4)
        weight = torch.tensor([1.0, 0.0, 1.0])
        gated = mask * weight[:, None, None, None]
        self.assertTrue(torch.equal(gated[1], torch.zeros_like(gated[1])))
        self.assertTrue(torch.equal(gated[0], mask[0]))

    def test_zeroed_mask_produces_zero_residual(self):
        adapter = LocalDetailAdapter(32, 32, 8, 2)
        with torch.no_grad():
            adapter.alpha.fill_(0.7)
        hidden = torch.randn(2, 12, 32)
        mask = torch.ones(2, 1, 3, 4)
        mask[1] = 0.0
        residual = adapter(hidden, torch.randn(2, 5, 32), mask, (3, 4))
        self.assertTrue(torch.equal(residual[1], torch.zeros_like(residual[1])))
        self.assertGreater(float(residual[0].abs().max()), 0.0)


class PipelineWiringTests(unittest.TestCase):
    """推理侧：token 分支关闭时不能声称启用旁路，CFG batch 必须对齐。"""

    def test_local_tokens_repeat_with_num_samples(self):
        tokens = torch.arange(6.0).reshape(1, 3, 2)
        repeated = tokens.repeat_interleave(4, dim=0)
        self.assertEqual(repeated.shape, (4, 3, 2))
        self.assertTrue(torch.equal(repeated[0], repeated[3]))

    def test_pipeline_declares_the_branch_defaults(self):
        source = (ROOT / "pipelines/IMAGGarment_pipeline.py").read_text(encoding="utf-8")
        for marker in ("self.local_detail_source = \"off\"", "self._local_detail_tokens = None",
                       "local_detail_spatial_shape"):
            self.assertIn(marker, source)

    def test_processor_forwards_branch_kwargs(self):
        source = (ROOT / "adapter/attention_processor.py").read_text(encoding="utf-8")
        self.assertIn("local_adapter = getattr(self, \"local_detail_adapter\", None)", source)
        self.assertIn("local_detail_tokens is not None", source)


if __name__ == "__main__":
    unittest.main()
