"""核验 E9 推理装配：来源只读 checkpoint、权重必须完整、显式关闭回到原 E5 路径。"""

import ast
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from checkpoint_utils import extract_texture_metadata
from models.local_detail_adapter import (
    DEFAULT_LOCAL_DETAIL_LAYER, LocalDetailAdapter, attach_local_detail_adapter,
)


ROOT = Path(__file__).resolve().parents[1]
LAYER = DEFAULT_LOCAL_DETAIL_LAYER


def inference_function(name):
    """隔离执行真实装配函数，避免 CPU 测试依赖 diffusers。"""
    path = ROOT / "inference_IMAGGarment-1.py"
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if not nodes:
        raise AssertionError("推理脚本缺少函数：%s" % name)
    namespace = {
        "torch": torch, "nn": nn, "extract_texture_metadata": extract_texture_metadata,
        "DEFAULT_LOCAL_DETAIL_LAYER": DEFAULT_LOCAL_DETAIL_LAYER,
        "attach_local_detail_adapter": attach_local_detail_adapter,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FakeProcessor(nn.Module):
    def __init__(self, hidden_size=32, cross_attention_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.local_detail_adapter = None


class FakeUNet:
    def __init__(self):
        # 顺序决定 texture_adapter 的处理器索引前缀。
        self.attn_processors = {"other.processor": FakeProcessor(), LAYER: FakeProcessor()}


def branch_state(inner_dim=8, num_heads=2, hidden=32, context=32):
    torch.manual_seed(0)
    return LocalDetailAdapter(hidden, context, inner_dim, num_heads).state_dict()


def checkpoint(source="local", inner_dim=8, num_heads=2, include=("unet", "texture_adapter"),
               grid=16, layer=LAYER, output_constraint="off", highpass_kernel=3):
    state = {"meta": {"local_detail_source": source, "local_detail_grid": grid,
                      "local_detail_layer": layer, "local_detail_dim": inner_dim,
                      "local_detail_heads": num_heads, "region_kernel_size": 9,
                      "local_detail_output_constraint": output_constraint,
                      "local_detail_highpass_kernel": highpass_kernel},
             "unet": {}, "texture_adapter": {}}
    if source != "off":
        weights = branch_state(inner_dim, num_heads)
        prefixes = {"unet": layer + ".local_detail_adapter.",
                    "texture_adapter": "1.local_detail_adapter."}
        for component in include:
            state[component].update({prefixes[component] + key: value.clone()
                                     for key, value in weights.items()})
    return state


class ConfigureLocalDetailTests(unittest.TestCase):
    def setUp(self):
        self.configure = inference_function("configure_local_detail_from_checkpoint")
        self.unet = FakeUNet()

    def test_metadata_drives_the_source_and_loads_weights(self):
        state = checkpoint("local")
        config = self.configure(self.unet, state, -1)
        self.assertEqual(config["source"], "local")
        self.assertEqual(config["grid"], 16)
        self.assertEqual(config["region_kernel_size"], 9)
        module = self.unet.attn_processors[LAYER].local_detail_adapter
        self.assertIsInstance(module, LocalDetailAdapter)
        self.assertFalse(module.training)
        expected = branch_state(8, 2)
        actual = module.state_dict()
        self.assertTrue(all(torch.equal(expected[key], actual[key]) for key in expected))

    def test_resampled_source_is_accepted(self):
        config = self.configure(self.unet, checkpoint("resampled"), -1)
        self.assertEqual(config["source"], "resampled")

    def test_highpass_constraint_is_loaded_from_checkpoint(self):
        config = self.configure(self.unet, checkpoint("local", output_constraint="highpass"), -1)
        self.assertEqual(config["output_constraint"], "highpass")
        self.assertEqual(self.unet.attn_processors[LAYER].local_detail_adapter.output_constraint, "highpass")

    def test_explicit_off_returns_e5_path_without_attaching(self):
        config = self.configure(self.unet, checkpoint("local"), 0)
        self.assertEqual(config["source"], "off")
        self.assertIsNone(self.unet.attn_processors[LAYER].local_detail_adapter)

    def test_e5_checkpoint_stays_off(self):
        config = self.configure(self.unet, checkpoint("off"), -1)
        self.assertEqual(config["source"], "off")
        self.assertIsNone(self.unet.attn_processors[LAYER].local_detail_adapter)

    def test_requesting_branch_on_e5_checkpoint_fails(self):
        with self.assertRaises(ValueError):
            self.configure(self.unet, checkpoint("off"), 1)

    def test_stray_weights_without_metadata_fail(self):
        state = checkpoint("local")
        state["meta"]["local_detail_source"] = "off"
        with self.assertRaises(ValueError):
            self.configure(self.unet, state, -1)

    def test_declared_source_without_weights_fails(self):
        state = checkpoint("local", include=())
        with self.assertRaises(ValueError):
            self.configure(self.unet, state, -1)

    def test_unknown_source_and_bad_grid_fail(self):
        state = checkpoint("local")
        state["meta"]["local_detail_source"] = "sideways"
        with self.assertRaises(ValueError):
            self.configure(self.unet, state, -1)
        state = checkpoint("local", grid=0)
        with self.assertRaises(ValueError):
            self.configure(FakeUNet(), state, -1)

    def test_dimension_mismatch_with_attached_module_fails(self):
        self.configure(self.unet, checkpoint("local", inner_dim=8, num_heads=2), -1)
        with self.assertRaises(ValueError):
            self.configure(self.unet, checkpoint("local", inner_dim=16, num_heads=2), -1)

    def test_weights_on_unexpected_layer_fail(self):
        state = checkpoint("local")
        state["unet"] = {"down_blocks.0.attn2.processor.local_detail_adapter.alpha": torch.zeros(())}
        with self.assertRaises(ValueError):
            self.configure(self.unet, state, -1)

    def test_single_component_export_still_loads(self):
        config = self.configure(self.unet, checkpoint("local", include=("unet",)), -1)
        self.assertEqual(config["source"], "local")
        self.assertIsNotNone(self.unet.attn_processors[LAYER].local_detail_adapter)

    def test_layer_from_metadata_is_respected(self):
        unet = FakeUNet()
        state = checkpoint("local", layer="other.processor")
        state["texture_adapter"] = {"0.local_detail_adapter." + key: value
                                    for key, value in branch_state(8, 2).items()}
        self.configure(unet, state, -1)
        self.assertIsNotNone(unet.attn_processors["other.processor"].local_detail_adapter)
        self.assertIsNone(unet.attn_processors[LAYER].local_detail_adapter)


class InferenceContractTests(unittest.TestCase):
    def test_cli_flag_offers_default_off_and_on(self):
        source = (ROOT / "inference_IMAGGarment-1.py").read_text(encoding="utf-8")
        self.assertIn("--use_local_detail_adapter", source)
        self.assertIn("choices=[-1, 0, 1]", source)

    def test_token_branch_is_required_for_the_bypass(self):
        source = (ROOT / "inference_IMAGGarment-1.py").read_text(encoding="utf-8")
        self.assertIn("E9 局部旁路要求 BF 纹理分支及 token/hybrid 条件模式", source)

    def test_benchmark_records_the_flag_in_run_config(self):
        source = (ROOT / "tools/run_fixed_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("use_local_detail_adapter: {int(experiment_flags['use_local_detail_adapter'])}",
                      source)
        self.assertIn("\"--use_local_detail_adapter\"", source)


if __name__ == "__main__":
    unittest.main()
