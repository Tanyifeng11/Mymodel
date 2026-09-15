"""执行真实训练辅助函数，核验冻结、数据错配、严格加载与保存恢复。"""

import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import shutil
import unittest
import uuid

import torch

from models.local_detail_adapter import DEFAULT_LOCAL_DETAIL_LAYER
from models.text_guided_queries import guidance_config_from_checkpoint, text_content_mask
from models.text_texture_film import film_config_from_checkpoint
from models.nexus_texture_adapter import nexus_config_from_checkpoint
from test_text_guided_resampler import small_joint_models, joint_state


ROOT = Path(__file__).resolve().parents[1]


def training_namespace():
    path = ROOT / "train_GAM_texture_joint.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "make_film_text_indices", "film_condition_ids", "film_training_config",
        "nexus_training_config", "validate_nexus_training_args", "validate_nexus_source", "freeze_nexus_only",
        "validate_film_training_args", "validate_film_source", "freeze_film_only",
        "load_partial_state", "load_joint_checkpoint_into_models",
        "_is_palette_key", "_is_balanced_gate_key", "local_detail_config",
        "training_run_config", "local_detail_stats", "save_training_checkpoint",
        "_opt_int", "_opt_float",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "nn": torch.nn, "random": random, "os": os,
                 "film_config_from_checkpoint": film_config_from_checkpoint,
        "nexus_config_from_checkpoint": nexus_config_from_checkpoint,
                 "guidance_config_from_checkpoint": guidance_config_from_checkpoint,
                 "text_content_mask": text_content_mask, "argparse": argparse,
                 "DEFAULT_LOCAL_DETAIL_LAYER": DEFAULT_LOCAL_DETAIL_LAYER}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    # 使用真实 argparse 定义，验证默认值和启动参数，不导入缺失的扩散模型依赖。
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    parser_nodes = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
        parser_nodes.append(node)
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def film_args(namespace, **overrides):
    args = namespace["ap"].parse_args([
        "--pretrained_model_name_or_path", "sd", "--pretrained_vae_model_path", "vae",
        "--dataset_json_path", "training.json", "--data_root_path", "BF",
        "--texture_adapter_ckpt", "texture.pt", "--gam_init_ckpt", "e5.pt",
        "--film_hidden_dim", "16", "--use_tcpm_lite", "1", "--use_texture_gate", "1",
        "--layer_group_enabled", "1", "--texture_condition_mode", "token",
        "--texture_mode", "patch_resampled", "--val_vis_steps", "0",
        "--vis_every_n_steps", "0", "--max_train_steps", "100", "--seed", "42",
    ])
    args.training_data_sha256 = "sample-hash"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class FilmTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = training_namespace()

    def test_shuffled_is_reproducible_derangement_and_preserves_rng(self):
        before = random.getstate()
        mapping = self.ns["make_film_text_indices"](11, "shuffled", 42)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(sorted(mapping), list(range(11)))
        self.assertTrue(all(i != j for i, j in enumerate(mapping)))
        self.assertEqual(mapping, self.ns["make_film_text_indices"](11, "shuffled", 42))
        self.assertIsNone(self.ns["make_film_text_indices"](1, "matched", 42))
        with self.assertRaises(ValueError):
            self.ns["make_film_text_indices"](1, "shuffled", 42)

    def test_text_dropout_is_shared_without_mutating_main_text(self):
        correct = torch.tensor([[0, 3, 2, 2], [0, 2, 2, 2]])
        shuffled = torch.tensor([[0, 8, 2, 2], [0, 9, 2, 2]])
        before, null = correct.clone(), torch.tensor([0, 2, 2, 2])
        result = self.ns["film_condition_ids"](correct, shuffled, 2, null)
        self.assertTrue(torch.equal(result[0], shuffled[0]))
        self.assertTrue(torch.equal(result[1], null))
        self.assertTrue(torch.equal(correct, before))
        self.assertEqual(shuffled[1, 1].item(), 9)
        self.assertIs(self.ns["film_condition_ids"](correct, None, 2, null), correct)

    def test_cli_requires_isolated_e5_configuration(self):
        args = film_args(self.ns)
        self.ns["validate_film_training_args"](args)
        for changes in ({"use_tcpm_lite": 0}, {"film_hidden_dim": -1},
                        {"resampler_training": "text_only"}, {"local_detail_source": "local"},
                        {"ctd_prob": 0.5}, {"reload_texture_adapter_after_gam_init": True},
                        {"film_hidden_dim": 0, "film_text_mode": "shuffled"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.ns["validate_film_training_args"](film_args(self.ns, **changes))

    def test_strict_load_allows_only_new_film_keys(self):
        reference = joint_state(small_joint_models())
        candidate = small_joint_models()
        candidate["bf"].configure_film(16)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["load_joint_checkpoint_into_models"](reference, **candidate, strict_base=True)
        for name, value in reference["bf_texture_conditioner"].items():
            self.assertTrue(torch.equal(value, candidate["bf"].state_dict()[name]))
        for component, key in (("bf_texture_conditioner", "resampler_queries"),
                               ("tcpm_lite", "residual_scale"), ("texture_adapter", "0.weight")):
            broken = copy.deepcopy(reference)
            del broken[component][key]
            with self.subTest(component=component), contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, RuntimeError)):
                self.ns["load_joint_checkpoint_into_models"](broken, **candidate, strict_base=True)

    def test_freeze_does_not_change_existing_weights(self):
        models = small_joint_models()
        models["bf"].configure_film(16)
        before = joint_state(models)
        self.ns["freeze_film_only"](models["bf"], [m for key, m in models.items() if key != "bf"])
        for key, module in models.items():
            for name, parameter in module.named_parameters():
                self.assertEqual(parameter.requires_grad, key == "bf" and name.startswith("film."))
        after = joint_state(models)
        for key in before:
            if key != "meta":
                self.assertTrue(all(torch.equal(v, after[key][k]) for k, v in before[key].items()))

    def test_source_guards_configuration_and_resume_mapping(self):
        args = film_args(self.ns)
        reference = joint_state(small_joint_models())
        self.ns["validate_film_source"](reference, args)
        wrong_gate = copy.deepcopy(reference)
        wrong_gate["meta"]["gate_min"] = args.gate_min + 0.2
        with self.assertRaisesRegex(ValueError, "gate_min"):
            self.ns["validate_film_source"](wrong_gate, args)
        models = small_joint_models()
        models["bf"].configure_film(16)
        saved = joint_state(models)
        saved["meta"].update(self.ns["film_training_config"](args), train_global_step=25,
                             training_data_sha256=args.training_data_sha256)
        self.assertEqual(self.ns["validate_film_source"](saved, args, resume=True), 25)
        with self.assertRaises(ValueError):
            self.ns["validate_film_source"](saved, args)
        for changes in ({"film_text_mode": "shuffled"}, {"max_train_samples": 20},
                        {"training_data_sha256": "changed"}, {"film_hidden_dim": 32}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.ns["validate_film_source"](saved, film_args(self.ns, **changes), resume=True)

    def test_real_checkpoint_save_and_strict_restore(self):
        args = film_args(self.ns)
        models = small_joint_models()
        models["bf"].configure_film(16)
        with torch.no_grad():
            models["bf"].film.mlp[-1].bias.fill_(0.2)
        output = ROOT / "tmp" / ("film-save-test-" + uuid.uuid4().hex)
        output.mkdir(parents=True)
        try:
            self.ns["save_training_checkpoint"](
                accelerator=None, **models, output_dir=str(output), global_step=25,
                args=args, resolved_image_encoder_path="clip",
            )
            saved = torch.load(output / "checkpoint-25/joint_model.pt", weights_only=False)
            self.assertEqual(saved["meta"]["film_hidden_dim"], 16)
            self.assertEqual(saved["meta"]["film_text_mode"], "matched")
            restored = small_joint_models()
            restored["bf"].configure_film(16)
            with contextlib.redirect_stdout(io.StringIO()):
                self.ns["load_joint_checkpoint_into_models"](saved, **restored, strict_base=True)
            for name, value in models["bf"].state_dict().items():
                self.assertTrue(torch.equal(value, restored["bf"].state_dict()[name]))
            incomplete = copy.deepcopy(saved)
            del incomplete["bf_texture_conditioner"]["film.mlp.3.bias"]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                self.ns["load_joint_checkpoint_into_models"](incomplete, **restored, strict_base=True)
        finally:
            # 路径由本测试在仓库 tmp 下直接创建，删除前验证范围。
            self.assertEqual(output.resolve().parent, (ROOT / "tmp").resolve())
            shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
