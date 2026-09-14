"""CPU 检查 FiLM 的 E5 兼容性、条件隔离、冻结梯度和 checkpoint。"""

import copy
import io
import unittest

import torch

from models.bf_texture_module import BFTextureConditioner
from models.tcpm_lite import TCPMLite
from models.text_guided_queries import text_content_mask
from models.text_texture_film import TextTextureFiLM, film_config_from_checkpoint


def small_conditioner(film=False):
    return BFTextureConditioner(
        clip_embeddings_dim=24, cross_attention_dim=32, num_tokens=16,
        stage_channels=(8, 8, 8, 8), stage_token_hw=(2, 2),
        film_hidden_dim=16 if film else 0,
    )


class TextTextureFiLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(41)

    def inputs(self):
        return {
            "texture_images": torch.randn(2, 3, 32, 32),
            "clip_vision_tokens": torch.randn(2, 9, 24),
            "text_embeds": torch.randn(2, 7, 32),
            "text_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]).bool(),
        }

    def activate(self, module):
        with torch.no_grad():
            module.mlp[-1].weight.normal_(std=0.1)
            module.mlp[-1].bias.fill_(0.03)

    def test_default_parameter_count_and_zero_init(self):
        module = TextTextureFiLM()
        self.assertEqual(sum(p.numel() for p in module.parameters()), 132992)
        self.assertEqual(module.mlp[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(module.mlp[-1].bias.count_nonzero().item(), 0)
        self.assertFalse(any(name == "gate" for name, _ in module.named_parameters()))

    def test_old_keys_and_zero_init_match_all_bf_paths(self):
        base, candidate = small_conditioner(), small_conditioner(True)
        old_keys = set(base.state_dict())
        self.assertFalse(any(key.startswith("film.") for key in old_keys))
        self.assertEqual({key for key in candidate.state_dict() if not key.startswith("film.")}, old_keys)
        self.assertIn("stage3.0.weight", old_keys)
        self.assertIn("stage3.1.weight", old_keys)
        missing, unexpected = candidate.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(missing and all(key.startswith("film.") for key in missing))
        self.assertFalse(unexpected)
        data = self.inputs()
        for mode in ("patch_resampled", "legacy_pooled"):
            for source in ("off", "resampled", "local"):
                with self.subTest(mode=mode, source=source):
                    kwargs = dict(texture_mode=mode, local_detail_source=source, local_detail_grid=3)
                    expected, actual = base(**data, **kwargs), candidate(**data, **kwargs)
                    self.assertEqual(actual[0].shape, (2, 16, 32))
                    self.assertEqual(expected[1], actual[1])
                    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
                    if source != "off":
                        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_explicit_disable_restores_e5_after_learning(self):
        base, candidate = small_conditioner(), small_conditioner(True)
        candidate.load_state_dict(base.state_dict(), strict=False)
        self.activate(candidate.film)
        data = self.inputs()
        for mode in ("patch_resampled", "legacy_pooled"):
            expected = base(**data, texture_mode=mode)[0]
            self.assertGreater((candidate(**data, texture_mode=mode)[0] - expected).abs().sum().item(), 0)
            disabled = candidate(**data, texture_mode=mode, apply_film=False)[0]
            torch.testing.assert_close(expected, disabled, rtol=0, atol=0)
        candidate.film_enabled = False
        torch.testing.assert_close(base(**data)[0], candidate(**data)[0], rtol=0, atol=0)
        # 关闭 FiLM 时，无需准备额外文本条件。
        data.pop("text_embeds")
        data.pop("text_mask")
        torch.testing.assert_close(base(**data)[0], candidate(**data)[0], rtol=0, atol=0)

    def test_film_is_before_stage3_activation_and_changes_local_tokens(self):
        candidate = small_conditioner(True)
        self.activate(candidate.film)
        data = self.inputs()
        f1 = candidate.stage1(data["texture_images"])
        f2 = candidate.stage2(f1)
        before_activation = candidate.stage3[1](candidate.stage3[0](f2))
        expected_f3 = candidate.stage3[2](candidate.film(before_activation, data["text_embeds"], data["text_mask"]))
        features = candidate._encode_texture_features(
            data["texture_images"], data["text_embeds"], data["text_mask"])
        torch.testing.assert_close(features[2], expected_f3, rtol=0, atol=0)
        for mode in ("patch_resampled", "legacy_pooled"):
            expected = candidate(**data, texture_mode=mode, local_detail_source="local", local_detail_grid=3)
            changed = dict(data, text_embeds=data["text_embeds"].flip(0))
            actual = candidate(**changed, texture_mode=mode, local_detail_source="local", local_detail_grid=3)
            self.assertGreater((actual[0] - expected[0]).abs().sum().item(), 0)
            self.assertGreater((actual[2] - expected[2]).abs().sum().item(), 0)

    def test_padding_does_not_contribute_and_empty_text_is_identity(self):
        module = TextTextureFiLM(32, 8, 16)
        self.activate(module)
        features, text = torch.randn(2, 8, 4, 4), torch.randn(2, 5, 32)
        mask = text_content_mask(torch.tensor([[0, 2, 2, 2, 2], [0, 9, 2, 2, 2]]), 2)
        self.assertEqual(mask.sum(dim=1).tolist(), [0, 1])
        result = module(features, text, mask)
        torch.testing.assert_close(result[0], features[0], rtol=0, atol=0)
        self.assertGreater((result[1] - features[1]).abs().sum().item(), 0)
        changed = text.clone()
        changed[~mask] = float("nan")
        torch.testing.assert_close(module(features, changed, mask), result, rtol=0, atol=0)
        self.assertEqual(module.last_stats["text_present_frac"], 0.5)

    def test_masked_mean_ignores_content_order(self):
        module = TextTextureFiLM(32, 8, 16)
        self.activate(module)
        features, text = torch.randn(2, 8, 4, 4), torch.randn(2, 5, 32)
        mask = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0]]).bool()
        result = module(features, text, mask)
        order = torch.tensor([4, 2, 0, 3, 1])
        torch.testing.assert_close(module(features, text[:, order], mask[:, order]), result, rtol=0, atol=0)

    def test_gradients_pass_through_frozen_cnn_resampler_tcpm_and_downstream(self):
        candidate = small_conditioner(True)
        candidate.train_film_only()
        tcpm = TCPMLite(32, residual_scale_init=0.2).requires_grad_(False)
        downstream = torch.nn.Sequential(torch.nn.Linear(32, 8), torch.nn.SiLU(),
                                         torch.nn.Linear(8, 3)).requires_grad_(False)
        modules = {"bf": candidate, "tcpm": tcpm, "downstream": downstream}
        before = {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}
        trainable = [name for name, parameter in candidate.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable and all(name.startswith("film.") for name in trainable))
        optimizer = torch.optim.AdamW(candidate.film.parameters(), lr=1e-2)
        data, target = self.inputs(), torch.randn(2, 3)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = downstream(tcpm(candidate(**data)[0], data["text_embeds"]).mean(dim=1))
            torch.nn.functional.mse_loss(prediction, target).backward()
            for name, parameter in candidate.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(candidate.film.mlp[-1].weight.grad.abs().sum().item(), 0)
            if step > 0:
                self.assertGreater(candidate.film.mlp[1].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        changed = []
        for module_name, module in modules.items():
            for name, value in module.state_dict().items():
                if module_name == "bf" and name.startswith("film."):
                    if not torch.equal(value, before[module_name][name]):
                        changed.append(name)
                else:
                    torch.testing.assert_close(value, before[module_name][name], rtol=0, atol=0)
            if module_name != "bf":
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertIn("film.mlp.1.weight", changed)
        self.assertIn("film.mlp.3.weight", changed)

    def test_empty_half_and_autocast_have_finite_gradients(self):
        for autocast in (False, True):
            with self.subTest(autocast=autocast):
                module = TextTextureFiLM(32, 8, 16)
                self.activate(module)
                features, text = torch.randn(2, 8, 4, 4), torch.zeros(2, 5, 32)
                if not autocast:
                    module, features, text = module.half(), features.half(), text.half()
                with torch.autocast("cpu", dtype=torch.float16, enabled=autocast):
                    result = module(features, text, torch.zeros(2, 5).bool())
                    loss = result.float().square().mean()
                loss.backward()
                torch.testing.assert_close(result, features, rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in module.parameters()))

    def test_checkpoint_roundtrip_and_old_metadata(self):
        base = small_conditioner()
        self.assertEqual(film_config_from_checkpoint(base.state_dict(), {}), {"film_hidden_dim": 0})
        candidate = small_conditioner(True)
        self.activate(candidate.film)
        buffer = io.BytesIO()
        torch.save({"bf": candidate.state_dict(), "meta": candidate.film_config()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored = small_conditioner()
        restored.configure_film(**film_config_from_checkpoint(state["bf"], state["meta"]))
        restored.load_state_dict(state["bf"], strict=True)
        data = self.inputs()
        torch.testing.assert_close(candidate(**data)[0], restored(**data)[0], rtol=0, atol=0)
        restored.configure_film(0)
        self.assertIsNone(restored.film)

    def test_checkpoint_rejects_incomplete_or_inconsistent_film(self):
        candidate = small_conditioner(True)
        state, metadata = candidate.state_dict(), candidate.film_config()
        for problem in ("no_metadata", "wrong_dim", "no_weights", "missing_weight", "wrong_channels", "unknown_weight"):
            with self.subTest(problem=problem):
                broken, meta = copy.deepcopy(state), dict(metadata)
                if problem == "no_metadata":
                    meta = {}
                elif problem == "wrong_dim":
                    meta["film_hidden_dim"] += 1
                elif problem == "no_weights":
                    broken = small_conditioner().state_dict()
                elif problem == "missing_weight":
                    del broken["film.mlp.3.bias"]
                elif problem == "wrong_channels":
                    broken["stage3.1.weight"] = torch.ones(16)
                else:
                    broken["film.gate"] = torch.zeros(())
                with self.assertRaisesRegex(ValueError, "FiLM"):
                    film_config_from_checkpoint(broken, meta)

    def test_validation_for_unconfigured_training_and_missing_text(self):
        with self.assertRaisesRegex(ValueError, "film_only"):
            small_conditioner().train_film_only()
        with self.assertRaisesRegex(ValueError, "FiLM"):
            TextTextureFiLM(hidden_dim=0)
        model = small_conditioner(True)
        data = self.inputs()
        data.pop("text_mask")
        with self.assertRaisesRegex(ValueError, "FiLM"):
            model(**data)


if __name__ == "__main__":
    unittest.main()
