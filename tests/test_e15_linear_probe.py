"""E15 线性探针的离线单测：只覆盖纯 numpy/sklearn 部分，不加载模型。"""

import unittest

import numpy as np

from tools.e15_linear_probe import (gram_blocks, kernel_pca, orientation_angle, axis_label,
                                    classification_scores, prepare_folds, spectral_basis,
                                    ridge_predict)


def stripes(height, width, axis, period=8):
    """生成理想正/负条纹灰度图：vertical 表示图案沿 x 变化。"""
    y, x = np.mgrid[0:height, 0:width]
    coordinate = x if axis == "vertical" else y
    return 0.5 + 0.4 * np.cos(2 * np.pi * coordinate / period)


class OrientationTests(unittest.TestCase):
    def test_vertical_and_horizontal(self):
        vertical = orientation_angle(stripes(128, 128, "vertical"))
        horizontal = orientation_angle(stripes(128, 128, "horizontal"))
        self.assertEqual(axis_label(vertical[0]), "vertical")
        self.assertEqual(axis_label(horizontal[0]), "horizontal")
        self.assertGreater(vertical[1], 1.0)

    def test_flat_image_has_no_axis(self):
        angle, anisotropy = orientation_angle(np.full((64, 64), 0.5))
        self.assertLess(anisotropy, 0.5)
        self.assertIsNotNone(axis_label(angle))

    def test_diagonal_outside_tolerance(self):
        self.assertIsNone(axis_label(45.0))
        self.assertEqual(axis_label(20.0), "vertical")
        self.assertEqual(axis_label(160.0), "vertical")
        self.assertEqual(axis_label(110.0), "horizontal")


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.train = self.rng.normal(size=(40, 12)).astype(np.float64)
        self.test = self.rng.normal(size=(10, 12)).astype(np.float64)
        self.target = self.rng.normal(size=(40, 2))

    def test_ridge_matches_sklearn(self):
        from sklearn.linear_model import Ridge

        mean = self.train.mean(axis=0)
        scale = self.train.std(axis=0)
        standard_train = (self.train - mean) / scale
        standard_test = (self.test - mean) / scale
        alpha = 0.7
        expected = Ridge(alpha=alpha).fit(standard_train, self.target).predict(standard_test)
        k_train, k_test = gram_blocks(self.train, self.test)
        weights, vectors = spectral_basis(k_train)
        got = ridge_predict(weights, vectors, k_test, self.target, alpha)
        self.assertTrue(np.allclose(got, expected, atol=1e-5))

    def test_kernel_pca_matches_sklearn(self):
        from sklearn.decomposition import PCA

        mean = self.train.mean(axis=0)
        scale = self.train.std(axis=0)
        standard_train = (self.train - mean) / scale
        standard_test = (self.test - mean) / scale
        keep = 5
        pca = PCA(n_components=keep).fit(standard_train)
        expected = pca.transform(standard_test)
        k_train, k_test = gram_blocks(self.train, self.test)
        weights, vectors = spectral_basis(k_train)
        _, got = kernel_pca(weights, vectors, k_test, keep)
        self.assertTrue(np.allclose(np.abs(got), np.abs(expected), atol=1e-6))


class ProbeTests(unittest.TestCase):
    def test_separable_signal_beats_permutation(self):
        rng = np.random.default_rng(1)
        cluster = np.repeat(np.eye(2) * 6.0, 30, axis=0)
        matrix = (cluster + rng.normal(scale=0.5, size=(60, 2))).astype(np.float32)
        target = np.array(["a"] * 30 + ["b"] * 30)
        prepared = prepare_folds(matrix, target, 5, 42, stratified=True)
        result = classification_scores(prepared, target, [0.1, 1.0, 10.0], 42, 8)
        self.assertGreater(result["balanced_accuracy"], 0.95)
        shuffled = classification_scores(prepared, np.random.default_rng(2).permutation(target),
                                         [0.1, 1.0, 10.0], 42, 8)
        self.assertLess(shuffled["balanced_accuracy"], 0.85)


if __name__ == "__main__":
    unittest.main()