"""Tests for cumulative confusion counts and exact score-based AUC."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import ClassificationMetrics


def binary_probabilities(positive):
    positive = np.asarray(positive)
    return np.column_stack((1 - positive, positive))


class TestClassificationMetrics(unittest.TestCase):
    def test_t40_accuracy_and_ratios_use_global_counts_in_fixed_label_order(self):
        metrics = ClassificationMetrics([1, 0])
        metrics.update([1], [1]).update([1, 1, 1], [0, 0, 0])
        result = metrics.result()
        self.assertEqual(result["n_samples"], 4)
        self.assertEqual(result["accuracy"], 0.25)
        assert_array_equal(result["labels"], [1, 0])
        assert_array_equal(result["confusion_matrix"], [[1, 3], [0, 0]])
        assert_allclose(result["precision"], [1, 0])
        assert_allclose(result["recall"], [0.25, 0])
        assert_allclose(result["f1"], [0.4, 0])

    def test_t41_rolling_confusion_window_splits_batches(self):
        metrics = ClassificationMetrics([0, 1], window_size=3)
        metrics.update([0, 1], [0, 1]).update([1, 0], [0, 0])
        cumulative, rolling = metrics.result(), metrics.result(rolling=True)
        assert_array_equal(cumulative["confusion_matrix"], [[2, 0], [1, 1]])
        assert_array_equal(rolling["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertEqual(cumulative["accuracy"], 0.75)
        self.assertAlmostEqual(rolling["accuracy"], 2 / 3)
        assert_allclose(rolling["precision"], [0.5, 1])
        assert_allclose(rolling["recall"], [1, 0.5])
        assert_allclose(rolling["f1"], [2 / 3, 2 / 3])

    def test_t42_default_zero_division_and_reset(self):
        metrics = ClassificationMetrics([0, 1], window_size=2)
        metrics.update([0], [0], y_proba=[[1, 0]])
        result = metrics.result()
        for name in ("precision", "recall", "f1"):
            self.assertEqual(result[name][1], 0)
        metrics.reset()
        for rolling in (False, True):
            result = metrics.result(rolling=rolling)
            self.assertEqual((result["n_samples"], result["n_scored"]), (0, 0))
            assert_array_equal(result["confusion_matrix"], np.zeros((2, 2)))
            self.assertTrue(np.isnan(result["accuracy"]))
            self.assertTrue(np.isnan(result["auc"]))

    def test_t43_auc_uses_cross_batch_pairs_and_counts_ties_as_half(self):
        metrics = ClassificationMetrics([0, 1], pos_label=1)
        scores = binary_probabilities([0.5, 0.5])
        metrics.update([1, 0], [0, 0], y_proba=scores)
        scores[:] = [0, 1]
        metrics.update([1, 0], [1, 1], y_proba=binary_probabilities([0.9, 0.8]))
        result = metrics.result()
        self.assertEqual(result["n_scored"], 4)
        self.assertAlmostEqual(result["auc"], 2.5 / 4)

    def test_t44_rolling_auc_becomes_undefined_when_one_class_remains(self):
        metrics = ClassificationMetrics([0, 1], pos_label=1, window_size=2)
        metrics.update([0, 1], [0, 1], y_proba=binary_probabilities([0.1, 0.9]))
        self.assertEqual(metrics.result(rolling=True)["auc"], 1)
        metrics.update([1, 1], [0, 1], y_proba=binary_probabilities([0.7, 0.8]))
        self.assertEqual(metrics.result()["auc"], 1)
        rolling = metrics.result(rolling=True)
        self.assertTrue(np.isnan(rolling["auc"]))
        self.assertEqual(rolling["accuracy"], 0.5)

    def test_t45_multiclass_auc_excludes_unscored_rows_in_both_domains(self):
        metrics = ClassificationMetrics([0, 1, 2], window_size=3)
        metrics.update([0, 1, 2], [0, 1, 2], y_proba=np.eye(3))
        metrics.update([0], [1])
        result, rolling = metrics.result(), metrics.result(rolling=True)
        self.assertEqual((result["n_samples"], result["n_scored"]), (4, 3))
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["auc"], 1)
        assert_array_equal(result["auc_per_class"], [1, 1, 1])
        self.assertEqual((rolling["n_samples"], rolling["n_scored"]), (3, 2))
        self.assertEqual(rolling["auc"], 1)
        assert_array_equal(rolling["auc_per_class"], [np.nan, 1, 1])

    def test_t46_invalid_label_or_probability_preserves_both_domains(self):
        metrics = ClassificationMetrics([0, 1], pos_label=1, window_size=2)
        metrics.update([0, 1], [0, 1], y_proba=[[1, 0], [0, 1]])
        for y, probs, message in (
            ([7], [[1, 0]], "unknown label"),
            ([0], [[np.inf, 0]], "y_proba"),
            ([0], [[0.3, 0.3]], "sum to one"),
            ([0], [[1]], "shape"),
            ([0], [[1.1, -0.1]], "probabilities"),
        ):
            with self.subTest(y=y, probs=probs), self.assertRaisesRegex(ValueError, message):
                metrics.update(y, [0], y_proba=probs)
            for rolling in (True, False):
                result = metrics.result(rolling=rolling)
                assert_array_equal(result["confusion_matrix"], np.eye(2))
                self.assertEqual((result["n_scored"], result["auc"]), (2, 1))
