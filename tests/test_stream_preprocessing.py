"""Tests of incremental transforms and snapshot independence."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import (
    DecisionTreeClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    TabularPreprocessor,
)


class TestIncrementalPreprocessing(unittest.TestCase):
    def test_t23_scaler_ignores_nan_and_uses_unit_scale_for_constant_column(self):
        scaler = StandardScaler()
        self.assertIs(scaler.partial_fit([[0, 4], [2, 4]]), scaler)
        scaler.partial_fit([[4, np.nan]])
        assert_array_equal(scaler.count_, [3, 2])
        assert_array_equal(scaler.mean_, [2, 4])
        assert_allclose(scaler.scale_, [np.sqrt(8 / 3), 1])

        X = np.array([[4, 4], [2, np.nan]])
        assert_allclose(scaler.transform(X), [[2 / np.sqrt(8 / 3), 0], [0, np.nan]])
        assert_array_equal(X, [[4, 4], [2, np.nan]])
        assert_array_equal(scaler.count_, [3, 2])

    def test_t24_imputer_uses_fallback_until_it_observes_real_values(self):
        imputer = SimpleImputer(fill_value=-5)
        imputer.partial_fit([[np.nan], [np.nan]])
        assert_array_equal(imputer.transform([[np.nan]]), [[-5]])
        assert_array_equal(imputer.count_, [0])
        imputer.partial_fit([[2], [4]])
        assert_array_equal(imputer.statistics_, [3])
        assert_array_equal(imputer.transform([[np.nan], [9]]), [[3], [9]])
        assert_array_equal(imputer.count_, [2])

    def test_t25_scaler_keeps_small_variance_at_large_offset(self):
        offset = 1e12
        scaler = StandardScaler().partial_fit([[offset + 1], [offset + 2]])
        scaler.partial_fit([[offset + 3]])
        assert_array_equal(scaler.mean_, [offset + 2])
        assert_allclose(scaler.var_, [2 / 3])
        assert_array_equal(scaler.transform([[offset + 2]]), [[0]])

    def test_t26_unrepresentable_moments_leave_scaler_usable(self):
        scaler = StandardScaler().fit([[0]])
        largest = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "running moments"):
            scaler.partial_fit([[largest], [-largest]])
        assert_array_equal(scaler.mean_, [0])
        assert_array_equal(scaler.count_, [1])
        assert_array_equal(scaler.scale_, [1])
        assert_array_equal(scaler.transform([[3]]), [[3]])

    def test_t27_new_categories_append_across_columns(self):
        encoder = OneHotEncoder()
        encoder.partial_fit([[2, 8]])
        encoder.partial_fit([[1, 7]])
        self.assertEqual(encoder.n_features_out_, 4)
        assert_array_equal(encoder.transform([[2, 8], [1, 7]]), [[1, 1, 0, 0], [0, 0, 1, 1]])
        assert_array_equal(encoder.categories_[0], [2, 1])
        assert_array_equal(encoder.categories_[1], [8, 7])

    def test_t28_unknown_category_policy_does_not_learn_at_transform(self):
        ignore = OneHotEncoder(handle_unknown="ignore").fit([[2, 8]])
        error = OneHotEncoder(handle_unknown="error").fit([[2, 8]])
        assert_array_equal(ignore.transform([[99, 8]]), [[0, 1]])
        with self.assertRaisesRegex(ValueError, "unknown categories"):
            error.transform([[99, 8]])
        for encoder in (ignore, error):
            self.assertEqual(encoder.n_features_out_, 2)
            assert_array_equal(encoder.transform([[2, 8]]), [[1, 1]])

    def test_t29_invalid_category_updates_leave_existing_mapping_intact(self):
        encoder = OneHotEncoder().fit([[2, 8]])
        for value, error in ((np.nan, ValueError), (np.inf, ValueError), ("other", TypeError)):
            with self.subTest(value=value), self.assertRaisesRegex(error, "categor"):
                encoder.partial_fit([[value, value]])
        assert_array_equal(encoder.transform([[2, 8]]), [[1, 1]])
        self.assertEqual(encoder.n_features_out_, 2)

    def test_t30_fit_resets_all_transformer_state(self):
        scaler = StandardScaler().partial_fit([[0]]).partial_fit([[2]])
        imputer = SimpleImputer().partial_fit([[0]]).partial_fit([[2]])
        encoder = OneHotEncoder().partial_fit([["old"]]).partial_fit([["much longer"]])
        assert_array_equal(encoder.transform([["old"], ["much longer"]]), np.eye(2))
        self.assertIs(scaler.fit([[10]]), scaler)
        self.assertIs(imputer.fit([[10]]), imputer)
        self.assertIs(encoder.fit([["new"]]), encoder)
        assert_array_equal(scaler.mean_, [10])
        assert_array_equal(scaler.count_, [1])
        assert_array_equal(imputer.statistics_, [10])
        assert_array_equal(imputer.count_, [1])
        assert_array_equal(encoder.transform([["old"], ["new"]]), [[0], [1]])

    def test_b01_scaler_snapshot_is_reusable_with_a_tree(self):
        scaler = StandardScaler()
        scaler.partial_fit(np.empty((0, 3)))
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            scaler.snapshot()
        training = [[0], [2]]
        scaler.partial_fit(training)
        frozen = scaler.snapshot()
        tree = DecisionTreeClassifier().fit(frozen.transform(training), [0, 1])

        scaler.partial_fit([[4]])
        assert_array_equal(frozen.transform([[5]]), [[4]])
        assert_allclose(scaler.transform([[5]]), [[3 / np.sqrt(8 / 3)]])
        assert_array_equal(tree.predict(frozen.transform(training)), [0, 1])
        self.assertEqual(frozen.transform(np.empty((0, 1))).shape, (0, 1))

    def test_b02_tabular_snapshot_freezes_numeric_and_categorical_coordinates(self):
        prep = TabularPreprocessor([0], [1, 2])
        prep.partial_fit([[0, 2, 8], [2, 2, 8], [np.nan, 2, 8]])
        frozen = prep.snapshot()
        probe = [[5, 1, 8], [np.nan, 2, 7]]
        expected_old = [[4, 0, 1], [0, 1, 0]]
        assert_array_equal(frozen.transform(probe), expected_old)

        prep.partial_fit([[10, 1, 7], [np.nan, 2, 8]])
        expected_new = [[1 / np.sqrt(56 / 3), 0, 1, 1, 0], [0, 1, 0, 0, 1]]
        assert_allclose(prep.transform(probe), expected_new)
        assert_array_equal(frozen.transform(probe), expected_old)
        prep.fit([[20, 5, 9]])
        assert_array_equal(prep.transform([[np.nan, 5, 9]]), [[0, 1, 1]])
        assert_array_equal(frozen.transform(probe), expected_old)

    def test_b03_failed_mixed_update_cannot_publish_numeric_only_state(self):
        prep = TabularPreprocessor([0], [1]).fit([[0, 2], [2, 2]])
        for learn in (prep.partial_fit, prep.fit):
            with self.subTest(learn=learn), self.assertRaisesRegex(ValueError, "categories"):
                learn([[100, np.nan]])
            assert_array_equal(prep.transform([[5, 2]]), [[4, 1]])
        prep.partial_fit([[4, 1]])
        assert_allclose(prep.transform([[5, 2]]), [[3 / np.sqrt(8 / 3), 1, 0]])

    def test_b04_tabular_supports_each_single_branch_and_unobserved_columns(self):
        numeric = TabularPreprocessor([1, 0], [], fill_value=-3).fit([[2, np.nan], [2, np.nan]])
        assert_array_equal(numeric.transform([[2, np.nan], [4, 5]]), [[-3, 0], [5, 2]])
        categorical = TabularPreprocessor([], [1]).fit([[500, 9], [600, 4]])
        assert_array_equal(categorical.transform([[np.nan, 9], [5, 4]]), [[0, 1], [1, 0]])
        assert_array_equal(categorical.transform([[5, 7]]), [[0, 0]])
