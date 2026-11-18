"""Behavior and failure protection for the buffered decision tree."""

import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import DecisionTreeClassifier
from numcompute_stream.tree import _candidate_gains


def column(values):
    return np.asarray(values)[:, None]


class TestDecisionTree(unittest.TestCase):
    def test_t04_rejects_wrong_input_shapes(self):
        tree = DecisionTreeClassifier()
        X, y = column([0, 1, 2]), [0, 1, 0]
        for features, labels, message in (
            (X[:, 0], y, "X.*2 dimensions"),
            (X, column(y), "y.*1-D"),
            (X, y[:2], "same number of samples"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                tree.fit(features, labels)
        self.assertFalse(tree.is_fitted_)

    def test_t05_rejects_invalid_types_and_missing_feature_columns(self):
        tree = DecisionTreeClassifier()
        with self.assertRaisesRegex(TypeError, "X.*real numeric"):
            tree.fit(column([1j, 2j]), [0, 1])
        with self.assertRaisesRegex(ValueError, "X.*at least one feature"):
            tree.fit(np.empty((2, 0)), [0, 1])
        with self.assertRaisesRegex(TypeError, "buffer_size"):
            DecisionTreeClassifier(buffer_size=True)

    def test_t06_empty_updates_and_predictions(self):
        tree = DecisionTreeClassifier(buffer_size=2)
        self.assertIs(tree.partial_fit(np.empty((0, 9)), []), tree)
        self.assertEqual(tree.pending_samples_, 0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tree.fit(np.empty((0, 2)), [])
        tree.partial_fit([[0, 0], [1, 1]], [0, 1])
        self.assertEqual(tree.predict(np.empty((0, 2))).shape, (0,))
        self.assertEqual(tree.predict_proba(np.empty((0, 2))).shape, (0, 2))
        tree.partial_fit(np.empty((0, 2)), [])
        self.assertEqual(tree.retrain_count_, 1)
        with self.assertRaisesRegex(ValueError, "X.*2 features"):
            tree.partial_fit(np.empty((0, 3)), [])

    def test_t07_finds_perfect_gini_split_and_uses_observed_boundary(self):
        X, y = column([0, 1, 2, 3]), [0, 0, 1, 1]
        tree = DecisionTreeClassifier(max_depth=1)
        self.assertIs(tree.fit(X, y), tree)
        self.assertEqual(tree._root.threshold, 1)
        assert_array_equal(tree.predict(X), y)
        assert_array_equal(tree.predict_proba([[1], [1.5]]), [[1, 0], [0, 1]])

    def test_t08_leaf_frequency_columns_follow_sorted_classes(self):
        tree = DecisionTreeClassifier(max_depth=0).fit(column([0, 1, 2]), ["z", "a", "z"])
        assert_array_equal(tree.classes_, ["a", "z"])
        assert_allclose(tree.predict_proba([[10], [-10]]), [[1 / 3, 2 / 3]] * 2)
        assert_array_equal(tree.predict([[10], [-10]]), ["z", "z"])

    def test_t09_constant_feature_and_leaf_vote_tie(self):
        tree = DecisionTreeClassifier().fit(column([7, 7, 7, 7]), [3, 1, 3, 1])
        self.assertIsNone(tree._root.feature)
        assert_array_equal(tree.classes_, [1, 3])
        assert_array_equal(tree.predict_proba([[7]]), [[0.5, 0.5]])
        assert_array_equal(tree.predict([[7]]), [1])

    def test_t10_feature_and_threshold_ties_prefer_smallest(self):
        tree = DecisionTreeClassifier(max_depth=1).fit([[0, 0], [1, 1], [2, 2]], [0, 1, 0])
        self.assertEqual((tree._root.feature, tree._root.threshold), (0, 0))
        assert_array_equal(tree.predict_proba([[0, 2], [2, 0]]), [[1, 0], [0.5, 0.5]])

    def test_t11_duplicate_values_with_no_positive_gain(self):
        tree = DecisionTreeClassifier().fit(column([0, 0, 1, 1]), [0, 1, 0, 1])
        self.assertIsNone(tree._root.feature)
        assert_array_equal(tree.predict_proba([[0], [1]]), [[0.5, 0.5]] * 2)

    def test_t12_insufficient_rows_for_min_samples_split(self):
        tree = DecisionTreeClassifier(min_samples_split=5).fit(column([0, 1, 2, 3]), [0, 0, 1, 1])
        self.assertIsNone(tree._root.feature)
        assert_array_equal(tree.predict_proba([[0], [3]]), [[0.5, 0.5]] * 2)

    def test_t13_single_class_produces_one_probability_column(self):
        X = column([0, 1, 2])
        tree = DecisionTreeClassifier().fit(X, ["yes"] * 3)
        self.assertIsNone(tree._root.feature)
        assert_array_equal(tree.classes_, ["yes"])
        assert_array_equal(tree.predict_proba(X), np.ones((3, 1)))
        assert_array_equal(tree.predict(X), ["yes"] * 3)

    def test_t14_extreme_finite_values_avoid_threshold_overflow(self):
        largest = np.finfo(np.float64).max
        X = column([-largest, largest])
        with np.errstate(over="raise", invalid="raise"):
            tree = DecisionTreeClassifier().fit(X, [0, 1])
            self.assertEqual(tree._root.threshold, -largest)
            assert_array_equal(tree.predict(X), [0, 1])

    def test_t15_sampling_is_per_node_without_replacement_and_repeatable(self):
        X = np.repeat(column(np.arange(6)), 3, axis=1)
        y = [0, 0, 1, 1, 0, 0]
        tree = DecisionTreeClassifier(max_depth=2, max_features=0.5, random_state=7)
        fake_rng = Mock()
        fake_rng.choice.return_value = np.array([2, 0])
        with patch("numcompute_stream.tree.np.random.default_rng", return_value=fake_rng) as rng:
            tree.fit(X, y)
        rng.assert_called_once_with(7)
        self.assertEqual(fake_rng.choice.call_args_list, [call(3, size=2, replace=False)] * 2)

        other = DecisionTreeClassifier(max_depth=2, max_features=0.5, random_state=7)
        tree.fit(X, y)
        other.fit(X, y)
        for a, b in ((tree._root, other._root), (tree._root.right, other._root.right)):
            self.assertEqual((a.feature, a.threshold), (b.feature, b.threshold))
        assert_array_equal(tree.predict_proba(X), other.predict_proba(X))
        assert_array_equal(tree.predict(X), y)

    def test_t16_max_features_validation_and_all_features(self):
        X, y = [[0, 0], [0, 1], [0, 2], [0, 3]], [0, 0, 1, 1]
        for value in (3, 0.0, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_features"):
                DecisionTreeClassifier(max_features=value).fit(X, y)
        all_features = DecisionTreeClassifier().fit(X, y)
        all_fraction = DecisionTreeClassifier(max_features=1.0).fit(X, y)
        self.assertEqual(all_features._root.feature, 1)
        self.assertEqual(all_fraction._root.feature, 1)
        assert_array_equal(all_features.predict(X), all_fraction.predict(X))

    def test_t17_cold_start_and_exact_buffer_threshold(self):
        tree = DecisionTreeClassifier(buffer_size=4)
        self.assertIs(tree.partial_fit([[0], [1]], [0, 0]), tree)
        self.assertEqual(tree.pending_samples_, 2)
        self.assertEqual(tree.retrain_count_, 0)
        for predict in (tree.predict, tree.predict_proba):
            with self.assertRaisesRegex(RuntimeError, "not fitted"):
                predict([[0]])
        tree.partial_fit([[2], [3]], [1, 1])
        self.assertTrue(tree.is_fitted_)
        self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (1, 4, 0))
        assert_array_equal(tree.predict(column([0, 3])), [0, 1])

    def test_t18_overshoot_trains_on_entire_incoming_chunk(self):
        tree = DecisionTreeClassifier(buffer_size=3, max_depth=0)
        tree.partial_fit(column([0, 1]), [0, 0])
        tree.partial_fit(column([2, 3, 4]), [1, 1, 1])
        self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (1, 5, 0))
        assert_allclose(tree.predict_proba([[0]]), [[0.4, 0.6]])
        assert_array_equal(tree.predict([[0]]), [1])

    def test_t19_keeps_published_tree_until_later_chunk_fills_tail(self):
        tree = DecisionTreeClassifier(buffer_size=3)
        tree.partial_fit(column([0, 1, 2]), [0, 0, 0])
        tree.partial_fit(column([3, 4]), [1, 1])
        self.assertEqual(tree.pending_samples_, 2)
        self.assertEqual(tree.retrain_count_, 1)
        assert_array_equal(tree.classes_, [0])
        assert_array_equal(tree.predict([[4]]), [0])
        tree.partial_fit([[5]], [1])
        self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (2, 3, 0))
        assert_array_equal(tree.classes_, [1])
        assert_array_equal(tree.predict([[4]]), [1])

    def test_t20_explicit_fit_resets_stream_even_below_threshold(self):
        tree = DecisionTreeClassifier(buffer_size=2)
        tree.partial_fit(column([0, 1]), [0, 0])
        tree.partial_fit(column([2, 3]), [1, 1])
        tree.partial_fit([[4]], [1])
        self.assertEqual(tree.retrain_count_, 2)
        self.assertEqual(tree.pending_samples_, 1)
        tree.fit([[8, 9]], ["new"])
        self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (1, 1, 0))
        assert_array_equal(tree.classes_, ["new"])
        assert_array_equal(tree.predict([[8, 9]]), ["new"])

    def test_t21_rejects_nonfinite_values_without_state_changes(self):
        tree = DecisionTreeClassifier(buffer_size=2).fit(column([0, 1]), [0, 1])
        tree.partial_fit([[2]], [0])
        for X, y, name in (
            ([[np.nan]], [0], "X"), ([[np.inf]], [0], "X"),
            ([[1]], [np.nan], "y"), ([[1]], [np.inf], "y"),
        ):
            with self.subTest(X=X, y=y), self.assertRaisesRegex(ValueError, name):
                tree.partial_fit(X, y)
            self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (1, 2, 1))
            assert_array_equal(tree.classes_, [0, 1])
            assert_array_equal(tree.predict([[0], [1]]), [0, 1])

    def test_t22_pending_rows_and_labels_are_copied(self):
        X_backing = column([0.0, 100.0, 1.0, 100.0])
        y_backing = np.array([0, 9, 0, 9])
        X, y = X_backing[::2], y_backing[::2]
        tree = DecisionTreeClassifier(buffer_size=3)
        tree.partial_fit(X, y)
        assert_array_equal(X_backing, column([0, 100, 1, 100]))
        assert_array_equal(y_backing, [0, 9, 0, 9])
        X_backing[:] = 999
        y_backing[:] = 8
        tree.partial_fit([[2]], [1])
        assert_array_equal(tree.classes_, [0, 1])
        assert_array_equal(tree.predict(column([0, 1, 2])), [0, 0, 1])

    def test_a01_build_failure_preserves_buffer_and_model_for_retry(self):
        tree = DecisionTreeClassifier(buffer_size=3).fit(column([0, 1]), [0, 1])
        tree.partial_fit([[4]], [2])
        with patch.object(tree, "_build_tree", side_effect=RuntimeError("injected failure")):
            for train in (tree.fit, tree.partial_fit):
                with self.subTest(train=train), self.assertRaisesRegex(RuntimeError, "injected"):
                    train(column([5, 6]), [2, 2])
                self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (1, 2, 1))
                assert_array_equal(tree.classes_, [0, 1])
                assert_array_equal(tree.predict(column([0, 1])), [0, 1])
        tree.partial_fit(column([5, 6]), [2, 2])
        self.assertEqual((tree.retrain_count_, tree.last_train_samples_, tree.pending_samples_), (2, 3, 0))
        assert_array_equal(tree.classes_, [2])

    def test_a02_pending_stream_checks_width_and_preserves_integer_labels(self):
        tree = DecisionTreeClassifier(buffer_size=2)
        first, second = 2**63, 2**63 + 1
        tree.partial_fit([[0]], np.array([first], dtype=np.uint64))
        with self.assertRaisesRegex(ValueError, "X.*1 features"):
            tree.partial_fit([[1, 2]], np.array([second], dtype=np.uint64))
        with self.assertRaisesRegex(TypeError, "y dtype kind"):
            tree.partial_fit([[1]], np.array([1], dtype=np.int64))
        self.assertEqual(tree.pending_samples_, 1)
        tree.partial_fit([[1]], np.array([second], dtype=np.uint64))
        exposed_classes = tree.classes_
        self.assertEqual(exposed_classes.dtype, np.uint64)
        exposed_classes[0] = 0
        assert_array_equal(tree.predict([[0], [1]]), np.array([first, second], dtype=np.uint64))

    def test_a03_vectorized_multiclass_gains_match_independent_counting(self):
        values = [3, 0, 1, 1, 2, 3, 4, 2]
        labels = [2, 0, 0, 1, 2, 1, 2, 1]

        def gini(group):
            return 1 - sum((group.count(label) / len(group)) ** 2 for label in set(group))

        expected_thresholds = sorted(set(values))[:-1]
        expected_gains = []
        for threshold in expected_thresholds:
            left = [y for x, y in zip(values, labels) if x <= threshold]
            right = [y for x, y in zip(values, labels) if x > threshold]
            expected_gains.append(
                gini(labels) - (len(left) * gini(left) + len(right) * gini(right)) / len(labels)
            )
        thresholds, gains = _candidate_gains(np.array(values), np.array(labels), np.bincount(labels), gini(labels))
        assert_array_equal(thresholds, expected_thresholds)
        assert_allclose(gains, expected_gains, atol=1e-14, rtol=0)
