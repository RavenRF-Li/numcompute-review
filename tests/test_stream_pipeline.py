"""Tests for publication, raw buffering and stage composition."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import DecisionTreeClassifier, OneHotEncoder, Pipeline, StandardScaler, TabularPreprocessor


def numeric_pipeline():
    return Pipeline([
        ("numeric", TabularPreprocessor([0], [])),
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ])


class TestPipeline(unittest.TestCase):
    def test_t31_scaler_keeps_published_coordinate_system_between_builds(self):
        pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=2))])
        pipe.partial_fit([[0], [2]], [0, 1])
        before = pipe.predict_proba([[5]])
        assert_array_equal(pipe.transform([[5]]), [[4]])
        pipe.partial_fit([[10]], [1])
        assert_array_equal(pipe.transform([[5]]), [[4]])
        assert_array_equal(pipe.predict_proba([[5]]), before)
        self.assertEqual((pipe.retrain_count_, pipe.pending_samples_), (1, 1))
        with self.assertRaisesRegex(ValueError, "NaN"):
            pipe.partial_fit([[np.nan]], [0])
        self.assertEqual(pipe.pending_samples_, 1)

    def test_t32_refits_all_raw_rows_under_the_new_snapshot(self):
        pipe = numeric_pipeline()
        pipe.partial_fit([[0], [2]], [0, 1]).partial_fit([[10]], [1])
        captured = []
        original = DecisionTreeClassifier.fit

        def fit(tree, X, y):
            captured.append(X.copy())
            return original(tree, X, y)

        with patch.object(DecisionTreeClassifier, "fit", new=fit):
            pipe.partial_fit([[np.nan]], [0])
        self.assertEqual(len(captured), 1)
        assert_allclose(captured[0], [[6 / np.sqrt(56 / 3)], [0]])
        assert_allclose(pipe.transform([[10], [np.nan]]), captured[0])
        self.assertEqual((pipe.retrain_count_, pipe.last_train_samples_, pipe.pending_samples_), (2, 2, 0))

    def test_t33_new_category_changes_width_only_after_retraining(self):
        pipe = Pipeline([
            ("categories", TabularPreprocessor([], [0])),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        pipe.partial_fit([[2], [2]], [0, 0])
        pipe.partial_fit([[1]], [1])
        assert_array_equal(pipe.transform([[1], [2]]), [[0], [1]])
        assert_array_equal(pipe.predict([[1], [2]]), [0, 0])
        original = DecisionTreeClassifier.fit

        def fit(tree, X, y):
            assert_array_equal(X, [[0, 1], [1, 0]])
            return original(tree, X, y)

        with patch.object(DecisionTreeClassifier, "fit", new=fit):
            pipe.partial_fit([[2]], [0])
        assert_array_equal(pipe.transform([[1], [2]]), [[0, 1], [1, 0]])
        assert_array_equal(pipe.predict([[1], [2]]), [1, 0])

    def test_t34_fit_failure_preserves_buffer_and_learning_state_for_retry(self):
        pipe, control = numeric_pipeline(), numeric_pipeline()
        for model in (pipe, control):
            model.partial_fit([[0], [2]], [0, 1]).partial_fit([[4]], [0])
        timings = pipe.last_timing_
        with patch.object(DecisionTreeClassifier, "fit", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                pipe.partial_fit([[10]], [1])
        self.assertEqual((pipe.retrain_count_, pipe.pending_samples_), (1, 1))
        self.assertEqual(pipe.last_timing_, timings)
        assert_array_equal(pipe.transform([[5]]), [[4]])
        assert_array_equal(pipe.predict([[0], [5]]), [0, 1])
        for model in (pipe, control):
            model.partial_fit([[10]], [1])
        assert_allclose(pipe.transform([[5]]), [[1 / np.sqrt(14)]])
        assert_allclose(pipe.transform([[5]]), control.transform([[5]]))
        assert_array_equal(pipe.predict([[4], [10]]), control.predict([[4], [10]]))

    def test_t35_validates_step_names_combinations_and_rows(self):
        class DropRow:
            def transform(self, X):
                return X[:-1]

        tree = DecisionTreeClassifier(buffer_size=2)
        pipe = Pipeline([("drop", DropRow()), ("tree", tree)])
        with self.assertRaisesRegex(ValueError, "drop.*preserve"):
            pipe.partial_fit([[1]], [0])
        self.assertEqual(pipe.pending_samples_, 0)
        self.assertFalse(pipe.is_fitted_)
        with self.assertRaisesRegex(ValueError, "duplicate name"):
            Pipeline([("step", StandardScaler()), ("step", tree)])
        with self.assertRaisesRegex(ValueError, "only one"):
            Pipeline([("scale", StandardScaler()), ("num", TabularPreprocessor([0], [])), ("tree", tree)])
        with self.assertRaisesRegex(TypeError, "stateless"):
            Pipeline([("categories", OneHotEncoder()), ("tree", tree)])

    def test_t36_fit_restarts_a_mixed_pipeline_below_threshold(self):
        pipe = Pipeline([
            ("mixed", TabularPreprocessor([0], [1])),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        pipe.partial_fit([[0, 2], [2, 2]], [0, 1])
        pipe.partial_fit([[3, 2], [5, 1]], [0, 1])
        pipe.partial_fit([[6, 2]], [0])
        self.assertEqual((pipe.retrain_count_, pipe.pending_samples_), (2, 1))
        self.assertIs(pipe.fit([[10, 9]], [1]), pipe)
        self.assertEqual((pipe.retrain_count_, pipe.last_train_samples_, pipe.pending_samples_), (1, 1, 0))
        assert_array_equal(pipe.transform([[np.nan, 2], [11, 9]]), [[0, 0], [1, 1]])
        assert_array_equal(pipe.classes_, [1])

    def test_c01_tree_only_overshoot_never_calls_tree_partial_fit(self):
        pipe = Pipeline([("tree", DecisionTreeClassifier(buffer_size=3, max_depth=0))])
        with patch.object(DecisionTreeClassifier, "partial_fit", side_effect=AssertionError("duplicate buffer")):
            pipe.partial_fit([[0], [1]], [0, 0])
            with self.assertRaisesRegex(RuntimeError, "not fitted"):
                pipe.predict([[0]])
            pipe.partial_fit([[2], [3], [4], [5]], [1, 1, 1, 1])
            self.assertEqual((pipe.retrain_count_, pipe.last_train_samples_, pipe.pending_samples_), (1, 6, 0))
            assert_allclose(pipe.predict_proba([[0]]), [[1 / 3, 2 / 3]])
            pipe.partial_fit([[6], [7], [8]], [0, 0, 0])
        self.assertEqual((pipe.retrain_count_, pipe.last_train_samples_, pipe.pending_samples_), (2, 3, 0))
        assert_array_equal(pipe.predict([[4]]), [0])

    def test_c02_stateless_steps_wrap_the_learner_in_order(self):
        class Double:
            def transform(self, X):
                X *= 2
                return X

        class AddTen:
            def transform(self, X):
                return X + 10

        scaler, tree = StandardScaler(), DecisionTreeClassifier()
        pipe = Pipeline([("double", Double()), ("scale", scaler), ("offset", AddTen()), ("tree", tree)])
        X = np.array([[0.0], [2.0]])
        pipe.fit(X, [0, 1])
        assert_array_equal(X, [[0], [2]])
        assert_array_equal(pipe.transform(X), [[9], [11]])
        assert_array_equal(pipe.predict(X), [0, 1])
        self.assertIsNone(scaler.n_features_in_)
        self.assertFalse(tree.is_fitted_)
