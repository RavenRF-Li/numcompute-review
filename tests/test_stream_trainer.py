"""Tests for test-before-train evaluation, timing and logging."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from numcompute_stream import DecisionTreeClassifier, Pipeline, StreamTrainer
from numcompute_stream.stream import _array_storage_bytes


def trainer_for_tree(*, labels=(0, 1), pos_label=None, window_size=3, buffer_size=2):
    pipe = Pipeline([("tree", DecisionTreeClassifier(buffer_size=buffer_size, max_depth=1))])
    return StreamTrainer(pipe, labels=labels, pos_label=pos_label, window_size=window_size)


class TestStreamTrainer(unittest.TestCase):
    def test_t47_scores_with_old_model_and_excludes_warmup_from_metrics(self):
        trainer = trainer_for_tree()
        X = [[0], [1]]
        first = trainer.fit_chunk(X, [0, 1])
        second = trainer.fit_chunk(X, [1, 0])
        third = trainer.fit_chunk(X, [1, 0])
        self.assertEqual(first["status"], "warmup")
        self.assertTrue(np.isnan(first["metrics"]["accuracy"]))
        self.assertEqual(first["y_proba"].shape, (0, 2))
        self.assertEqual(first["evaluated_samples"], 0)
        self.assertEqual(second["metrics"]["accuracy"], 0)
        self.assertEqual(third["metrics"]["accuracy"], 1)
        self.assertEqual(third["log"]["cumulative"]["n_samples"], 4)
        self.assertEqual(third["log"]["cumulative"]["accuracy"], 0.5)
        self.assertAlmostEqual(third["log"]["rolling"]["accuracy"], 2 / 3)
        self.assertEqual([row["evaluated_version"] for row in trainer.logs], [None, 1, 2])
        self.assertEqual([row["retrain_count"] for row in trainer.logs], [1, 2, 3])
        # The old model ranked the second chunk backwards, the newer one ranked
        # the third correctly. Two pairs tie, one wins, one loses.
        self.assertAlmostEqual(third["log"]["cumulative"]["auc"], 0.5)

    def test_t48_maps_sparse_model_class_columns_into_configured_order(self):
        trainer = trainer_for_tree(labels=[1, 0], pos_label=1)
        trainer.fit_chunk([[0], [1]], [0, 0])
        result = trainer.fit_chunk([[0], [1]], [1, 0])
        assert_array_equal(result["y_pred"], [0, 0])
        assert_array_equal(result["y_proba"], [[0, 1], [0, 1]])
        assert_array_equal(result["metrics"]["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertAlmostEqual(result["metrics"]["auc"], 0.5)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.5)

    def test_t49_score_is_pure_and_unknown_label_prevents_learning(self):
        trainer = trainer_for_tree()
        X = [[0], [1]]
        trainer.fit_chunk(X, [0, 1])
        first = trainer.score_chunk(X, [0, 1])
        second = trainer.score_chunk(X, [0, 1])
        assert_array_equal(first["y_pred"], second["y_pred"])
        assert_array_equal(first["y_proba"], second["y_proba"])
        self.assertEqual(first["metrics"]["accuracy"], 1)
        with self.assertRaisesRegex(ValueError, "unknown label"):
            trainer.fit_chunk([[0]], [7])
        self.assertEqual((trainer.pipeline.retrain_count_, trainer.pipeline.pending_samples_), (1, 0))
        self.assertEqual(len(trainer.logs), 1)
        self.assertEqual(trainer.metrics.result()["n_samples"], 0)
        self.assertEqual(trainer.fit_chunk([[0]], [0])["log"]["cumulative"]["n_samples"], 1)

    def test_t50_byte_estimate_counts_shared_storage_and_reports_timing_scopes(self):
        original = np.arange(8, dtype=np.float64)
        distinct = np.ones(3, dtype=np.int64)
        strided = np.lib.stride_tricks.as_strided(original, shape=(2,))
        views = [original[1::2], original[::-1], strided, distinct[:1]]
        self.assertEqual(_array_storage_bytes(views), original.nbytes + distinct.nbytes)
        trainer = trainer_for_tree()
        first = trainer.fit_chunk([[0]], [0])["log"]
        second = trainer.fit_chunk([[1]], [1])["log"]
        self.assertEqual(first["memory"]["pending_bytes"], 16)
        self.assertEqual(second["memory"]["pending_bytes"], 0)
        self.assertEqual(first["trained_samples"], 0)
        self.assertEqual(second["trained_samples"], 2)
        for row in (first, second):
            memory = row["memory"]
            self.assertEqual(memory["total_bytes"], sum(n for key, n in memory.items() if key != "total_bytes"))
            timing = row["timing"]
            self.assertGreaterEqual(timing["update_s"], timing["retrain_s"])
            self.assertGreaterEqual(timing["retrain_s"], timing["tree_fit_s"])
            self.assertGreaterEqual(timing["update_s"], timing["preprocessing_s"])
        self.assertEqual(first["timing"]["tree_fit_s"], 0)
        self.assertEqual(first["timing"]["retrain_s"], 0)

    def test_t51_failed_build_does_not_count_evaluation_until_successful_retry(self):
        trainer = trainer_for_tree()
        trainer.fit_chunk([[0], [1]], [0, 1])
        trainer.fit_chunk([[0]], [0])
        with patch.object(DecisionTreeClassifier, "fit", side_effect=RuntimeError("failed build")):
            with self.assertRaisesRegex(RuntimeError, "failed build"):
                trainer.fit_chunk([[1]], [1])
        self.assertEqual(len(trainer.logs), 2)
        self.assertEqual(trainer.metrics.result()["n_samples"], 1)
        self.assertEqual(trainer.pipeline.pending_samples_, 1)
        retry = trainer.fit_chunk([[1]], [1])
        self.assertEqual(retry["log"]["cumulative"]["n_samples"], 2)
        self.assertEqual(retry["log"]["rolling"]["n_samples"], 2)
        self.assertEqual(len(trainer.logs), 3)
        self.assertEqual(retry["log"]["retrain_count"], 2)

    def test_c03_reset_and_empty_chunk_restart_the_entire_stream(self):
        trainer = trainer_for_tree(window_size=None)
        trainer.fit_chunk([[0], [1]], [0, 1])
        trainer.fit_chunk([[0]], [0])
        result = trainer.fit_chunk(np.empty((0, 1)), [])
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["log"])
        self.assertEqual(len(trainer.logs), 2)
        self.assertIsNone(trainer.logs[-1]["rolling"])
        trainer.reset()
        self.assertEqual((trainer.pipeline.retrain_count_, trainer.pipeline.pending_samples_), (0, 0))
        self.assertEqual(trainer.logs, [])
        self.assertEqual(trainer.metrics.result()["n_samples"], 0)
        fresh = trainer.fit_chunk([[0, 0], [1, 1]], [0, 1])
        self.assertEqual(fresh["status"], "warmup")
        self.assertEqual(fresh["log"]["retrain_count"], 1)
