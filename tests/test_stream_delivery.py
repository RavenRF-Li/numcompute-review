"""Reproduce V01 and V02 on small data with hand-checked results."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from benchmark.run_stream import buffer_sweep, split_case
from demo.stream_workflow import run_stream, stream_chunks
from numcompute_stream.visualise import plot_metric_over_time, plot_predictions_vs_ground_truth


class TestStreamDelivery(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_v01_real_csv_to_pre_update_predictions_and_plots(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "tiny.csv"
            source.write_text("a,b,c,y\n0,0,0,0\n1,1,0,1\n,2,1,1\n3,3,1,1\n4,,2,0\n5,5,2,1\n",
                              encoding="utf-8")
            truth, predictions = [], []

            def observe(X, y, result):
                if result["status"] == "scored":
                    truth.extend(y.tolist())
                    predictions.extend(result["y_pred"].tolist())

            trainer = run_stream(stream_chunks(source, chunk_size=2), buffer_size=2, on_chunk=observe)
            self.assertEqual([row["status"] for row in trainer.logs], ["warmup", "scored", "scored"])
            self.assertEqual(trainer.pipeline.retrain_count_, 3)
            self.assertEqual(trainer.pipeline.pending_samples_, 0)
            self.assertEqual(trainer.metrics.result()["accuracy"], 0.25)
            assert_array_equal(truth, [1, 1, 0, 1])
            assert_array_equal(predictions, [0, 0, 1, 1])
            plot_metric_over_time([row["cumulative"]["accuracy"] for row in trainer.logs],
                                  "Accuracy", "Accuracy", save_path=root / "metric.png")
            plot_predictions_vs_ground_truth(truth, predictions, save_path=root / "predictions.png")
            for name in ("metric.png", "predictions.png"):
                self.assertGreater((root / name).stat().st_size, 0)

    def test_v02_split_reference_and_shared_evaluation_rows(self):
        case = split_case(np.array([0, 1, 2, 3], dtype=float), np.array([0, 0, 1, 1]))
        case.check(case.operation())
        case.check(case.baseline())
        self.assertEqual(case.metadata["best_threshold"], 1)
        self.assertEqual(case.metadata["best_gain"], 0.5)

        X = np.array([[0, 0, 0], [1, 1, 1]])
        chunks = [(X, np.array(y)) for y in ([0, 0], [1, 1], [1, 1], [1, 0])]
        report = buffer_sweep(chunks, [2, 4], warmups=0, repeats=1)
        self.assertEqual(report["common_chunks_one_based"], [3, 4])
        small, large = report["models"]
        self.assertEqual([small["first_scored_chunk"], large["first_scored_chunk"]], [2, 3])
        self.assertEqual([small["coverage"], large["coverage"]], [0.75, 0.5])
        self.assertEqual([small["common"]["accuracy"], large["common"]["accuracy"]], [0.75, 0.25])
        self.assertEqual([small["common"]["rows"], large["common"]["rows"]], [4, 4])
        self.assertEqual(set(small["median_seconds"]), {"tree_fit_s", "retrain_s", "overall_s"})
