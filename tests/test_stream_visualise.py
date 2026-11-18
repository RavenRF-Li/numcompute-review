"""Headless checks for the three stream plotting functions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from numcompute_stream.visualise import (
    compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth,
)


class TestVisualise(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_t52_metric_plot_saves_and_preserves_warmup_gap(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "metric.png"
            figure, axes = plot_metric_over_time(
                [np.nan, 0.5, 1], "Pre-update accuracy", "Accuracy", save_path=target
            )
            self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIs(axes.figure, figure)
        self.assertEqual(axes.get_title(), "Pre-update accuracy")
        self.assertEqual(axes.get_ylabel(), "Accuracy")
        assert_array_equal(axes.get_xlim(), [0.5, 3.5])
        assert_array_equal(axes.lines[0].get_xdata(), [1, 2, 3])
        assert_array_equal(axes.lines[0].get_ydata(), [np.nan, 0.5, 1])

    def test_t53_comparison_label_display_and_bad_length(self):
        with patch.object(plt, "show") as display:
            figure, axes = compare_models([0.5, 1], [0.4, 0.9], ["Buffer 64", "Buffer 128"], show=True)
            display.assert_called_once_with()
        self.assertIs(axes.figure, figure)
        self.assertEqual([text.get_text() for text in axes.get_legend().get_texts()],
                         ["Buffer 64", "Buffer 128"])
        assert_array_equal(axes.lines[1].get_ydata(), [0.4, 0.9])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            compare_models([0.5], [0.4, 0.9], ["Buffer 64", "Buffer 128"])

    def test_t54_predictions_and_truth_share_one_tick_mapping(self):
        _, axes = plot_predictions_vs_ground_truth(["b", "a"], ["a", "b"])
        ticks = {label.get_text(): float(tick) for tick, label in zip(axes.get_yticks(), axes.get_yticklabels())}
        self.assertEqual(set(ticks), {"a", "b"})
        truth, predicted = (collection.get_offsets() for collection in axes.collections)
        assert_array_equal(truth, [[1, ticks["b"]], [2, ticks["a"]]])
        assert_array_equal(predicted, [[1, ticks["a"]], [2, ticks["b"]]])
