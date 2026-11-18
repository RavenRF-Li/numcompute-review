"""Tests for cumulative moments, quantiles and fixed-edge histograms."""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import StreamingStats


class TestStreamingStats(unittest.TestCase):
    def test_t37_accumulated_moments_and_exact_quantiles(self):
        stats = StreamingStats(bin_edges=[0, 3, 5])
        self.assertIs(stats.update_stats([[0, np.nan], [2, 2]]), stats)
        stats.update([[4, 4]])
        result = stats.result()
        assert_array_equal(result["count"], [3, 2])
        assert_array_equal(result["mean"], [2, 3])
        assert_allclose(result["var"], [8 / 3, 1])
        assert_array_equal(result["quantile"], [2, 3])
        assert_array_equal(stats.result([0.25, 0.75])["quantile"], [[1, 2.5], [3, 3.5]])

    def test_t38_fixed_histogram_tracks_edges_and_out_of_range_values(self):
        stats = StreamingStats(bin_edges=[0, 1, 2])
        stats.update_stats([[-1], [0], [1]]).update_stats([[2], [3], [np.nan]])
        result = stats.result()
        assert_array_equal(result["histogram"], [[1, 2]])
        assert_array_equal(result["underflow"], [1])
        assert_array_equal(result["overflow"], [1])
        assert_array_equal(result["edges"], [0, 1, 2])
        assert_array_equal(result["count"], [5])

    def test_t39_all_missing_ddof_and_reset(self):
        stats = StreamingStats(bin_edges=[0, 3, 5]).update([[np.nan, 3]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = stats.result(q=[0.25, 0.75], ddof=1)
        assert_array_equal(result["count"], [0, 1])
        assert_array_equal(result["mean"], [np.nan, 3])
        assert_array_equal(result["var"], [np.nan, np.nan])
        assert_array_equal(result["quantile"], [[np.nan, 3], [np.nan, 3]])
        assert_array_equal(result["histogram"], [[0, 0], [0, 1]])
        stats.reset()
        with self.assertRaisesRegex(RuntimeError, "No data"):
            stats.result()
        stats.update([[4, 5]])
        result = stats.result()
        assert_array_equal(result["count"], [1, 1])
        assert_array_equal(result["mean"], [4, 5])
        assert_array_equal(result["histogram"], [[0, 1], [0, 1]])
        assert_array_equal(result["edges"], [0, 3, 5])

    def test_b05_invalid_updates_and_callers_cannot_modify_history(self):
        with self.assertRaisesRegex(ValueError, "bin_edges"):
            StreamingStats(bin_edges=[0, 0, 2])
        edges = np.array([0.0, 1.0, 2.0])
        X = np.array([[0.5], [1.5]])
        stats = StreamingStats(bin_edges=edges).update(np.empty((0, 3))).update(X)
        edges[:] = 99
        X[:] = 99
        for invalid in ([[np.inf]], [[0, 1]], [[np.finfo(float).max], [-np.finfo(float).max]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                stats.update(invalid)
        result = stats.result()
        assert_array_equal(result["count"], [2])
        assert_array_equal(result["mean"], [1])
        assert_array_equal(result["quantile"], [1])
        assert_array_equal(result["histogram"], [[1, 1]])
        assert_array_equal(result["edges"], [0, 1, 2])
        result["histogram"][:] = 0
        result["edges"][:] = 0
        assert_array_equal(stats.result()["histogram"], [[1, 1]])
        assert_array_equal(stats.result()["edges"], [0, 1, 2])
