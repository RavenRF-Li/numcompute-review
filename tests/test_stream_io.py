"""Tests for supervised CSV input (T01–T03)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from numcompute_stream import iter_supervised_csv


class TestSupervisedCSV(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "stream.csv"

    def test_t01_header_blank_lines_missing_feature_and_short_tail(self):
        self.path.write_text(
            "first;second;label\n1;;0\n  \n2;4;1\n3;5;0\n", encoding="utf-8"
        )

        batches = list(
            iter_supervised_csv(
                self.path, target_col=-1, chunk_size=2, delimiter=";", skip_header=1
            )
        )

        self.assertEqual(len(batches), 2)
        (X_first, y_first), (X_tail, y_tail) = batches
        assert_array_equal(X_first, [[1, np.nan], [2, 4]])
        assert_array_equal(y_first, [0, 1])
        assert_array_equal(X_tail, [[3, 5]])
        assert_array_equal(y_tail, [0])
        self.assertEqual(X_first.dtype, np.float64)
        self.assertEqual(y_first.dtype, np.float64)

        self.path.write_text("10,0,20\n13,1,23\n", encoding="utf-8")
        [(X, y)] = list(iter_supervised_csv(self.path, target_col=np.int64(1)))
        assert_array_equal(X, [[10, 20], [13, 23]])
        assert_array_equal(y, [0, 1])

    def test_t02_inconsistent_width_surfaces_on_the_affected_chunk(self):
        self.path.write_text("1,2,0\n3,1\n", encoding="utf-8")
        batches = iter_supervised_csv(self.path, target_col=-1, chunk_size=1)

        X, y = next(batches)
        assert_array_equal(X, [[1, 2]])
        assert_array_equal(y, [0])
        with self.assertRaisesRegex(ValueError, "same number of fields"):
            next(batches)

        with self.assertRaisesRegex(ValueError, "columns"):
            next(iter_supervised_csv(self.path, target_col=-1, chunk_size=2))

    def test_t03_rejects_invalid_target_columns_and_values(self):
        self.path.write_text("1,2,\n", encoding="utf-8")
        for target in (3, -4):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "target_col"):
                next(iter_supervised_csv(self.path, target_col=target))
        for target in (True, np.bool_(False), 1.0):
            with self.subTest(target=target), self.assertRaisesRegex(TypeError, "target_col"):
                next(iter_supervised_csv(self.path, target_col=target))

        for label in ("", "nan", "inf", "-inf"):
            self.path.write_text(f"1,2,{label}\n", encoding="utf-8")
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "target labels"):
                next(iter_supervised_csv(self.path, target_col=-1))

        self.path.write_text("1,2,invalid\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            next(iter_supervised_csv(self.path, target_col=-1))

        self.path.write_text("1,inf,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "features"):
            next(iter_supervised_csv(self.path, target_col=-1))

        self.path.write_text("0\n1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "at least one feature column"):
            next(iter_supervised_csv(self.path, target_col=0))
