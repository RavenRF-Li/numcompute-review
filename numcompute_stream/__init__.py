"""Data input for NumCompute-Stream."""

from .io import iter_csv_chunks, iter_supervised_csv, load_csv

__all__ = ["load_csv", "iter_csv_chunks", "iter_supervised_csv"]
