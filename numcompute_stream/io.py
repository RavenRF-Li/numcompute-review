"""Read numeric CSV data for supervised streams."""

import numpy as np

from numcompute.io import iter_csv_chunks, load_csv

from ._validation import column_index, numeric_array

__all__ = ["load_csv", "iter_csv_chunks", "iter_supervised_csv"]


def iter_supervised_csv(
    path, *, target_col, chunk_size=1000, delimiter=",", skip_header=0
):
    """Yield pairs of float64 feature and label arrays from a numeric CSV.

    Parameters
    ----------
    path : str or path-like
        Path to the UTF-8 CSV file.
    target_col : int
        Zero-based label column. Negative indices count from the right.
    chunk_size : int, default=1000
        Maximum number of records per chunk. The final chunk may be shorter.
    delimiter : str, default=","
        Field separator passed to :func:`iter_csv_chunks`.
    skip_header : int, default=0
        Number of initial lines to skip once.

    Yields
    ------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature columns in their original order. Missing features remain NaN.
    y : numpy.ndarray, shape (n_samples,)
        Finite numeric labels from the target column.

    Raises
    ------
    TypeError
        An integer option has an invalid type.
    ValueError
        Invalid option value, empty or malformed data, fewer than two
        columns, an invalid target index, nonfinite labels, or infinity in X.

    Notes
    -----
    Reading and validation occur during iteration. Blank lines are ignored.
    Splitting and validation use O(n_samples * n_features) time and space
    per chunk, excluding arrays retained by the caller.
    """
    for chunk in iter_csv_chunks(
        path,
        chunk_size=chunk_size,
        delimiter=delimiter,
        skip_header=skip_header,
        missing="propagate",
    ):
        n_columns = chunk.shape[1]
        if n_columns < 2:
            raise ValueError("CSV must contain a target column and at least one feature column")
        target = column_index(target_col, name="target_col", n_columns=n_columns)
        y = numeric_array(chunk[:, target], name="y (target labels)", ndim=1)
        X = numeric_array(
            np.delete(chunk, target, axis=1), name="X (features)", ndim=2, allow_nan=True
        )
        yield X, y.copy()
