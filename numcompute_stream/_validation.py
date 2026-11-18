"""Shared input validation for NumCompute-Stream."""

import numpy as np

from numcompute._validation import integer
from numcompute._validation import real_array as _real_array


def column_index(value, *, name, n_columns):
    """Validate and normalize a possibly negative column index."""
    index = integer(value, name, minimum=-n_columns)
    if index >= n_columns:
        raise ValueError(f"{name}={index} is out of range for {n_columns} CSV columns")
    return index % n_columns


def numeric_array(
    values, *, name, ndim, allow_nan=False, nonempty=True, check_values=True
):
    """Validate a real array and identify the input in errors."""
    try:
        return _real_array(
            values, ndim=ndim, allow_nan=allow_nan,
            nonempty=nonempty, check_values=check_values,
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc


def check_feature_width(X, *, expected_width):
    """Check the width of a two-dimensional feature array."""
    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature column")
    if expected_width is not None and X.shape[1] != expected_width:
        raise ValueError(f"X must have {expected_width} features, got shape {X.shape}")


def feature_matrix(values, *, expected_width=None, allow_nan=False):
    """Validate a two-dimensional float64 matrix, optionally allowing NaN."""
    X = numeric_array(
        values, name="X", ndim=2, nonempty=False, check_values=False
    )
    check_feature_width(X, expected_width=expected_width)
    with np.errstate(over="ignore"):
        X = X.astype(np.float64, copy=False)
    if allow_nan:
        if np.isinf(X).any():
            raise ValueError("X contains infinity or values outside the float64 range")
    elif not np.isfinite(X).all():
        raise ValueError("X contains NaN, infinity or values outside the float64 range")
    return X


def label_vector(values, *, name="y"):
    """Validate a one-dimensional label array without coercing its dtype."""
    y = np.asarray(values)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {y.shape}")
    if y.dtype.kind not in "biufUS":
        raise TypeError(f"{name} must contain real numeric, boolean or string labels")
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        raise ValueError(f"{name} contains NaN or infinite labels")
    return y
