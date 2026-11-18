"""Shared input validation for NumCompute-Stream."""

from numcompute._validation import integer
from numcompute._validation import real_array as _real_array


def column_index(value, *, name, n_columns):
    """Validate and normalize a possibly negative column index."""
    index = integer(value, name, minimum=-n_columns)
    if index >= n_columns:
        raise ValueError(f"{name}={index} is out of range for {n_columns} CSV columns")
    return index % n_columns


def numeric_array(values, *, name, ndim, allow_nan=False):
    """Validate a nonempty real array and identify the input in errors."""
    try:
        return _real_array(values, ndim=ndim, allow_nan=allow_nan)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc
