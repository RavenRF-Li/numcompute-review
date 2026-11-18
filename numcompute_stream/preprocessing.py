"""Incremental feature scaling, mean imputation and categorical encoding."""

from dataclasses import dataclass

import numpy as np

from ._validation import check_feature_width, feature_matrix, integer, numeric_array
from .stats import _updated_moments


def _require_fitted(width):
    if width is None:
        raise RuntimeError("Call fit or partial_fit with a nonempty batch first")


def _learn_numeric(X, previous=None, *, reset):
    width = None if previous is None or reset else previous.n_features
    X = feature_matrix(X, expected_width=width, allow_nan=True)
    if X.shape[0]:
        return _updated_moments(None if reset else previous, X)
    if reset:
        raise ValueError("X must not be empty for fit")
    return previous


def _scale_from_variance(variance):
    return np.where(variance > 0, np.sqrt(variance), 1.0)


def _impute(X, fill_values):
    return np.where(np.isnan(X), fill_values, X)


def _validate_fill_value(value):
    value = numeric_array(value, name="fill_value", ndim=0, check_values=False)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("fill_value must be finite in float64")
    return value


@dataclass(frozen=True)
class _NumericSnapshot:
    _mean: np.ndarray
    _scale: np.ndarray
    _fill_values: np.ndarray | None

    def __post_init__(self):
        self._mean.setflags(write=False)
        self._scale.setflags(write=False)
        if self._fill_values is not None:
            self._fill_values.setflags(write=False)

    def _transform(self, X):
        if self._fill_values is not None:
            X = _impute(X, self._fill_values)
        with np.errstate(over="ignore"):
            result = (X - self._mean) / self._scale
        if np.isinf(result).any():
            raise ValueError("X produces nonfinite scaled features in float64")
        return result

    def transform(self, X):
        X = feature_matrix(X, expected_width=self._mean.size, allow_nan=True)
        return self._transform(X)


def _numeric_snapshot(moments, *, fill_value=None):
    mean = moments.mean
    missing = np.isnan(mean)
    fill_values = None if fill_value is None else np.where(missing, fill_value, mean)
    return _NumericSnapshot(np.where(missing, 0.0, mean),
                            _scale_from_variance(moments.var()), fill_values)


class StandardScaler:
    """Standardize each feature using its cumulative mean and population variance.

    Fit and partial_fit return self. NaNs are ignored while learning and
    preserved by transform. Unobserved columns use center zero. Unobserved
    and constant columns use scale one. Input and output have shape (n, d).
    """

    def __init__(self):
        self._moments = None

    @property
    def n_features_in_(self):
        """Number of learned columns, or None before learning."""
        return None if self._moments is None else self._moments.n_features

    @property
    def count_(self):
        """Number of raw observations in each column, excluding NaN."""
        _require_fitted(self.n_features_in_)
        return self._moments.count

    @property
    def mean_(self):
        """Per-column center, zero for an unobserved column."""
        _require_fitted(self.n_features_in_)
        return np.nan_to_num(self._moments.mean, nan=0.0)

    @property
    def var_(self):
        """Per-column population variance, NaN for an unobserved column."""
        _require_fitted(self.n_features_in_)
        return self._moments.var()

    @property
    def scale_(self):
        """Per-column standard deviations, with unit fallback."""
        return _scale_from_variance(self.var_)

    def fit(self, X):
        """Replace learned state with moments from a nonempty batch."""
        self._moments = _learn_numeric(X, reset=True)
        return self

    def partial_fit(self, X):
        """Accumulate raw observations from a batch."""
        self._moments = _learn_numeric(X, self._moments, reset=False)
        return self

    def transform(self, X):
        """Apply the current center and scale without learning from X."""
        return self.snapshot().transform(X)

    def snapshot(self):
        """Return a transform-only view with independent, frozen parameters."""
        _require_fitted(self.n_features_in_)
        return _numeric_snapshot(self._moments)


class SimpleImputer:
    """Fill NaN with the cumulative observed mean in each column.

    Before a column has any observations, use the finite `fill_value` (zero
    by default). Fit and partial_fit return self. Input and output are (n, d).
    """

    def __init__(self, fill_value=0.0):
        self._fill_value = _validate_fill_value(fill_value)
        self._moments = None

    @property
    def n_features_in_(self):
        """Number of learned columns, or None before learning."""
        return None if self._moments is None else self._moments.n_features

    @property
    def count_(self):
        """Number of raw observations in each column, excluding NaN."""
        _require_fitted(self.n_features_in_)
        return self._moments.count

    @property
    def statistics_(self):
        """Per-column replacement values."""
        _require_fitted(self.n_features_in_)
        return np.nan_to_num(self._moments.mean, nan=self._fill_value)

    def fit(self, X):
        """Replace learned state with moments from a nonempty batch."""
        self._moments = _learn_numeric(X, reset=True)
        return self

    def partial_fit(self, X):
        """Accumulate raw observations without counting replacements."""
        self._moments = _learn_numeric(X, self._moments, reset=False)
        return self

    def transform(self, X):
        """Replace NaN without changing the learned state."""
        fill_values = self.statistics_
        X = feature_matrix(X, expected_width=fill_values.size, allow_nan=True)
        return _impute(X, fill_values)


def _category_matrix(X, previous):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D categorical array, got shape {X.shape}")
    width = None if previous is None else previous.width
    check_feature_width(X, expected_width=width)
    if X.dtype.kind not in "biufUS":
        raise TypeError("X must contain real numeric, boolean or string categories")
    if X.dtype.kind == "f" and not np.isfinite(X).all():
        raise ValueError("X numeric categories must be finite")
    if X.shape[0] and previous is not None and X.dtype.kind != previous.values.dtype.kind:
        raise TypeError("X category dtype kind must match the learned categories")
    return X


@dataclass(frozen=True)
class _Encoding:
    width: int
    columns: np.ndarray
    values: np.ndarray
    handle_unknown: str

    def __post_init__(self):
        self.columns.setflags(write=False)
        self.values.setflags(write=False)

    def transform(self, X):
        X = _category_matrix(X, self)
        if X.shape[0] == 0:
            return np.empty((0, self.values.size), dtype=float)
        matches = X[:, self.columns] == self.values
        if self.handle_unknown == "error" and np.any(matches.sum(axis=1) != self.width):
            raise ValueError("X contains unknown categories")
        return matches.astype(float)


def _learn_encoding(X, previous, *, handle_unknown, reset):
    previous = None if reset else previous
    X = _category_matrix(X, previous)
    if X.shape[0] == 0:
        if reset:
            raise ValueError("X must not be empty for fit")
        return previous

    columns, values = [], []
    for index, column in enumerate(X.T):
        new = np.unique(column)
        if previous is not None:
            known = previous.values[previous.columns == index]
            new = new[~np.isin(new, known)]
        if new.size:
            columns.append(np.full(new.size, index, dtype=np.intp))
            values.append(new)
    if not values:
        return previous
    if previous is not None:
        columns.insert(0, previous.columns)
        values.insert(0, previous.values)
    return _Encoding(X.shape[1], np.concatenate(columns), np.concatenate(values), handle_unknown)


class OneHotEncoder:
    """Encode a homogeneous 2-D categorical array as (n_samples, n_categories).

    Each new batch appends previously unseen keys, sorted by input column
    and category within that batch. Existing output columns keep their
    positions. Transform encodes unknown values as zero for that input
    column with handle_unknown="ignore"; with "error" it raises ValueError.
    """

    def __init__(self, handle_unknown="ignore"):
        if not isinstance(handle_unknown, str):
            raise TypeError("handle_unknown must be a string")
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self._handle_unknown = handle_unknown
        self._encoding = None

    @property
    def n_features_in_(self):
        """Number of learned input columns, or None before learning."""
        return None if self._encoding is None else self._encoding.width

    @property
    def n_features_out_(self):
        """Number of learned one-hot columns."""
        return 0 if self._encoding is None else self._encoding.values.size

    @property
    def categories_(self):
        """Copies of the categories for each input column, in append order."""
        _require_fitted(self.n_features_in_)
        mapping = self._encoding
        return [mapping.values[mapping.columns == index] for index in range(mapping.width)]

    def fit(self, X):
        """Replace all category mappings using a nonempty batch, returning self."""
        self._encoding = _learn_encoding(X, None, handle_unknown=self._handle_unknown, reset=True)
        return self

    def partial_fit(self, X):
        """Append new categories, returning self."""
        self._encoding = _learn_encoding(
            X, self._encoding, handle_unknown=self._handle_unknown, reset=False
        )
        return self

    def transform(self, X):
        """Encode using the current category mapping without learning from X."""
        _require_fitted(self.n_features_in_)
        return self._encoding.transform(X)


@dataclass(frozen=True)
class _TabularSnapshot:
    _width: int
    _numeric_columns: tuple
    _categorical_columns: tuple
    _numeric: _NumericSnapshot | None
    _encoding: _Encoding | None

    def transform(self, X):
        X = feature_matrix(X, expected_width=self._width, allow_nan=True)
        blocks = []
        if self._numeric is not None:
            blocks.append(self._numeric._transform(X[:, self._numeric_columns]))
        if self._encoding is not None:
            blocks.append(self._encoding.transform(X[:, self._categorical_columns]))
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


class TabularPreprocessor:
    """Impute and scale selected numeric columns, then append encoded columns.

    Column positions are distinct, nonnegative integers. Either group may be
    empty and unselected columns are omitted. Numeric input is converted to
    float64, with categorical columns supplied as finite numeric codes.
    Both imputation and scaling use a single set of moments of the raw
    observed values. The encoded part uses the specified categorical column
    order and ignores unknown categories during transform.

    Transform returns (n_samples, n_numeric_columns + n_onehot_columns).
    Fit and partial_fit return self. Snapshot returns a transform-only
    object with the numeric parameters and category layout fixed at creation.
    """

    def __init__(self, numeric_columns, categorical_columns, *, fill_value=0.0):
        self._numeric_columns = tuple(integer(i, "numeric_columns") for i in numeric_columns)
        self._categorical_columns = tuple(integer(i, "categorical_columns") for i in categorical_columns)
        columns = self._numeric_columns + self._categorical_columns
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Specify at least one column and select each column only once")
        self._max_column = max(columns)
        self._fill_value = _validate_fill_value(fill_value)
        self._width = None
        self._moments = None
        self._encoding = None

    @property
    def n_features_in_(self):
        """Number of input columns before selection, or None before learning."""
        return self._width

    def _learn(self, X, *, reset):
        X = feature_matrix(X, expected_width=None if reset else self._width, allow_nan=True)
        if X.shape[0] == 0:
            if reset:
                raise ValueError("X must not be empty for fit")
            return self
        if self._max_column >= X.shape[1]:
            raise ValueError(f"Selected column {self._max_column} is out of range for X shape {X.shape}")

        moments, encoding = None, None
        if self._numeric_columns:
            moments = _updated_moments(
                None if reset else self._moments, X[:, self._numeric_columns]
            )
        if self._categorical_columns:
            encoding = _learn_encoding(
                X[:, self._categorical_columns], self._encoding,
                handle_unknown="ignore", reset=reset,
            )
        self._width, self._moments, self._encoding = X.shape[1], moments, encoding
        return self

    def fit(self, X):
        """Replace the numeric and categorical state from a nonempty batch."""
        return self._learn(X, reset=True)

    def partial_fit(self, X):
        """Accumulate numeric moments and append category columns."""
        return self._learn(X, reset=False)

    def transform(self, X):
        """Apply the current numeric parameters and category mapping."""
        return self.snapshot().transform(X)

    def snapshot(self):
        """Return fixed numeric parameters and a fixed, append-order mapping."""
        _require_fitted(self._width)
        numeric = None
        if self._moments is not None:
            numeric = _numeric_snapshot(self._moments, fill_value=self._fill_value)
        return _TabularSnapshot(
            self._width, self._numeric_columns, self._categorical_columns,
            numeric, self._encoding,
        )
