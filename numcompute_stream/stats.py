"""Cumulative column statistics for finite numeric observations and NaNs."""

import warnings

import numpy as np

from numcompute.stats import RunningStats, quantile

from ._validation import feature_matrix, numeric_array


def _updated_moments(previous, X):
    """Build and validate a candidate accumulator before callers publish it."""
    candidate = RunningStats() if previous is None else previous.copy()
    with np.errstate(over="ignore"):
        candidate.update(X)
    observed = candidate.count > 0
    variance = candidate.var()[observed]
    if (not np.isfinite(candidate.mean[observed]).all()
            or not np.isfinite(variance).all() or (variance < 0).any()):
        raise ValueError("X produces nonfinite or invalid running moments in float64")
    return candidate


class StreamingStats:
    """Accumulate exact per-column statistics along axis 0.

    `bin_edges` is a strictly increasing finite vector shared by all columns.
    The last bin includes its right edge. Underflow and overflow counts track
    observations outside the edges. NaN contributes to none of the counts.

    Exact linear quantiles retain copies of all chunks, using O(N*d) storage
    for N rows and d columns. The moments and b-bin counts use O(d*(b+1))
    storage, independent of N. Quantile queries need O(N*d) temporary storage
    and at most O(d*N*log(N)) time for reduction, excluding quantile output.
    """

    def __init__(self, *, bin_edges):
        edges = numeric_array(
            bin_edges, name="bin_edges", ndim=1, check_values=False
        )
        with np.errstate(over="ignore"):
            edges = edges.astype(np.float64, copy=True)
        if (edges.size < 2 or not np.isfinite(edges).all()
                or np.any(edges[1:] <= edges[:-1])):
            raise ValueError("bin_edges must contain at least two finite, strictly increasing edges")
        self._edges = edges
        self.reset()

    def reset(self):
        """Clear observations and counts, retaining the fixed bin edges."""
        self._moments = None
        self._history = []
        self._histogram = self._underflow = self._overflow = None

    def update_stats(self, X_chunk):
        """Update from shape (n_samples, n_features) and return self.

        Empty chunks are no-ops. Invalid inputs or unrepresentable moments
        raise without changing accumulated statistics.
        """
        previous = self._moments
        width = previous.n_features if previous is not None else None
        X = feature_matrix(X_chunk, expected_width=width, allow_nan=True)
        if X.shape[0] == 0:
            return self

        moments = _updated_moments(previous, X)
        histogram = np.stack([
            np.histogram(column[~np.isnan(column)], bins=self._edges)[0]
            for column in X.T
        ]).astype(np.int64, copy=False)
        underflow = np.sum(X < self._edges[0], axis=0, dtype=np.int64)
        overflow = np.sum(X > self._edges[-1], axis=0, dtype=np.int64)
        if previous is not None:
            histogram += self._histogram
            underflow += self._underflow
            overflow += self._overflow
        self._history.append(X.copy())
        self._moments = moments
        self._histogram, self._underflow, self._overflow = histogram, underflow, overflow
        return self

    update = update_stats

    def result(self, q=0.5, *, ddof=0):
        """Return independent arrays for count, mean, var and quantile.

        The result also includes histogram (d, b), edges (b+1,), underflow (d,)
        and overflow (d,). Moments have shape (d,). Quantiles have shape (d,)
        for scalar q and (len(q), d) for vector q. Unobserved columns and
        variance with count <= ddof return NaN. Before any nonempty update,
        including after reset, raise RuntimeError.
        """
        if self._moments is None:
            raise RuntimeError("No data has been seen; call update_stats before result")
        count = self._moments.count
        variance = self._moments.var(ddof=ddof)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="all-NaN slice encountered", category=RuntimeWarning
            )
            quantiles = quantile(
                np.concatenate(self._history, axis=0), q, axis=0, nan_policy="omit"
            )
        if not np.isfinite(quantiles[..., count > 0]).all():
            raise ValueError("Quantile interpolation exceeded the float64 range")
        return {
            "count": count,
            "mean": self._moments.mean,
            "var": variance,
            "quantile": quantiles,
            "histogram": self._histogram.copy(),
            "edges": self._edges.copy(),
            "underflow": self._underflow.copy(),
            "overflow": self._overflow.copy(),
        }
