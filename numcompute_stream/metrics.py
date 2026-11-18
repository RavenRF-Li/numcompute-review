"""Cumulative and sample-window classification metrics."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from numcompute.metrics import roc_auc_score

from ._validation import integer, label_vector, numeric_array


@dataclass(frozen=True)
class _Batch:
    pairs: np.ndarray
    confusion: np.ndarray
    probabilities: np.ndarray | None


def _ratio(numerator, denominator, fallback):
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, fallback, dtype=float),
        where=denominator != 0,
    )


class ClassificationMetrics:
    """Accumulate metrics in the fixed order supplied by one-dimensional labels.

    The confusion matrix has true labels on rows and predictions on columns.
    Precision, recall and F1 are per class, with denominator zero mapped to
    zero_division (0.0, 1.0 or NaN). Accuracy is NaN before any evaluation.
    Probabilities are optional and use columns in the configured label order.
    Missing probabilities do not affect the other metrics.

    With pos_label, AUC is exact one-vs-rest for that class. Otherwise it is
    the macro mean of defined one-vs-rest AUCs. A class needs both a positive
    and a negative scored example. Exact cumulative AUC stores O(M*k) values
    for M scored rows and k labels and sorts scores on each result query.
    A configured rolling window retains the last W evaluated rows.
    """

    def __init__(self, labels, *, pos_label=None, window_size=None, zero_division=0.0):
        labels = label_vector(labels, name="labels")
        self._indices = {label: i for i, label in enumerate(labels.tolist())}
        self._k = labels.size
        if self._k == 0 or len(self._indices) != self._k:
            raise ValueError("labels must be nonempty and unique")
        self._labels = labels.copy()
        self._pos_index = (
            None if pos_label is None
            else self._codes(label_vector([pos_label], name="pos_label"), "pos_label")[0]
        )
        self._window_size = (
            None if window_size is None else integer(window_size, "window_size", minimum=1)
        )
        self._zero = float(numeric_array(zero_division, name="zero_division", ndim=0, allow_nan=True))
        if self._zero not in (0.0, 1.0) and not np.isnan(self._zero):
            raise ValueError("zero_division must be 0.0, 1.0 or NaN")
        self.reset()

    @property
    def labels_(self):
        """Copy of the configured probability-column and matrix-axis labels."""
        return self._labels.copy()

    def reset(self):
        """Clear cumulative and rolling data, retaining configuration."""
        self._matrix = np.zeros((self._k, self._k), dtype=np.int64)
        self._scores = []
        self._window = deque()
        self._window_rows = 0
        self._window_matrix = np.zeros_like(self._matrix) if self._window_size is not None else None

    def _codes(self, values, name):
        # Map only distinct values in Python. Exact scalar equality keeps large
        # integral labels distinct even with differently typed numeric inputs.
        unique, inverse = np.unique(values, return_inverse=True)
        try:
            indices = np.array([self._indices[value] for value in unique.tolist()], dtype=np.intp)
        except KeyError as exc:
            raise ValueError(f"{name} contains unknown label {exc.args[0]!r}") from exc
        return indices[inverse]

    def _probabilities(self, values, rows):
        if values is None:
            return None
        matrix = numeric_array(values, name="y_proba", ndim=2, nonempty=False)
        expected = (rows, self._k)
        if matrix.shape != expected:
            raise ValueError(f"y_proba must have shape {expected}, got {matrix.shape}")
        if ((matrix < 0).any() or (matrix > 1).any()
                or not np.isclose(matrix.sum(axis=1, dtype=float), 1, atol=1e-6, rtol=0).all()):
            raise ValueError("y_proba rows must contain probabilities in [0, 1] that sum to one")
        return matrix.astype(float, copy=False)

    def _count_pairs(self, pairs):
        return np.bincount(pairs, minlength=self._k**2).reshape(self._matrix.shape)

    def _batch(self, true_codes, pred_codes, probabilities):
        pairs = true_codes * self._k + pred_codes
        return _Batch(pairs, self._count_pairs(pairs), probabilities)

    def update(self, y_true_chunk, y_pred_chunk, *, y_proba=None):
        """Fold equal-length label vectors and optional (n, k) scores into state."""
        true = label_vector(y_true_chunk, name="y_true")
        pred = label_vector(y_pred_chunk, name="y_pred")
        if len(true) != len(pred):
            raise ValueError("y_true and y_pred must have the same length")
        probabilities = self._probabilities(y_proba, len(true))
        batch = self._batch(self._codes(true, "y_true"), self._codes(pred, "y_pred"), probabilities)
        self._commit(batch)
        return self

    def _commit(self, batch):
        if not batch.pairs.size:
            return
        scores = None if batch.probabilities is None else batch.probabilities.copy()
        self._matrix += batch.confusion
        if scores is not None:
            self._scores.append((batch.pairs, scores))
        if self._window_size is None:
            return

        self._window_matrix += batch.confusion
        self._window.append((batch.pairs, scores))
        self._window_rows += batch.pairs.size
        while self._window_rows > self._window_size:
            pairs, scores = self._window.popleft()
            drop = min(pairs.size, self._window_rows - self._window_size)
            self._window_matrix -= self._count_pairs(pairs[:drop])
            self._window_rows -= drop
            if drop < pairs.size:
                # The window may cut into a large chunk. Keep just its tail.
                self._window.appendleft((pairs[drop:].copy(), None if scores is None else scores[drop:]))

    def _auc(self, scored):
        by_class = np.full(self._k, np.nan)
        if not scored:
            return float("nan"), by_class, 0
        if len(scored) == 1:
            pairs, scores = scored[0]
        else:
            pair_chunks, score_chunks = zip(*scored)
            pairs, scores = np.concatenate(pair_chunks), np.concatenate(score_chunks)
        truth = pairs // self._k
        classes = range(self._k) if self._pos_index is None else (self._pos_index,)
        for index in classes:
            positive = truth == index
            if positive.any() and not positive.all():
                by_class[index] = roc_auc_score(positive, scores[:, index], pos_label=True)
        defined = by_class[~np.isnan(by_class)]
        return (float(defined.mean()) if defined.size else float("nan")), by_class, len(truth)

    def _report(self, matrix, scored):
        count = int(matrix.sum())
        correct = matrix.diagonal()
        predicted, actual = matrix.sum(axis=0), matrix.sum(axis=1)
        auc, auc_by_class, n_scored = self._auc(scored)
        return {
            "labels": self.labels_,
            "n_samples": count,
            "n_scored": n_scored,
            "accuracy": float(correct.sum() / count) if count else float("nan"),
            "precision": _ratio(correct, predicted, self._zero),
            "recall": _ratio(correct, actual, self._zero),
            "f1": _ratio(2 * correct, predicted + actual, self._zero),
            "confusion_matrix": matrix.copy(),
            "auc": auc,
            "auc_per_class": auc_by_class,
        }

    def _batch_result(self, batch):
        scored = () if batch.probabilities is None else ((batch.pairs, batch.probabilities),)
        return self._report(batch.confusion, scored)

    def result(self, *, rolling=False):
        """Return an independent snapshot of cumulative or last-W-row metrics."""
        if not rolling:
            return self._report(self._matrix, self._scores)
        if self._window_size is None:
            raise ValueError("Rolling metrics require a configured window_size")
        scored = [(pairs, scores) for pairs, scores in self._window if scores is not None]
        return self._report(self._window_matrix, scored)
