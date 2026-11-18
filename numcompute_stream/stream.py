"""Prequential evaluation, learning and summaries for a streaming pipeline."""

from collections import deque
from time import perf_counter

import numpy as np

from .metrics import ClassificationMetrics
from .pipeline import Pipeline


def _array_storage_bytes(root, seen_storage=None):
    """Estimate reachable array backing bytes, counting each owner once."""
    if seen_storage is None:
        seen_storage = set()
    objects = [root]
    visited = set()
    total = 0
    while objects:
        value = objects.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(value, np.ndarray):
            owner = value
            while True:
                base = owner.obj if isinstance(owner, memoryview) else getattr(owner, "base", None)
                if base is not None:
                    owner = base
                else:
                    break
            if id(owner) not in seen_storage:
                seen_storage.add(id(owner))
                total += owner.nbytes if isinstance(owner, np.ndarray) else memoryview(owner).nbytes
        elif isinstance(value, dict):
            objects.extend(value.values())
        elif isinstance(value, (list, tuple, deque)):
            objects.extend(value)
        elif hasattr(value, "__dict__"):
            objects.extend(vars(value).values())
    return total


def _summary(report):
    return {
        "n_samples": report["n_samples"],
        "n_scored": report["n_scored"],
        "accuracy": report["accuracy"],
        "macro_precision": float(report["precision"].mean()),
        "macro_recall": float(report["recall"].mean()),
        "macro_f1": float(report["f1"].mean()),
        "auc": report["auc"],
    }


class StreamTrainer:
    """Score each chunk with the published pipeline before learning from it.

    Labels configure the metrics and probability-column order. Warmup chunks
    train the pipeline but contribute no evaluations. Score_chunk is read-only.
    Fit_chunk returns the pre-update scoring result plus a log record; empty
    chunks do not add a record. The trainer and pipeline are single-threaded.

    Logs store scalar summaries and timing in seconds. The memory estimate
    counts retained NumPy backing buffers for the current tree/snapshot,
    learning state, raw buffer and metrics, with shared owners counted once.
    Python objects, temporary allocations, logs and external stateless steps
    are excluded.
    """

    def __init__(self, pipeline, *, labels, pos_label=None, window_size=None):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a streaming Pipeline")
        self.pipeline = pipeline
        self.metrics = ClassificationMetrics(labels, pos_label=pos_label, window_size=window_size)
        self._rolling = window_size is not None
        self.logs = []

    def reset(self):
        """Clear the pipeline, the scored history and the log."""
        self.pipeline.reset()
        self.metrics.reset()
        self.logs.clear()

    def _memory(self):
        seen_storage = set()
        memory = {
            name: _array_storage_bytes(state, seen_storage)
            for name, state in (
                ("published_bytes", self.pipeline._published),
                ("learning_bytes", self.pipeline._learner),
                ("pending_bytes", self.pipeline._pending),
                ("metrics_bytes", self.metrics),
            )
        }
        memory["total_bytes"] = sum(memory.values())
        return memory

    def _score(self, X, y):
        true = self.metrics._codes(y, "y")
        if not len(y) or not self.pipeline.is_fitted_:
            status = "empty" if not len(y) else "warmup"
            version = None
            predictions = y[:0].copy()
            probabilities = np.empty((0, self.metrics._k), dtype=float)
            batch = self.metrics._batch(true[:0], true[:0], None)
        else:
            status = "scored"
            version = self.pipeline.retrain_count_
            classes = self.pipeline.classes_
            columns = self.metrics._codes(classes, "Pipeline classes_")
            scores = self.pipeline._predict_proba(X, validated=True)
            winners = scores.argmax(axis=1)
            predictions = classes[winners]
            probabilities = np.zeros((len(y), self.metrics._k), dtype=float)
            probabilities[:, columns] = scores
            batch = self.metrics._batch(true, columns[winners], probabilities)
        report = self.metrics._batch_result(batch)
        return {
            "status": status,
            "evaluated_version": version,
            "evaluated_samples": report["n_samples"],
            "y_pred": predictions,
            "y_proba": probabilities,
            "metrics": report,
        }, batch

    def score_chunk(self, X, y):
        """Return predictions and batch metrics without learning or logging."""
        X, y = self.pipeline._batch(X, y)
        result, _ = self._score(X, y)
        return result

    def fit_chunk(self, X, y):
        """Score, learn, then commit metrics and a log if learning succeeded."""
        evaluation_start = perf_counter()
        X, y = self.pipeline._batch(X, y)
        result, batch = self._score(X, y)
        evaluation_s = perf_counter() - evaluation_start
        if result["status"] == "empty":
            return {**result, "log": None}

        previous_version = self.pipeline.retrain_count_
        self.pipeline.partial_fit(X, y)
        self.metrics._commit(batch)
        version = self.pipeline.retrain_count_
        timing = self.pipeline.last_timing_
        timing["evaluation_s"] = evaluation_s
        record = {
            "status": result["status"],
            "chunk_samples": len(y),
            "evaluated_version": result["evaluated_version"],
            "retrain_count": version,
            "pending_samples": self.pipeline.pending_samples_,
            "last_train_samples": self.pipeline.last_train_samples_,
            "trained_samples": self.pipeline.last_train_samples_ if version != previous_version else 0,
            "batch": _summary(result["metrics"]),
            "cumulative": _summary(self.metrics.result()),
            "rolling": _summary(self.metrics.result(rolling=True)) if self._rolling else None,
            "timing": timing,
            "memory": self._memory(),
        }
        self.logs.append(record)
        return {**result, "log": record}
