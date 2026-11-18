"""Matplotlib plots for chunk metrics and pre-update predictions."""

import matplotlib.pyplot as plt
import numpy as np

from ._validation import label_vector, numeric_array


def _series(values, name):
    return numeric_array(values, name=name, ndim=1, allow_nan=True, nonempty=False)


def _finish(figure, axes, save_path, show):
    if save_path is not None:
        figure.savefig(save_path, dpi=140)
    if show:
        plt.show()
    return figure, axes


def _chunk_axes(chunks, title, ylabel):
    figure, axes = plt.subplots(figsize=(8, 3.7), layout="constrained")
    axes.set(title=title, xlabel="Chunk", ylabel=ylabel)
    if chunks:
        axes.set_xlim(0.5, chunks + 0.5)
    axes.grid(axis="y", alpha=0.25)
    return figure, axes


def plot_metric_over_time(metric_values, title, ylabel, *, save_path=None, show=False):
    """Plot a 1-D metric by one-based chunk number, leaving NaN gaps."""
    values = _series(metric_values, "metric_values")
    figure, axes = _chunk_axes(values.size, title, ylabel)
    axes.plot(np.arange(1, values.size + 1), values, color="#24686d")
    return _finish(figure, axes, save_path, show)


def compare_models(metric1, metric2, labels, *, save_path=None, show=False):
    """Compare two equally sized metric series on the same chunk axis."""
    first, second = _series(metric1, "metric1"), _series(metric2, "metric2")
    if first.shape != second.shape:
        raise ValueError("metric1 and metric2 must have equal lengths")
    names = () if isinstance(labels, str) else tuple(labels)
    if len(names) != 2 or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("labels must provide two nonempty model names")
    figure, axes = _chunk_axes(first.size, "Model comparison", "Metric value")
    chunks = np.arange(1, first.size + 1)
    axes.plot(chunks, first, label=names[0], color="#24686d")
    axes.plot(chunks, second, label=names[1], color="#ae512e", linestyle="--")
    axes.legend(loc="best")
    return _finish(figure, axes, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False):
    """Plot equal-length 1-D label vectors on one shared categorical scale."""
    truth = label_vector(y_true, name="y_true")
    predictions = label_vector(y_pred, name="y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have equal lengths")
    if (truth.dtype.kind in "US") != (predictions.dtype.kind in "US"):
        raise TypeError("y_true and y_pred must both contain numeric labels or both contain text labels")
    if truth.size and {truth.dtype.kind, predictions.dtype.kind} == {"S", "U"}:
        raise TypeError("y_true and y_pred must use the same text label representation")
    labels = dict.fromkeys(truth.tolist() + predictions.tolist())
    coordinates = {label: i for i, label in enumerate(labels)}
    samples = np.arange(1, truth.size + 1)
    figure, axes = plt.subplots(figsize=(8, 3.7), layout="constrained")
    axes.scatter(samples, [coordinates[label] for label in truth.tolist()],
                 label="Ground truth", facecolors="none", edgecolors="#24686d", s=34)
    axes.scatter(samples, [coordinates[label] for label in predictions.tolist()],
                 label="Prediction", color="#ae512e", marker="x", s=22)
    axes.set(title="Predictions before learning vs. ground truth", xlabel="Evaluated sample",
             ylabel="Class")
    axes.set_yticks(list(coordinates.values()), [str(label) for label in labels])
    axes.grid(axis="y", alpha=0.25)
    axes.legend(loc="best")
    return _finish(figure, axes, save_path, show)
