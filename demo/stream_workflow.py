"""Shared CSV streaming entry points for the notebook and benchmarks."""

from pathlib import Path

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier, Pipeline, StreamTrainer, TabularPreprocessor,
    iter_supervised_csv,
)


DATA_PATH = Path(__file__).resolve().parent / "data" / "stream_sample.csv"
CHUNK_SIZE = 32
SEED = 2026
WINDOW_SIZE = 128
TREE_OPTIONS = {"max_depth": 4, "min_samples_split": 8, "max_features": None}


def stream_chunks(path=DATA_PATH, *, chunk_size=CHUNK_SIZE):
    """Read ordered (X, y) chunks with the target in the final CSV column."""
    return iter_supervised_csv(path, target_col=-1, chunk_size=chunk_size, skip_header=1)


def run_stream(chunks, *, buffer_size, seed=SEED, on_chunk=None):
    """Run a fresh trainer, optionally observing each pre-update scoring result."""
    pipe = Pipeline([
        ("preprocess", TabularPreprocessor(numeric_columns=[0, 1], categorical_columns=[2])),
        ("tree", DecisionTreeClassifier(buffer_size=buffer_size, random_state=seed, **TREE_OPTIONS)),
    ])
    trainer = StreamTrainer(pipe, labels=[0, 1], pos_label=1, window_size=WINDOW_SIZE)
    for X, y in chunks:
        result = trainer.fit_chunk(X, y)
        if on_chunk is not None:
            on_chunk(X, y, result)
    return trainer


def common_evaluation(logs_by_buffer):
    """Return the common scored-chunk mask for aligned replay logs."""
    rows = [np.array([entry["chunk_samples"] for entry in log]) for log in logs_by_buffer.values()]
    if not rows or any(not np.array_equal(rows[0], other) for other in rows[1:]):
        raise ValueError("Expected logs for the same ordered chunks")
    return np.logical_and.reduce([
        [entry["batch"]["n_samples"] > 0 for entry in log]
        for log in logs_by_buffer.values()
    ], axis=0)
