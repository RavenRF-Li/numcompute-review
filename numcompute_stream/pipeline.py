"""Named incremental preprocessing and an atomically published decision tree."""

from copy import copy
from time import perf_counter

import numpy as np

from numcompute.pipeline import _named_objects, _transformed

from ._validation import feature_matrix, label_vector
from .preprocessing import StandardScaler, TabularPreprocessor
from .tree import DecisionTreeClassifier


def _transform_steps(steps, X):
    for name, step in steps:
        X = _transformed(step, name, X)
    return X


class Pipeline:
    """Connect named, row-preserving transforms to a DecisionTreeClassifier.

    Supports one StandardScaler or TabularPreprocessor, with optional
    stateless, deterministic, row-wise transform(X) steps on either side. The tree
    may also be the only step. Supplied estimator and learner objects are
    configuration templates; fitting this pipeline leaves them untouched.
    Use TabularPreprocessor to impute numeric NaN before fitting the tree.

    X is a real array of shape (n_samples, n_features); y is a label vector
    of shape (n_samples,). Fit starts a new stream. Partial fit keeps one
    raw buffer; all pending rows are transformed with the same snapshot
    when a new tree is trained. Predictions and transform use the last
    successfully published pair. A failing update preserves the live state.
    """

    def __init__(self, steps):
        steps = _named_objects(steps, "steps")
        tree = steps[-1][1]
        if type(tree) is not DecisionTreeClassifier:
            raise TypeError("The last step must be a DecisionTreeClassifier")
        self._tree_options = {
            name: getattr(tree, name)
            for name in ("max_depth", "min_samples_split", "max_features", "buffer_size", "random_state")
        }
        self._learner = None
        before, after = [], []
        for name, step in steps[:-1]:
            if type(step) in (StandardScaler, TabularPreprocessor):
                if self._learner is not None:
                    raise ValueError("Pipeline accepts only one incremental preprocessing step")
                self._learner = copy(step)
            else:
                if any(callable(getattr(step, method, None)) for method in ("fit", "partial_fit", "snapshot")):
                    raise TypeError(f"{name!r} must be stateless, a StandardScaler or a TabularPreprocessor")
                if not callable(getattr(step, "transform", None)):
                    raise TypeError(f"{name!r} must provide transform()")
                (before if self._learner is None else after).append((name, step))
        self._before, self._after = tuple(before), tuple(after)
        self._allow_nan = isinstance(self._learner, TabularPreprocessor)
        self.reset()

    def reset(self):
        """Clear training, pending data and timings, retaining configuration."""
        if self._learner is not None:
            self._learner._reset()
        self._pending = []
        self._pending_samples = 0
        self._width = self._label_kind = None
        self._published = None
        self._retrain_count = 0
        self._last_timing = dict.fromkeys(
            ("preprocessing_s", "tree_fit_s", "retrain_s", "update_s"), 0.0
        )

    @property
    def is_fitted_(self):
        """Whether there is a published snapshot and tree."""
        return self._published is not None

    @property
    def classes_(self):
        """Latest tree classes, ordered by probability column."""
        return self._model()[1].classes_

    @property
    def pending_samples_(self):
        """Number of raw samples below the retraining threshold."""
        return self._pending_samples

    @property
    def retrain_count_(self):
        """Number of successful builds since construction, reset or fit."""
        return self._retrain_count

    @property
    def last_train_samples_(self):
        """Number of samples in the most recently fitted tree, zero at startup."""
        return self._published[1].last_train_samples_ if self.is_fitted_ else 0

    @property
    def last_timing_(self):
        """Seconds for the last nonempty successful update.

        Preprocessing covers learning and feature transforms. Tree fit includes
        the estimator's input checks. Retraining includes merging the buffer,
        snapshot and full-buffer transform, fitting and publishing. Overall
        update includes raw validation and buffering, excluding scoring and logs.
        """
        return self._last_timing.copy()

    def _input(self, X, *, expected_width):
        return feature_matrix(X, expected_width=expected_width, allow_nan=self._allow_nan)

    def _batch(self, X, y, *, reset=False):
        X = self._input(X, expected_width=None if reset else self._width)
        y = label_vector(y)
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of samples, got {len(X)} and {len(y)}")
        if len(X) and not reset and self._label_kind is not None and y.dtype.kind != self._label_kind:
            raise TypeError("y dtype kind must match the stream")
        return X, y

    def _features(self, X, snapshot, *, before_applied=False):
        if not before_applied:
            X = _transform_steps(self._before, X)
        if snapshot is not None:
            X = snapshot.transform(X)
        return _transform_steps(self._after, X)

    def _learn(self, X, y, *, reset):
        update_start = perf_counter()
        X, y = self._batch(X, y, reset=reset)
        if len(X) == 0:
            if reset:
                raise ValueError("X and y must not be empty for fit")
            return self

        pending = [] if reset else self._pending
        rows = len(X) if reset else self._pending_samples + len(X)
        prep_start = perf_counter()
        incoming = _transform_steps(self._before, X)
        learner = self._learner
        if learner is not None:
            # These built-in learners replace their state on successful updates.
            learner = copy(learner)
            (learner.fit if reset else learner.partial_fit)(incoming)
        preprocessing = perf_counter() - prep_start
        tree_fit = retrain = 0.0
        if reset or rows >= self._tree_options["buffer_size"]:
            retrain_start = perf_counter()
            if pending:
                features, labels = zip(*pending, (X, y))
                training_X, training_y = np.concatenate(features), np.concatenate(labels)
            else:
                training_X, training_y = incoming, y

            prep_start = perf_counter()
            snapshot = learner.snapshot() if learner is not None else None
            transformed = self._features(training_X, snapshot, before_applied=not pending)
            preprocessing += perf_counter() - prep_start
            tree = DecisionTreeClassifier(**self._tree_options)
            fit_start = perf_counter()
            tree.fit(transformed, training_y)
            tree_fit = perf_counter() - fit_start

            self._published = snapshot, tree
            self._retrain_count = 1 if reset else self._retrain_count + 1
            self._pending = []
            self._pending_samples = 0
            retrain = perf_counter() - retrain_start
        else:
            self._pending.append((X.copy(), y.copy()))
            self._pending_samples = rows

        self._learner = learner
        self._width, self._label_kind = X.shape[1], y.dtype.kind
        self._last_timing = {
            "preprocessing_s": preprocessing,
            "tree_fit_s": tree_fit,
            "retrain_s": retrain,
            "update_s": perf_counter() - update_start,
        }
        return self

    def fit(self, X, y):
        """Fit immediately on a nonempty batch and discard old buffered rows."""
        return self._learn(X, y, reset=True)

    def partial_fit(self, X_chunk, y_chunk):
        """Update the learner and retrain once if the raw buffer is full."""
        return self._learn(X_chunk, y_chunk, reset=False)

    def _model(self):
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted; call fit or fill the training buffer")
        return self._published

    def transform(self, X):
        """Apply the published transforms to X and return (n_samples, n_output)."""
        snapshot, _ = self._model()
        X = self._input(X, expected_width=self._width)
        return self._features(X, snapshot)

    def predict(self, X):
        """Return labels of shape (n_samples,) from the published pair."""
        return self._model()[1].predict(self.transform(X))

    def predict_proba(self, X):
        """Return (n_samples, len(classes_)) from the published pair."""
        return self._predict_proba(X)

    def _predict_proba(self, X, *, validated=False):
        snapshot, tree = self._model()
        if not validated:
            X = self._input(X, expected_width=self._width)
        return tree.predict_proba(self._features(X, snapshot))
