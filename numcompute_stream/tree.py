"""A Gini decision tree with buffered retraining."""

from dataclasses import dataclass

import numpy as np

from ._validation import feature_matrix, integer, label_vector


_GAIN_TOLERANCE = 1e-12


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    probabilities: np.ndarray | None = None


def _candidate_gains(values, labels, counts, parent_gini):
    """Return ascending thresholds and Gini gains for one candidate feature."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    boundaries = np.flatnonzero(ordered[:-1] < ordered[1:])
    if boundaries.size == 0:
        return ordered[:0], np.empty(0, dtype=float)

    prefix = np.zeros((ordered.size, counts.size), dtype=np.intp)
    prefix[np.arange(ordered.size), labels[order]] = 1
    np.cumsum(prefix, axis=0, out=prefix)
    left_counts = prefix[boundaries]
    n_left = boundaries + 1
    n_right = ordered.size - n_left

    left_p = left_counts / n_left[:, None]
    right_p = (counts - left_counts) / n_right[:, None]
    left_gini = 1.0 - np.sum(left_p**2, axis=1)
    right_gini = 1.0 - np.sum(right_p**2, axis=1)
    weighted_gini = (n_left * left_gini + n_right * right_gini) / ordered.size
    return ordered[boundaries], parent_gini - weighted_gini


def _best_split(X, rows, labels, counts, features, parent_gini):
    """Find the best eligible split, treating near-equal gains as ties."""
    def candidates(feature):
        return _candidate_gains(X[rows, feature], labels, counts, parent_gini)

    # Retain one maximum per feature so the tie band uses the global maximum.
    maxima = np.array([candidates(feature)[1].max(initial=-np.inf) for feature in features])
    best_gain = maxima.max()
    if best_gain <= _GAIN_TOLERANCE:
        return None

    tied_features = (maxima > _GAIN_TOLERANCE) & (best_gain - maxima <= _GAIN_TOLERANCE)
    feature = features[np.flatnonzero(tied_features)[0]]
    thresholds, gains = candidates(feature)
    tied_thresholds = (gains > _GAIN_TOLERANCE) & (best_gain - gains <= _GAIN_TOLERANCE)
    return int(feature), float(thresholds[np.flatnonzero(tied_thresholds)[0]])


def _training_arrays(X, y, *, expected_width=None):
    X = feature_matrix(X, expected_width=expected_width)
    y = label_vector(y)
    if X.shape[0] != y.size:
        raise ValueError(
            f"X and y must have the same number of samples, got {X.shape[0]} and {y.size}"
        )
    return X, y


class DecisionTreeClassifier:
    """Train one classification tree, either immediately or from a sample buffer.

    The root is at depth zero; `max_depth=0` produces one leaf.
    A node becomes a leaf at the depth limit, with fewer than
    `min_samples_split` rows, when all labels agree, or when no valid split
    has sufficient gain. `max_features` is None (all features), a positive
    count, or a fraction in (0, 1] rounded up. When using a subset, each
    eligible node samples without replacement. Every build restarts the
    Generator at the nonnegative integer `random_state` and visits left
    children first.

    `buffer_size` is a positive sample count. An update that reaches or
    exceeds it trains once on all pending rows, including the full incoming
    chunk. Predictions use the latest fitted tree while more rows accumulate.

    Features are finite float64 values after conversion; very large integers
    can lose precision. Labels accept real, boolean or string dtypes. Each
    incremental stream has one feature width and label dtype kind. An
    explicit fit starts a fresh stream. Failing validation or building a new
    tree leaves the published model and pending data intact.

    The threshold is the last observed value in the lower group, routed left
    by <=. Gains must exceed 1e-12. Within 1e-12 of the best gain, prefer the
    smallest feature index, then the smallest threshold. Prediction ties
    choose the first sorted class.
    """

    def __init__(
        self, max_depth=8, min_samples_split=2, max_features=None,
        buffer_size=128, random_state=0,
    ):
        self.max_depth = integer(max_depth, "max_depth")
        self.min_samples_split = integer(min_samples_split, "min_samples_split", minimum=2)
        self.buffer_size = integer(buffer_size, "buffer_size", minimum=1)
        self.random_state = integer(random_state, "random_state")
        if isinstance(max_features, (float, np.floating)):
            if not 0 < max_features <= 1:
                raise ValueError("max_features as a fraction must be in (0, 1]")
        elif max_features is not None:
            max_features = integer(max_features, "max_features", minimum=1)
        self.max_features = max_features
        self._root = None
        self._classes = None
        self._width = None
        self._label_kind = None
        self._pending = []
        self._pending_samples = 0
        self._retrain_count = 0
        self._last_train_samples = 0

    @property
    def is_fitted_(self):
        """Whether a trained tree is available."""
        return self._root is not None

    @property
    def classes_(self):
        """Copy of the latest sorted classes, in probability-column order."""
        self._require_fitted()
        return self._classes.copy()

    @property
    def pending_samples_(self):
        """Number of rows awaiting training."""
        return self._pending_samples

    @property
    def retrain_count_(self):
        """Number of successful builds since construction or the latest fit."""
        return self._retrain_count

    @property
    def last_train_samples_(self):
        """Number of rows used in the latest build, or zero before training."""
        return self._last_train_samples

    def _feature_count(self, width):
        if self.max_features is None:
            return width
        if isinstance(self.max_features, (float, np.floating)):
            return int(np.ceil(self.max_features * np.float64(width)))
        if self.max_features > width:
            raise ValueError(f"max_features={self.max_features} exceeds X width {width}")
        return self.max_features

    def _build_tree(self, X, labels, n_classes, feature_count):
        rng = np.random.default_rng(self.random_state)
        width = X.shape[1]
        all_features = range(width)
        root = _Node()
        stack = [(root, np.arange(X.shape[0]), 0)]
        while stack:
            node, rows, depth = stack.pop()
            targets = labels[rows]
            counts = np.bincount(targets, minlength=n_classes)
            probabilities = counts / rows.size
            if (depth >= self.max_depth or rows.size < self.min_samples_split
                    or np.count_nonzero(counts) == 1):
                node.probabilities = probabilities
                continue

            features = all_features
            if feature_count < width:
                features = np.sort(rng.choice(width, size=feature_count, replace=False))
            parent_gini = 1.0 - np.sum(probabilities**2)
            split = _best_split(X, rows, targets, counts, features, parent_gini)
            if split is None:
                node.probabilities = probabilities
                continue

            node.feature, node.threshold = split
            left = X[rows, node.feature] <= node.threshold
            node.left, node.right = _Node(), _Node()
            stack.append((node.right, rows[~left], depth + 1))
            stack.append((node.left, rows[left], depth + 1))
        return root

    def _train(self, X, y, feature_count, *, retrain_count):
        """Build a candidate before replacing any published or pending state."""
        classes, labels = np.unique(y, return_inverse=True)
        root = self._build_tree(X, labels, classes.size, feature_count)
        self._root = root
        self._classes = classes
        self._width = X.shape[1]
        self._label_kind = y.dtype.kind
        self._pending.clear()
        self._pending_samples = 0
        self._retrain_count = retrain_count
        self._last_train_samples = X.shape[0]

    def fit(self, X, y):
        """Fit on nonempty X (n_samples, n_features) and y (n_samples,).

        On success, discard the old buffer and start the build count at one.
        Return self.
        """
        X, y = _training_arrays(X, y)
        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty for fit")
        feature_count = self._feature_count(X.shape[1])
        self._train(X, y, feature_count, retrain_count=1)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Accumulate a chunk of the same shapes as fit and return self.

        A valid empty chunk changes nothing. A nonempty tail below the
        threshold stays pending until more rows arrive or fit resets it.
        """
        X, y = _training_arrays(X_chunk, y_chunk, expected_width=self._width)
        if X.shape[0] == 0:
            return self
        feature_count = self._feature_count(X.shape[1])
        if self._label_kind is not None and y.dtype.kind != self._label_kind:
            raise TypeError(
                f"y dtype kind must match the stream ({self._label_kind!r}), "
                f"got {y.dtype.kind!r}"
            )

        n_samples = self._pending_samples + X.shape[0]
        if n_samples < self.buffer_size:
            self._pending.append((X.copy(), y.copy()))
            self._pending_samples = n_samples
            self._width = X.shape[1]
            self._label_kind = y.dtype.kind
        else:
            if self._pending:
                feature_chunks, label_chunks = zip(*self._pending, (X, y))
                X, y = np.concatenate(feature_chunks), np.concatenate(label_chunks)
            self._train(X, y, feature_count, retrain_count=self._retrain_count + 1)
        return self

    def _require_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Decision tree is not fitted; call fit or fill the training buffer")

    def _leaf_groups(self, X):
        """Yield row indices and probabilities for each reached leaf."""
        stack = [(self._root, np.arange(X.shape[0]))] if X.shape[0] else []
        while stack:
            node, rows = stack.pop()
            if node.feature is None:
                yield rows, node.probabilities
            else:
                left = X[rows, node.feature] <= node.threshold
                if np.any(left):
                    stack.append((node.left, rows[left]))
                if not np.all(left):
                    stack.append((node.right, rows[~left]))

    def predict(self, X):
        """Return labels of shape (n_samples,) from the latest trained tree."""
        self._require_fitted()
        X = feature_matrix(X, expected_width=self._width)
        predictions = np.empty(X.shape[0], dtype=self._classes.dtype)
        for rows, probabilities in self._leaf_groups(X):
            predictions[rows] = self._classes[np.argmax(probabilities)]
        return predictions

    def predict_proba(self, X):
        """Return leaf class frequencies, shape (n_samples, len(classes_))."""
        self._require_fitted()
        X = feature_matrix(X, expected_width=self._width)
        probabilities = np.empty((X.shape[0], self._classes.size), dtype=float)
        for rows, leaf_probabilities in self._leaf_groups(X):
            probabilities[rows] = leaf_probabilities
        return probabilities
