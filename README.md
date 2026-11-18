# NumCompute-Stream

NumCompute-Stream is a modular framework for streaming classification, developed for Assignment 2.2. Its NumPy decision tree accepts batches through `partial_fit` and retrains on the entire buffer once it reaches the configured threshold.

- `numcompute/` contains the numerical tools and original interfaces from Assignment 2.1.
- `numcompute_stream/` provides batched numeric CSV input, a Gini decision tree, buffered retraining, incremental preprocessing and statistics, a pipeline, classification metrics, `StreamTrainer`, and visualizations.
- The Assignment 2.2 notebook includes its executed outputs, and the benchmark results are checked in. `demo/quickstart.ipynb` and `benchmark/results.json` preserve the Assignment 2.1 examples and measurements.

## Installation and examples

From the repository root, install the project using Python 3.10 or later. It requires NumPy 1.24 or later and Matplotlib 3.6 or later.

```sh
python -m pip install .
```

Run the examples from the repository root in the existing Conda `aiml` environment. The first example reads one batch of the sample CSV and uses the last column as the label. Both arrays use `float64`, and missing feature values remain `NaN`.

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import iter_supervised_csv

batches = iter_supervised_csv(
    "demo/data/sample.csv", target_col=-1, chunk_size=2, skip_header=1
)
try:
    X, y = next(batches)
    print(X.shape, y.tolist())  # (2, 2) [0.0, 1.0]
finally:
    batches.close()
PY
```

This example trains the tree after the buffer reaches four rows. Probability columns follow `classes_`. The standalone tree requires finite numeric features.

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import DecisionTreeClassifier

tree = DecisionTreeClassifier(buffer_size=4, max_depth=1)
tree.partial_fit([[0], [1]], ["low", "low"])
tree.partial_fit([[2], [3]], ["high", "high"])
print(tree.predict([[0], [3]]).tolist())  # ['low', 'high']
print(tree.classes_.tolist(), tree.predict_proba([[3]]).tolist())  # ['high', 'low'] [[1.0, 0.0]]
PY
```

A preprocessing snapshot fixes the numeric transformation and output columns. This example updates a mixed-column preprocessor with two batches and compares the snapshot with the current transformation for the same row.

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import TabularPreprocessor

prep = TabularPreprocessor(numeric_columns=[0], categorical_columns=[1])
prep.partial_fit([[0, 1], [2, 1]])
frozen = prep.snapshot()
prep.partial_fit([[4, 2]])
print(frozen.transform([[2, 2]]).tolist())  # [[1.0, 0.0]]
print(prep.transform([[2, 2]]).tolist())    # [[0.0, 0.0, 1.0]]
PY
```

`StreamingStats(bin_edges=...)` tracks per-column means, variances, and fixed-bin histograms. Use `result(q=0.5)` to query the exact median. Exact quantile queries retain historical samples, so memory use grows with the number of rows.

The trainer scores each batch with the current model before learning from that batch. The second batch introduces a numeric `NaN` and a new category. The third triggers the second tree build.

```sh
conda run --no-capture-output -n aiml python -B - <<'PY'
from numcompute_stream import DecisionTreeClassifier, Pipeline, StreamTrainer, TabularPreprocessor

pipe = Pipeline([
    ("prep", TabularPreprocessor([0], [1])),
    ("tree", DecisionTreeClassifier(buffer_size=2)),
])
trainer = StreamTrainer(pipe, labels=[0, 1], pos_label=1, window_size=2)
for X, y in [
    ([[0, 1], [2, 1]], [0, 1]),
    ([[float("nan"), 2]], [1]),
    ([[4, 2]], [0]),
]:
    row = trainer.fit_chunk(X, y)["log"]
    print(row["status"], row["retrain_count"], row["pending_samples"])
# warmup 1 0
# scored 1 1
# scored 2 0
PY
```

The pipeline supports a standalone tree or a tree preceded by one `StandardScaler` or `TabularPreprocessor`. It also supports stateless, row-wise transformations that preserve row count before or after the incremental preprocessor. Use `TabularPreprocessor` to impute missing numeric values. Cumulative and rolling metrics count only rows for which the trainer made predictions. `score_chunk` evaluates a batch without updating the model.

The log stores scalar summaries. Call `trainer.metrics.result()` for the full confusion matrix and per-class metrics. Exact cumulative AUC retains historical scores and sorts them at query time. The log records separate timings for preprocessing, tree fitting, retraining, the full update, and evaluation. Its memory field estimates the storage of retained NumPy arrays.

## Tests

```sh
# Assignment 2.2 tests
conda run -n aiml python -B -m unittest discover -s tests -p 'test_stream_*.py' -v

# Assignment 2.1 regression tests
conda run -n aiml python -B -m pytest -p no:cacheprovider tests --ignore-glob='tests/test_stream_*.py'

# Run both suites
conda run -n aiml python -B -m pytest -p no:cacheprovider tests
```

## Notebook, benchmarks, and installation check

The [executed notebook](demo/stream_demo.ipynb) replays 2,000 synthetic observations in CSV order. It demonstrates evaluation before learning, warmup, buffered retraining, rolling metrics, and batched statistics. The generation rules are documented separately. [CSV data notes](demo/data/README.md)

Run these commands from the repository root. The notebook runner uses the `aiml` Jupyter kernel and local IPC, so it needs permission to open local sockets. The `aiml` environment used here includes `jupyter_client`, `ipykernel`, and `matplotlib_inline`.

```sh
conda run -n aiml python -B -m demo.execute_stream_notebook
conda run -n aiml python -B -m benchmark.run_stream --output "$(mktemp -d)/stream_results.json"
conda run -n aiml python -B -m demo.check_stream_install
```

- The notebook includes saved outputs. Its figures are also available as PNGs. [Cumulative accuracy](demo/figures/stream_metric.png), [Model comparison on shared batches](demo/figures/stream_compare.png), [Predictions and ground truth](demo/figures/stream_predictions.png)
- The benchmark prints a timing table and writes a fresh JSON file to a temporary directory. It checks that the Python loop and NumPy agree on candidate thresholds and Gini gains before timing. It compares accuracy for three buffer sizes on the same evaluable batches. Overall runtime includes scoring, exact AUC queries, and memory logging. [Recorded benchmark results](benchmark/stream_results.json)
- The installation check builds a wheel offline, installs it in a temporary virtual environment, then imports both packages from outside the checkout. It verifies training and plotting with the Agg backend using the NumPy and Matplotlib already available in `aiml`.

The full suite passed 709 tests in `aiml` with Python 3.12.13, NumPy 2.5.2, and Matplotlib 3.11.1. This includes 642 legacy tests and 67 Assignment 2.2 tests.

## Report and submission

- [Technical report (3-page PDF)](report/Assignment2.2-report.pdf)
- [Submission branch raven/assignment2.2](https://github.com/RavenRF-Li/numcompute-review/tree/raven/assignment2.2)

## API and archive

- The API contract covers the original `numcompute/` package. [Legacy API contract](API_CONTRACT.md)
- The Assignment 2.1 archive contains the original README, assignment requirements, task allocation, contribution record, and historical report. Commands and paths for code, tests, demos, and benchmarks in the archived documents are relative to the repository root. [Archive](archive/assignment2.1/), [Original README](archive/assignment2.1/README.md), [Report PDF](archive/assignment2.1/report/NumCompute-report.pdf), [Report DOCX](archive/assignment2.1/report/NumCompute-report.docx)

## Deliverables

- The complete `numcompute_stream/` package, README, and at least 30 substantive Assignment 2.2 unit tests.
- An executable `demo/stream_demo.ipynb` that demonstrates batched CSV input, incremental training, metrics, and plotting.
- Reproducible loop and vectorized benchmarks, plus measured results for multiple `buffer_size` settings in `benchmark/`.
- A technical report of at most four pages and a demonstration video of at most five minutes.
- The work lives on `raven/assignment2.2` in the personal fork. `main` retains the original group project.
