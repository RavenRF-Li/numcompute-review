# Assignment 2.2 validation record

This page records the final checks for NumCompute-Stream and supports the technical report.

| Check | Result |
| --- | --- |
| Full regression suite | 709 passed, including 67 Assignment 2.2 tests and 642 legacy tests. Four existing warnings came from NumPy configuration output. |
| Demo notebook | The `aiml` Jupyter kernel ran all five code cells from a clean state. The saved notebook contains successful outputs and three embedded plots. |
| Installation | An offline build produced a wheel containing both packages. In a temporary environment, imports from outside the repository, minimal stream training and prediction, and PNG export with Agg all passed. |
| Benchmarks | A fresh run checked Gini thresholds and gains and reproduced the three buffer accuracies on 1,744 shared evaluation rows. The retained measurement file matches all nine recorded source hashes and the synthetic CSV hash. |

The Assignment 2.2 tests cover buffered retraining, missing values, category expansion and frozen preprocessing, score-before-update evaluation, rolling metrics, retry after a failed build, and artifact generation. The suite includes small reference examples for Gini gain and the common evaluation set.

The checks used Python 3.12.13, NumPy 2.5.2, and Matplotlib 3.11.1 in the existing `aiml` environment. The [README](../README.md) provides the commands. Replay outputs and measured workloads are available in the [notebook](../demo/stream_demo.ipynb) and [benchmark record](../benchmark/stream_results.json).
