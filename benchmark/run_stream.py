"""Measure split scoring and buffered prequential training for Assignment 2.2."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from demo.stream_workflow import (
    CHUNK_SIZE, DATA_PATH, SEED, TREE_OPTIONS, WINDOW_SIZE, common_evaluation, run_stream, stream_chunks,
)
from numcompute.benchmarking import BenchmarkCase, run_benchmarks
from numcompute_stream.tree import _candidate_gains


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark" / "stream_results.json"


def loop_candidate_gains(values, labels, counts, parent_gini):
    """Independent candidate scoring using a sorted Python sweep."""
    ordered = sorted(zip(values.tolist(), labels.tolist()), key=lambda row: row[0])
    left, right = [0] * len(counts), counts.tolist()
    thresholds, gains = [], []
    for index, (value, label) in enumerate(ordered[:-1]):
        left[label] += 1
        right[label] -= 1
        if value == ordered[index + 1][0]:
            continue
        n_left, n_right = index + 1, len(ordered) - index - 1
        left_gini = 1 - sum((count / n_left) ** 2 for count in left)
        right_gini = 1 - sum((count / n_right) ** 2 for count in right)
        thresholds.append(value)
        gains.append(parent_gini - (n_left * left_gini + n_right * right_gini) / len(ordered))
    return np.asarray(thresholds, dtype=values.dtype), np.asarray(gains, dtype=float)


def _best_threshold(thresholds, gains):
    """Select a threshold with the tree's per-feature tie policy for checks."""
    best = gains.max(initial=-np.inf)
    if best <= 1e-12:
        return None
    return thresholds[np.flatnonzero((gains > 1e-12) & (best - gains <= 1e-12))[0]]


def split_case(values, labels):
    counts = np.bincount(labels)
    parent = 1 - np.sum((counts / len(labels)) ** 2)
    args = values, labels, counts, parent
    expected_thresholds, expected_gains = loop_candidate_gains(*args)
    expected_best = _best_threshold(expected_thresholds, expected_gains)

    def check(result):
        thresholds, gains = result
        np.testing.assert_array_equal(thresholds, expected_thresholds)
        np.testing.assert_allclose(gains, expected_gains, rtol=0, atol=1e-12)
        best = _best_threshold(thresholds, gains)
        if best != expected_best:
            raise AssertionError(f"Expected best threshold {expected_best}, got {best}")
        if best is not None:
            np.testing.assert_array_equal(values <= best, values <= expected_best)

    return BenchmarkCase(
        name=f"tree_candidate_gini_{len(values)}",
        operation=lambda: _candidate_gains(*args),
        baseline=lambda: loop_candidate_gains(*args),
        check=check,
        metadata={
            "rows": len(values), "dtype": str(values.dtype), "classes": len(counts),
            "candidate_thresholds": len(expected_thresholds),
            "best_threshold": float(expected_best) if expected_best is not None else None,
            "best_gain": float(expected_gains.max()) if expected_gains.size else None,
            "gain_atol": 1e-12,
        },
    )


def _accuracy(log, mask):
    rows = [entry["batch"] for entry, include in zip(log, mask) if include]
    scored = sum(row["n_samples"] for row in rows)
    correct = sum(round(row["accuracy"] * row["n_samples"]) for row in rows if row["n_samples"])
    return {"rows": scored, "correct": correct, "accuracy": correct / scored if scored else None}


def _timed_replay(chunks, buffer_size, seed):
    start = perf_counter()
    trainer = run_stream(chunks, buffer_size=buffer_size, seed=seed)
    overall = perf_counter() - start
    log = trainer.logs
    retrains = [row for row in log if row["trained_samples"]]
    tree_fit = [row["timing"]["tree_fit_s"] for row in retrains]
    retrain = [row["timing"]["retrain_s"] for row in retrains]
    return log, {
        "tree_fit_s": sum(tree_fit),
        "retrain_s": sum(retrain),
        "overall_s": overall,
        "tree_fit_per_retrain_s": tree_fit,
        "retrain_per_retrain_s": retrain,
    }


def _outcomes(log):
    return [
        (row["chunk_samples"], row["trained_samples"], row["batch"]["n_samples"],
         row["batch"]["accuracy"] if row["batch"]["n_samples"] else None)
        for row in log
    ]


def buffer_sweep(chunks, buffer_sizes, *, seed=SEED, repeats=3, warmups=1):
    """Replay prepared identical chunks and compare the common scored rows."""
    sizes = tuple(buffer_sizes)
    if len(sizes) < 2 or len(sizes) != len(set(sizes)) or repeats < 1 or warmups < 0:
        raise ValueError("Use distinct buffer sizes, positive repeats and nonnegative warmups")
    chunks = tuple(chunks)
    samples = {size: [] for size in sizes}
    logs = {}
    for _ in range(warmups):
        for size in sizes:
            run_stream(chunks, buffer_size=size, seed=seed)
    for repeat in range(repeats):
        offset = repeat % len(sizes)
        for size in sizes[offset:] + sizes[:offset]:
            log, timings = _timed_replay(chunks, size, seed)
            if size in logs:
                if _outcomes(log) != _outcomes(logs[size]):
                    raise AssertionError(f"Replay outcomes changed for buffer_size={size}")
            else:
                logs[size] = log
            samples[size].append(timings)

    common = common_evaluation(logs)
    total_rows = sum(len(y) for _, y in chunks)
    models = []
    for size, log in logs.items():
        scored = [row["batch"]["n_samples"] > 0 for row in log]
        all_available = _accuracy(log, scored)
        models.append({
            "buffer_size": size,
            "first_scored_chunk": next((i for i, yes in enumerate(scored, 1) if yes), None),
            "warmup_rows": total_rows - all_available["rows"],
            "coverage": all_available["rows"] / total_rows if total_rows else None,
            "all_available": all_available,
            "common": _accuracy(log, common),
            "retrain_count": log[-1]["retrain_count"] if log else 0,
            "training_rows_per_retrain": [row["trained_samples"] for row in log if row["trained_samples"]],
            "pending_rows": log[-1]["pending_samples"] if log else 0,
            "max_batch_end_array_bytes": max((row["memory"]["total_bytes"] for row in log), default=0),
            "timing_samples": samples[size],
            "median_seconds": {
                key: median(sample[key] for sample in samples[size])
                for key in ("tree_fit_s", "retrain_s", "overall_s")
            },
        })
    return {
        "repeats": repeats,
        "warmups_per_model": warmups,
        "sample_order": "rotate buffer order by one position per measured repeat",
        "total_rows": total_rows,
        "common_chunks_one_based": (np.flatnonzero(common) + 1).tolist(),
        "timing_scope": {
            "tree_fit_s": "Sum of candidate tree.fit calls, including estimator validation.",
            "retrain_s": "Sum of merge, snapshot, full-buffer transform, tree fit and publication.",
            "overall_s": "Fresh setup and full Trainer replay, including scoring, AUC and memory logs.",
            "excluded_from_overall": "CSV reading, rendering, and this report's aggregation and serialization.",
        },
        "models": models,
    }


def _provenance(data_path):
    data_path = data_path.resolve()
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    sources = (
        "numcompute/benchmarking.py", "numcompute_stream/tree.py", "numcompute_stream/pipeline.py",
        "numcompute_stream/preprocessing.py", "numcompute_stream/metrics.py", "numcompute_stream/stream.py",
        "demo/stream_workflow.py", "demo/generate_stream_data.py", "benchmark/run_stream.py",
    )
    return {
        "git_base_commit": result.stdout.strip() if result.returncode == 0 else None,
        "data_path": str(data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "benchmark" / "results.json").resolve():
        parser.error("Assignment 2.1 benchmark/results.json is reserved")
    if len(set(args.buffer_sizes)) < 3 or len(args.buffer_sizes) != len(set(args.buffer_sizes)):
        parser.error("provide at least three distinct buffer sizes")
    if min(args.buffer_sizes) < 1 or args.chunk_size < 1 or args.seed < 0:
        parser.error("buffer and chunk sizes must be positive, and seed nonnegative")

    chunks = list(stream_chunks(args.data, chunk_size=args.chunk_size))
    rng = np.random.default_rng(args.seed)
    cases = [
        split_case(np.round(rng.normal(size=rows), 2), rng.integers(0, 3, size=rows))
        for rows in (256, 2048)
    ]
    split = run_benchmarks(cases, repeats=7, number=5, warmups=2, seed=args.seed)
    buffers = buffer_sweep(chunks, args.buffer_sizes, seed=args.seed)
    report = {
        "provenance": _provenance(args.data),
        "stream_config": {
            "seed": args.seed, "chunk_size": args.chunk_size, "tree": TREE_OPTIONS,
            "rolling_window": WINDOW_SIZE, "buffer_sizes": args.buffer_sizes,
        },
        "candidate_splits": split,
        "buffer_sweep": buffers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in split["results"]:
        numpy_us = row["operation"]["median_seconds"] * 1e6
        loop_us = row["baseline"]["median_seconds"] * 1e6
        print(f"{row['name']:<27} NumPy {numpy_us:7.2f} µs | loop {loop_us:8.2f} µs")
    print("\nBuffer  Common accuracy (rows)  Coverage  Builds  Tree fit   Retrain   Overall")
    for row in buffers["models"]:
        common = row["common"]
        accuracy = "undefined" if common["accuracy"] is None else f"{common['accuracy']:.2%}"
        times = row["median_seconds"]
        print(f"{row['buffer_size']:>6}  {accuracy:>9} ({common['rows']:>4})"
              f"  {row['coverage']:>8.2%}  {row['retrain_count']:>6}"
              f"  {times['tree_fit_s']:>7.4f}s  {times['retrain_s']:>7.4f}s  {times['overall_s']:>7.4f}s")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
