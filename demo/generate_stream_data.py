"""Regenerate the seeded numeric CSV used in the Assignment 2.2 demo."""

import argparse
import csv

import numpy as np

from .stream_workflow import DATA_PATH, SEED


def generate(path, *, rows=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    signal, context = rng.normal(size=(2, rows))
    categories = rng.integers(0, 2, size=rows)
    for start, category in ((3 * rows // 10, 2), (7 * rows // 10, 3)):
        suffix = categories[start:]
        suffix[rng.random(suffix.size) < 0.25] = category
    after_drift = np.arange(rows) >= rows // 2
    score = np.where(after_drift, 0.65 * signal + 0.85 * context, 1.10 * signal - 0.65 * context)
    score += np.array([-0.35, 0.30, 0.70, -0.60])[categories] + rng.normal(0, 0.35, rows)
    targets = (score >= 0).astype(int)
    missing_signal = rng.random(rows) < 0.07
    missing_context = rng.random(rows) < 0.05
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["signal", "context", "category_code", "target"])
        writer.writerows(
            (
                "" if missing_x else f"{x:.6f}",
                "" if missing_z else f"{z:.6f}",
                category,
                target,
            )
            for x, z, category, target, missing_x, missing_z
            in zip(signal, context, categories, targets, missing_signal, missing_context)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DATA_PATH)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
