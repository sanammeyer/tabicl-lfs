#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_RESULTS_PATH = Path("results/cc18_pdl-extratrees.csv")


def compute_metric_stats(
    csv_path: Path,
) -> Tuple[Dict[str, Tuple[float, float, int]], int]:
    """
    Compute mean and std (over successful runs) for all numeric columns.

    Returns:
        metrics: mapping metric_name -> (mean, std, count)
        n_successful: number of successful rows (error column empty)
    """
    sums: Dict[str, float] = {}
    sums_sq: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    # Columns that are clearly identifiers / non-metrics and should be skipped
    skip_columns = {"task_id", "dataset_id", "seed", "error"}

    n_successful = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in CSV file {csv_path}")

        fieldnames = reader.fieldnames

        for row in reader:
            error = row.get("error")
            # Keep only successful runs (no error message)
            if error not in (None, "", "nan", "NaN"):
                continue

            n_successful += 1

            for col in fieldnames:
                if col in skip_columns:
                    continue

                value = row.get(col)
                if value in (None, "", "nan", "NaN"):
                    continue

                try:
                    x = float(value)
                except ValueError:
                    # Non-numeric column, ignore
                    continue

                sums[col] = sums.get(col, 0.0) + x
                sums_sq[col] = sums_sq.get(col, 0.0) + x * x
                counts[col] = counts.get(col, 0) + 1

    if not counts:
        raise RuntimeError(f"No numeric metric columns found in {csv_path}")

    metrics: Dict[str, Tuple[float, float, int]] = {}
    for col, cnt in counts.items():
        mean = sums[col] / cnt
        if cnt > 1:
            # Sample standard deviation over successful runs
            variance = (sums_sq[col] - cnt * mean * mean) / (cnt - 1)
            std = variance**0.5
        else:
            std = 0.0

        metrics[col] = (mean, std, cnt)

    return metrics, n_successful


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        csv_path = Path(argv[0])
    else:
        csv_path = DEFAULT_RESULTS_PATH

    if not csv_path.is_file():
        raise SystemExit(f"Results file not found: {csv_path}")

    metric_stats, n = compute_metric_stats(csv_path)

    print(f"File: {csv_path}")
    print(f"Datasets used: {n}")

    # If present, highlight the main classification metrics first
    for key in ("accuracy_mean", "f1_macro_mean"):
        if key in metric_stats:
            mean, std, cnt = metric_stats[key]
            print(f"{key:20s} mean={mean:.6f} std={std:.6f} (n={cnt})")

    print("\nAll metrics (successful runs only):")
    for name in sorted(metric_stats.keys()):
        mean, std, cnt = metric_stats[name]
        print(f"{name:20s} mean={mean:.6f} std={std:.6f} (n={cnt})")


if __name__ == "__main__":
    main()
