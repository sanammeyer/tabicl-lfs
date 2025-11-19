#!/usr/bin/env python
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple


DEFAULT_RESULTS_PATH = Path("results/cc18_tabicl_step-1000_ea.csv")


def compute_average_metrics(csv_path: Path) -> Tuple[float, float, int]:
    """Compute average accuracy_mean and f1_macro_mean over successful runs."""
    accuracies: List[float] = []
    f1_macros: List[float] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            error = row.get("error")
            # Keep only successful runs (no error message)
            if error not in (None, "", "nan", "NaN"):
                continue

            try:
                acc = float(row["accuracy_mean"])
                f1 = float(row["f1_macro_mean"])
            except (KeyError, ValueError):
                # Skip rows that do not have the expected metrics
                continue

            accuracies.append(acc)
            f1_macros.append(f1)

    if not accuracies or not f1_macros:
        raise RuntimeError(
            "No valid rows with 'accuracy_mean' and 'f1_macro_mean' found "
            f"in {csv_path}"
        )

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_f1_macro = sum(f1_macros) / len(f1_macros)
    return avg_accuracy, avg_f1_macro, len(accuracies)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        csv_path = Path(argv[0])
    else:
        csv_path = DEFAULT_RESULTS_PATH

    if not csv_path.is_file():
        raise SystemExit(f"Results file not found: {csv_path}")

    avg_acc, avg_f1, n = compute_average_metrics(csv_path)
    print(f"File: {csv_path}")
    print(f"Datasets used: {n}")
    print(f"Average accuracy_mean:  {avg_acc:.4f}")
    print(f"Average f1_macro_mean: {avg_f1:.4f}")


if __name__ == "__main__":
    main()

