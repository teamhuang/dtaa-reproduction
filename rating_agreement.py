"""Recalculate dimension-specific agreement from two complete item-level CSV forms."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = ("D1_timing_class", "D3_causal_role", "D4_reproducibility")


def load_form(path):
    form = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["variable_id", *DIMENSIONS]
    form = form.loc[:, required].apply(lambda column: column.str.strip())
    if form.eq("").any().any() or form.variable_id.duplicated().any():
        raise ValueError(f"{path}: expected one complete row per variable_id")
    return form.set_index("variable_id")


def agreement(a, b, categories):
    observed = np.mean(a == b)
    expected = np.dot(np.bincount(a, minlength=categories) / len(a),
                      np.bincount(b, minlength=categories) / len(b))
    # Match the archived protocol's perfect-agreement convention for a constant resample.
    return (1.0 if expected == 1 else (observed - expected) / (1 - expected)), observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--ref", type=Path, help="Optional second comparison form")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    rater = load_form(args.rater)
    if len(rater) < 2:
        parser.error("At least two items are required")
    rng = np.random.default_rng(args.seed)
    results = []
    comparisons = [("vs answer key", args.key)]
    if args.ref:
        comparisons.append(("vs original rater", args.ref))
    for label, path in comparisons:
        other = load_form(path)
        if set(rater.index) != set(other.index):
            parser.error(f"{path}: variable_id sets differ")
        other = other.loc[rater.index]
        for dim in DIMENSIONS:
            levels, codes = np.unique(np.r_[rater[dim], other[dim]], return_inverse=True)
            a, b = np.split(codes, 2)
            kappa, observed = agreement(a, b, len(levels))
            estimates = []
            for _ in range(args.n_boot):
                sample = rng.choice(len(a), size=len(a), replace=True)
                estimates.append(agreement(a[sample], b[sample], len(levels))[0])
            low, high = np.quantile(estimates, [0.025, 0.975])
            results.append([label, dim, kappa, observed, low, high, len(a)])
    columns = ["comparison", "dimension", "kappa", "agreement", "ci_low", "ci_high", "n_items"]
    print(pd.DataFrame(results, columns=columns).to_csv(index=False, float_format="%.3f"), end="")


if __name__ == "__main__":
    main()
