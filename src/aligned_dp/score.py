"""CLI for computing Metric Acceptability scores over privatized text CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .metric_acceptability import (
    Evaluator,
    ScoreAggregator,
    BERTScoreAdapter,
    BLEURTAdapter,
    PerplexityAdapter,
    CoLAAdapter,
    DCRSAdapter,
    CLIAdapter,
    compute_per_dataset_stats,
)

# <config>_eps<int>_<dataset>.csv
_FILENAME_RE = re.compile(r"^(.*)_eps(\d+(?:\.\d+)?)_(.*?)\.csv$")


def parse_filename_metadata(filename: str) -> tuple[str, float, str]:
    """Extract (config, epsilon, dataset) from a privatize.py output filename."""
    m = _FILENAME_RE.match(filename)
    if not m:
        return "unknown", 0.0, "unknown"
    return m.group(1), float(m.group(2)), m.group(3)




_PRIV_METRICS = [
    BERTScoreAdapter(),
    BLEURTAdapter(),
    PerplexityAdapter(),
    CoLAAdapter(),
    DCRSAdapter(),
    CLIAdapter(),
]

_ORIG_METRICS = [
    PerplexityAdapter(),
    CoLAAdapter(),
    DCRSAdapter(),
    CLIAdapter(),
]

_FACETS = {
    "Utility": ["bertscore", "bleurt"],
    "Struct": ["ppl", "cola"],
    "Readability": ["dcrs", "cli"],
}

_RAW_METRIC_COLS = [f"raw_{m.name}" for m in _PRIV_METRICS]
_ORIG_METRIC_COLS = [f"orig_{m.name}" for m in _ORIG_METRICS]
_EXTERNAL_BASELINE_COLS = ["original_text"] + [f"{m.name}_baseline" for m in _ORIG_METRICS]

_OUTPUT_COLS = [
    "config", "epsilon", "dataset", "original_text", "privatized_text",
] + _RAW_METRIC_COLS + [f"Z_{m.name}" for m in _PRIV_METRICS] + [
    f"Score_{f}" for f in _FACETS
] + ["Metric_Acceptability"]




def compute_cmd(args: argparse.Namespace) -> None:
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {args.input_dir}")

    evaluator = Evaluator()

    for path in tqdm(files, desc="Files"):
        cache_path = args.cache_dir / f"{path.stem}_metrics.csv"
        if cache_path.exists() and not args.force:
            continue

        config, epsilon, dataset = parse_filename_metadata(path.name)
        df = pd.read_csv(path)
        df["original_text"] = df["original_text"].fillna("").astype(str)
        df["privatized_text"] = df["privatized_text"].fillna("").astype(str)

        priv = df["privatized_text"].tolist()
        orig = df["original_text"].tolist()


        priv_results = evaluator.evaluate(_PRIV_METRICS, priv, orig)
        for name, vals in priv_results.items():
            df[f"raw_{name}"] = vals


        orig_results = evaluator.evaluate(_ORIG_METRICS, orig)
        for name, vals in orig_results.items():
            df[f"orig_{name}"] = vals

        df["config"] = config
        df["epsilon"] = epsilon
        df["dataset"] = dataset

        df.to_csv(cache_path, index=False)




def aggregate_cmd(args: argparse.Namespace) -> None:
    df = _load_cached_metrics(args.cache_dir)
    _coerce_numeric(df, _RAW_METRIC_COLS + _ORIG_METRIC_COLS)

    if args.baseline_dir is not None:
        baseline_stats = _baseline_stats_from_external(df, args.baseline_dir)
    else:
        baseline_stats = _baseline_stats_from_cache(df)

    # Normalize semantic metrics against batch distribution.
    semantic_stats = compute_per_dataset_stats(df, {
        "raw_bertscore": "bertscore",
        "raw_bleurt": "bleurt",
    })
    baseline_stats = baseline_stats.merge(semantic_stats, on="dataset", how="outer")

    aggregator = ScoreAggregator(_PRIV_METRICS, _FACETS, clip=args.clip)
    scored = aggregator.aggregate(df, baseline_stats)

    print()
    print("NaN counts (lower is better; non-zero indicates merge issues):")
    for facet in _FACETS:
        print(f"  Score_{facet:15s}: {scored[f'Score_{facet}'].isna().sum()}")

    final = scored[[c for c in _OUTPUT_COLS if c in scored.columns]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)

    print(f"\nWrote {len(final):,} rows to {args.output}")
    print("\nMean Metric_Acceptability by (config, epsilon):")
    print(final.groupby(["config", "epsilon"])["Metric_Acceptability"].mean().to_string())




def _load_cached_metrics(cache_dir: Path) -> pd.DataFrame:
    cache_files = sorted(cache_dir.glob("*_metrics.csv"))
    if not cache_files:
        sys.exit(f"No cached metric files found in {cache_dir}")

    print(f"Loading {len(cache_files)} cached metric files")
    parts = []
    for path in tqdm(cache_files, desc="Loading"):
        parts.append(pd.read_csv(
            path,
            engine="python",
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
        ))
    return pd.concat(parts, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _baseline_stats_from_cache(df: pd.DataFrame) -> pd.DataFrame:
    unique = df.drop_duplicates(subset=["dataset", "original_text"])
    metric_cols = {f"orig_{m.name}": m.name for m in _ORIG_METRICS}
    return compute_per_dataset_stats(unique, metric_cols)


def _baseline_stats_from_external(
    df: pd.DataFrame,
    baseline_dir: Path,
) -> pd.DataFrame:
    baseline_files = sorted(baseline_dir.glob("baseline_stats_*.csv"))
    if not baseline_files:
        print(f"No baseline_stats_*.csv in {baseline_dir}; falling back to "
              f"orig_ columns from cache.", file=sys.stderr)
        return _baseline_stats_from_cache(df)

    df_base = pd.concat(
        [pd.read_csv(f, engine="python", on_bad_lines="warn") for f in baseline_files],
        ignore_index=True,
    )

    missing = set(_EXTERNAL_BASELINE_COLS) - set(df_base.columns)
    if missing:
        sys.exit(f"Baseline files in {baseline_dir} are missing columns: {missing}")

    df["original_text"] = df["original_text"].astype(str).str.strip()
    df_base["original_text"] = df_base["original_text"].astype(str).str.strip()
    df_base = df_base[_EXTERNAL_BASELINE_COLS].drop_duplicates(subset=["original_text"])

    merged = df.merge(df_base, on="original_text", how="left")
    matched = int(merged[f"{_ORIG_METRICS[0].name}_baseline"].notna().sum())
    total = len(merged)
    print(f"External baseline merge: {matched:,}/{total:,} rows matched.")
    if matched == 0:
        sys.exit("No rows matched the external baselines; check text formatting.")

    unique = merged.drop_duplicates(subset=["dataset", "original_text"])
    metric_cols = {f"{m.name}_baseline": m.name for m in _ORIG_METRICS}
    return compute_per_dataset_stats(unique, metric_cols)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compute = sub.add_parser(
        "compute",
        help="Compute raw metrics for each input CSV and cache to disk.",
    )
    compute.add_argument("--input-dir", type=Path, required=True,
                         help="Directory of privatized text CSVs from privatize.py.")
    compute.add_argument("--cache-dir", type=Path, required=True,
                         help="Where to write per-file cached metric CSVs.")
    compute.add_argument("--force", action="store_true",
                         help="Re-compute even if a cache file already exists.")
    compute.set_defaults(func=compute_cmd)

    aggregate = sub.add_parser(
        "aggregate",
        help="Normalize cached metrics and write final acceptability scores.",
    )
    aggregate.add_argument("--cache-dir", type=Path, required=True,
                           help="Directory of cached metric CSVs from `compute`.")
    aggregate.add_argument("--output", type=Path, required=True,
                           help="Path to the final scores CSV.")
    aggregate.add_argument("--baseline-dir", type=Path, default=None,
                           help="Optional directory of baseline_stats_*.csv files. "
                                "If omitted, baselines are derived from the orig_ "
                                "columns inside the cache.")
    aggregate.add_argument("--clip", type=float, default=3.0,
                           help="Z-score clip range [-clip, +clip] (default 3.0).")
    aggregate.set_defaults(func=aggregate_cmd)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
