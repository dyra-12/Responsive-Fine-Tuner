#!/usr/bin/env python3

"""Generate a README-ready stability–plasticity table and an optional plot.

Usage:
  python scripts/stability_plasticity_report.py \
    --results data/experiments/stability_plasticity/results.csv \
    --out-html data/experiments/stability_plasticity/plot.html

The script prints a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


def _format_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(no results yet)"

    view = df.copy()
    if len(view) > max_rows:
        view = view.tail(max_rows)

    for col in ("stability_accuracy", "plasticity_accuracy"):
        if col in view.columns:
            view[col] = view[col].apply(
                lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
            )

    cols = [
        c
        for c in [
            "cycle",
            "feedback_samples",
            "stability_accuracy",
            "plasticity_accuracy",
            "learning_rate",
            "timestamp",
        ]
        if c in view.columns
    ]

    return view[cols].to_markdown(index=False)


def _write_plot_html(df: pd.DataFrame, out_html: str) -> None:
    import plotly.express as px

    view = df.copy()
    # ensure numeric
    for col in ("stability_accuracy", "plasticity_accuracy"):
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce")

    fig = px.line(
        view,
        x="cycle",
        y=["stability_accuracy", "plasticity_accuracy"],
        markers=True,
        title="Stability vs Plasticity Across Feedback Cycles",
    )
    fig.update_layout(yaxis=dict(range=[0, 1]))

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="data/experiments/stability_plasticity/results.csv",
        help="Path to results CSV (written by the app)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Max rows to include in the printed Markdown table",
    )
    p.add_argument(
        "--out-html",
        default=None,
        help="Optional: write an interactive plot to this HTML path",
    )

    args = p.parse_args(argv)

    if not os.path.exists(args.results):
        print(f"No results found at: {args.results}")
        return 1

    df = pd.read_csv(args.results)

    print(_format_markdown_table(df, max_rows=args.max_rows))

    if args.out_html:
        _write_plot_html(df, args.out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
