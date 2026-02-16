#!/usr/bin/env python3
"""
Simple bar chart for the *base test only* (default_test / BridgeSim).

Reads ``metrics_zero_shot.json`` from:
    <base_dir>/<policy>/default_test/metrics_zero_shot.json

Generates one plot per metric.

Usage:
    python bar_plot_default.py --base_dir ./eval_merged_latest
    python bar_plot_default.py --base_dir ./eval_merged_latest --policies octo spatial robovlm cogact
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from plot_config import (
    DEFAULT_DPI,
    DEFAULT_METRICS,
    DEFAULT_OUTPUT_DIR,
    METRIC_REGISTRY,
    add_legend,
    apply_style,
    get_policy_color,
    get_policy_display,
    integer_y_formatter,
    style_axis,
)
from plot_utils import (
    auto_ylim,
    collect_results,
    sanitize_filename,
    save_figure,
)

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", type=str, required=True,
                   help="Root dir containing <policy>/default_test/metrics_zero_shot.json")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   choices=list(METRIC_REGISTRY.keys()))
    p.add_argument("--policies", nargs="+",
                   default=["octo", "spatial", "robovlm", "cogact"],
                   help="Policy dir names (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    apply_style()

    # --- Load data ---
    df = collect_results(args.base_dir, filename="results.json")
    if df.empty:
        print("No data found. Exiting.")
        return

    # Keep only default_test
    df = df[df["test"] == "default_test"].copy()
    df = df[df["policy"].isin(args.policies)]

    if df.empty:
        print("No base-test data for selected policies. Exiting.")
        return

    # --- Plot each metric ---
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipping '{metric}' (not in data).")
            continue

        meta = METRIC_REGISTRY[metric]
        plot_data = df.set_index("policy")[metric].reindex(args.policies)

        fig, ax = plt.subplots(figsize=(7, 6))
        display_names = [get_policy_display(p) for p in plot_data.index]
        colors = [get_policy_color(p) for p in plot_data.index]

        bars = ax.bar(display_names, plot_data.values, color=colors,
                      alpha=0.9, width=0.55)

        # Annotate
        for bar in bars:
            h = bar.get_height()
            if pd.notna(h):
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12.5,
                )

        # Styling
        ax.set_title(meta["title"], fontsize=16, pad=20)
        ax.set_xlabel("Policy", fontsize=16, labelpad=10)
        style_axis(ax, ylabel=meta["ylabel"], ylabel_size=16, ytick_size=14)
        ax.set_xticklabels(display_names, fontsize=16)

        valid = plot_data.dropna().values
        if valid.size > 0:
            ax.set_ylim(valid.min() * 0.95, valid.max() * 1.035)
        else:
            ax.set_ylim(0, 100)
        integer_y_formatter(ax, math_mode=False)

        fig.tight_layout()
        fname = f"{sanitize_filename(meta['title'])}_base_test.png"
        save_figure(fig, os.path.join(args.output_dir, fname), dpi=args.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
