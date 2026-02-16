#!/usr/bin/env python3
"""
Grouped bar chart comparing policies across perturbation tests.

Reads ``results.json`` files from:
    <base_dir>/<policy>/<test>/results.json

Generates one plot per metric (GVL_AM, GVL_T30, GVL_L30).

Usage:
    python bar_plot.py --base_dir ./eval_merged_latest
    python bar_plot.py --base_dir ./eval_merged_latest --metrics avg_last_10 --error_type std
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (
    DEFAULT_BAR_WIDTH,
    DEFAULT_DPI,
    DEFAULT_ERROR_TYPE,
    DEFAULT_FIG_SIZE,
    DEFAULT_METRICS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_ORDER,
    DEFAULT_TEST_ORDER,
    METRIC_REGISTRY,
    POLICY_REGISTRY,
    add_legend,
    apply_style,
    get_policy_color,
    get_policy_display,
    get_short_to_label_map,
    integer_y_formatter,
    style_axis,
)
from plot_utils import (
    auto_ylim,
    collect_results,
    plot_grouped_bars,
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
                   help="Root directory containing <policy>/<test>/results.json")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for saved plots (default: %(default)s)")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   choices=list(METRIC_REGISTRY.keys()),
                   help="Metrics to plot (default: all)")
    p.add_argument("--policies", nargs="+", default=None,
                   help="Policy dir names to include (default: all found)")
    p.add_argument("--tests", nargs="+", default=DEFAULT_TEST_ORDER,
                   help="Short test names to include (default: %(default)s)")
    p.add_argument("--error_type", choices=["sem", "std", "none"],
                   default=DEFAULT_ERROR_TYPE,
                   help="Error-bar type (default: %(default)s)")
    p.add_argument("--fig_size", nargs=2, type=float, default=list(DEFAULT_FIG_SIZE),
                   metavar=("W", "H"), help="Figure size in inches")
    p.add_argument("--bar_width", type=float, default=DEFAULT_BAR_WIDTH)
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

    # --- Map test directory names to short names ---
    from plot_config import get_dir_to_short_test_map
    df["test"] = df["test"].replace(get_dir_to_short_test_map())

    # --- Filter ---
    policy_order = args.policies or [p for p in DEFAULT_POLICY_ORDER if p in df["policy"].unique()]
    df = df[df["test"].isin(args.tests) & df["policy"].isin(policy_order)]
    df.dropna(subset=args.metrics, how="all", inplace=True)

    if df.empty:
        print("No matching data after filtering. Exiting.")
        return

    # Derived look-ups
    label_map = get_short_to_label_map()
    test_labels = [label_map.get(t, t) for t in args.tests]
    policy_colors = {p: get_policy_color(p) for p in policy_order}
    policy_display = {p: get_policy_display(p) for p in policy_order}

    # --- Plot each metric ---
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipping '{metric}' (not in data).")
            continue

        meta = METRIC_REGISTRY[metric]

        pivot = df.pivot_table(index="policy", columns="test", values=metric)
        pivot = pivot.reindex(index=policy_order, columns=args.tests)
        pivot.index = [policy_display.get(p, p) for p in pivot.index]

        # Error bars
        err_col = f"{metric}_{args.error_type}" if args.error_type != "none" else None
        err_pivot = None
        if err_col and err_col in df.columns:
            err_pivot = df.pivot_table(index="policy", columns="test", values=err_col)
            err_pivot = err_pivot.reindex(index=policy_order, columns=args.tests)
            err_pivot.index = pivot.index

        colors = {policy_display.get(p, p): get_policy_color(p) for p in policy_order}

        fig, ax = plt.subplots(figsize=tuple(args.fig_size))

        x_pos = plot_grouped_bars(
            ax, pivot, colors,
            bar_width=args.bar_width,
            group_spacing=1.2,
            error_pivot=err_pivot,
        )

        # X-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(test_labels, fontsize=20)

        # Y-axis
        ymin, ymax = auto_ylim(pivot)
        ax.set_ylim(ymin, ymax)
        integer_y_formatter(ax)

        # Styling
        style_axis(ax, ylabel="VLM Score", ylabel_size=25, ytick_size=16)
        add_legend(ax, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        # Save
        fname = f"{sanitize_filename(meta['title'])}_grouped_bar_annotated_scaled.png"
        save_figure(fig, os.path.join(args.output_dir, fname), dpi=args.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
