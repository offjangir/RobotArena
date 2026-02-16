#!/usr/bin/env python3
"""
Multi-dataset comparison plots.

Generates two figures:
  1. **Dataset comparison** -- grouped bars for each dataset (BridgeSim,
     DroidSim, RH20TSim) with policies as bars, base-test only.
  2. **Test-condition comparison** -- grouped bars for each perturbation
     test with policies as bars (BridgeSim data only).

Consolidates the former ``plot_dataset_Perturbation.py`` and
``multidataset_plot.py``.

Usage:
    python plot_multi_dataset.py \\
        --base_dirs ./eval_merged ./eval_droid ./eval_rh20t \\
        --dataset_names BridgeSim DroidSim RH20TSim
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (
    DEFAULT_DPI,
    DEFAULT_ERROR_TYPE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_ORDER,
    DEFAULT_TEST_ORDER,
    METRIC_REGISTRY,
    add_legend,
    apply_style,
    get_policy_display,
    get_policy_color,
    get_short_to_label_map,
    integer_y_formatter,
    style_axis,
)
from plot_utils import (
    apply_name_mappings,
    auto_ylim,
    collect_results,
    plot_grouped_bars,
    sanitize_filename,
    save_figure,
)

# ============================================================================
# Defaults
# ============================================================================

FONT_SIZES = {"ylabel": 22, "xtick": 20, "ytick": 18, "legend": 20, "annotate": 12}
FIG_SIZE = (12, 8)

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dirs", nargs="+", required=True,
                   help="Base directories, one per dataset")
    p.add_argument("--dataset_names", nargs="+", required=True,
                   help="Display names, one per --base_dirs entry")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--metric", type=str, default="avg_last_10",
                   choices=list(METRIC_REGISTRY.keys()))
    p.add_argument("--error_type", choices=["sem", "std", "none"],
                   default=DEFAULT_ERROR_TYPE)
    p.add_argument("--policies", nargs="+", default=None,
                   help="Policies to include (display names). Default: all 6.")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args()


# ============================================================================
# Plot: Dataset Comparison
# ============================================================================

def create_dataset_comparison_plot(df, metric, error_type, policies,
                                   dataset_order, output_dir, dpi):
    """Grouped bars: x-axis = datasets, bars = policies (base test only)."""
    df_base = df[df["test"] == "base"].copy()

    pivot = df_base.pivot_table(index="dataset", columns="policy", values=metric)
    pivot = pivot.reindex(index=dataset_order, columns=policies)

    err_pivot = None
    err_col = f"{metric}_{error_type}" if error_type != "none" else None
    if err_col and err_col in df_base.columns:
        err_pivot = df_base.pivot_table(index="dataset", columns="policy", values=err_col)
        err_pivot = err_pivot.reindex(index=dataset_order, columns=policies)

    colors = {p: get_policy_color(p) for p in policies}

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x_pos = plot_grouped_bars(
        ax, pivot.T,   # transpose: index=policies, columns=datasets
        colors, bar_width=0.08, group_spacing=0.65,
        error_pivot=err_pivot.T if err_pivot is not None else None,
        capsize=4, annot_fontsize=FONT_SIZES["annotate"],
    )
    # Re-draw because pivot is transposed for plot_grouped_bars but we want
    # datasets on x-axis.  Easier to just build it directly:
    ax.clear()

    num_policies = len(policies)
    bar_width = 0.08
    x_pos = np.arange(len(dataset_order)) * 0.65

    for i, pol in enumerate(policies):
        scores = pivot[pol].values
        yerr = err_pivot[pol].values if err_pivot is not None else None
        pos = x_pos - (bar_width * num_policies / 2) + (i * bar_width) + bar_width / 2
        bars = ax.bar(pos, scores, bar_width, color=colors[pol], alpha=0.9,
                       label=pol, yerr=yerr, capsize=4,
                       error_kw=dict(linewidth=1, capthick=1))
        for bar, s in zip(bars, scores):
            if pd.notna(s):
                ax.annotate(f"{s:.2f}", xy=(bar.get_x() + bar.get_width() / 2, s),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=FONT_SIZES["annotate"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_order, fontsize=FONT_SIZES["xtick"])
    style_axis(ax, ylabel="Task Progression Score (%)",
               ylabel_size=FONT_SIZES["ylabel"], ytick_size=FONT_SIZES["ytick"])
    ymin, ymax = auto_ylim(pivot)
    ax.set_ylim(ymin, ymax)
    integer_y_formatter(ax)
    add_legend(ax, ncol=num_policies, fontsize=FONT_SIZES["legend"], bbox_y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_figure(fig, os.path.join(output_dir, "datasets_comparison_plot.png"), dpi=dpi)
    plt.close(fig)


# ============================================================================
# Plot: Test-Condition Comparison (BridgeSim data only)
# ============================================================================

def create_test_comparison_plot(df, metric, error_type, policies,
                                test_order, output_dir, dpi):
    """Grouped bars: x-axis = test conditions, bars = policies."""
    label_map = get_short_to_label_map()
    test_labels = [label_map.get(t, t) for t in test_order]
    colors = {p: get_policy_color(p) for p in policies}

    df_filt = df[df["test"].isin(test_order)].copy()
    pivot = df_filt.pivot_table(index="policy", columns="test", values=metric)
    pivot = pivot.reindex(index=policies, columns=test_order)

    err_pivot = None
    err_col = f"{metric}_{error_type}" if error_type != "none" else None
    if err_col and err_col in df_filt.columns:
        err_pivot = df_filt.pivot_table(index="policy", columns="test", values=err_col)
        err_pivot = err_pivot.reindex(index=policies, columns=test_order)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    num_policies = len(policies)
    bar_width = 0.20
    x_pos = np.arange(len(test_order))

    for i, pol in enumerate(policies):
        scores = pivot.loc[pol].values
        yerr = err_pivot.loc[pol].values if err_pivot is not None else None
        pos = x_pos - (bar_width * num_policies / 2) + (i * bar_width) + bar_width / 2
        bars = ax.bar(pos, scores, bar_width, color=colors[pol], alpha=0.9,
                       label=pol, yerr=yerr, capsize=4,
                       error_kw=dict(linewidth=1, capthick=1))
        for bar, s in zip(bars, scores):
            if pd.notna(s):
                ax.annotate(f"{s:.2f}", xy=(bar.get_x() + bar.get_width() / 2, s),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=FONT_SIZES["annotate"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_labels, fontsize=FONT_SIZES["xtick"])
    style_axis(ax, ylabel="Task Progression Score (%)",
               ylabel_size=FONT_SIZES["ylabel"], ytick_size=FONT_SIZES["ytick"])
    ymin, ymax = auto_ylim(pivot)
    ax.set_ylim(ymin, ymax)
    integer_y_formatter(ax)
    add_legend(ax, ncol=num_policies, fontsize=FONT_SIZES["legend"], bbox_y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_figure(fig, os.path.join(output_dir, "tests_comparison_plot.png"), dpi=dpi)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    apply_style()

    if len(args.base_dirs) != len(args.dataset_names):
        raise ValueError("--base_dirs and --dataset_names must have the same length.")

    # --- Load and merge all datasets ---
    frames = []
    for bdir, dname in zip(args.base_dirs, args.dataset_names):
        df = collect_results(bdir, filename="results.json")
        if df.empty:
            print(f"Warning: no data from {bdir}")
            continue
        df = apply_name_mappings(df)
        df["dataset"] = dname
        frames.append(df)

    if not frames:
        print("No data loaded. Exiting.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    df_all.dropna(subset=[args.metric], inplace=True)

    # Determine policy list
    if args.policies:
        policies = args.policies
    else:
        ordered = [get_policy_display(p) for p in DEFAULT_POLICY_ORDER]
        policies = [p for p in ordered if p in df_all["policy"].unique()]

    # --- Dataset comparison (all datasets, base test) ---
    create_dataset_comparison_plot(
        df_all, args.metric, args.error_type, policies,
        args.dataset_names, args.output_dir, args.dpi,
    )

    # --- Test-condition comparison (first dataset only) ---
    df_first = df_all[df_all["dataset"] == args.dataset_names[0]].copy()
    # For test comparison, limit to the 4 core policies by default
    test_policies = [p for p in policies if p in df_first["policy"].unique()]
    create_test_comparison_plot(
        df_first, args.metric, args.error_type, test_policies,
        DEFAULT_TEST_ORDER, args.output_dir, args.dpi,
    )


if __name__ == "__main__":
    main()
