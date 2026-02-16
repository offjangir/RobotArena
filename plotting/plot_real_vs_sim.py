#!/usr/bin/env python3
"""
Real World vs Simulation comparison plots.

Reads ``metrics_zero_shot.json`` from:
    <base_dir>/<policy>/<test>/metrics_zero_shot.json

where <test> is expected to be ``real`` and ``sim``.

Generates one plot per metric (GVL_AM, GVL_T30, GVL_L30).

Consolidates the former ``plot_realsim.py`` and the real/sim portion of
``plot_Real_sim_RobArena.py``.

Usage:
    python plot_real_vs_sim.py --base_dir ./eval_realsim
    python plot_real_vs_sim.py --base_dir ./eval_realsim --policies octo spatial robovlm
"""

import argparse
import os

import matplotlib.pyplot as plt
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
    get_policy_name_mapping,
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
# Constants
# ============================================================================

TEST_ORDER = ["real", "sim"]
LABEL_MAP = {"real": "Real World", "sim": "Simulation Environment"}
DEFAULT_POLICIES = ["octo", "spatial", "robovlm"]

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", required=True,
                   help="Root dir with <policy>/<real|sim>/metrics_zero_shot.json")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   choices=list(METRIC_REGISTRY.keys()))
    p.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES,
                   help="Policy dir names (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    apply_style()

    # --- Load ---
    df = collect_results(args.base_dir, filename="metrics_zero_shot.json")
    if df.empty:
        print("No data found. Exiting.")
        return

    # Map policy names
    name_map = get_policy_name_mapping()
    df["policy"] = df["policy"].replace(name_map)

    policy_display = [get_policy_display(p) for p in args.policies]
    df = df[df["test"].isin(TEST_ORDER) & df["policy"].isin(policy_display)]
    df.dropna(subset=args.metrics, how="all", inplace=True)

    if df.empty:
        print("No matching data after filtering. Exiting.")
        return

    test_labels = [LABEL_MAP[t] for t in TEST_ORDER]
    colors = {p: get_policy_color(p) for p in policy_display}

    # --- Plot each metric ---
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipping '{metric}' (not in data).")
            continue

        meta = METRIC_REGISTRY[metric]
        pivot = df.pivot_table(index="policy", columns="test", values=metric)
        pivot = pivot.reindex(index=policy_display, columns=TEST_ORDER)

        fig, ax = plt.subplots(figsize=(9, 7))
        num_policies = len(pivot.index)
        total_group_width = 0.47
        bar_width = total_group_width / num_policies
        group_spacing = 0.7
        x_pos = np.arange(len(TEST_ORDER)) * group_spacing

        for i, pol in enumerate(pivot.index):
            scores = pivot.loc[pol]
            offset = total_group_width / 2
            pos = x_pos - offset + (i * bar_width) + bar_width / 2

            bars = ax.bar(pos, scores, bar_width, color=colors.get(pol, "gray"),
                          alpha=0.9, label=pol)

            for bar in bars:
                h = bar.get_height()
                if pd.notna(h):
                    ax.annotate(f"{h:.2f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=14)

        # Axis styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(test_labels, fontsize=16)
        style_axis(ax, ylabel="Task Progression Score (%)",
                   ylabel_size=16, ytick_size=14)

        # x-limits to centre the groups
        plot_center = x_pos.max() / 2
        plot_width = x_pos.max() + total_group_width + group_spacing * 0.4
        ax.set_xlim(plot_center - plot_width / 2, plot_center + plot_width / 2)

        ymin, ymax = auto_ylim(pivot, lower_factor=0.95, upper_factor=1.05)
        ax.set_ylim(ymin, ymax)
        integer_y_formatter(ax)

        add_legend(ax, ncol=num_policies, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f"{sanitize_filename(meta['title'])}_realsim.png"
        save_figure(fig, os.path.join(args.output_dir, fname), dpi=args.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
