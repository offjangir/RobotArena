#!/usr/bin/env python3
"""
BridgeSim vs SimplerSim (or any two simulator environments) comparison.

Reads ``results.json`` from the ``default_test`` directory of each base
directory, then produces a grouped bar chart with simulators on the x-axis
and policies as bars.

Replaces the former ``bb4vbball.py``.

Usage:
    python plot_sim_comparison.py \\
        --base_dir_a ./eval_merged --name_a "BridgeSim(70 Envs)" \\
        --base_dir_b ./eval_iclr_default --name_b "SimplerSim(4 Envs)"
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
    add_legend,
    apply_style,
    get_policy_color,
    get_policy_display,
    get_policy_name_mapping,
    integer_y_formatter,
    style_axis,
)
from plot_utils import (
    collect_single_test,
    save_figure,
)

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir_a", required=True, help="First simulator base dir")
    p.add_argument("--name_a", required=True, help="Display name for first simulator")
    p.add_argument("--base_dir_b", required=True, help="Second simulator base dir")
    p.add_argument("--name_b", required=True, help="Display name for second simulator")
    p.add_argument("--metric", default="avg_last_10",
                   help="Metric to compare (default: %(default)s)")
    p.add_argument("--policies", nargs="+", default=None,
                   help="Policy dir names (default: all found)")
    p.add_argument("--error_type", choices=["sem", "std", "none"],
                   default=DEFAULT_ERROR_TYPE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output_name", default=None,
                   help="Output filename (auto-generated if omitted)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    apply_style()

    name_map = get_policy_name_mapping()
    datasets = [args.name_b, args.name_a]  # SimplerSim first, then BridgeSim

    # --- Collect data ---
    df_a = collect_single_test(args.base_dir_a, metric=args.metric,
                               dataset_name=args.name_a)
    df_b = collect_single_test(args.base_dir_b, metric=args.metric,
                               dataset_name=args.name_b)

    if df_a.empty or df_b.empty:
        print("Missing data from one or both directories. Exiting.")
        return

    # Map directory names to display names
    df_a["Policy"] = df_a["Policy"].map(name_map)
    df_b["Policy"] = df_b["Policy"].map(name_map)

    df = pd.merge(df_b, df_a, on="Policy")

    # Determine policy order
    if args.policies:
        policy_order = [get_policy_display(p) for p in args.policies]
    else:
        all_display = [get_policy_display(p) for p in DEFAULT_POLICY_ORDER]
        policy_order = [p for p in all_display if p in df["Policy"].values]

    df["Policy"] = pd.Categorical(df["Policy"], categories=policy_order, ordered=True)
    df = df.sort_values("Policy").reset_index(drop=True)
    df = df[df["Policy"].notna()].reset_index(drop=True)

    # --- Plotting ---
    num_policies = len(df)
    num_datasets = len(datasets)
    fig, ax = plt.subplots(figsize=(10, 6))

    total_group_width = 0.8
    bar_width = total_group_width / num_policies
    x = np.arange(num_datasets)

    for i, policy in enumerate(df["Policy"]):
        color = get_policy_color(str(policy))
        offset = total_group_width / 2
        pos = x - offset + i * bar_width + bar_width / 2
        heights = df.loc[i, datasets].values.astype(float)

        yerr = None
        if args.error_type != "none":
            yerr = [df.loc[i, f"{ds}_{args.error_type}"] for ds in datasets]

        ax.bar(pos, heights, bar_width, color=color, alpha=1.0, label=str(policy),
               yerr=yerr, capsize=4, error_kw=dict(linewidth=1, capthick=1))

        for xp, h in zip(pos, heights):
            if pd.notna(h):
                ax.annotate(f"{h:.2f}", xy=(xp, h), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=12.5)

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16)
    style_axis(ax, ylabel="Task Progression Score (%)", ylabel_size=16,
               ytick_size=14)

    all_vals = df[datasets].values.flatten()
    all_vals = all_vals[~pd.isna(all_vals)]
    if all_vals.size > 0:
        ax.set_ylim(all_vals.min() * 0.65, all_vals.max() * 1.04)
    integer_y_formatter(ax, math_mode=False)

    add_legend(ax, ncol=num_policies, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fname = args.output_name or f"GVL_L30_{args.name_a.split('(')[0]}_vs_{args.name_b.split('(')[0]}.png"
    save_figure(fig, os.path.join(args.output_dir, fname), dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
