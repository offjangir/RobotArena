#!/usr/bin/env python3
"""
Bradley-Terry (BT) score visualisation.

Exponentiates BT scores to produce relative-skill bar charts with
asymmetric confidence-interval error bars.

Consolidates the former ``bt.py`` and ``bt1.py``.

Usage:
    python plot_bt_scores.py
    python plot_bt_scores.py --output_dir ./plots --output_name custom_bt.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (
    DEFAULT_DPI,
    DEFAULT_OUTPUT_DIR,
    add_legend,
    apply_style,
    get_policy_color,
    integer_y_formatter,
    style_axis,
)
from plot_utils import save_figure

# ============================================================================
# Default BT Data (update with new results as needed)
# ============================================================================

DEFAULT_BT_DATA = {
    "model":    ["Cogact", "RoboVLM", "SpatialVLA", "Octo"],
    "bt_score": [0.300938,  0.275418,  -0.284302,   -0.292054],
    "ci_lower": [0.264468,  0.238849,  -0.320521,   -0.328493],
    "ci_upper": [0.337408,  0.311987,  -0.248083,   -0.255614],
}

POLICY_ORDER = ["Octo", "SpatialVLA", "RoboVLM", "Cogact"]

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bt_csv", default=None,
                   help="CSV with columns: model, bt_score, ci_lower, ci_upper. "
                        "If omitted, built-in defaults are used.")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output_name", default="bt_scores.png")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    apply_style()

    # --- Load data ---
    if args.bt_csv:
        df = pd.read_csv(args.bt_csv)
    else:
        df = pd.DataFrame(DEFAULT_BT_DATA)

    # Exponentiate
    df["score_exp"]    = np.exp(df["bt_score"])
    df["ci_lower_exp"] = np.exp(df["ci_lower"])
    df["ci_upper_exp"] = np.exp(df["ci_upper"])
    df["err_lower"]    = df["score_exp"] - df["ci_lower_exp"]
    df["err_upper"]    = df["ci_upper_exp"] - df["score_exp"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bar_width = 0.5
    positions = np.arange(len(POLICY_ORDER))

    for i, name in enumerate(POLICY_ORDER):
        row = df[df["model"] == name].iloc[0]
        color = get_policy_color(name)

        ax.bar(
            positions[i], row["score_exp"],
            width=bar_width, color=color,
            yerr=[[row["err_lower"]], [row["err_upper"]]],
            capsize=5,
            error_kw=dict(linewidth=1.5, capthick=1.5),
            alpha=0.95, label=name, zorder=3,
        )
        ax.annotate(
            f"{row['score_exp']:.3f}",
            xy=(positions[i], row["score_exp"]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=13, zorder=4,
        )

    # Styling
    ax.grid(True, axis="y", linestyle=":", linewidth=1,
            color="lightgray", zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(POLICY_ORDER, fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"Relative Skill ($e^{\mathrm{BT \ score}}$)",
                   fontsize=16, labelpad=10)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(bottom=0)

    for spine in ax.spines.values():
        spine.set_visible(True)

    add_legend(ax, ncol=len(POLICY_ORDER), fontsize=16, bbox_y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, os.path.join(args.output_dir, args.output_name),
                dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
