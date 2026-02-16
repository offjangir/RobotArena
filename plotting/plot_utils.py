"""
Shared utility functions for RobotArena plotting scripts.

Provides data-loading helpers for both ``results.json`` (with stds/sems)
and ``metrics_zero_shot.json`` (means only), plus reusable plotting
primitives (bar groups, annotations, auto y-limits).
"""

import json
import os

import numpy as np
import pandas as pd

from plot_config import (
    get_dir_to_short_test_map,
    get_policy_name_mapping,
)

# ============================================================================
# Data Collection
# ============================================================================

def collect_results(base_dir: str, filename: str = "results.json") -> pd.DataFrame:
    """
    Walk ``base_dir/<policy>/<test>/`` and load *filename* from each leaf.

    For ``results.json`` the loader extracts ``averaged``, ``stds``, and
    ``sems`` sections.  For ``metrics_zero_shot.json`` it extracts only
    ``averaged``.  Any other JSON layout is loaded the same way as
    ``metrics_zero_shot.json``.

    Returns a flat DataFrame with columns:
        policy, test, <metric_keys>, [<metric_keys>_std, <metric_keys>_sem]
    """
    if not os.path.isdir(base_dir):
        print(f"Warning: directory not found: '{base_dir}'")
        return pd.DataFrame()

    rows = []
    for policy in sorted(os.listdir(base_dir)):
        policy_path = os.path.join(base_dir, policy)
        if not os.path.isdir(policy_path):
            continue
        for test in sorted(os.listdir(policy_path)):
            filepath = os.path.join(policy_path, test, filename)
            if not os.path.isfile(filepath):
                continue
            with open(filepath, "r") as fh:
                data = json.load(fh)

            row = {"policy": policy, "test": test}
            averaged = data.get("averaged", {})
            row.update(averaged)

            stds = data.get("stds", {})
            sems = data.get("sems", {})
            row.update({f"{k}_std": v for k, v in stds.items()})
            row.update({f"{k}_sem": v for k, v in sems.items()})
            rows.append(row)

    if not rows:
        print(f"Warning: no '{filename}' files found under {base_dir}")
    return pd.DataFrame(rows)


def collect_single_test(base_dir: str, test_dir: str = "default_test",
                        filename: str = "results.json",
                        metric: str = "avg_last_10",
                        dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Load a single metric (plus std/sem) from ``<policy>/<test_dir>/<filename>``.

    Returns a DataFrame with columns:
        Policy, <dataset_name>, <dataset_name>_std, <dataset_name>_sem
    """
    if not os.path.isdir(base_dir):
        print(f"Warning: directory not found: '{base_dir}'")
        return pd.DataFrame()

    rows = []
    for policy in sorted(os.listdir(base_dir)):
        fpath = os.path.join(base_dir, policy, test_dir, filename)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r") as fh:
            data = json.load(fh)
        rows.append({
            "Policy": policy,
            dataset_name: data.get("averaged", {}).get(metric),
            f"{dataset_name}_std": data.get("stds", {}).get(metric),
            f"{dataset_name}_sem": data.get("sems", {}).get(metric),
        })
    return pd.DataFrame(rows)


# ============================================================================
# DataFrame Helpers
# ============================================================================

def apply_name_mappings(df: pd.DataFrame, map_tests: bool = True,
                        map_policies: bool = True) -> pd.DataFrame:
    """Apply standard directory-name -> display-name mappings in-place."""
    if map_tests and "test" in df.columns:
        df["test"] = df["test"].replace(get_dir_to_short_test_map())
    if map_policies and "policy" in df.columns:
        df["policy"] = df["policy"].replace(get_policy_name_mapping())
    return df


# ============================================================================
# Plotting Helpers
# ============================================================================

def plot_grouped_bars(ax, pivot, policy_colors, bar_width=0.15,
                      group_spacing=1.2, error_pivot=None, capsize=3,
                      annotate=True, annot_fontsize=10.5, alpha=0.9):
    """
    Draw grouped bars on *ax* from a pivoted DataFrame.

    Parameters
    ----------
    pivot : DataFrame
        Index = policies (or groups), columns = categories.
    policy_colors : dict
        {name: hex_color} for each index entry.
    error_pivot : DataFrame or None
        Same shape as *pivot*; used for yerr.
    """
    num_groups = len(pivot.columns)
    num_bars = len(pivot.index)
    x_pos = np.arange(num_groups) * group_spacing

    for i, name in enumerate(pivot.index):
        scores = pivot.loc[name]
        color = policy_colors.get(name, "gray")
        pos = x_pos - (bar_width * num_bars / 2) + (i * bar_width) + (bar_width / 2)

        yerr = None
        if error_pivot is not None and name in error_pivot.index:
            yerr = error_pivot.loc[name].values

        bars = ax.bar(
            pos, scores, bar_width,
            color=color, alpha=alpha, label=name,
            yerr=yerr, capsize=capsize,
            error_kw=dict(linewidth=1, capthick=1),
        )
        if annotate:
            _annotate_bars(ax, bars, fontsize=annot_fontsize)

    return x_pos


def _annotate_bars(ax, bars, fontsize=10.5, fmt=".2f"):
    """Place value labels above each bar."""
    for bar in bars:
        h = bar.get_height()
        if pd.notna(h):
            ax.annotate(
                f"{h:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=fontsize,
            )


def auto_ylim(pivot, lower_factor=0.95, upper_factor=1.03):
    """Compute y-axis limits from pivot values, ignoring NaNs."""
    vals = pivot.values.flatten()
    vals = vals[~pd.isna(vals)]
    if vals.size == 0:
        return (0, 100)
    return (float(vals.min()) * lower_factor, float(vals.max()) * upper_factor)


def save_figure(fig, output_path, dpi=300):
    """Save figure and print confirmation."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved: {output_path}")


def sanitize_filename(title: str) -> str:
    """Turn a metric title string into a filesystem-safe name."""
    for ch in (" ", "$", "{", "}"):
        title = title.replace(ch, "_" if ch == " " else "")
    return title
