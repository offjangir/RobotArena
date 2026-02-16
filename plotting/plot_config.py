"""
Shared configuration for all RobotArena plotting scripts.

Centralizes policy definitions, color palettes, test/metric registries,
and matplotlib styling so every plot has a consistent appearance.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================================
# Policy Registry
# ============================================================================
# Each entry: dir_name -> {display, color}
# Add new policies here and they will be picked up by all scripts.

POLICY_REGISTRY = {
    "octo":         {"display": "Octo",       "color": "#ef5675"},
    "spatial":      {"display": "SpatialVLA",  "color": "#374c80"},
    "robovlm":      {"display": "RoboVLM",    "color": "#7a5195"},
    "cogact":       {"display": "Cogact",      "color": "#ff764a"},
    "open_pi_zero": {"display": "OpenPIZero",  "color": "#2ca3a8"},
    "xvla":         {"display": "X-VLA",       "color": "#ffbf3f"},
}

# Default ordering used when no explicit list is provided.
DEFAULT_POLICY_ORDER = [
    "octo", "spatial", "robovlm", "cogact", "open_pi_zero", "xvla",
]

# ============================================================================
# Test Registry
# ============================================================================
# Maps directory name -> short key and display label.

TEST_REGISTRY = {
    "default_test":        {"short": "base",       "label": "BridgeSim"},
    "permute_test":        {"short": "pose",       "label": "\u0394ObjPose"},
    "adv_background_test": {"short": "color",      "label": "\u0394Color"},
    "background_test":     {"short": "background", "label": "\u0394BG"},
    "camera_test":         {"short": "camera",     "label": "\u0394Cam"},
    "pose_test":           {"short": "pose_var",   "label": "\u0394Pose"},
    "asset_test":          {"short": "asset",      "label": "\u0394Asset"},
}

DEFAULT_TEST_ORDER = ["base", "pose", "color", "background"]

# ============================================================================
# Metric Registry
# ============================================================================

METRIC_REGISTRY = {
    "avg_m":       {"title": r"$GVL_{AM}$ Score",  "ylabel": r"$GVL_{AM}$ Score"},
    "avg_ts":      {"title": r"$GVL_{T30}$ Score", "ylabel": r"$GVL_{T30}$ Score"},
    "avg_last_10": {"title": r"$GVL_{L30}$ Score", "ylabel": r"$GVL_{L30}$ Score"},
}

DEFAULT_METRICS = ["avg_m", "avg_ts", "avg_last_10"]

# ============================================================================
# Plot Defaults
# ============================================================================

DEFAULT_DPI = 300
DEFAULT_ERROR_TYPE = "sem"     # "sem" or "std"
DEFAULT_FIG_SIZE = (14, 7)
DEFAULT_BAR_WIDTH = 0.15
DEFAULT_OUTPUT_DIR = "./plots"

# ============================================================================
# Helper look-ups (derived from registries)
# ============================================================================

def get_policy_color(name: str) -> str:
    """Return hex color for a policy (accepts dir_name or display name)."""
    if name in POLICY_REGISTRY:
        return POLICY_REGISTRY[name]["color"]
    for info in POLICY_REGISTRY.values():
        if info["display"] == name:
            return info["color"]
    return "gray"


def get_policy_display(dir_name: str) -> str:
    """Map a directory name to its display name."""
    return POLICY_REGISTRY.get(dir_name, {}).get("display", dir_name.capitalize())


def get_dir_to_short_test_map() -> dict:
    """Return {dir_name: short_name} for test directories."""
    return {k: v["short"] for k, v in TEST_REGISTRY.items()}


def get_short_to_label_map() -> dict:
    """Return {short_name: display_label} for tests."""
    return {v["short"]: v["label"] for v in TEST_REGISTRY.values()}


def get_policy_name_mapping() -> dict:
    """Return {dir_name: display_name} for all policies."""
    return {k: v["display"] for k, v in POLICY_REGISTRY.items()}


# ============================================================================
# Matplotlib / Seaborn Styling
# ============================================================================

def apply_style():
    """Apply the project-wide matplotlib style. Call once at script start."""
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"


def style_axis(ax, ylabel="VLM Score", ylabel_size=22, xtick_size=20,
               ytick_size=16, grid=True):
    """Apply common axis styling."""
    ax.set_ylabel(ylabel, fontsize=ylabel_size, labelpad=10)
    ax.tick_params(axis="y", labelsize=ytick_size)
    if grid:
        ax.grid(True, which="major", axis="y",
                linestyle=":", linewidth=0.7, color="lightgray")
    for spine in ax.spines.values():
        spine.set_visible(True)


def add_legend(ax, ncol=None, fontsize=16, bbox_y=1.0, **kwargs):
    """Add a horizontal legend above the plot."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    by_label = dict(zip(labels, handles))
    if ncol is None:
        ncol = len(by_label)
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.5,
        handletextpad=0.5,
        handlelength=1.5,
        **kwargs,
    )


def integer_y_formatter(ax, math_mode=True):
    """Format y-axis ticks as integers, optionally wrapped in $...$."""
    if math_mode:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${int(x)}$"))
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
