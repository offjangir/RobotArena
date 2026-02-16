# Plotting

Visualization scripts for **RobotArena** evaluation results. All scripts share a common configuration module (`plot_config.py`) and utility library (`plot_utils.py`) to ensure consistent styling, colors, and data loading across every figure.

## Directory Structure

```
plotting/
├── plot_config.py           # Shared constants: policies, colors, tests, metrics, styling
├── plot_utils.py            # Shared helpers: data loaders, bar-chart primitives, I/O
├── bar_plot.py              # Policies × perturbation tests (grouped bars)
├── bar_plot_default.py      # Base test only (simple bars)
├── plot_multi_dataset.py    # Multi-dataset & test-condition comparison
├── plot_sim_comparison.py   # BridgeSim vs SimplerSim (two-simulator comparison)
├── plot_real_vs_sim.py      # Real World vs Simulation
├── plot_bt_scores.py        # Bradley-Terry relative-skill scores
└── README.md
```

## Prerequisites

```bash
pip install matplotlib seaborn pandas numpy
```

## Expected Data Layout

Most scripts read JSON metric files produced by the evaluation pipeline. The two formats are:

| File | Contents | Used by |
|------|----------|---------|
| `results.json` | `{ "averaged": {...}, "stds": {...}, "sems": {...} }` | `bar_plot.py`, `plot_multi_dataset.py`, `plot_sim_comparison.py` |
| `metrics_zero_shot.json` | `{ "averaged": {...} }` | `bar_plot_default.py`, `plot_real_vs_sim.py` |

These files are expected at:

```
<base_dir>/
  <policy>/            # e.g. octo, spatial, robovlm, cogact, ...
    <test>/            # e.g. default_test, permute_test, ...
      results.json
      metrics_zero_shot.json
```

## Quick Start

Every script uses `argparse`. Run any script with `--help` to see all options.

```bash
cd plotting/

# Perturbation tests (grouped bar chart with error bars)
python bar_plot.py --base_dir ./eval_merged_latest

# Base test only (simple bar chart)
python bar_plot_default.py --base_dir ./eval_merged_latest

# Multi-dataset comparison (BridgeSim / DroidSim / RH20TSim)
python plot_multi_dataset.py \
    --base_dirs ./eval_merged ./eval_droid ./eval_rh20t \
    --dataset_names BridgeSim DroidSim RH20TSim

# BridgeSim vs SimplerSim
python plot_sim_comparison.py \
    --base_dir_a ./eval_merged       --name_a "BridgeSim(70 Envs)" \
    --base_dir_b ./eval_iclr_default --name_b "SimplerSim(4 Envs)"

# Real World vs Simulation
python plot_real_vs_sim.py --base_dir ./eval_realsim

# Bradley-Terry scores (uses built-in data by default)
python plot_bt_scores.py
```

All plots are saved to `./plots/` by default (override with `--output_dir`).

## Scripts

### `bar_plot.py` -- Perturbation Test Comparison

Generates grouped bar charts comparing policies across perturbation test conditions (BridgeSim, &Delta;ObjPose, &Delta;Color, &Delta;BG). Produces one plot per metric.

```
python bar_plot.py --base_dir <DIR> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base_dir` | *required* | Root directory with `<policy>/<test>/results.json` |
| `--output_dir` | `./plots` | Output directory for saved PNGs |
| `--metrics` | all three | Which metrics to plot (`avg_m`, `avg_ts`, `avg_last_10`) |
| `--policies` | auto-detected | Policy directory names to include |
| `--tests` | `base pose color background` | Short test names to include |
| `--error_type` | `sem` | Error bars: `sem`, `std`, or `none` |
| `--fig_size` | `14 7` | Figure width and height (inches) |
| `--bar_width` | `0.15` | Width of each bar |
| `--dpi` | `300` | Output resolution |

### `bar_plot_default.py` -- Base Test Only

Simple bar chart showing each policy's score on the default (base) test only. Reads `metrics_zero_shot.json`.

```
python bar_plot_default.py --base_dir <DIR> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base_dir` | *required* | Root directory |
| `--policies` | `octo spatial robovlm cogact` | Policies to include |
| `--metrics` | all three | Metrics to plot |

### `plot_multi_dataset.py` -- Multi-Dataset Comparison

Produces two figures from multiple evaluation directories:
1. **Dataset comparison** -- base-test scores across datasets (x-axis = datasets, bars = policies)
2. **Test-condition comparison** -- perturbation scores from the first dataset (x-axis = tests, bars = policies)

```
python plot_multi_dataset.py --base_dirs <D1> <D2> <D3> --dataset_names <N1> <N2> <N3> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base_dirs` | *required* | One directory per dataset |
| `--dataset_names` | *required* | Display name for each directory |
| `--metric` | `avg_last_10` | Single metric to plot |
| `--error_type` | `sem` | Error bars |
| `--policies` | all 6 | Policies to include (display names) |

### `plot_sim_comparison.py` -- Simulator Comparison

Compares two simulator environments (e.g., BridgeSim vs SimplerSim) using default-test results.

```
python plot_sim_comparison.py \
    --base_dir_a <DIR> --name_a <NAME> \
    --base_dir_b <DIR> --name_b <NAME> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base_dir_a/b` | *required* | Eval directories for each simulator |
| `--name_a/b` | *required* | Display names |
| `--metric` | `avg_last_10` | Metric to compare |
| `--error_type` | `sem` | Error bars |
| `--output_name` | auto | Custom output filename |

### `plot_real_vs_sim.py` -- Real vs Simulation

Compares real-world and simulation evaluation scores. Expects test directories named `real` and `sim`.

```
python plot_real_vs_sim.py --base_dir <DIR> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base_dir` | *required* | Root dir with `<policy>/<real\|sim>/` structure |
| `--policies` | `octo spatial robovlm` | Policies to include |
| `--metrics` | all three | Metrics to plot |

### `plot_bt_scores.py` -- Bradley-Terry Scores

Visualizes exponentiated Bradley-Terry scores as relative skill levels with confidence-interval error bars.

```
python plot_bt_scores.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bt_csv` | built-in | CSV file with columns: `model`, `bt_score`, `ci_lower`, `ci_upper` |
| `--output_name` | `bt_scores.png` | Output filename |

## Shared Modules

### `plot_config.py`

Central configuration that all scripts import:

- **`POLICY_REGISTRY`** -- maps directory name to display name and hex color for every supported policy. Add a new policy here and it will appear in all plots automatically.
- **`TEST_REGISTRY`** -- maps test directory names (e.g. `default_test`) to short keys (`base`) and display labels (`BridgeSim`).
- **`METRIC_REGISTRY`** -- maps metric keys to LaTeX titles and y-axis labels.
- **`apply_style()`** -- applies the project-wide matplotlib/seaborn theme.
- **`style_axis()`**, **`add_legend()`**, **`integer_y_formatter()`** -- reusable axis formatting helpers.

### `plot_utils.py`

Shared data-loading and plotting utilities:

- **`collect_results(base_dir, filename)`** -- walks `<policy>/<test>/` directories and loads the specified JSON file into a flat DataFrame.
- **`collect_single_test(base_dir, ...)`** -- loads a single metric from one test directory per policy (used by simulator-comparison plots).
- **`apply_name_mappings(df)`** -- maps directory names to display names using the registries.
- **`plot_grouped_bars(ax, pivot, colors, ...)`** -- draws grouped bar charts with optional error bars and annotations.
- **`auto_ylim(pivot)`** -- computes sensible y-axis limits from data.
- **`save_figure(fig, path, dpi)`** -- saves and prints confirmation.

## Adding a New Policy

1. Add an entry to `POLICY_REGISTRY` in `plot_config.py`:
   ```python
   "new_policy": {"display": "NewPolicy", "color": "#aabbcc"},
   ```
2. Append the directory name to `DEFAULT_POLICY_ORDER`.
3. All scripts will pick up the new policy automatically.

## Adding a New Test Condition

1. Add an entry to `TEST_REGISTRY` in `plot_config.py`:
   ```python
   "new_test": {"short": "new", "label": "NewTest"},
   ```
2. Append `"new"` to `DEFAULT_TEST_ORDER` if it should appear by default.
