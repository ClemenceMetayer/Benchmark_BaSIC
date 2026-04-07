"""Heatmap visualization of benchmark results per hyperparameter type.

For each hyperparameter type, produces a figure with one subplot per metric.
Each subplot is a heatmap where:
  - rows = parameter variations + baseline(s)
  - columns = ODE systems
  - cell color = relative change vs baseline (green = better, red = worse)
  - cell text = raw metric value

Usage:
    python visualization/analyze_recommendations.py
    python visualization/analyze_recommendations.py --results all_benchmark_results.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent

SYSTEMS = ["lotka_volterra", "chain", "seir", "goldbeter", "yeast_glycolysis"]
SYSTEM_LABELS = {
    "lotka_volterra": "Lotka-Volterra",
    "chain": "Chain",
    "seir": "SEIR",
    "goldbeter": "Goldbeter",
    "yeast_glycolysis": "Yeast glycolysis",
}

BASELINE_HS = 12
BASELINE_SS = 13

# Metric categories with distinct sequential colormaps (single-hue gradients).
# Convention: light = low value, dark = high value (never reversed).
METRIC_CATEGORIES = [
    ("Structure", "Blues", [
        {"col": "F1",           "label": "F1",        "higher_better": True,  "fmt": ".2f"},
        {"col": "Precision",    "label": "Precision",  "higher_better": True,  "fmt": ".2f"},
        {"col": "Recall",       "label": "Recall",     "higher_better": True,  "fmt": ".2f"},
        {"col": "Hamming",      "label": "Hamming",    "higher_better": False, "fmt": ".3f"},
    ]),
    ("Estimation", "Oranges", [
        {"col": "NMSPE_active", "label": "NMSPE active", "higher_better": False, "fmt": ".3f"},
        {"col": "FP_MSE",      "label": "FP MSE",       "higher_better": False, "fmt": ".4f"},
    ]),
    ("Prediction", "Greens", [
        {"col": "NMSE_training", "label": "NMSE training", "higher_better": False, "fmt": ".3f"},
        {"col": "NMSE_new_x0",  "label": "NMSE new x₀",   "higher_better": False, "fmt": ".3f"},
    ]),
    ("Log-likelihood", "Purples", [
        {"col": "log_lik_median", "label": "Median log-lik", "higher_better": True, "fmt": ".1f"},
    ]),
    ("MCMC diagnostics", "Reds", [
        {"col": "rhat_max_sparse", "label": "R-hat max", "higher_better": False, "fmt": ".3f"},
        {"col": "ess_min_sparse",  "label": "ESS min",   "higher_better": True,  "fmt": ".0f"},
    ]),
    ("Timing", "Greys", [
        {"col": "time_total_s", "label": "Total time (s)", "higher_better": False, "fmt": ".0f"},
    ]),
]

# Flat list of all metrics (for data building and ranking)
METRICS = [m for _, _, metrics in METRIC_CATEGORIES for m in metrics]

# Metrics used for ranking (subset without timing)
RANKING_METRICS = [m for m in METRICS if m["col"] != "time_total_s"]

# Map metric col -> (category_label, base_colormap_name)
_METRIC_CMAP = {}
for cat_label, cmap_name, metrics in METRIC_CATEGORIES:
    for m in metrics:
        _METRIC_CMAP[m["col"]] = (cat_label, cmap_name)

# ── Parameter groups ───────────────────────────────────────────────────────
# Each entry: name (filename), title, hs_runs (run_ids), ss_runs (run_ids).
# Empty list = prior not applicable for this parameter.

PARAM_GROUPS = [
    {"name": "segments",
     "title": "Segments (integrator)",
     "hs_runs": [0, 2], "ss_runs": [1, 3]},

    {"name": "max_steps",
     "title": "Max steps (integrator)",
     "hs_runs": [4, 91], "ss_runs": [5, 92]},

    {"name": "chains",
     "title": "MCMC chains",
     "hs_runs": [6, 87, 89], "ss_runs": [7, 88, 90]},

    {"name": "replicates",
     "title": "Replicates (data)",
     "hs_runs": [8, 85], "ss_runs": [9, 86]},

    {"name": "T",
     "title": "Time horizon T (data)",
     "hs_runs": [10, 83], "ss_runs": [11, 84]},

    {"name": "tau0",
     "title": "τ₀ (horseshoe only)",
     "hs_runs": [14, 15, 16], "ss_runs": []},

    {"name": "slab_scale",
     "title": "Slab scale (horseshoe only)",
     "hs_runs": [17, 18], "ss_runs": []},

    {"name": "spike_sd",
     "title": "Spike SD (spike-and-slab only)",
     "hs_runs": [], "ss_runs": [19, 20]},

    {"name": "slab_sd",
     "title": "Slab SD (spike-and-slab only)",
     "hs_runs": [], "ss_runs": [21, 22]},

    {"name": "theta",
     "title": "θ_a & θ_b (spike-and-slab only)",
     "hs_runs": [], "ss_runs": [23, 24]},

    {"name": "degree_penalty",
     "title": "Degree penalty",
     "hs_runs": [25], "ss_runs": [26]},

    {"name": "warmup_samples",
     "title": "Warmup & Samples (MCMC)",
     "hs_runs": [27, 28, 29, 30, 31, 32, 33, 34],
     "ss_runs": [41, 42, 43, 44, 45, 46, 47, 48]},

    {"name": "thinning",
     "title": "Thinning (MCMC)",
     "hs_runs": [35, 36], "ss_runs": [49, 50]},

    {"name": "target_accept",
     "title": "Target accept (MCMC)",
     "hs_runs": [37, 38], "ss_runs": [51, 52]},

    {"name": "max_treedepth",
     "title": "Max treedepth (MCMC)",
     "hs_runs": [39, 40], "ss_runs": [53, 54]},

    {"name": "ci_level",
     "title": "CI level (horseshoe only)",
     "hs_runs": [55, 56], "ss_runs": []},

    {"name": "incl_prob",
     "title": "Inclusion threshold (spike-and-slab only)",
     "hs_runs": [], "ss_runs": [57, 58]},

    {"name": "rtol_atol",
     "title": "Tolerances rtol & atol (integrator)",
     "hs_runs": [59, 60], "ss_runs": [61, 62]},

    {"name": "shooting_scale",
     "title": "Shooting scale",
     "hs_runs": [63, 64, 65, 66], "ss_runs": [67, 68, 69, 70]},

    {"name": "init_median",
     "title": "Initialization strategy (median vs regression)",
     "hs_runs": [71], "ss_runs": [74]},

    {"name": "init_std_noise",
     "title": "Initialization noise (std_noise)",
     "hs_runs": [72, 73], "ss_runs": [75, 76]},

    {"name": "noise",
     "title": "Noise fraction (data)",
     "hs_runs": [77, 78, 79], "ss_runs": [80, 81, 82]},

    {"name": "single_shooting",
     "title": "Single shooting (integrator)",
     "hs_runs": [93], "ss_runs": [94]},

    {"name": "library",
     "title": "Library (include_mm, include_bias, degree)",
     "hs_runs": [95, 97, 99, 101, 103],
     "ss_runs": [96, 98, 100, 102, 104]},
]


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(results_csv, plan_csv):
    df = pd.read_csv(results_csv)
    plan = pd.read_csv(plan_csv)
    plan_sub = plan[["run_id", "category", "description"]].drop_duplicates(
        subset=["run_id"]
    )
    df = df.merge(plan_sub, on="run_id", how="left", suffixes=("", "_plan"))
    if "category_plan" in df.columns:
        df["category"] = df["category"].fillna(df["category_plan"])
        df.drop(columns=["category_plan"], inplace=True, errors="ignore")
    if "description_plan" in df.columns:
        df["description"] = df["description"].fillna(df["description_plan"])
        df.drop(columns=["description_plan"], inplace=True, errors="ignore")
    return df, plan


# ── Heatmap data construction ──────────────────────────────────────────────

def build_heatmap_data(df, group, plan_df):
    """Build heatmap matrices for a parameter group.

    Returns dict with:
        row_labels, col_labels, prior_labels (per row),
        data: {metric_col -> 2D ndarray (n_rows × n_systems)}
    """
    row_labels = []
    row_run_ids = []
    prior_labels = []

    desc_map = dict(zip(plan_df["run_id"], plan_df["description"]))

    def _clean_label(label):
        """Strip trailing (HS) / (SS) from description."""
        return re.sub(r"\s*\((HS|SS)\)\s*$", "", label)

    for prior_tag, runs, bl_id in [("HS", group["hs_runs"], BASELINE_HS),
                                    ("SS", group["ss_runs"], BASELINE_SS)]:
        if not runs:
            continue
        # Baseline row first
        row_labels.append(f"baseline ({prior_tag})")
        row_run_ids.append(bl_id)
        prior_labels.append(prior_tag)
        # Variation rows
        for rid in runs:
            label = _clean_label(desc_map.get(rid, f"run {rid}"))
            row_labels.append(label)
            row_run_ids.append(rid)
            prior_labels.append(prior_tag)

    col_labels = [SYSTEM_LABELS[s] for s in SYSTEMS]

    # Build metric matrices
    data = {}
    for metric in METRICS:
        col = metric["col"]
        matrix = np.full((len(row_labels), len(SYSTEMS)), np.nan)
        for i, rid in enumerate(row_run_ids):
            for j, sys_name in enumerate(SYSTEMS):
                vals = df.loc[
                    (df["run_id"] == rid) & (df["system_name"] == sys_name), col
                ]
                if not vals.empty:
                    matrix[i, j] = vals.iloc[0]
        data[col] = matrix

    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "prior_labels": prior_labels,
        "row_run_ids": row_run_ids,
        "data": data,
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def _find_baseline_indices(hmap_data):
    """Return (hs_baseline_idx, ss_baseline_idx) or None if absent."""
    hs_idx = ss_idx = None
    for i, (label, prior) in enumerate(
        zip(hmap_data["row_labels"], hmap_data["prior_labels"])
    ):
        if "baseline" in label:
            if prior == "HS":
                hs_idx = i
            else:
                ss_idx = i
    return hs_idx, ss_idx


def _compute_delta_matrix(matrix, prior_labels, hs_bl_idx, ss_bl_idx, higher_better):
    """Relative change from baseline; sign flipped so positive = improvement."""
    delta = np.zeros_like(matrix)
    n_rows = matrix.shape[0]
    for i in range(n_rows):
        bl_idx = hs_bl_idx if prior_labels[i] == "HS" else ss_bl_idx
        if bl_idx is None:
            continue
        bl_vals = matrix[bl_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(bl_vals != 0,
                           (matrix[i] - bl_vals) / np.abs(bl_vals),
                           0.0)
            rel = np.where(np.isfinite(rel), rel, 0.0)
        delta[i] = rel
    if not higher_better:
        delta = -delta
    return delta


def _auto_fmt(matrix, base_fmt):
    """Increase decimal precision if the base format makes distinct values look identical."""
    finite = matrix[np.isfinite(matrix)].ravel()
    if len(finite) == 0:
        return base_fmt
    m = re.match(r"\.(\d+)f", base_fmt)
    if not m:
        return base_fmt
    decimals = int(m.group(1))
    for d in range(decimals, decimals + 4):
        fmt = f".{d}f"
        displayed = {f"{v:{fmt}}" for v in finite}
        if len(displayed) >= min(len(finite), 3):
            return fmt
    return f".{decimals + 3}f"


def plot_param_heatmap(group, hmap_data, outdir):
    """One figure per parameter group, subplots in 3-column grid.

    - Each metric category has its own sequential colormap (light=low, dark=high).
    - Y-axis labels shown only on the left column.
    - X-axis labels shown only on the bottom row of each column.
    """
    n_metrics = len(METRICS)
    n_rows = len(hmap_data["row_labels"])
    n_cols = len(hmap_data["col_labels"])

    n_subplot_cols = 3
    n_subplot_rows = int(np.ceil(n_metrics / n_subplot_cols))

    # Adaptive sizing
    subplot_w = max(3.8, n_cols * 0.85 + 1.5)
    subplot_h = max(2.2, n_rows * 0.42 + 0.6)
    fig, axes = plt.subplots(
        n_subplot_rows, n_subplot_cols,
        figsize=(subplot_w * n_subplot_cols, subplot_h * n_subplot_rows + 1.0),
    )
    axes = np.atleast_2d(axes)

    fig.suptitle(group["title"], fontsize=14, fontweight="bold", y=0.99)

    hs_bl_idx, ss_bl_idx = _find_baseline_indices(hmap_data)

    # Adaptive font size for cell annotations
    annot_size = 8 if n_rows <= 10 else (7 if n_rows <= 16 else 6)

    # Which subplot rows contain the last visible metric in each column
    last_row_per_col = [0] * n_subplot_cols
    for ax_idx in range(n_metrics):
        c = ax_idx % n_subplot_cols
        r = ax_idx // n_subplot_cols
        last_row_per_col[c] = r

    for ax_idx, metric in enumerate(METRICS):
        row_idx = ax_idx // n_subplot_cols
        col_idx = ax_idx % n_subplot_cols
        ax = axes[row_idx, col_idx]
        col = metric["col"]
        matrix = hmap_data["data"][col]

        # Sequential colormap on raw values (light = low, dark = high)
        _, cmap_name = _METRIC_CMAP[col]

        vmin = np.nanmin(matrix) if np.any(np.isfinite(matrix)) else 0
        vmax = np.nanmax(matrix) if np.any(np.isfinite(matrix)) else 1
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        im = ax.imshow(matrix, cmap=cmap_name, vmin=vmin, vmax=vmax,
                        aspect="auto")

        # Determine annotation format: increase decimals if values look identical
        fmt = _auto_fmt(matrix, metric["fmt"])

        # Annotate cells with raw values
        cmap_obj = plt.get_cmap(cmap_name)
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if np.isfinite(val):
                    text = f"{val:{fmt}}"
                    # Text color: white on dark background, black on light
                    normed = (val - vmin) / (vmax - vmin)
                    rgba = cmap_obj(normed)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.55 else "black"
                else:
                    text = "—"
                    text_color = "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=annot_size, color=text_color)

        # X-axis: only on the last row that has a subplot in this column
        ax.set_xticks(range(n_cols))
        if row_idx == last_row_per_col[col_idx]:
            ax.set_xticklabels(hmap_data["col_labels"], fontsize=8,
                               rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

        # Y-axis: only on the left column
        ax.set_yticks(range(n_rows))
        if col_idx == 0:
            ax.set_yticklabels(hmap_data["row_labels"], fontsize=8)
            # Bold baseline row labels
            for i, label in enumerate(hmap_data["row_labels"]):
                if "baseline" in label:
                    ax.get_yticklabels()[i].set_fontweight("bold")
        else:
            ax.set_yticklabels([])

        ax.set_title(metric["label"], fontsize=10, fontweight="bold")

        # Separator between HS and SS groups
        if hs_bl_idx is not None and ss_bl_idx is not None:
            sep_y = ss_bl_idx - 0.5
            ax.axhline(sep_y, color="black", linewidth=2)

    # Hide unused subplots
    for ax_idx in range(n_metrics, n_subplot_rows * n_subplot_cols):
        row_idx = ax_idx // n_subplot_cols
        col_idx = ax_idx % n_subplot_cols
        axes[row_idx, col_idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        outdir / f"heatmap_{group['name']}.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved heatmap_{group['name']}.png")


# ── Per-parameter ranking ──────────────────────────────────────────────────

def _parse_varied_params(varied_str):
    """Parse varied_params column into a list of (param_name, baseline_val, tested_val).

    Examples:
        'mcmc.num_warmup: 2000 -> 500'           -> [('mcmc.num_warmup', '2000', '500')]
        'integrator.rtol: 1e-07 -> 1e-05 | ...'  -> [('integrator.rtol', ...), ...]
        'none (baseline)'                          -> []
    """
    if pd.isna(varied_str) or "baseline" in str(varied_str).lower():
        return []
    parts = [p.strip() for p in str(varied_str).split("|")]
    parsed = []
    for part in parts:
        m = re.match(r"([\w.]+):\s*(.*?)\s*->\s*(.*)", part)
        if m:
            name = m.group(1).strip()
            bl_val = m.group(2).strip() or "None"
            tested_val = m.group(3).strip() or "None"
            parsed.append((name, bl_val, tested_val))
    return parsed


def compute_per_parameter_ranking(df, plan_df, outdir):
    """For each individual parameter, rank all tested values (incl. baseline).

    Ranking is done per (metric, system) pair, then averaged.
    Produces CSV files with the full ranking and the best config per parameter.
    """
    metric_cols = [m["col"] for m in RANKING_METRICS]
    metric_dirs = {m["col"]: m["higher_better"] for m in RANKING_METRICS}

    desc_map = dict(zip(plan_df["run_id"], plan_df["description"]))

    # Build groups: (param_name, prior) -> [(run_id, tested_value)]
    param_groups = {}

    for _, row in plan_df.iterrows():
        parsed = _parse_varied_params(row.get("varied_params", ""))
        if not parsed:
            continue

        cat = str(row.get("category", ""))
        if cat.endswith("_hs"):
            prior = "HS"
        elif cat.endswith("_ss"):
            prior = "SS"
        else:
            continue

        run_id = row["run_id"]
        if run_id not in df["run_id"].values:
            continue

        # For multi-param runs (e.g. rtol+atol), create a combined key
        if len(parsed) > 1:
            param_key = " + ".join(p[0] for p in parsed)
            tested_label = " | ".join(f"{p[0]}={p[2]}" for p in parsed)
            bl_label = " | ".join(f"{p[0]}={p[1]}" for p in parsed)
        else:
            param_key = parsed[0][0]
            tested_label = parsed[0][2]
            bl_label = parsed[0][1]

        key = (param_key, prior)
        if key not in param_groups:
            param_groups[key] = {"variations": [], "bl_label": bl_label}
        param_groups[key]["variations"].append((run_id, tested_label))

    # Rank each parameter group
    all_rows = []

    for (param_name, prior), group_info in sorted(param_groups.items()):
        bl_id = BASELINE_HS if prior == "HS" else BASELINE_SS
        bl_label = group_info["bl_label"]
        variations = group_info["variations"]

        # All configs: baseline + variations
        configs = [(bl_id, f"{bl_label} (baseline)")]
        for rid, label in variations:
            configs.append((rid, label))

        run_ids = [rid for rid, _ in configs]
        labels = {rid: lab for rid, lab in configs}

        sub = df[df["run_id"].isin(run_ids)].copy()
        if sub.empty:
            continue

        # Rank for each (metric, system) pair
        rank_records = []
        for metric_col in metric_cols:
            ascending = not metric_dirs[metric_col]
            for sys_name in SYSTEMS:
                sys_sub = sub.loc[
                    sub["system_name"] == sys_name, ["run_id", metric_col]
                ].dropna().drop_duplicates(subset="run_id")
                if sys_sub.empty:
                    continue
                sys_sub["_rank"] = sys_sub[metric_col].rank(
                    ascending=ascending, method="average",
                )
                for _, r in sys_sub.iterrows():
                    rank_records.append((r["run_id"], r["_rank"]))

        if not rank_records:
            continue

        rank_df = pd.DataFrame(rank_records, columns=["run_id", "rank"])
        mean_ranks = rank_df.groupby("run_id")["rank"].mean()

        best_rid = mean_ranks.idxmin()

        for rid in run_ids:
            if rid in mean_ranks.index:
                all_rows.append({
                    "parameter": param_name,
                    "prior": prior,
                    "config": labels[rid],
                    "run_id": int(rid),
                    "mean_rank": round(mean_ranks[rid], 3),
                    "is_best": rid == best_rid,
                    "is_baseline": rid == bl_id,
                })

    if not all_rows:
        print("  No ranking results computed.")
        return

    results_df = pd.DataFrame(all_rows)

    # Save full ranking
    results_df.to_csv(outdir / "per_parameter_ranking.csv", index=False)
    print(f"  Saved per_parameter_ranking.csv")

    # Save best config summary
    best_df = results_df[results_df["is_best"]].copy()
    best_df = best_df.sort_values(["prior", "parameter"]).reset_index(drop=True)
    best_df.to_csv(outdir / "best_config_per_parameter.csv", index=False)
    print(f"  Saved best_config_per_parameter.csv")

    # Print summary
    for prior in ["HS", "SS"]:
        prior_label = "Horseshoe" if prior == "HS" else "Spike-and-Slab"
        sub = best_df[best_df["prior"] == prior]
        if sub.empty:
            continue
        print(f"\n  {'─' * 90}")
        print(f"  BEST CONFIG PER PARAMETER — {prior_label}")
        print(f"  {'─' * 90}")
        print(f"  {'Parameter':<45s} {'Best config':<35s} {'Mean rank':<10s} Note")
        print(f"  {'─' * 90}")
        for _, row in sub.iterrows():
            note = "= baseline" if row["is_baseline"] else ""
            print(
                f"  {row['parameter']:<45s} {row['config']:<35s} "
                f"{row['mean_rank']:<10.3f} {note}"
            )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results",
                        default=str(BENCHMARK_DIR / "all_benchmark_results.csv"))
    parser.add_argument("--plan",
                        default=str(BENCHMARK_DIR / "config" / "experiment_plan.csv"))
    parser.add_argument("--outdir",
                        default=str(BENCHMARK_DIR / "figures" / "heatmaps"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-HYPERPARAMETER HEATMAPS — BaSIC BENCHMARK")
    print("=" * 70)

    df, plan_df = load_data(args.results, args.plan)
    print(f"Loaded {len(df)} rows\n")

    # Heatmaps
    print("--- Heatmaps per parameter type ---")
    for group in PARAM_GROUPS:
        hmap_data = build_heatmap_data(df, group, plan_df)
        if not hmap_data["row_labels"]:
            print(f"  [SKIP] {group['title']} — no data")
            continue
        plot_param_heatmap(group, hmap_data, outdir)

    # Per-parameter ranking
    print("\n--- Per-parameter ranking ---")
    compute_per_parameter_ranking(df, plan_df, outdir)

    print(f"\n{'=' * 70}")
    print(f"Figures and rankings saved to: {outdir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
