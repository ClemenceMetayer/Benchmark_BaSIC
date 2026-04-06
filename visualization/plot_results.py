"""Visualize benchmark results for BaSIC hyperparameter evaluation.

Reads all_benchmark_results.csv and config/experiment_plan.csv, then
generates figures: heatmaps, sensitivity plots, PCA, timing analysis,
HS vs SS comparisons, log-likelihood analysis, MCMC diagnostics, and
aggregated summaries.

Usage:
    python visualization/plot_results.py
    python visualization/plot_results.py --results all_benchmark_results.csv
    python visualization/plot_results.py --outdir figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# Paths and constants
# ────────────────────────────────────────────────────────────

BENCHMARK_DIR = Path(__file__).resolve().parent.parent

SYSTEMS = ["lotka_volterra", "chain", "seir", "goldbeter", "yeast_glycolysis"]
SYSTEM_LABELS = {
    "lotka_volterra": "Lotka-Volterra",
    "chain": "Chain",
    "seir": "SEIR",
    "goldbeter": "Goldbeter",
    "yeast_glycolysis": "Yeast glycolysis",
}

METRICS = {
    "F1": {"label": "F1-Score", "higher_better": True, "vmin": 0, "vmax": 1},
    "NMSPE_active": {"label": "NMSPE (active)", "higher_better": False, "vmin": 0, "vmax": None},
    "NMSE_new_x0": {"label": "NMSE (new x₀)", "higher_better": False, "vmin": 0, "vmax": None},
    "CI_coverage_active": {"label": "CI coverage (95%)", "higher_better": None, "vmin": 0, "vmax": 1},
    "NMSE_training": {"label": "NMSE (training)", "higher_better": False, "vmin": 0, "vmax": None},
    "Hamming": {"label": "Hamming distance", "higher_better": False, "vmin": 0, "vmax": 1},
    "log_lik_median": {"label": "Log-vraisemblance (médiane)", "higher_better": True, "vmin": None, "vmax": None},
}

CATEGORY_GROUPS = {
    "Prior (HS)": ["prior_hs"],
    "Prior (SS)": ["prior_ss"],
    "MCMC (HS)": ["mcmc_hs"],
    "MCMC (SS)": ["mcmc_ss"],
    "Intégrateur (HS)": ["integrator_hs", "shooting_hs"],
    "Intégrateur (SS)": ["integrator_ss", "shooting_ss"],
    "Sélection (HS)": ["selection_hs"],
    "Sélection (SS)": ["selection_ss"],
    "Données (HS)": ["data_hs"],
    "Données (SS)": ["data_ss"],
    "Initialisation (HS)": ["init_hs"],
    "Initialisation (SS)": ["init_ss"],
    "Librairie (HS)": ["library_hs"],
    "Librairie (SS)": ["library_ss"],
}

SYSTEM_COLORS = dict(zip(SYSTEMS, plt.cm.Set2.colors[:len(SYSTEMS)]))

BASELINE_HS = 12
BASELINE_SS = 13


# ────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────

def load_data(results_csv, plan_csv):
    df = pd.read_csv(results_csv)
    plan = pd.read_csv(plan_csv)

    plan_cols = ["run_id", "category", "description", "varied_params",
                 "sparse_prior.type"]
    plan_sub = plan[[c for c in plan_cols if c in plan.columns]].drop_duplicates(subset=["run_id"])
    df = df.merge(plan_sub, on="run_id", how="left", suffixes=("", "_plan"))

    if "category_plan" in df.columns:
        df["category"] = df["category"].fillna(df["category_plan"])
        df.drop(columns=["category_plan"], inplace=True, errors="ignore")
    if "description_plan" in df.columns:
        df["description"] = df["description"].fillna(df["description_plan"])
        df.drop(columns=["description_plan"], inplace=True, errors="ignore")

    if "sparse_prior.type" in df.columns:
        df["prior_type"] = df["sparse_prior.type"]
    elif "sparse_prior.type_plan" in df.columns:
        df["prior_type"] = df["sparse_prior.type_plan"]
    else:
        df["prior_type"] = df["category"].apply(
            lambda c: "spike_and_slab" if str(c).endswith("_ss") else "horseshoe"
        )
    return df


# ────────────────────────────────────────────────────────────
# 1. Heatmaps
# ────────────────────────────────────────────────────────────

def plot_heatmap(df, metric, outdir):
    meta = METRICS[metric]
    pivot = df.pivot_table(index="system_name", columns="run_id", values=metric)
    systems_present = [s for s in SYSTEMS if s in pivot.index]
    pivot = pivot.loc[systems_present]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.25), 1 + len(systems_present) * 0.8))

    vmin, vmax = meta["vmin"], meta["vmax"]
    if vmax is None:
        vmax = np.nanpercentile(pivot.values, 95)
    if vmin is None:
        vmin = np.nanpercentile(pivot.values, 5)

    cmap = "RdYlGn" if meta["higher_better"] else "RdYlGn_r"
    if meta["higher_better"] is None:
        cmap = "coolwarm"

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(systems_present)))
    ax.set_yticklabels([SYSTEM_LABELS.get(s, s) for s in systems_present], fontsize=9)

    xtick_step = max(1, len(pivot.columns) // 30)
    ax.set_xticks(range(0, len(pivot.columns), xtick_step))
    ax.set_xticklabels(pivot.columns[::xtick_step], fontsize=7, rotation=90)
    ax.set_xlabel("Run ID", fontsize=10)
    ax.set_title(f"{meta['label']} — tous les runs", fontsize=13, fontweight="bold")

    # Highlight baselines
    for bl_id, color in [(BASELINE_HS, "blue"), (BASELINE_SS, "red")]:
        if bl_id in pivot.columns:
            col_idx = list(pivot.columns).index(bl_id)
            ax.axvline(col_idx, color=color, linewidth=1.5, linestyle="--", alpha=0.7)

    plt.colorbar(im, ax=ax, shrink=0.8, label=meta["label"])
    fig.tight_layout()
    fig.savefig(outdir / f"heatmap_{metric}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap_{metric}.png")


# ────────────────────────────────────────────────────────────
# 2. Sensitivity plots per category
# ────────────────────────────────────────────────────────────

def plot_sensitivity(df, group_name, categories, outdir):
    sub = df[df["category"].isin(categories)].copy()
    if sub.empty:
        return

    is_ss = group_name.endswith("(SS)")
    baseline_id = BASELINE_SS if is_ss else BASELINE_HS
    bl = df[df["run_id"] == baseline_id]

    metrics_to_plot = ["F1", "NMSPE_active", "NMSE_new_x0", "CI_coverage_active", "log_lik_median"]
    metrics_to_plot = [m for m in metrics_to_plot if m in sub.columns and sub[m].notna().any()]

    if not metrics_to_plot:
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    descriptions = sub.drop_duplicates("run_id").sort_values("run_id")["description"].tolist()
    run_ids = sub.drop_duplicates("run_id").sort_values("run_id")["run_id"].tolist()
    x = np.arange(len(run_ids))

    for ax, metric in zip(axes, metrics_to_plot):
        meta = METRICS.get(metric, {"label": metric})

        for sys_name in SYSTEMS:
            sys_data = sub[sub["system_name"] == sys_name].sort_values("run_id")
            if sys_data.empty:
                continue
            y = [sys_data[sys_data["run_id"] == rid][metric].values[0]
                 if rid in sys_data["run_id"].values else np.nan
                 for rid in run_ids]
            ax.plot(x, y, "o-", color=SYSTEM_COLORS[sys_name],
                    label=SYSTEM_LABELS.get(sys_name, sys_name),
                    markersize=4, linewidth=1.2, alpha=0.8)

        # Baseline line
        bl_vals = bl.groupby("system_name")[metric].mean()
        for sys_name, val in bl_vals.items():
            if sys_name in SYSTEM_COLORS:
                ax.axhline(val, color=SYSTEM_COLORS[sys_name], linestyle=":", alpha=0.4)

        ax.set_ylabel(meta.get("label", metric), fontsize=9)
        ax.grid(alpha=0.2)
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="best", ncol=2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(descriptions, rotation=45, ha="right", fontsize=7)

    fig.suptitle(f"Sensibilité — {group_name}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    safe_name = group_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(outdir / f"sensitivity_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sensitivity_{safe_name}.png")


# ────────────────────────────────────────────────────────────
# 3. HS vs SS comparison
# ────────────────────────────────────────────────────────────

def plot_hs_vs_ss(df, outdir):
    metrics_compare = ["F1", "NMSPE_active", "NMSE_new_x0", "CI_coverage_active"]

    # Baseline comparison
    fig, axes = plt.subplots(1, len(metrics_compare), figsize=(4 * len(metrics_compare), 4))
    for ax, metric in zip(axes, metrics_compare):
        meta = METRICS.get(metric, {"label": metric})
        for sys_name in SYSTEMS:
            hs = df[(df["run_id"] == BASELINE_HS) & (df["system_name"] == sys_name)][metric]
            ss = df[(df["run_id"] == BASELINE_SS) & (df["system_name"] == sys_name)][metric]
            if not hs.empty and not ss.empty:
                ax.scatter(hs.values[0], ss.values[0], color=SYSTEM_COLORS[sys_name],
                           s=80, label=SYSTEM_LABELS.get(sys_name, sys_name), zorder=3)
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Horseshoe", fontsize=9)
        ax.set_ylabel(f"Spike-and-Slab", fontsize=9)
        ax.set_title(meta.get("label", metric), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    fig.suptitle("Baselines — Horseshoe vs Spike-and-Slab", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "hs_vs_ss_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved hs_vs_ss_baseline.png")

    # Structure: precision vs recall
    fig, ax = plt.subplots(figsize=(8, 6))
    for prior, marker, bl_id in [("horseshoe", "o", BASELINE_HS),
                                  ("spike_and_slab", "s", BASELINE_SS)]:
        sub = df[df["prior_type"] == prior]
        for sys_name in SYSTEMS:
            sys_sub = sub[sub["system_name"] == sys_name]
            if sys_sub.empty:
                continue
            ax.scatter(sys_sub["Precision"], sys_sub["Recall"],
                       color=SYSTEM_COLORS[sys_name], marker=marker,
                       alpha=0.3, s=20)
        # Highlight baseline
        bl = df[(df["run_id"] == bl_id)]
        for sys_name in SYSTEMS:
            bl_sys = bl[bl["system_name"] == sys_name]
            if not bl_sys.empty:
                ax.scatter(bl_sys["Precision"].values[0], bl_sys["Recall"].values[0],
                           color=SYSTEM_COLORS[sys_name], marker=marker,
                           s=150, edgecolors="black", linewidth=1.5, zorder=5)

    ax.set_xlabel("Precision", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Precision vs Recall (o=HS, □=SS, gros=baseline)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "hs_vs_ss_structure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved hs_vs_ss_structure.png")


# ────────────────────────────────────────────────────────────
# 4. Log-likelihood analysis
# ────────────────────────────────────────────────────────────

def plot_log_likelihood(df, outdir):
    if "log_lik_median" not in df.columns or df["log_lik_median"].isna().all():
        print("  [SKIP] No log-likelihood data.")
        return

    # Log-likelihood vs F1
    fig, ax = plt.subplots(figsize=(8, 6))
    for sys_name in SYSTEMS:
        sub = df[df["system_name"] == sys_name]
        ax.scatter(sub["F1"], sub["log_lik_median"],
                   color=SYSTEM_COLORS[sys_name], alpha=0.4, s=20,
                   label=SYSTEM_LABELS.get(sys_name, sys_name))
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_ylabel("Log-vraisemblance (médiane)", fontsize=11)
    ax.set_title("F1 vs Log-vraisemblance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "log_likelihood_vs_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved log_likelihood_vs_f1.png")


# ────────────────────────────────────────────────────────────
# 5. Timing analysis
# ────────────────────────────────────────────────────────────

def plot_timing(df, outdir):
    if "time_total_s" not in df.columns:
        return

    # Timing breakdown by category
    agg = df.groupby("category")[["time_sparse_s", "time_refit_s"]].mean().sort_values("time_sparse_s", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.35)))
    y = np.arange(len(agg))
    ax.barh(y, agg["time_sparse_s"], height=0.7, label="Sparse", color="#4C72B0")
    ax.barh(y, agg["time_refit_s"], height=0.7, left=agg["time_sparse_s"],
            label="Refit", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=8)
    ax.set_xlabel("Temps moyen (s)", fontsize=10)
    ax.set_title("Temps d'exécution par catégorie", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(outdir / "timing_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved timing_breakdown.png")


# ────────────────────────────────────────────────────────────
# 6. MCMC diagnostics
# ────────────────────────────────────────────────────────────

def plot_mcmc_diagnostics(df, outdir):
    if "rhat_max_sparse" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R-hat
    ax = axes[0]
    for sys_name in SYSTEMS:
        sub = df[df["system_name"] == sys_name]
        ax.scatter(sub["run_id"], sub["rhat_max_sparse"],
                   color=SYSTEM_COLORS[sys_name], alpha=0.4, s=15,
                   label=SYSTEM_LABELS.get(sys_name, sys_name))
    ax.axhline(1.05, color="red", linestyle="--", alpha=0.6, label="R̂ = 1.05")
    ax.set_xlabel("Run ID", fontsize=10)
    ax.set_ylabel("R̂ max (sparse)", fontsize=10)
    ax.set_title("Convergence MCMC — R̂", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    # ESS
    ax = axes[1]
    for sys_name in SYSTEMS:
        sub = df[df["system_name"] == sys_name]
        ax.scatter(sub["run_id"], sub["ess_min_sparse"],
                   color=SYSTEM_COLORS[sys_name], alpha=0.4, s=15,
                   label=SYSTEM_LABELS.get(sys_name, sys_name))
    ax.axhline(400, color="orange", linestyle="--", alpha=0.6, label="ESS = 400")
    ax.set_xlabel("Run ID", fontsize=10)
    ax.set_ylabel("ESS min (sparse)", fontsize=10)
    ax.set_title("Convergence MCMC — ESS", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(outdir / "mcmc_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mcmc_diagnostics.png")


# ────────────────────────────────────────────────────────────
# 7. PCA
# ────────────────────────────────────────────────────────────

def plot_pca(df, outdir):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  [SKIP] scikit-learn not installed.")
        return

    pc_metrics = ["F1", "NMSPE_active", "NMSE_new_x0", "CI_coverage_active", "log_lik_median"]
    pc_metrics = [m for m in pc_metrics if m in df.columns and df[m].notna().any()]
    if len(pc_metrics) < 3:
        print("  [SKIP] Not enough metrics for PCA.")
        return

    sub = df[["run_id", "system_name", "category"] + pc_metrics].dropna()
    if len(sub) < 10:
        return

    X = StandardScaler().fit_transform(sub[pc_metrics].values)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    # Biplot
    fig, ax = plt.subplots(figsize=(10, 8))
    for sys_name in SYSTEMS:
        mask = sub["system_name"] == sys_name
        ax.scatter(Z[mask, 0], Z[mask, 1], color=SYSTEM_COLORS[sys_name],
                   alpha=0.4, s=20, label=SYSTEM_LABELS.get(sys_name, sys_name))

    # Loading vectors
    loadings = pca.components_.T
    scale = np.abs(Z).max() * 0.8
    for i, metric in enumerate(pc_metrics):
        ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
                metric, color="red", fontsize=8, fontweight="bold")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
    ax.set_title("PCA — espace des métriques (biplot)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "pca_biplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pca_biplot.png")


# ────────────────────────────────────────────────────────────
# 8. Aggregated summary
# ────────────────────────────────────────────────────────────

def plot_aggregated_summary(df, outdir):
    metrics_summary = ["F1", "NMSPE_active", "NMSE_new_x0", "log_lik_median"]
    metrics_summary = [m for m in metrics_summary if m in df.columns]

    agg = df.groupby("category")[metrics_summary].mean()

    fig, axes = plt.subplots(1, len(metrics_summary),
                              figsize=(5 * len(metrics_summary), max(5, len(agg) * 0.3)))
    if len(metrics_summary) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_summary):
        meta = METRICS.get(metric, {"label": metric})
        sorted_agg = agg.sort_values(metric, ascending=not meta.get("higher_better", True))
        colors = ["#4C72B0" if "hs" in cat or cat == "baseline" else "#DD8452"
                  for cat in sorted_agg.index]
        ax.barh(range(len(sorted_agg)), sorted_agg[metric], color=colors)
        ax.set_yticks(range(len(sorted_agg)))
        ax.set_yticklabels(sorted_agg.index, fontsize=8)
        ax.set_xlabel(meta.get("label", metric), fontsize=9)

        # Baseline lines
        for bl_id, ls in [(BASELINE_HS, "--"), (BASELINE_SS, ":")]:
            bl_val = df[df["run_id"] == bl_id][metric].mean()
            if not np.isnan(bl_val):
                ax.axvline(bl_val, color="black", linestyle=ls, alpha=0.5, linewidth=1)

        ax.grid(alpha=0.2, axis="x")

    fig.suptitle("Résumé agrégé par catégorie", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "aggregated_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved aggregated_summary.png")


# ────────────────────────────────────────────────────────────
# Main
# ──��─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(BENCHMARK_DIR / "all_benchmark_results.csv"))
    parser.add_argument("--plan", default=str(BENCHMARK_DIR / "config" / "experiment_plan.csv"))
    parser.add_argument("--outdir", default=str(BENCHMARK_DIR / "figures"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VISUALISATION DES RÉSULTATS — BENCHMARK BaSIC")
    print("=" * 70)

    df = load_data(args.results, args.plan)
    print(f"Loaded {len(df)} rows, {df['system_name'].nunique()} systems, "
          f"{df['run_id'].nunique()} runs\n")

    # 1. Heatmaps
    print("--- Heatmaps ---")
    for metric in ["F1", "NMSPE_active", "NMSE_new_x0", "CI_coverage_active", "log_lik_median"]:
        if metric in df.columns and df[metric].notna().any():
            plot_heatmap(df, metric, outdir)

    # 2. Sensitivity plots
    print("\n--- Sensibilité par catégorie ---")
    for group_name, categories in CATEGORY_GROUPS.items():
        plot_sensitivity(df, group_name, categories, outdir)

    # 3. HS vs SS
    print("\n--- HS vs SS ---")
    plot_hs_vs_ss(df, outdir)

    # 4. Log-likelihood
    print("\n--- Log-vraisemblance ---")
    plot_log_likelihood(df, outdir)

    # 5. Timing
    print("\n--- Temps d'exécution ---")
    plot_timing(df, outdir)

    # 6. MCMC diagnostics
    print("\n--- Diagnostics MCMC ---")
    plot_mcmc_diagnostics(df, outdir)

    # 7. PCA
    print("\n--- PCA ---")
    plot_pca(df, outdir)

    # 8. Aggregated summary
    print("\n--- Résumé agrégé ---")
    plot_aggregated_summary(df, outdir)

    print(f"\n{'=' * 70}")
    print(f"Toutes les figures sauvegardées dans: {outdir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
