"""Analyze benchmark results and produce per-hyperparameter recommendations.

For each hyperparameter category, computes a composite score (weighted
combination of F1, NMSPE, NMSE, log-likelihood) and identifies the best
configuration. Produces:
  - recommendations_HS.png / recommendations_SS.png: ranked bar charts
  - delta_vs_baseline_HS.png / delta_vs_baseline_SS.png: delta from baseline
  - per_param_recommendations_HS.png / _SS.png: best value per parameter
  - recommendations.txt: text summary with best value per parameter

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

# Metrics used for ranking.  Sign indicates direction only:
#   positive = higher is better,  negative = lower is better.
SCORE_WEIGHTS = {
    "F1":             1,   # higher is better
    "NMSPE_active":  -1,   # lower is better
    "NMSE_new_x0":   -1,   # lower is better
    "log_lik_median":  1,  # higher is better
    "CI_coverage_active": 1,  # higher is better
}

SYSTEM_COLORS = dict(zip(SYSTEMS, plt.cm.Set2.colors[:len(SYSTEMS)]))

# Categories by prior type
HS_CATEGORIES = ["prior_hs", "mcmc_hs", "integrator_hs", "shooting_hs",
                 "selection_hs", "library_hs", "data_hs", "init_hs"]
SS_CATEGORIES = ["prior_ss", "mcmc_ss", "integrator_ss", "shooting_ss",
                 "selection_ss", "library_ss", "data_ss", "init_ss"]

CATEGORY_LABELS = {
    "prior_hs": "Prior", "prior_ss": "Prior",
    "mcmc_hs": "MCMC", "mcmc_ss": "MCMC",
    "integrator_hs": "Intégrateur", "integrator_ss": "Intégrateur",
    "shooting_hs": "Shooting", "shooting_ss": "Shooting",
    "selection_hs": "Sélection", "selection_ss": "Sélection",
    "library_hs": "Librairie", "library_ss": "Librairie",
    "data_hs": "Données", "data_ss": "Données",
    "init_hs": "Initialisation", "init_ss": "Initialisation",
}


def load_data(results_csv, plan_csv):
    df = pd.read_csv(results_csv)
    plan = pd.read_csv(plan_csv)
    plan_sub = plan[["run_id", "category", "description"]].drop_duplicates(subset=["run_id"])
    df = df.merge(plan_sub, on="run_id", how="left", suffixes=("", "_plan"))
    if "category_plan" in df.columns:
        df["category"] = df["category"].fillna(df["category_plan"])
        df.drop(columns=["category_plan"], inplace=True, errors="ignore")
    if "description_plan" in df.columns:
        df["description"] = df["description"].fillna(df["description_plan"])
        df.drop(columns=["description_plan"], inplace=True, errors="ignore")
    return df


def compute_composite_score(df):
    """Add a 'composite_score' column (lower = better) based on mean rank.

    1. Average each metric per run across systems.
    2. Rank runs for each metric (rank 1 = best).
    3. Average the ranks across metrics → mean_rank per run.
    4. Broadcast back to every row so the rest of the code works unchanged.
    """
    df = df.copy()

    # Metrics to use (only those present)
    metrics = [m for m in SCORE_WEIGHTS if m in df.columns]

    # Mean metric value per run (across systems)
    run_means = df.groupby("run_id")[metrics].mean()

    # Rank runs for each metric (rank 1 = best)
    run_ranks = pd.DataFrame(index=run_means.index)
    for metric in metrics:
        ascending = SCORE_WEIGHTS[metric] < 0  # lower is better → rank ascending
        run_ranks[metric] = run_means[metric].rank(
            ascending=ascending, method="average", na_option="bottom",
        )

    # Mean rank across all metrics (simple average, no weighting)
    run_ranks["mean_rank"] = run_ranks[metrics].mean(axis=1)

    # Broadcast back: lower mean_rank = better, but existing code expects
    # higher composite_score = better, so invert.
    max_rank = run_ranks["mean_rank"].max()
    run_ranks["composite_score"] = max_rank - run_ranks["mean_rank"]

    df = df.merge(
        run_ranks[["composite_score", "mean_rank"]],
        left_on="run_id", right_index=True, how="left",
    )
    return df


def plot_recommendations(df, prior_label, categories, baseline_id, outdir):
    """Bar chart of best run per category, ranked by composite score."""
    df = compute_composite_score(df)

    bl_score = df[df["run_id"] == baseline_id].groupby("system_name")["composite_score"].mean().mean()

    results = []
    for cat in categories:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        # Best run = highest mean composite score across systems
        mean_scores = sub.groupby(["run_id", "description"])["composite_score"].mean()
        if mean_scores.empty:
            continue
        best_idx = mean_scores.idxmax()
        best_run_id, best_desc = best_idx
        results.append({
            "category": CATEGORY_LABELS.get(cat, cat),
            "description": best_desc,
            "run_id": best_run_id,
            "score": mean_scores[best_idx],
            "delta": mean_scores[best_idx] - bl_score,
        })

    if not results:
        print(f"  [SKIP] No results for {prior_label}")
        return

    res_df = pd.DataFrame(results).sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(res_df) * 0.5)))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in res_df["delta"]]
    y = np.arange(len(res_df))
    ax.barh(y, res_df["score"], color=colors, height=0.7)
    ax.axvline(bl_score, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline ({prior_label})")

    labels = [f"{row['category']}: {row['description']}" for _, row in res_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Score composite", fontsize=10)
    ax.set_title(f"Meilleure config par catégorie — {prior_label}\n"
                 f"(vert = meilleur que baseline, rouge = pire)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(outdir / f"recommendations_{prior_label}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved recommendations_{prior_label}.png")


def plot_delta_vs_baseline(df, prior_label, categories, baseline_id, outdir):
    """For each run in the categories, plot delta from baseline per system."""
    df = compute_composite_score(df)

    bl = df[df["run_id"] == baseline_id].set_index("system_name")["composite_score"]
    sub = df[df["category"].isin(categories)].copy()
    if sub.empty:
        return

    sub["delta"] = sub.apply(
        lambda row: row["composite_score"] - bl.get(row["system_name"], 0), axis=1
    )

    agg = sub.groupby("description")["delta"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.35)))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in agg.values]
    y = np.arange(len(agg))
    ax.barh(y, agg.values, color=colors, height=0.7)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=7)
    ax.set_xlabel("Δ score composite vs baseline", fontsize=10)
    ax.set_title(f"Écart au baseline — {prior_label}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(outdir / f"delta_vs_baseline_{prior_label}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved delta_vs_baseline_{prior_label}.png")


# ────────────────────────────────────────────────────────────
# Per-parameter analysis: best value for each hyperparameter
# ────────────────────────────────────────────────────────────

def _parse_varied_param(varied_str):
    """Extract param name and tested value from varied_params column.

    'mcmc.num_warmup: 2000 -> 500'  -> ('mcmc.num_warmup', '500')
    'none (baseline)'               -> (None, None)
    """
    if pd.isna(varied_str) or "baseline" in str(varied_str).lower():
        return None, None
    first = str(varied_str).split("|")[0].strip()
    m = re.match(r"([\w.]+):\s*.+?\s*->\s*(.+)", first)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _get_baseline_value(param_name, plan_df):
    """Get the baseline (default) value for a parameter from the plan."""
    for _, row in plan_df.iterrows():
        vp = str(row.get("varied_params", ""))
        if param_name in vp and "->" in vp:
            first = vp.split("|")[0].strip()
            m = re.match(r"[\w.]+:\s*(.+?)\s*->\s*.+", first)
            if m:
                return m.group(1).strip()
    return "?"


def analyze_per_parameter(df, plan_df, outdir):
    """For each hyperparameter, find the best value across all systems.

    Groups runs by the parameter they vary, includes the baseline, and
    picks the value with the highest mean composite score across systems.
    """
    df = compute_composite_score(df)

    # Build groups: (param_name, prior_type) -> [(run_id, tested_value)]
    groups = {}
    for _, row in plan_df.iterrows():
        param_name, tested_val = _parse_varied_param(row.get("varied_params", ""))
        if param_name is None:
            continue
        cat = str(row.get("category", ""))
        if cat.endswith("_ss"):
            prior = "SS"
        elif cat.endswith("_hs"):
            prior = "HS"
        else:
            continue
        run_id = row["run_id"]
        if run_id not in df["run_id"].values:
            continue
        key = (param_name, prior)
        if key not in groups:
            groups[key] = []
        groups[key].append((run_id, tested_val))

    # Analyze each group
    all_recs = []
    for (param_name, prior_type), run_list in sorted(groups.items()):
        bl_id = BASELINE_SS if prior_type == "SS" else BASELINE_HS
        bl_value = _get_baseline_value(param_name, plan_df)

        # Run IDs: baseline + tested values
        entries = [(bl_id, f"{bl_value} (baseline)")]
        for rid, val in run_list:
            entries.append((rid, str(val)))

        # Mean composite score across systems for each value
        best_score = -np.inf
        best_label = None
        best_rid = None
        bl_score = None
        value_scores = []

        for rid, label in entries:
            sub = df[df["run_id"] == rid]
            if sub.empty:
                continue
            mean_score = sub["composite_score"].mean()
            value_scores.append((label, mean_score))
            if rid == bl_id:
                bl_score = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_label = label
                best_rid = rid

        if bl_score is None:
            continue

        is_baseline = best_rid == bl_id
        delta = best_score - bl_score

        all_recs.append({
            "param": param_name,
            "prior": prior_type,
            "best_value": best_label,
            "best_run_id": best_rid,
            "best_score": best_score,
            "baseline_score": bl_score,
            "delta": delta,
            "is_baseline_best": is_baseline,
            "n_tested": len(entries),
            "all_scores": value_scores,
        })

    if not all_recs:
        print("  No per-parameter recommendations computed.")
        return

    # ── Text output ──
    _print_per_param_text(all_recs, outdir)

    # ── Table figures ──
    _plot_per_param_table(all_recs, outdir)

    # ── Detail plots per parameter ──
    detail_dir = outdir / "recommendation_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for rec in all_recs:
        _plot_param_detail(rec, detail_dir)


def _print_per_param_text(all_recs, outdir):
    """Print and save text recommendations per parameter."""
    lines = []
    lines.append("=" * 95)
    lines.append("RECOMMANDATIONS PAR HYPERPARAMÈTRE")
    lines.append("(rang moyen sur les 5 métriques: F1, NMSPE, NMSE, log-lik, CI_coverage)")
    lines.append("(chaque métrique classée séparément, puis moyenne des rangs — rang bas = meilleur)")
    lines.append("=" * 95)

    for prior_type in ["HS", "SS"]:
        prior_label = "Horseshoe" if prior_type == "HS" else "Spike-and-Slab"
        sub = [r for r in all_recs if r["prior"] == prior_type]
        if not sub:
            continue

        lines.append(f"\n{'─' * 95}")
        lines.append(f"  {prior_label}")
        lines.append(f"{'─' * 95}")
        lines.append(
            f"  {'Paramètre':<35s} {'Meilleure valeur':<28s} "
            f"{'Score':<10s} {'Δ vs BL':<12s} {'Verdict'}"
        )
        lines.append(f"  {'─' * 90}")

        for rec in sorted(sub, key=lambda r: r["param"]):
            delta = rec["delta"]
            if rec["is_baseline_best"]:
                verdict = "baseline OK"
            elif delta > 0.005:
                verdict = "AMÉLIORATION"
            elif delta < -0.005:
                verdict = "dégradation"
            else:
                verdict = "≈ équivalent"

            delta_str = f"{delta:+.4f}" if not rec["is_baseline_best"] else "—"
            lines.append(
                f"  {rec['param']:<35s} {rec['best_value']:<28s} "
                f"{rec['best_score']:<10.4f} {delta_str:<12s} {verdict}"
            )

    # Summary
    changes = [r for r in all_recs if not r["is_baseline_best"] and r["delta"] > 0.005]
    lines.append(f"\n{'=' * 95}")
    if changes:
        lines.append("PARAMÈTRES À MODIFIER (amélioration vs baseline) :")
        for r in sorted(changes, key=lambda x: -x["delta"]):
            p = "HS" if r["prior"] == "HS" else "SS"
            lines.append(
                f"  [{p}] {r['param']:<30s} → {r['best_value']:<25s} "
                f"(Δ = {r['delta']:+.4f})"
            )
    else:
        lines.append("La baseline est optimale (ou quasi-optimale) pour tous les paramètres.")
    lines.append("=" * 95)

    summary_text = "\n".join(lines)
    print(summary_text)

    (outdir / "recommendations.txt").write_text(summary_text, encoding="utf-8")
    print(f"\n  Saved recommendations.txt")


def _plot_per_param_table(all_recs, outdir):
    """Summary table figure per prior type."""
    for prior_type in ["HS", "SS"]:
        prior_label = "Horseshoe" if prior_type == "HS" else "Spike-and-Slab"
        sub = sorted(
            [r for r in all_recs if r["prior"] == prior_type],
            key=lambda r: r["param"],
        )
        if not sub:
            continue

        table_data = []
        cell_colors = []
        for rec in sub:
            delta = rec["delta"]
            if rec["is_baseline_best"]:
                delta_str = "= baseline"
                color = "#d4edda"  # green
            elif delta > 0.005:
                delta_str = f"{delta:+.4f}"
                color = "#cce5ff"  # blue = improvement
            elif delta < -0.005:
                delta_str = f"{delta:+.4f}"
                color = "#f8d7da"  # red = worse
            else:
                delta_str = f"{delta:+.4f}"
                color = "#fff3cd"  # yellow = negligible

            table_data.append([
                rec["param"],
                rec["best_value"],
                f"{rec['best_score']:.4f}",
                f"{rec['baseline_score']:.4f}",
                delta_str,
            ])
            cell_colors.append(["white", color, "white", "white", color])

        col_labels = ["Paramètre", "Meilleure valeur", "Score", "Score BL", "Δ"]

        fig_height = max(3, 0.45 * len(table_data) + 1.5)
        fig, ax = plt.subplots(figsize=(15, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=cell_colors,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        header_color = "#4C72B0" if prior_type == "HS" else "#DD8452"
        for j in range(len(col_labels)):
            table[0, j].set_facecolor(header_color)
            table[0, j].set_text_props(color="white", fontweight="bold")

        fig.suptitle(
            f"Recommandation par hyperparamètre — {prior_label}\n"
            f"(vert = baseline optimale, bleu = amélioration possible)",
            fontsize=12, fontweight="bold", y=0.98,
        )
        fig.tight_layout()
        fig.savefig(
            outdir / f"per_param_recommendations_{prior_type}.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved per_param_recommendations_{prior_type}.png")


def _plot_param_detail(rec, detail_dir):
    """Bar chart for one parameter: composite score for each tested value."""
    scores = rec["all_scores"]
    if not scores:
        return

    labels = [s[0] for s in scores]
    vals = [s[1] for s in scores]
    prior_type = rec["prior"]

    fig, ax = plt.subplots(figsize=(max(4, len(scores) * 1.2), 4))

    colors = []
    for label in labels:
        if "(baseline)" in label:
            colors.append("#2ca02c")
        elif prior_type == "HS":
            colors.append("#4C72B0")
        else:
            colors.append("#DD8452")

    bars = ax.bar(range(len(scores)), vals, color=colors, alpha=0.85)

    # Highlight best
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Score composite", fontsize=9)
    prior_label = "Horseshoe" if prior_type == "HS" else "Spike-and-Slab"
    ax.set_title(f"{rec['param']} ({prior_label})", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    safe_name = rec["param"].replace(".", "_")
    fig.savefig(detail_dir / f"{safe_name}_{prior_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    print("ANALYSE DES RECOMMANDATIONS — BENCHMARK BaSIC")
    print("=" * 70)

    df = load_data(args.results, args.plan)
    plan_df = pd.read_csv(args.plan)
    print(f"Loaded {len(df)} rows\n")

    # Per-category recommendations
    print("--- Recommandations par catégorie ---")
    plot_recommendations(df, "HS", HS_CATEGORIES, BASELINE_HS, outdir)
    plot_recommendations(df, "SS", SS_CATEGORIES, BASELINE_SS, outdir)

    # Delta vs baseline
    print("\n--- Delta vs baseline ---")
    plot_delta_vs_baseline(df, "HS", HS_CATEGORIES, BASELINE_HS, outdir)
    plot_delta_vs_baseline(df, "SS", SS_CATEGORIES, BASELINE_SS, outdir)

    # Per-parameter recommendations
    print("\n--- Recommandations par paramètre ---")
    analyze_per_parameter(df, plan_df, outdir)

    print(f"\n{'=' * 70}")
    print(f"Figures sauvegardées dans: {outdir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
