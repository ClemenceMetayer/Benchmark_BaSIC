"""Generate the experimental plan for the BaSIC hyperparameter benchmark.

Design: One-Factor-At-a-Time (OFAT) around two baselines (horseshoe and
spike-and-slab), with each hyperparameter varied independently.

Usage:
    python generate_plan.py                  # writes experiment_plan.csv
    python generate_plan.py --output my.csv  # custom output path
"""

import argparse
import csv
import copy
from pathlib import Path


# ────────────────────────────────────────────────────────────
# Baseline configurations
# ────────────────────────────────────────────────────────────

BASELINE_HORSESHOE = {
    # Prior
    "sparse_prior.type": "horseshoe",
    "sparse_prior.tau0": 1.0,
    "sparse_prior.slab_scale": 1.0,
    "sparse_prior.degree_penalty": 1.0,
    "sparse_prior.spike_sd": "",
    "sparse_prior.slab_sd": "",
    "sparse_prior.theta_a": "",
    "sparse_prior.theta_b": "",
    # MCMC
    "mcmc.num_warmup": 2000,
    "mcmc.num_samples": 1000,
    "mcmc.num_chains": 4,
    "mcmc.thinning": 1,
    "mcmc.target_accept": 0.85,
    "mcmc.max_treedepth": 6,
    # Selection
    "selection.strategy": "ci_nonzero_or",
    "selection.threshold": 0.0,
    "selection.hyperparameter": 0.95,
    # Library
    "library.degree": 2,
    "library.include_bias": False,
    "library.include_mm": False,
    # Integrator
    "integrator.shooting": "multiple",
    "integrator.segments": 5,
    "integrator.rtol": 1e-7,
    "integrator.atol": 1e-9,
    "integrator.max_steps": 4000,
    # Multiple shooting
    "shooting_scale": 0.02,
    # Initialization
    "initialization.strategy": "regression",
    "initialization.std_noise": 0.01,
    # Data generation
    "observations.noise_fraction": 0.1,
    "observations.replicates": 4,
    "time.T": 30,
}

BASELINE_SPIKE_AND_SLAB = copy.deepcopy(BASELINE_HORSESHOE)
BASELINE_SPIKE_AND_SLAB.update({
    "sparse_prior.type": "spike_and_slab",
    "sparse_prior.tau0": "",
    "sparse_prior.slab_scale": "",
    "sparse_prior.degree_penalty": 1.0,  # degree_penalty applies to both priors
    "sparse_prior.spike_sd": 0.01,
    "sparse_prior.slab_sd": 1.0,
    "sparse_prior.theta_a": 1.0,
    "sparse_prior.theta_b": 1.0,
    "selection.strategy": "incl_prob",
    "selection.threshold": 0.9,
    "selection.hyperparameter": "",
})


def _vary(baseline, overrides, category, description):
    """Create a run config by overriding specific fields of a baseline."""
    cfg = copy.deepcopy(baseline)
    cfg.update(overrides)
    cfg["category"] = category
    cfg["description"] = description
    # Build a human-readable summary of what changed vs the baseline
    if overrides:
        parts = []
        for k, v in overrides.items():
            baseline_val = baseline.get(k)
            if v != baseline_val:
                parts.append(f"{k}: {baseline_val} -> {v}")
        cfg["varied_params"] = " | ".join(parts) if parts else "none"
    else:
        cfg["varied_params"] = "none (baseline)"
    return cfg


def generate_plan():
    """Return a list of run configurations (dicts)."""
    runs = []
    HS = BASELINE_HORSESHOE
    SS = BASELINE_SPIKE_AND_SLAB

    # ── Baselines ──────────────────────────────────────────
    runs.append(_vary(HS, {}, "baseline", "Horseshoe baseline"))
    runs.append(_vary(SS, {}, "baseline", "Spike-and-slab baseline"))

    # ── Horseshoe-specific prior variations ────────────────
    runs.append(_vary(HS, {"sparse_prior.tau0": 0.1},
                      "prior_hs", "tau0=0.1 (strong sparsity)"))
    runs.append(_vary(HS, {"sparse_prior.tau0": 0.5},
                      "prior_hs", "tau0=0.5"))
    runs.append(_vary(HS, {"sparse_prior.tau0": 2.0},
                      "prior_hs", "tau0=2.0 (weak sparsity)"))
    runs.append(_vary(HS, {"sparse_prior.slab_scale": ""},
                      "prior_hs", "slab_scale=None (standard horseshoe)"))
    runs.append(_vary(HS, {"sparse_prior.slab_scale": 5.0},
                      "prior_hs", "slab_scale=5.0"))

    # ── Spike-and-slab-specific prior variations ───────────
    runs.append(_vary(SS, {"sparse_prior.spike_sd": 0.001},
                      "prior_ss", "spike_sd=0.001 (tighter spike)"))
    runs.append(_vary(SS, {"sparse_prior.spike_sd": 0.1},
                      "prior_ss", "spike_sd=0.1 (wider spike)"))
    runs.append(_vary(SS, {"sparse_prior.slab_sd": 0.5},
                      "prior_ss", "slab_sd=0.5 (narrower slab)"))
    runs.append(_vary(SS, {"sparse_prior.slab_sd": 5.0},
                      "prior_ss", "slab_sd=5.0 (wider slab)"))
    runs.append(_vary(SS, {"sparse_prior.theta_a": 1.0, "sparse_prior.theta_b": 5.0},
                      "prior_ss", "theta_b=5 (sparse Beta prior)"))
    runs.append(_vary(SS, {"sparse_prior.theta_a": 0.5, "sparse_prior.theta_b": 0.5},
                      "prior_ss", "theta_a=theta_b=0.5 (U-shaped Beta)"))

    # ── Shared prior variation: degree_penalty (both priors) ─
    runs.append(_vary(HS, {"sparse_prior.degree_penalty": 0.0},
                      "prior_hs", "degree_penalty=0 (no penalty, HS)"))
    runs.append(_vary(SS, {"sparse_prior.degree_penalty": 0.0},
                      "prior_ss", "degree_penalty=0 (no penalty, SS)"))

    # ── MCMC variations (both baselines) ───────────────────
    for label, base in [("mcmc_hs", HS), ("mcmc_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"mcmc.num_warmup": 500},
                          label, f"warmup=500 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_warmup": 1000},
                          label, f"warmup=1000 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_warmup": 5000},
                          label, f"warmup=5000 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_warmup": 10000},
                          label, f"warmup=10000 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_samples": 250},
                          label, f"samples=250 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_samples": 500},
                          label, f"samples=500 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_samples": 2000},
                          label, f"samples=2000 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_samples": 5000},
                          label, f"samples=5000 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_chains": 2},
                          label, f"chains=2 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_chains": 6},
                          label, f"chains=6 ({tag})"))
        runs.append(_vary(base, {"mcmc.num_chains": 8},
                          label, f"chains=8 ({tag})"))
        runs.append(_vary(base, {"mcmc.thinning": 2},
                          label, f"thinning=2 ({tag})"))
        runs.append(_vary(base, {"mcmc.thinning": 5},
                          label, f"thinning=5 ({tag})"))
        runs.append(_vary(base, {"mcmc.target_accept": 0.80},
                          label, f"target_accept=0.80 ({tag})"))
        runs.append(_vary(base, {"mcmc.target_accept": 0.95},
                          label, f"target_accept=0.95 ({tag})"))
        runs.append(_vary(base, {"mcmc.max_treedepth": 8},
                          label, f"max_treedepth=8 ({tag})"))
        runs.append(_vary(base, {"mcmc.max_treedepth": 10},
                          label, f"max_treedepth=10 ({tag})"))

    # ── Selection variations ───────────────────────────────
    # Horseshoe: only ci_nonzero_or (baseline), vary CI level
    runs.append(_vary(HS, {"selection.hyperparameter": 0.90},
                      "selection_hs", "CI level=0.90"))
    runs.append(_vary(HS, {"selection.hyperparameter": 0.99},
                      "selection_hs", "CI level=0.99"))
    # Spike-and-slab: incl_prob (baseline=0.9), vary threshold
    runs.append(_vary(SS, {"selection.threshold": 0.8},
                      "selection_ss", "incl_prob threshold=0.8"))
    runs.append(_vary(SS, {"selection.threshold": 0.95},
                      "selection_ss", "incl_prob threshold=0.95"))

    # ── Library variations (both baselines) ────────────────
    for label, base in [("library_hs", HS), ("library_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"library.degree": 3},
                          label, f"degree=3 ({tag})"))
        runs.append(_vary(base, {"library.include_bias": True},
                          label, f"include_bias=True ({tag})"))
        runs.append(_vary(base, {"library.include_mm": True},
                          label, f"include_mm=True ({tag})"))

    # ── Integrator variations (both baselines) ─────────────
    for label, base in [("integrator_hs", HS), ("integrator_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"integrator.shooting": "single"},
                          label, f"single shooting ({tag})"))
        runs.append(_vary(base, {"integrator.segments": 10},
                          label, f"segments=10 ({tag})"))
        runs.append(_vary(base, {"integrator.segments": 15},
                          label, f"segments=15 ({tag})"))
        runs.append(_vary(base, {"integrator.rtol": 1e-5, "integrator.atol": 1e-7},
                          label, f"loose tolerance rtol=1e-5 ({tag})"))
        runs.append(_vary(base, {"integrator.rtol": 1e-9, "integrator.atol": 1e-11},
                          label, f"tight tolerance rtol=1e-9 ({tag})"))
        runs.append(_vary(base, {"integrator.max_steps": 2000},
                          label, f"max_steps=2000 ({tag})"))
        runs.append(_vary(base, {"integrator.max_steps": 8000},
                          label, f"max_steps=8000 ({tag})"))

    # ── Multiple shooting scale (both baselines) ──────────
    for label, base in [("shooting_hs", HS), ("shooting_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"shooting_scale": 0.01},
                          label, f"shooting_scale=0.01 ({tag})"))
        runs.append(_vary(base, {"shooting_scale": 0.05},
                          label, f"shooting_scale=0.05 ({tag})"))
        runs.append(_vary(base, {"shooting_scale": 0.1},
                          label, f"shooting_scale=0.1 ({tag})"))
        runs.append(_vary(base, {"shooting_scale": 0.2},
                          label, f"shooting_scale=0.2 ({tag})"))

    # ── Initialization variations (both baselines) ─────────
    for label, base in [("init_hs", HS), ("init_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"initialization.strategy": "median"},
                          label, f"init=median ({tag})"))
        runs.append(_vary(base, {"initialization.std_noise": 0.001},
                          label, f"init_std_noise=0.001 ({tag})"))
        runs.append(_vary(base, {"initialization.std_noise": 0.1},
                          label, f"init_std_noise=0.1 ({tag})"))

    # ── Data generation variations (both baselines) ────────
    for label, base in [("data_hs", HS), ("data_ss", SS)]:
        tag = "HS" if base is HS else "SS"
        runs.append(_vary(base, {"observations.noise_fraction": 0.05},
                          label, f"noise=5% ({tag})"))
        runs.append(_vary(base, {"observations.noise_fraction": 0.2},
                          label, f"noise=20% ({tag})"))
        runs.append(_vary(base, {"observations.noise_fraction": 0.3},
                          label, f"noise=30% ({tag})"))
        runs.append(_vary(base, {"observations.replicates": 2},
                          label, f"replicates=2 ({tag})"))
        runs.append(_vary(base, {"observations.replicates": 8},
                          label, f"replicates=8 ({tag})"))
        runs.append(_vary(base, {"time.T": 15},
                          label, f"T=15 ({tag})"))
        runs.append(_vary(base, {"time.T": 50},
                          label, f"T=50 ({tag})"))

    # Sort by JAX compilation signature so consecutive runs reuse the cache.
    # Parameters that affect compilation (array shapes / static args):
    COMPILATION_KEYS = [
        "library.degree", "library.include_bias", "library.include_mm",
        "integrator.shooting", "integrator.segments", "integrator.max_steps",
        "mcmc.num_chains",
        "observations.replicates", "time.T",
    ]
    runs.sort(key=lambda r: tuple(str(r[k]) for k in COMPILATION_KEYS))

    # Assign run IDs after sorting
    for i, run in enumerate(runs):
        run["run_id"] = i

    return runs


# ────────────────────────────────────────────────────────────
# Column order for CSV
# ────────────────────────────────────────────────────────────

COLUMNS = [
    "run_id", "category", "description", "varied_params",
    # Prior
    "sparse_prior.type",
    "sparse_prior.tau0", "sparse_prior.slab_scale", "sparse_prior.degree_penalty",
    "sparse_prior.spike_sd", "sparse_prior.slab_sd",
    "sparse_prior.theta_a", "sparse_prior.theta_b",
    # MCMC
    "mcmc.num_warmup", "mcmc.num_samples", "mcmc.num_chains",
    "mcmc.thinning", "mcmc.target_accept", "mcmc.max_treedepth",
    # Selection
    "selection.strategy", "selection.threshold", "selection.hyperparameter",
    # Library
    "library.degree", "library.include_bias", "library.include_mm",
    # Integrator
    "integrator.shooting", "integrator.segments",
    "integrator.rtol", "integrator.atol", "integrator.max_steps",
    # Multiple shooting
    "shooting_scale",
    # Init
    "initialization.strategy", "initialization.std_noise",
    # Data
    "observations.noise_fraction", "observations.replicates", "time.T",
]


def write_csv(runs, output_path):
    """Write the experiment plan to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"Wrote {len(runs)} runs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="experiment_plan.csv",
                        help="Output CSV path (default: experiment_plan.csv)")
    args = parser.parse_args()

    runs = generate_plan()
    write_csv(runs, args.output)

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in runs)
    print(f"\nPlan summary ({len(runs)} configurations):")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"\nWith 5 systems → {len(runs) * 5} total experiments")


if __name__ == "__main__":
    main()
