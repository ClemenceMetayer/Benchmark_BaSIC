"""Patch existing benchmark results with log-likelihood values.

Scans all out_*_run* directories, loads the MCMC artifacts, computes the
Gaussian log-likelihood, patches each benchmark_metrics.json, and rebuilds
all_benchmark_results.csv.

No MCMC re-run needed — everything is computed from stored artifacts.

Usage:
    python metrics/patch_log_likelihood.py
"""

import json
import sys
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SYSTEMS = ["lotka_volterra", "chain", "seir", "goldbeter", "yeast_glycolysis"]


def compute_log_likelihood_from_artifacts(artifacts_path: Path) -> dict:
    """Compute log-likelihood from a sparse artifacts .npz file."""
    artifacts = dict(np.load(artifacts_path, allow_pickle=True))

    obs_idx = artifacts["obs_idx"]
    cond_names = artifacts["cond_names"].tolist()

    X_post = artifacts.get("X_post_draws")   # (S, C, T, n_eq)
    sigma = artifacts.get("sigma")            # (S, C, n_obs)

    if X_post is None or sigma is None:
        return {
            "log_lik_median": float("nan"),
            "log_lik_mean": float("nan"),
            "log_lik_std": float("nan"),
            "success": False,
            "error": "X_post_draws or sigma not found",
        }

    S = X_post.shape[0]
    log_lik_samples = np.zeros(S)

    for c_idx, cond_name in enumerate(cond_names):
        y_key = f"y_{cond_name}"
        if y_key not in artifacts:
            continue
        y_obs = artifacts[y_key]

        # Average over replicates if needed
        if y_obs.ndim == 3:
            y_mean = np.nanmean(y_obs, axis=1)
        else:
            y_mean = y_obs

        valid = np.isfinite(y_mean)

        for s in range(S):
            y_pred = X_post[s, c_idx, :, obs_idx]
            sigma_s = sigma[s, c_idx, :]

            residuals = y_mean - y_pred
            log_p = (
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma_s[None, :])
                - 0.5 * (residuals / sigma_s[None, :]) ** 2
            )
            log_lik_samples[s] += np.where(valid, log_p, 0.0).sum()

    return {
        "log_lik_median": float(np.median(log_lik_samples)),
        "log_lik_mean": float(np.mean(log_lik_samples)),
        "log_lik_std": float(np.std(log_lik_samples)),
        "success": True,
    }


def patch_one(outdir: Path) -> bool:
    """Patch a single experiment's benchmark_metrics.json with log-likelihood."""
    metrics_path = outdir / "benchmark_metrics.json"
    artifacts_path = outdir / "multi_cond_sparse_inference" / "artifacts_sparse.npz"

    if not metrics_path.exists():
        return False
    if not artifacts_path.exists():
        print(f"  [SKIP] no artifacts: {outdir.name}")
        return False

    # Check if already patched
    with open(metrics_path) as f:
        results = json.load(f)

    existing = results.get("log_likelihood", {})
    if existing.get("success") and existing.get("log_lik_median") is not None:
        med = existing["log_lik_median"]
        if np.isfinite(med):
            print(f"  [OK]   already patched: {outdir.name} (median={med:.1f})")
            return True

    # Compute
    print(f"  [CALC] {outdir.name} ... ", end="", flush=True)
    log_lik = compute_log_likelihood_from_artifacts(artifacts_path)

    if log_lik["success"]:
        print(f"median={log_lik['log_lik_median']:.1f}")
    else:
        print(f"FAILED: {log_lik.get('error', '?')}")

    # Patch JSON
    results["log_likelihood"] = {k: v for k, v in log_lik.items()}
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return log_lik["success"]


def rebuild_csv():
    """Rebuild the global CSV by calling run_all_experiments.rebuild_global_csv."""
    sys.path.insert(0, str(BENCHMARK_DIR))
    try:
        import run_all_experiments as rae
        plan = rae.load_plan()
        rae.rebuild_global_csv(SYSTEMS, plan)
    except Exception as e:
        print(f"\n[WARN] Could not rebuild CSV: {e}")


def main():
    print("=" * 60)
    print("PATCH LOG-LIKELIHOOD INTO EXISTING RESULTS")
    print("=" * 60)

    outdirs = sorted(BENCHMARK_DIR.glob("out_*_run*"))
    if not outdirs:
        print("\nNo out_*_run* directories found. Nothing to patch.")
        return

    print(f"\nFound {len(outdirs)} experiment directories.\n")

    n_patched = 0
    n_skipped = 0
    n_failed = 0

    for outdir in outdirs:
        if not outdir.is_dir():
            continue
        result = patch_one(outdir)
        if result:
            n_patched += 1
        elif (outdir / "benchmark_metrics.json").exists():
            n_failed += 1
        else:
            n_skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Patched: {n_patched}  |  Failed: {n_failed}  |  Skipped: {n_skipped}")
    print(f"{'=' * 60}")

    if n_patched > 0:
        print("\nRebuilding all_benchmark_results.csv ...")
        rebuild_csv()

    print("\nDone.")


if __name__ == "__main__":
    main()
