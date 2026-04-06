"""Run all benchmark experiments for BaSIC hyperparameter evaluation.

For each system and each run configuration from experiment_plan.csv:
  1. Creates a temporary YAML config with the specified hyperparameters
  2. Runs the BaSIC pipeline (main.py: sparse inference + refit)
  3. Runs benchmark_metrics.py to compute evaluation metrics
  4. Collects all results in all_benchmark_results.csv

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --systems lotka_volterra chain --runs 0 1 2
    python run_all_experiments.py --workers 4          # run 4 experiments in parallel
    python run_all_experiments.py --skip-refit          # sparse inference only
    python run_all_experiments.py --skip-metrics        # skip benchmark_metrics.py
    python run_all_experiments.py --plan my_plan.csv    # custom plan file
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────

BASIC_DIR = Path(__file__).resolve().parent.parent / "3. BaSIC"
BENCHMARK_DIR = Path(__file__).resolve().parent
JAX_CACHE_DIR = BENCHMARK_DIR / ".jax_cache"


def get_available_systems():
    """Get list of available ODE systems from BaSIC/systems/ directory."""
    systems_dir = BASIC_DIR / "systems"
    yaml_files = list(systems_dir.glob("*.yaml"))
    # Exclude non-system YAML files
    exclude = {"chain_real_test_csv"}
    systems = sorted(
        f.stem for f in yaml_files
        if f.stem not in exclude and f.stem != "__pycache__"
    )
    return systems


def load_plan(csv_path=None):
    if csv_path is None:
        csv_path = str(BENCHMARK_DIR / "config" / "experiment_plan.csv")
    """Load hyperparameter configurations from the experiment plan CSV."""
    runs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up empty strings and convert types
            cleaned = {}
            for k, v in row.items():
                if v == "" or v is None:
                    cleaned[k] = None
                elif v in ("True", "true"):
                    cleaned[k] = True
                elif v in ("False", "false"):
                    cleaned[k] = False
                else:
                    try:
                        # Try int first, then float
                        if "." in v or "e" in v.lower():
                            cleaned[k] = float(v)
                        else:
                            cleaned[k] = int(v)
                    except (ValueError, TypeError):
                        cleaned[k] = v
            runs.append(cleaned)
    return runs


def build_yaml_config(system_name, run_config, output_path):
    """Build a YAML config by merging the system base config with run overrides.

    Returns the path to the written temporary YAML file.
    """
    base_yaml = BASIC_DIR / "systems" / f"{system_name}.yaml"
    with open(base_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Map flat CSV columns to nested YAML keys
    mapping = {
        # Prior
        "sparse_prior.type":            ("sparse_prior", "type"),
        "sparse_prior.tau0":            ("sparse_prior", "tau0"),
        "sparse_prior.slab_scale":      ("sparse_prior", "slab_scale"),
        "sparse_prior.degree_penalty":  ("sparse_prior", "degree_penalty"),
        "sparse_prior.spike_sd":        ("sparse_prior", "spike_sd"),
        "sparse_prior.slab_sd":         ("sparse_prior", "slab_sd"),
        "sparse_prior.theta_a":         ("sparse_prior", "theta_a"),
        "sparse_prior.theta_b":         ("sparse_prior", "theta_b"),
        # MCMC
        "mcmc.num_warmup":      ("mcmc", "num_warmup"),
        "mcmc.num_samples":     ("mcmc", "num_samples"),
        "mcmc.num_chains":      ("mcmc", "num_chains"),
        "mcmc.thinning":        ("mcmc", "thinning"),
        "mcmc.target_accept":   ("mcmc", "target_accept"),
        "mcmc.max_treedepth":   ("mcmc", "max_treedepth"),
        # Selection
        "selection.strategy":       ("selection", "strategy"),
        "selection.threshold":      ("selection", "threshold"),
        "selection.hyperparameter": ("selection", "hyperparameter"),
        # Library
        "library.degree":       ("library", "degree"),
        "library.include_bias": ("library", "include_bias"),
        "library.include_mm":   ("library", "include_mm"),
        # Integrator
        "integrator.shooting":  ("integrator", "shooting"),
        "integrator.segments":  ("integrator", "segments"),
        "integrator.rtol":      ("integrator", "rtol"),
        "integrator.atol":      ("integrator", "atol"),
        "integrator.max_steps": ("integrator", "max_steps"),
        # Initialization
        "initialization.strategy":  ("initialization", "strategy"),
        "initialization.std_noise": ("initialization", "std_noise"),
        # Data
        "observations.noise_fraction": ("observations", "noise_fraction"),
        "observations.replicates":     ("observations", "replicates"),
        "time.T":                      ("time", "T"),
    }

    for csv_key, yaml_path in mapping.items():
        value = run_config.get(csv_key)
        if value is None:
            continue

        section, key = yaml_path
        if section not in cfg:
            cfg[section] = {}
        cfg[section][key] = value

    # shooting_scale is read from the integrator section by BaSIC
    shooting_scale = run_config.get("shooting_scale")
    if shooting_scale is not None:
        if "integrator" not in cfg:
            cfg["integrator"] = {}
        cfg["integrator"]["shooting_scale"] = shooting_scale

    # Clean up prior keys: remove horseshoe-specific keys for spike_and_slab and vice versa
    # degree_penalty is shared by both priors
    prior_type = cfg.get("sparse_prior", {}).get("type", "horseshoe")
    if prior_type == "spike_and_slab":
        for k in ("tau0", "slab_scale"):
            cfg["sparse_prior"].pop(k, None)
    elif prior_type == "horseshoe":
        for k in ("spike_sd", "slab_sd", "theta_a", "theta_b"):
            cfg["sparse_prior"].pop(k, None)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return output_path


def is_experiment_done(system_name, run_id):
    """Check if an experiment has already completed successfully."""
    outdir = BENCHMARK_DIR / f"out_{system_name}_run{run_id}"
    return (outdir / "benchmark_metrics.json").exists()


def rebuild_global_csv(systems, plan):
    """Rebuild all_benchmark_results.csv from individual benchmark_metrics.json files.

    This avoids duplicates by always writing a fresh CSV from the source JSONs.
    """
    global_csv = BENCHMARK_DIR / "all_benchmark_results.csv"
    rows = []

    for system in systems:
        for run_config in plan:
            run_id = run_config["run_id"]
            metrics_path = BENCHMARK_DIR / f"out_{system}_run{run_id}" / "benchmark_metrics.json"
            if not metrics_path.exists():
                continue

            with open(metrics_path, "r") as f:
                results = json.load(f)

            # Flatten nested results
            f1 = results.get("structure", {}).get("f1", {})
            param = results.get("parameters", {})
            ci = results.get("ci_coverage", {})
            pred = results.get("prediction", {})
            train = results.get("training_fit", {})
            diag = results.get("diagnostics", {})
            timings = results.get("timings", {})
            meta = results.get("metadata", {})

            row = {
                "run_id": run_id,
                "system_name": system,
                "category": run_config.get("category", ""),
                "description": run_config.get("description", ""),
                "timestamp": meta.get("timestamp", ""),
                "outdir": meta.get("outdir", ""),
                # Structure
                "F1": f1.get("F1"),
                "Precision": f1.get("Precision"),
                "Recall": f1.get("Recall"),
                "TP": f1.get("TP"),
                "FP": f1.get("FP"),
                "TN": f1.get("TN"),
                "FN": f1.get("FN"),
                "Hamming": results.get("structure", {}).get("hamming"),
                # Parameters
                "NMSPE_active": param.get("NMSPE_active"),
                "NMSPE_all": param.get("NMSPE_all"),
                "FP_MSE": param.get("FP_MSE"),
                "n_active_terms": param.get("n_active_terms"),
                # CI coverage
                "CI_coverage_active": ci.get("CI_coverage_active"),
                "CI_coverage_all": ci.get("CI_coverage_all"),
                "CI_mean_width_active": ci.get("CI_mean_width_active"),
                # Prediction
                "NMSE_training": train.get("NMSE_training"),
                "NMSE_new_x0": pred.get("NMSE"),
                # Diagnostics
                "rhat_max_sparse": diag.get("rhat_max_sparse"),
                "ess_min_sparse": diag.get("ess_min_sparse"),
                # Timings
                "time_sparse_s": timings.get("step1_sparse_s"),
                "time_refit_s": timings.get("step2_refit_s"),
                "time_total_s": timings.get("total_with_viz_s"),
                # Metadata
                "n_mcmc_samples": meta.get("n_mcmc_samples"),
                "n_species": meta.get("n_species"),
                "n_features": meta.get("n_features"),
            }
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(global_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRebuilt global CSV: {global_csv} ({len(rows)} rows)")
    else:
        print("\nNo completed experiments found — no CSV generated.")


def run_single_experiment(system_name, run_config, skip_refit=False,
                          skip_metrics=False, python=sys.executable):
    """Run one experiment: BaSIC pipeline + benchmark metrics.

    Returns (system_name, run_id, success, elapsed_seconds, error_message).
    """
    run_id = run_config["run_id"]
    outdir = BENCHMARK_DIR / f"out_{system_name}_run{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save run config for traceability
    config_record = {k: v for k, v in run_config.items() if v is not None}
    (outdir / "run_config.json").write_text(
        json.dumps(config_record, indent=2, default=str), encoding="utf-8"
    )

    # Build temporary YAML
    temp_yaml = outdir / "config.yaml"
    build_yaml_config(system_name, run_config, temp_yaml)

    t0 = time.time()
    log_path = outdir / "experiment.log"

    # Share JAX compilation cache across subprocesses
    env = os.environ.copy()
    JAX_CACHE_DIR.mkdir(exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = str(JAX_CACHE_DIR)

    try:
        # ── Step 1: BaSIC pipeline ──
        cmd = [
            python, str(BASIC_DIR / "main.py"),
            "--yaml", str(temp_yaml),
            "--outdir", str(outdir),
            "--no-viz",  # skip visualisation for benchmark
        ]
        if skip_refit:
            # Run only sparse inference by calling step 1 directly
            cmd = [
                python, str(BASIC_DIR / "1_multi_condition_sparse_inference.py"),
                "--yaml", str(temp_yaml),
                "--outdir", str(outdir),
            ]

        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                cwd=str(BASIC_DIR), env=env,
            )

        if result.returncode != 0:
            elapsed = time.time() - t0
            return system_name, run_id, False, elapsed, f"Pipeline failed (rc={result.returncode})"

        # ── Step 2: Benchmark metrics ──
        if not skip_metrics:
            cmd_metrics = [
                python, str(BENCHMARK_DIR / "metrics" / "benchmark_metrics.py"),
                "--outdir", str(outdir),
                "--yaml", str(temp_yaml),
            ]
            with open(log_path, "a") as log_file:
                log_file.write("\n\n" + "=" * 80 + "\nBENCHMARK METRICS\n" + "=" * 80 + "\n")
                result_m = subprocess.run(
                    cmd_metrics, stdout=log_file, stderr=subprocess.STDOUT,
                    cwd=str(BASIC_DIR), env=env,
                )
            if result_m.returncode != 0:
                elapsed = time.time() - t0
                return system_name, run_id, False, elapsed, f"Metrics failed (rc={result_m.returncode})"

        elapsed = time.time() - t0
        return system_name, run_id, True, elapsed, None

    except Exception as e:
        elapsed = time.time() - t0
        return system_name, run_id, False, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run all BaSIC benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Systems to run (default: all). E.g.: --systems lotka_volterra chain"
    )
    parser.add_argument(
        "--runs", nargs="+", type=int, default=None,
        help="Run IDs to execute (default: all). E.g.: --runs 0 1 2"
    )
    parser.add_argument(
        "--plan", default=str(BENCHMARK_DIR / "config" / "experiment_plan.csv"),
        help="Path to experiment plan CSV (default: config/experiment_plan.csv)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of experiments to run in parallel (default: 1)"
    )
    parser.add_argument(
        "--skip-refit", action="store_true",
        help="Skip the refit step (only run sparse inference)"
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip benchmark_metrics.py computation"
    )
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help=f"Python interpreter (default: {sys.executable})"
    )

    args = parser.parse_args()

    # ── Resolve systems ──
    available = get_available_systems()
    if args.systems:
        systems = [s for s in args.systems if s in available]
        missing = set(args.systems) - set(systems)
        if missing:
            print(f"WARNING: systems not found: {missing}")
    else:
        systems = available

    if not systems:
        print(f"ERROR: no valid systems. Available: {available}")
        sys.exit(1)

    # ── Load plan ──
    if not Path(args.plan).exists():
        print(f"ERROR: plan file not found: {args.plan}")
        print("Run 'python generate_plan.py' first to create it.")
        sys.exit(1)

    plan = load_plan(args.plan)

    if args.runs is not None:
        plan = [r for r in plan if r["run_id"] in args.runs]

    if not plan:
        print("ERROR: no runs selected.")
        sys.exit(1)

    # ── Summary ──
    total = len(systems) * len(plan)
    print("=" * 70)
    print("BASIC BENCHMARK — EXPERIMENT PLAN")
    print("=" * 70)
    print(f"Systems ({len(systems)}): {', '.join(systems)}")
    print(f"Configurations: {len(plan)}")
    print(f"Total experiments: {total}")
    print(f"Workers: {args.workers}")
    print(f"Skip refit: {args.skip_refit}")
    print(f"Skip metrics: {args.skip_metrics}")
    print(f"BaSIC directory: {BASIC_DIR}")
    print("=" * 70)

    # ── Build task list (skip already completed experiments) ──
    tasks = []
    skipped = 0
    for system in systems:
        for run_config in plan:
            if is_experiment_done(system, run_config["run_id"]):
                skipped += 1
            else:
                tasks.append((system, run_config))

    if skipped:
        print(f"\nSkipping {skipped} already completed experiments.")
    print(f"Remaining: {len(tasks)} experiments to run.\n")

    if not tasks:
        print("All experiments already completed!")
        # Rebuild global CSV before exiting
        rebuild_global_csv(systems, plan)
        sys.exit(0)

    # ── Run experiments ──
    completed = 0
    failed = 0
    failed_list = []
    t_global = time.time()

    if args.workers <= 1:
        # Sequential execution
        for system, run_config in tasks:
            run_id = run_config["run_id"]
            desc = run_config.get("description", "")
            print(f"\n--- {system} / run {run_id}: {desc}")

            sys_name, rid, success, elapsed, err = run_single_experiment(
                system, run_config,
                skip_refit=args.skip_refit,
                skip_metrics=args.skip_metrics,
                python=args.python,
            )

            if success:
                completed += 1
                print(f"    OK ({elapsed:.0f}s)")
            else:
                failed += 1
                failed_list.append(f"{sys_name}_run{rid}: {err}")
                print(f"    FAILED ({elapsed:.0f}s): {err}")

            print(f"    Progress: {completed + failed}/{total}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for system, run_config in tasks:
                future = executor.submit(
                    run_single_experiment,
                    system, run_config,
                    skip_refit=args.skip_refit,
                    skip_metrics=args.skip_metrics,
                    python=args.python,
                )
                futures[future] = (system, run_config["run_id"], run_config.get("description", ""))

            for future in as_completed(futures):
                system, run_id, desc = futures[future]
                sys_name, rid, success, elapsed, err = future.result()

                if success:
                    completed += 1
                    print(f"  OK  {system}/run{run_id} ({elapsed:.0f}s) — {desc}")
                else:
                    failed += 1
                    failed_list.append(f"{sys_name}_run{rid}: {err}")
                    print(f"  FAIL {system}/run{run_id} ({elapsed:.0f}s) — {err}")

                print(f"       Progress: {completed + failed}/{total}")

    # ── Rebuild global CSV from all completed experiments ──
    rebuild_global_csv(systems, plan)

    # ── Final summary ──
    total_time = time.time() - t_global
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total experiments: {total}")
    print(f"Completed (this run): {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    if failed_list:
        print(f"\nFailed experiments:")
        for exp in failed_list:
            print(f"  - {exp}")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
