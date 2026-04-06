"""Compute benchmark metrics for a BaSIC pipeline run.

Metrics computed:
  - Structure recovery: F1-score, Hamming distance (from pipeline selection vs true_coeffs_visu)
  - Parameter estimation: NMSPE (Normalized Mean Squared Parameter Error)
  - Prediction quality: NMSE on training data, NMSE on new initial condition
  - Pipeline diagnostics: MCMC convergence (R-hat, ESS, divergences), wall-clock timings

Usage:
    python benchmark_metrics.py --outdir out_lotka_volterra_run0 --yaml config.yaml
"""

import argparse
import csv
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Add BaSIC to sys.path so we can import its modules ──
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASIC_DIR = BENCHMARK_DIR.parent / "3. BaSIC"
if str(BASIC_DIR) not in sys.path:
    sys.path.insert(0, str(BASIC_DIR))

import jax
import jax.numpy as jnp
import diffrax as dfx
import yaml

from src.library.library_features import sindy_library_features
from src.library.library_names import sindy_feature_names
from src.simulate_ODEs.simulate import simulate_ode
from src.simulate_ODEs.integrator import integrate_vf


# ═════════════════════════════════════════════════════════════
# Loading artifacts
# ═════════════════════════════════════════════════════════════

def load_artifacts(outdir: Path):
    """Load all artifacts from a BaSIC pipeline run."""
    sparse_dir = outdir / "multi_cond_sparse_inference"

    artifacts = dict(np.load(sparse_dir / "artifacts_sparse.npz", allow_pickle=True))

    with open(sparse_dir / "structure.json", "r") as f:
        structure = json.load(f)

    meta_path = sparse_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    return artifacts, structure, meta


def load_timings(outdir: Path) -> Dict:
    """Load pipeline timing information."""
    timings_path = outdir / "timings_pipeline.json"
    if timings_path.exists():
        return json.load(open(timings_path))
    return {}


def load_convergence_diagnostics(outdir: Path) -> Dict:
    """Load MCMC convergence diagnostics from sparse inference and refit."""
    result = {}

    # Sparse inference diagnostics
    sparse_diag = outdir / "multi_cond_sparse_inference" / "convergence_diagnostics.json"
    if sparse_diag.exists():
        result["sparse"] = json.load(open(sparse_diag))

    # Refit diagnostics (one per condition)
    refit_dir = outdir / "refit"
    if refit_dir.exists():
        result["refit"] = {}
        for diag_file in refit_dir.glob("convergence_diagnostics_refit_*.json"):
            cond = diag_file.stem.replace("convergence_diagnostics_refit_", "")
            result["refit"][cond] = json.load(open(diag_file))

    return result


# ═════════════════════════════════════════════════════════════
# Structure recovery: F1-score and Hamming distance
# ═════════════════════════════════════════════════════════════

def build_true_structure_matrix(
    yaml_config: Dict,
    Theta_names: List[str],
    species: List[str],
) -> np.ndarray:
    """Build M_true from the true_coeffs_visu section of the YAML.

    M_true[j, k] = 1 if feature k appears in equation j of the true system.
    Uses the first condition's coefficients (structure is shared across conditions).
    """
    n_eq = len(species)
    n_features = len(Theta_names)
    M_true = np.zeros((n_eq, n_features), dtype=int)

    true_coeffs = yaml_config.get("true_coeffs_visu", {})
    if not true_coeffs:
        print("[WARN] No true_coeffs_visu in YAML — cannot compute structure metrics")
        return M_true

    # Use first condition
    first_cond = list(true_coeffs.values())[0]

    for eq_idx, sp in enumerate(species):
        eq_coeffs = first_cond.get(sp, {})
        for feat_name, coeff in eq_coeffs.items():
            if feat_name in Theta_names:
                feat_idx = Theta_names.index(feat_name)
                M_true[eq_idx, feat_idx] = 1

    return M_true


def build_learned_structure_matrix(
    structure: Dict,
    Theta_names: List[str],
    species: List[str],
) -> np.ndarray:
    """Build M_hat from the pipeline's own feature selection (structure.json).

    Uses eq_selected_indices which stores the result of BaSIC's selection strategy.
    Also includes known_terms and fixed_terms as selected.
    """
    n_eq = len(species)
    n_features = len(Theta_names)
    M_hat = np.zeros((n_eq, n_features), dtype=int)

    # Selected features from sparse inference
    eq_selected = structure.get("eq_selected_indices", {})
    for eq_str, feat_indices in eq_selected.items():
        eq_idx = int(eq_str)
        for fi in feat_indices:
            if fi < n_features:
                M_hat[eq_idx, fi] = 1

    # Known terms (estimated outside SINDy but part of the model)
    eq_known = structure.get("eq_known_terms", {})
    for eq_str, terms in eq_known.items():
        eq_idx = int(eq_str)
        for term in terms:
            fi = term["feat_idx"]
            if fi < n_features:
                M_hat[eq_idx, fi] = 1

    # Fixed terms (constant-coefficient terms, also part of the model)
    eq_fixed = structure.get("eq_fixed_terms", {})
    for eq_str, terms in eq_fixed.items():
        eq_idx = int(eq_str)
        for term in terms:
            fi = term["feat_idx"]
            if fi < n_features:
                M_hat[eq_idx, fi] = 1

    return M_hat


def compute_f1_score(M_true: np.ndarray, M_hat: np.ndarray) -> Dict[str, float]:
    """Compute F1-score for structure recovery."""
    m_true = M_true.flatten()
    m_hat = M_hat.flatten()

    TP = int(np.sum((m_true == 1) & (m_hat == 1)))
    FP = int(np.sum((m_true == 0) & (m_hat == 1)))
    TN = int(np.sum((m_true == 0) & (m_hat == 0)))
    FN = int(np.sum((m_true == 1) & (m_hat == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def compute_hamming_distance(M_true: np.ndarray, M_hat: np.ndarray) -> float:
    """Normalized Hamming distance (0 = identical, 1 = completely different)."""
    return float(np.sum(M_true != M_hat) / M_true.size)


# ═════════════════════════════════════════════════════════════
# Parameter estimation: NMSPE
# ═════════════════════════════════════════════════════════════

def compute_nmspe(
    theta_samples: np.ndarray,   # (S, n_eq, n_features)
    Theta_names: List[str],
    species: List[str],
    yaml_config: Dict,
    M_true: np.ndarray,
) -> Dict:
    """Compute NMSPE by comparing learned coefficients to true coefficients.

    Uses true_coeffs_visu to get ground-truth SINDy coefficients (first condition).
    NMSPE = mean((theta_hat - theta_true)^2) / mean(theta_true^2)   [active terms only]
    """
    n_eq = len(species)
    n_features = len(Theta_names)

    # Build true coefficient matrix from YAML
    Xi_true = np.zeros((n_eq, n_features))
    true_coeffs = yaml_config.get("true_coeffs_visu", {})
    if true_coeffs:
        first_cond = list(true_coeffs.values())[0]
        for eq_idx, sp in enumerate(species):
            eq_coeffs = first_cond.get(sp, {})
            for feat_name, coeff in eq_coeffs.items():
                if feat_name in Theta_names:
                    feat_idx = Theta_names.index(feat_name)
                    Xi_true[eq_idx, feat_idx] = float(coeff)

    # Learned coefficients: median over MCMC samples
    Xi_hat = np.median(theta_samples, axis=0)  # (n_eq, n_features)

    # NMSPE on active terms (where M_true == 1)
    active_mask = M_true.astype(bool)
    if active_mask.sum() > 0:
        errors_active = (Xi_hat[active_mask] - Xi_true[active_mask]) ** 2
        norms_active = Xi_true[active_mask] ** 2
        nmspe_active = float(np.mean(errors_active) / max(np.mean(norms_active), 1e-12))
    else:
        nmspe_active = float("nan")

    # Also compute on ALL terms (including zeros)
    errors_all = (Xi_hat - Xi_true) ** 2
    nmspe_all = float(np.mean(errors_all))

    # False positive penalty: MSE on terms where M_true == 0
    inactive_mask = ~active_mask
    fp_mse = float(np.mean(Xi_hat[inactive_mask] ** 2)) if inactive_mask.sum() > 0 else 0.0

    return {
        "NMSPE_active": nmspe_active,
        "NMSPE_all": nmspe_all,
        "FP_MSE": fp_mse,
        "n_active_terms": int(active_mask.sum()),
        "n_total_terms": int(M_true.size),
    }


def compute_ci_coverage(
    theta_samples: np.ndarray,   # (S, n_eq, n_features)
    Theta_names: List[str],
    species: List[str],
    yaml_config: Dict,
    M_true: np.ndarray,
    ci_level: float = 0.95,
) -> Dict:
    """Compute credible interval coverage: fraction of true coefficients falling within the CI.

    A well-calibrated Bayesian model should have coverage close to ci_level.
    Coverage is computed on active terms only (where M_true == 1).
    """
    n_eq = len(species)
    n_features = len(Theta_names)

    # Build true coefficient matrix
    Xi_true = np.zeros((n_eq, n_features))
    true_coeffs = yaml_config.get("true_coeffs_visu", {})
    if true_coeffs:
        first_cond = list(true_coeffs.values())[0]
        for eq_idx, sp in enumerate(species):
            eq_coeffs = first_cond.get(sp, {})
            for feat_name, coeff in eq_coeffs.items():
                if feat_name in Theta_names:
                    feat_idx = Theta_names.index(feat_name)
                    Xi_true[eq_idx, feat_idx] = float(coeff)

    # Compute CI bounds from MCMC samples
    alpha = 1 - ci_level
    ci_lower = np.percentile(theta_samples, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(theta_samples, 100 * (1 - alpha / 2), axis=0)

    # Coverage on active terms
    active_mask = M_true.astype(bool)
    if active_mask.sum() > 0:
        covered = (Xi_true[active_mask] >= ci_lower[active_mask]) & \
                  (Xi_true[active_mask] <= ci_upper[active_mask])
        coverage_active = float(np.mean(covered))
        n_covered_active = int(np.sum(covered))
    else:
        coverage_active = float("nan")
        n_covered_active = 0

    # Coverage on all terms
    covered_all = (Xi_true >= ci_lower) & (Xi_true <= ci_upper)
    coverage_all = float(np.mean(covered_all))

    # Mean CI width on active terms (measures uncertainty spread)
    ci_width = ci_upper - ci_lower
    mean_ci_width_active = float(np.mean(ci_width[active_mask])) if active_mask.sum() > 0 else float("nan")

    return {
        "CI_coverage_active": coverage_active,
        "CI_coverage_all": coverage_all,
        "CI_n_covered_active": n_covered_active,
        "CI_n_active": int(active_mask.sum()),
        "CI_mean_width_active": mean_ci_width_active,
        "CI_level": ci_level,
    }


# ═════════════════════════════════════════════════════════════
# Prediction: NMSE
# ═════════════════════════════════════════════════════════════

def simulate_learned_model(
    Xi: np.ndarray,            # (n_eq, n_features) median coefficients
    x0: np.ndarray,
    t: np.ndarray,
    species: List[str],
    yaml_config: Dict,
) -> np.ndarray:
    """Simulate the learned SINDy model: dx/dt = Xi @ Theta(x)."""
    lib_cfg = yaml_config.get("library", {})
    degree = lib_cfg.get("degree", 2)
    include_bias = lib_cfg.get("include_bias", False)
    include_mm = lib_cfg.get("include_mm", False)

    # Build custom_specs if present
    custom_specs = lib_cfg.get("custom_specs", None)

    Theta_fn = sindy_library_features(
        species=species,
        degree=degree,
        include_bias=include_bias,
        include_mm=include_mm,
        custom_specs=custom_specs,
    )

    Xi_jax = jnp.array(Xi)

    def vf(tt, y, args):
        theta_vec = Theta_fn(y)
        return jnp.dot(Xi_jax, theta_vec)

    ys, success = integrate_vf(
        vf, jnp.array(x0), jnp.array(t),
        args=None,
        saveat=dfx.SaveAt(ts=jnp.array(t)),
        max_steps=10000,
        method="tsit5",
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    return np.array(ys)


def compute_nmse_new_x0(
    theta_samples: np.ndarray,  # (S, n_eq, n_features)
    yaml_config: Dict,
    species: List[str],
    Theta_names: List[str],
    x0_new: np.ndarray | None = None,
) -> Dict:
    """Compute NMSE on a new initial condition (generalization test)."""
    # Build dense time grid
    T_dense = yaml_config["time"].get("TD", 801)
    t0 = yaml_config["time"]["t0"]
    t_end = yaml_config["time"]["t_end"]
    t_eval = np.linspace(t0, t_end, T_dense)

    # Default new x0: use test_x0 from YAML if available, else +10% perturbation
    x0_train = np.array(yaml_config["dynamics"]["x0"])
    if x0_new is None:
        test_x0 = yaml_config.get("dynamics", {}).get("test_x0")
        if test_x0 is not None:
            x0_new = np.array(test_x0)
        else:
            x0_new = x0_train * 1.1

    # Simulate true system (first condition)
    sys_module_name = yaml_config["system"]["module"]
    sysmod = importlib.import_module(sys_module_name)
    tp_all = yaml_config["dynamics"]["true_params"]
    first_val = list(tp_all.values())[0]
    true_params = first_val if isinstance(first_val, dict) else tp_all

    x_true = np.array(simulate_ode(
        jnp.array(t_eval), jnp.array(x0_new),
        sysmod.rhs_true, true_params,
    ))

    # Simulate learned model (median coefficients)
    Xi_median = np.median(theta_samples, axis=0)

    try:
        x_pred = simulate_learned_model(Xi_median, x0_new, t_eval, species, yaml_config)

        mse = float(np.mean((x_pred - x_true) ** 2))
        norm = float(np.mean(x_true ** 2))
        nmse = mse / max(norm, 1e-12)

        if not np.isfinite(nmse):
            nmse = float("nan")

        return {
            "NMSE": nmse,
            "MSE": mse,
            "normalization": norm,
            "x0_new": x0_new.tolist(),
            "success": True,
        }
    except Exception as e:
        return {
            "NMSE": float("nan"),
            "MSE": float("nan"),
            "normalization": float("nan"),
            "x0_new": x0_new.tolist(),
            "success": False,
            "error": str(e),
        }


def compute_nmse_training(
    theta_samples: np.ndarray,  # (S, n_eq, n_features)
    artifacts: Dict,
    yaml_config: Dict,
    species: List[str],
) -> Dict:
    """Compute NMSE on training data (in-sample fit quality)."""
    Xi_median = np.median(theta_samples, axis=0)
    obs_idx = artifacts["obs_idx"]
    cond_names = artifacts["cond_names"].tolist()
    t = artifacts.get("t", None)

    total_mse = 0.0
    total_norm = 0.0
    n_points = 0

    for cond_name in cond_names:
        y_key = f"y_{cond_name}"
        if y_key not in artifacts:
            continue
        y_obs = artifacts[y_key]

        if t is None:
            T = y_obs.shape[0]
            t0 = yaml_config["time"]["t0"]
            t_end = yaml_config["time"]["t_end"]
            t = np.linspace(t0, t_end, T)

        x0 = np.array(yaml_config["dynamics"]["x0"])

        try:
            x_pred = simulate_learned_model(Xi_median, x0, t, species, yaml_config)
            y_pred = x_pred[:, obs_idx]

            # Handle replicates: y_obs can be (T, n_obs) or (T, R, n_obs)
            if y_obs.ndim == 3:
                y_mean = np.nanmean(y_obs, axis=1)  # average over replicates
            else:
                y_mean = y_obs

            mask = ~np.isnan(y_mean)
            if mask.sum() == 0:
                continue

            mse = np.mean((y_pred[mask] - y_mean[mask]) ** 2)
            norm = np.mean(y_mean[mask] ** 2)
            total_mse += mse * mask.sum()
            total_norm += norm * mask.sum()
            n_points += mask.sum()

        except Exception as e:
            print(f"  [WARN] NMSE training failed for condition {cond_name}: {e}")
            continue

    if n_points == 0 or total_norm < 1e-12:
        return {"NMSE_training": float("nan"), "n_points": 0}

    return {
        "NMSE_training": float(total_mse / total_norm),
        "MSE_training": float(total_mse / n_points),
        "n_points": int(n_points),
    }


# ═════════════════════════════════════════════════════════════
# Extract theta samples from artifacts
# ═════════════════════════════════════════════════════════════

def extract_theta_samples(artifacts: Dict, species: List[str], Theta_names: List[str]) -> np.ndarray:
    """Extract and reshape Xi samples to (S, n_eq, n_features)."""
    xi_raw = artifacts["Xi"]

    # Remove condition dimension if present
    if xi_raw.ndim == 4:
        # (S, C, P, n_eq) — take first condition for structure comparison
        xi_raw = xi_raw[:, 0, ...]

    S = xi_raw.shape[0]
    eq_len = len(species)
    feat_len = len(Theta_names)

    # Detect axes
    remaining = list(range(1, xi_raw.ndim))
    eq_axes = [ax for ax in remaining if xi_raw.shape[ax] == eq_len]
    feat_axes = [ax for ax in remaining if xi_raw.shape[ax] == feat_len]

    if not eq_axes or not feat_axes:
        raise ValueError(f"Cannot identify eq/feature axes in Xi.shape={xi_raw.shape}")

    eq_axis = eq_axes[0]
    feat_axis = feat_axes[0]
    assert eq_axis != feat_axis

    return np.transpose(xi_raw, [0, eq_axis, feat_axis])


# ═════════════════════════════════════════════════════════════
# CSV export
# ═════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "timestamp", "system_name", "outdir",
    # Structure
    "F1", "Precision", "Recall", "TP", "FP", "TN", "FN", "Hamming",
    # Parameters
    "NMSPE_active", "NMSPE_all", "FP_MSE", "n_active_terms",
    # CI coverage
    "CI_coverage_active", "CI_coverage_all", "CI_mean_width_active",
    # Prediction
    "NMSE_training", "NMSE_new_x0",
    # MCMC diagnostics (sparse)
    "rhat_max_sparse", "ess_min_sparse",
    # Timings
    "time_sparse_s", "time_refit_s", "time_total_s",
    # Metadata
    "n_mcmc_samples", "n_species", "n_features",
]


def flatten_results(results: Dict) -> Dict:
    """Flatten nested results dict into a single-level dict for CSV."""
    f1 = results.get("structure", {}).get("f1", {})
    param = results.get("parameters", {})
    ci = results.get("ci_coverage", {})
    pred = results.get("prediction", {})
    train = results.get("training_fit", {})
    diag = results.get("diagnostics", {})
    timings = results.get("timings", {})
    meta = results.get("metadata", {})

    return {
        "timestamp": meta.get("timestamp", ""),
        "system_name": meta.get("system_name", ""),
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


def save_to_csv(results: Dict, csv_path: Path):
    """Append a row to the global benchmark CSV."""
    row = flatten_results(results)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=str, required=True,
                        help="BaSIC pipeline output directory")
    parser.add_argument("--yaml", type=str, required=True,
                        help="YAML config file used for this run")
    parser.add_argument("--new_x0", type=str, default=None,
                        help="New x0 for NMSE (format: 'v1,v2,...')")

    args = parser.parse_args()
    outdir = Path(args.outdir)

    print("=" * 70)
    print("BENCHMARK METRICS — BaSIC")
    print("=" * 70)
    print(f"  Output dir: {outdir}")
    print(f"  YAML: {args.yaml}")

    # ── Load everything ──
    with open(args.yaml, "r") as f:
        yaml_config = yaml.safe_load(f)

    artifacts, structure, meta = load_artifacts(outdir)
    timings = load_timings(outdir)
    convergence = load_convergence_diagnostics(outdir)

    Theta_names = structure["Theta_names"]
    species = structure["species"]

    theta_samples = extract_theta_samples(artifacts, species, Theta_names)
    print(f"  Samples: {theta_samples.shape[0]}, "
          f"Equations: {theta_samples.shape[1]}, "
          f"Features: {theta_samples.shape[2]}")

    # ── 1. Structure recovery ──
    print("\n[1/4] Structure recovery (F1, Hamming)...")
    M_true = build_true_structure_matrix(yaml_config, Theta_names, species)
    M_hat = build_learned_structure_matrix(structure, Theta_names, species)

    f1 = compute_f1_score(M_true, M_hat)
    hamming = compute_hamming_distance(M_true, M_hat)

    print(f"  M_true: {M_true.sum()} active terms | M_hat: {M_hat.sum()} selected terms")
    print(f"  F1={f1['F1']:.3f}  Precision={f1['Precision']:.3f}  Recall={f1['Recall']:.3f}")
    print(f"  Hamming={hamming:.4f}")

    # ── 2. Parameter estimation ──
    print("\n[2/5] Parameter estimation (NMSPE)...")
    nmspe = compute_nmspe(theta_samples, Theta_names, species, yaml_config, M_true)
    print(f"  NMSPE (active terms): {nmspe['NMSPE_active']:.6f}")
    print(f"  NMSPE (all terms):    {nmspe['NMSPE_all']:.6f}")

    # ── 3. CI coverage ──
    print("\n[3/5] Credible interval coverage...")
    ci_cov = compute_ci_coverage(theta_samples, Theta_names, species, yaml_config, M_true)
    print(f"  CI coverage (active, 95%): {ci_cov['CI_coverage_active']:.3f}")
    print(f"  CI mean width (active):    {ci_cov['CI_mean_width_active']:.6f}")

    # ── 4. NMSE training ──
    print("\n[4/5] NMSE on training data...")
    nmse_train = compute_nmse_training(theta_samples, artifacts, yaml_config, species)
    print(f"  NMSE_training: {nmse_train.get('NMSE_training', float('nan')):.6f}")

    # ── 5. NMSE new x0 ──
    print("\n[5/5] NMSE on new initial condition...")
    x0_new = None
    if args.new_x0:
        x0_new = np.array([float(x) for x in args.new_x0.split(",")])
    nmse_pred = compute_nmse_new_x0(theta_samples, yaml_config, species, Theta_names, x0_new)
    print(f"  NMSE_new_x0: {nmse_pred.get('NMSE', float('nan')):.6f}")
    print(f"  x0_new: {nmse_pred.get('x0_new')}")

    # ── Extract convergence diagnostics ──
    diag_dict = {}
    sparse_diag = convergence.get("sparse", {})
    if sparse_diag:
        # Extract worst-case diagnostics across all sites
        sites = sparse_diag.get("sites", {})
        if sites:
            rhat_vals = [s.get("rhat_max", 0) for s in sites.values() if "rhat_max" in s]
            ess_vals = [s.get("ess_bulk_min", float("inf")) for s in sites.values() if "ess_bulk_min" in s]
            diag_dict["rhat_max_sparse"] = max(rhat_vals) if rhat_vals else None
            diag_dict["ess_min_sparse"] = min(ess_vals) if ess_vals else None

    # ── Assemble results ──
    system_name = yaml_config.get("system", {}).get("module", Path(args.yaml).stem)
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_name": system_name,
            "outdir": str(outdir),
            "n_mcmc_samples": int(theta_samples.shape[0]),
            "n_species": len(species),
            "n_features": len(Theta_names),
        },
        "structure": {
            "f1": f1,
            "hamming": hamming,
            "M_true_active": int(M_true.sum()),
            "M_hat_selected": int(M_hat.sum()),
        },
        "parameters": nmspe,
        "ci_coverage": ci_cov,
        "training_fit": nmse_train,
        "prediction": nmse_pred,
        "diagnostics": diag_dict,
        "timings": timings,
    }

    # ── Save ──
    # Detailed JSON
    json_path = outdir / "benchmark_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved detailed results: {json_path}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  F1-Score:        {f1['F1']:.4f}")
    print(f"  Hamming:         {hamming:.4f}")
    print(f"  NMSPE (active):  {nmspe['NMSPE_active']:.6f}")
    print(f"  CI coverage:     {ci_cov['CI_coverage_active']:.4f}")
    print(f"  CI mean width:   {ci_cov['CI_mean_width_active']:.6f}")
    print(f"  NMSE training:   {nmse_train.get('NMSE_training', float('nan')):.6f}")
    print(f"  NMSE new x0:     {nmse_pred.get('NMSE', float('nan')):.6f}")
    if timings:
        print(f"  Time (sparse):   {timings.get('step1_sparse_s', 0):.0f}s")
        print(f"  Time (refit):    {timings.get('step2_refit_s', 0):.0f}s")
        print(f"  Time (total):    {timings.get('total_with_viz_s', 0):.0f}s")
    if diag_dict:
        print(f"  R-hat max:       {diag_dict.get('rhat_max_sparse', 'N/A')}")
        print(f"  ESS min:         {diag_dict.get('ess_min_sparse', 'N/A')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
