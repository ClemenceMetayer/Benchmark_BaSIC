"""Plot trajectories of all 5 BaSIC ODE systems for every condition."""

import sys, os
from pathlib import Path
os.environ["JAX_ENABLE_X64"] = "1"

# Run from the BaSIC directory so that imports work
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
BASIC_DIR = str(BENCHMARK_DIR.parent / "3. BaSIC")
os.chdir(BASIC_DIR)
sys.path.insert(0, BASIC_DIR)

import yaml
import importlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from src.simulate_ODEs.simulate import simulate_ode

# ── System definitions ──────────────────────────────────────────────────────

SYSTEMS = ["lotka_volterra", "chain", "seir", "goldbeter", "yeast_glycolysis"]
SYSTEM_DIR = os.path.join(BASIC_DIR, "systems")

OUTPUT_PATH = str(BENCHMARK_DIR / "figures" / "system_trajectories.png")

# ── Load configs and rhs functions ──────────────────────────────────────────

configs = {}
rhs_fns = {}

for name in SYSTEMS:
    with open(os.path.join(SYSTEM_DIR, f"{name}.yaml")) as f:
        configs[name] = yaml.safe_load(f)
    mod = importlib.import_module(f"systems.{name}")
    rhs_fns[name] = mod.rhs_true

# ── Determine grid size ────────────────────────────────────────────────────

n_systems = len(SYSTEMS)
max_conditions = max(
    len(configs[name]["dynamics"]["true_params"]) for name in SYSTEMS
)

# ── Create figure ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    n_systems,
    max_conditions,
    figsize=(4.5 * max_conditions, 4 * n_systems),
    squeeze=False,
)

# Nice colour palette (tab10 is clear for up to 10 species)
COLORS = plt.cm.tab10.colors

for row, name in enumerate(SYSTEMS):
    cfg = configs[name]
    dyn = cfg["dynamics"]
    time_cfg = cfg["time"]

    x0 = jnp.array(dyn["x0"], dtype=jnp.float64)
    test_x0 = jnp.array(dyn["test_x0"], dtype=jnp.float64) if "test_x0" in dyn else None
    species = cfg["observations"]["observe"]

    t0 = float(time_cfg["t0"])
    t_end = float(time_cfg["t_end"])
    TD = int(time_cfg["TD"])
    t = jnp.linspace(t0, t_end, TD)

    conditions = list(dyn["true_params"].keys())

    for col, cond in enumerate(conditions):
        ax = axes[row, col]
        params = dyn["true_params"][cond]

        # Simulate with default x0
        X = simulate_ode(t, x0, rhs_fns[name], params)

        for i, sp in enumerate(species):
            ax.plot(t, X[:, i], color=COLORS[i % len(COLORS)], label=sp, linewidth=1.4)

        # If test_x0 exists, overlay with dashed lines
        if test_x0 is not None:
            X_test = simulate_ode(t, test_x0, rhs_fns[name], params)
            for i, sp in enumerate(species):
                ax.plot(
                    t, X_test[:, i],
                    color=COLORS[i % len(COLORS)],
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.7,
                )

        ax.set_title(f"{name}  —  {cond}", fontsize=10, fontweight="bold")
        ax.set_xlabel("time", fontsize=8)
        ax.set_ylabel("state", fontsize=8)
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots on this row
    for col in range(len(conditions), max_conditions):
        axes[row, col].set_visible(False)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure to {OUTPUT_PATH}")
