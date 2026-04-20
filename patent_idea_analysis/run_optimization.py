"""
Run SSB optimization and save results to a folder.

Saved files
-----------
  <OUT_DIR>/<timestamp>/
    config.json        - hyperparameters used for this run
    result.json        - scalar outputs (betas, ratio, convergence info)
    phi_params.npy     - optimised phase parameters, shape (n_stages-1, n_phase)
    spectrum.png       - plot of the optimised spectrum

Warm-starting
-------------
  Set RESUME to a previous results folder to initialise the optimiser from
  that run's best solution.  The resumed run saves to a new timestamped
  sub-folder so prior results are never overwritten.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import optimize_ssb, plot_optical_spectrum

# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------
WANTED_HARMONIC = 1       # target sideband index
BETA_MAX        = 10.0    # maximum allowed modulation depth (rad)
N_MAX           = 30      # harmonic truncation order per PM stage
N_STAGES        = 2       # number of phase modulators
POLY_ORDER      = None    # None = fully free phase; integer = polynomial order
N_ITER          = 200     # basin-hopping iterations
SEED            = None    # integer for reproducibility, or None for random
OBJECTIVE       = "power" # "ratio": minimise unwanted/wanted power ratio
                          # "power": maximise power at wanted harmonic
PHI_MAX         = None    # maximum phase shift (rad) per harmonic; None = no limit
                          # free-phase mode: hard bound [0, PHI_MAX]
                          # polynomial mode: soft quadratic penalty

# Full path to the folder where timestamped result sub-folders will be saved.
OUT_DIR = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"

# Set to a previous result folder path to warm-start, or None to start fresh.
RESUME = None   # e.g. r"C:\path\to\results\20260417_153012"

# ---------------------------------------------------------------------------


def load_resume(folder: str) -> np.ndarray:
    """Reconstruct the optimiser parameter vector from a previous result folder."""
    result_path = os.path.join(folder, "result.json")
    phi_path    = os.path.join(folder, "phi_params.npy")
    if not os.path.exists(result_path) or not os.path.exists(phi_path):
        raise FileNotFoundError(f"Cannot find result.json or phi_params.npy in {folder}")

    with open(result_path) as f:
        prev = json.load(f)

    betas      = prev["betas"]
    phi_params = np.load(phi_path, allow_pickle=True)   # object array of per-stage arrays
    x0 = np.concatenate([betas] + [p for p in phi_params])

    n_range  = np.arange(-N_MAX, N_MAX + 1)
    n_phase  = (POLY_ORDER + 1) if POLY_ORDER is not None else len(n_range)
    n_disp   = N_STAGES - 1
    expected = N_STAGES + n_disp * n_phase
    if len(x0) != expected:
        raise ValueError(
            f"Resumed run has {len(x0)} parameters but current config needs {expected}. "
            "Make sure N_MAX, N_STAGES, and POLY_ORDER match the original run."
        )
    return x0


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------
x0 = None
if RESUME:
    x0 = load_resume(RESUME)
    with open(os.path.join(RESUME, "result.json")) as f:
        prev_ratio = json.load(f)["ratio"]
    print(f"Warm-starting from {RESUME}  (previous ratio: {prev_ratio:.6f})")

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------
config = dict(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_stages        = N_STAGES,
    poly_order      = POLY_ORDER,
    n_iter          = N_ITER,
    seed            = SEED,
    objective       = OBJECTIVE,
    phi_max         = PHI_MAX,
    resumed_from    = RESUME,
)
print("Running optimisation with config:", config)

res = optimize_ssb(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_stages        = N_STAGES,
    poly_order      = POLY_ORDER,
    n_iter          = N_ITER,
    seed            = SEED,
    x0              = x0,
    objective       = OBJECTIVE,
    phi_max         = PHI_MAX,
)

total_power  = float(np.sum(np.abs(res["amplitudes"]) ** 2))
betas_str    = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(res["betas"]))
print(f"Done.  {betas_str}  ratio={res['ratio']:.6f}  wanted power={res['wanted_power']:.6f}  total power={total_power:.6f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_folder = os.path.join(OUT_DIR, timestamp)
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

result_data = dict(
    betas         = [float(b) for b in res["betas"]],
    ratio         = float(res["ratio"]),
    wanted_power  = float(res["wanted_power"]),
    nit           = int(res["opt_result"].nit),
    nfev          = int(res["opt_result"].nfev),
    message       = str(res["opt_result"].message),
)
with open(os.path.join(out_folder, "result.json"), "w") as f:
    json.dump(result_data, f, indent=2)

# phi_params.npy: object array (ragged for free phase; uniform for polynomial)
phi_arr = np.empty(len(res["phi_params_list"]), dtype=object)
for i, p in enumerate(res["phi_params_list"]):
    phi_arr[i] = p
np.save(os.path.join(out_folder, "phi_params.npy"), phi_arr, allow_pickle=True)

fig, _ = plot_optical_spectrum(res["harmonics"], res["amplitudes"])
fig.savefig(os.path.join(out_folder, "spectrum.png"), dpi=150)
plt.show()

print(f"Results saved to {out_folder}")
