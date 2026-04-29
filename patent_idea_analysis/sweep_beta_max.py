"""
Sweep BETA_MAX through a range of values for each wanted harmonic and save
all results into a single parent folder compatible with plot_sweep_comparison.py.

Output layout
-------------
  <OUT_DIR>/<batch_name>/
    harmonic_<N>/
      config.json
      sweep_beta_max.json
"""

import json
import os
from datetime import datetime

import numpy as np

from ssb_spectrum import optimize_ssb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WANTED_HARMONICS = [1, 2, 3, 4, 5, 6]   # list of target harmonic indices to sweep

BETA_MAX_MIN    = 0.5     # start of beta_max sweep (rad)
BETA_MAX_MAX    = 15.0    # end of beta_max sweep (rad)
N_POINTS        = 20      # number of sweep points (linearly spaced)
N_MAX           = 20      # harmonic truncation order (fixed for all calls)
N_STAGES        = 3
POLY_ORDER      = None
N_ITER          = 100
SEED            = None
OBJECTIVE       = "power"
PHI_MAX         = None
DISP_N_MAX      = None    # fixed dispersion limit (None = all harmonics free)

OUT_DIR    = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"
BATCH_NAME = f"beta_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ---------------------------------------------------------------------------

batch_folder    = os.path.join(OUT_DIR, BATCH_NAME)
beta_max_values = list(np.linspace(BETA_MAX_MIN, BETA_MAX_MAX, N_POINTS))

os.makedirs(batch_folder, exist_ok=True)
print(f"Batch folder: {batch_folder}\n")

for wanted_harmonic in WANTED_HARMONICS:
    sub_folder = os.path.join(batch_folder, f"harmonic_{wanted_harmonic}")
    os.makedirs(sub_folder, exist_ok=True)

    config = dict(
        wanted_harmonic = wanted_harmonic,
        beta_max_min    = BETA_MAX_MIN,
        beta_max_max    = BETA_MAX_MAX,
        n_points        = N_POINTS,
        n_max           = N_MAX,
        n_stages        = N_STAGES,
        poly_order      = POLY_ORDER,
        n_iter          = N_ITER,
        seed            = SEED,
        objective       = OBJECTIVE,
        phi_max         = PHI_MAX,
        disp_n_max      = DISP_N_MAX,
    )
    with open(os.path.join(sub_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"=== Harmonic {wanted_harmonic} ===")
    sweep_data = []
    for i, bmax in enumerate(beta_max_values):
        print(f"  beta_max={bmax:.3f}  ({i+1}/{N_POINTS}) ...", end="", flush=True)
        res = optimize_ssb(
            wanted_harmonic = wanted_harmonic,
            beta_max        = bmax,
            n_max           = N_MAX,
            n_stages        = N_STAGES,
            poly_order      = POLY_ORDER,
            n_iter          = N_ITER,
            seed            = SEED,
            objective       = OBJECTIVE,
            phi_max         = PHI_MAX,
            disp_n_max      = DISP_N_MAX,
        )
        power        = np.abs(res["amplitudes"]) ** 2
        power_others = power.copy()
        wanted_idx   = np.where(res["harmonics"] == wanted_harmonic)[0]
        if len(wanted_idx):
            power_others[wanted_idx[0]] = 0.0
        best_other = float(np.max(power_others))

        sweep_data.append({
            "beta_max":         bmax,
            "wanted_power":     float(res["wanted_power"]),
            "best_other_power": best_other,
            "ratio":            float(res["ratio"]),
        })
        print(f"  wanted_power={res['wanted_power']:.6f}  best_other={best_other:.6f}"
              f"  ratio={res['ratio']:.4f}")

    with open(os.path.join(sub_folder, "sweep_beta_max.json"), "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"  Saved to {sub_folder}\n")

print(f"All done. Point plot_sweep_comparison.py at:\n  {batch_folder}")
