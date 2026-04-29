"""
Run the disp_n_max sweep for multiple wanted harmonics and save all results
into a single parent folder that can be read by plot_sweep_comparison.py.

Output layout
-------------
  <OUT_DIR>/<batch_name>/
    harmonic_<N>/
      config.json
      sweep_disp_n_max.json
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

BETA_MAX        = 5.0
N_MAX           = 20     # harmonic truncation order — fixed for all optimizer calls
N_STAGES        = 3
POLY_ORDER      = None
N_ITER          = 100
SEED            = None
OBJECTIVE       = "power"
PHI_MAX         = None
DISP_N_MAX_MAX  = 8      # sweep DISP_N_MAX from 1 up to this value (≤ N_MAX)

OUT_DIR    = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"
BATCH_NAME = f"harmonic_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ---------------------------------------------------------------------------

batch_folder = os.path.join(OUT_DIR, BATCH_NAME)
os.makedirs(batch_folder, exist_ok=True)
print(f"Batch folder: {batch_folder}\n")

disp_n_max_values = list(range(1, DISP_N_MAX_MAX + 1))

for wanted_harmonic in WANTED_HARMONICS:
    sub_folder = os.path.join(batch_folder, f"harmonic_{wanted_harmonic}")
    os.makedirs(sub_folder, exist_ok=True)

    config = dict(
        wanted_harmonic = wanted_harmonic,
        beta_max        = BETA_MAX,
        n_max           = N_MAX,
        n_stages        = N_STAGES,
        poly_order      = POLY_ORDER,
        n_iter          = N_ITER,
        seed            = SEED,
        objective       = OBJECTIVE,
        phi_max         = PHI_MAX,
        disp_n_max_max  = DISP_N_MAX_MAX,
    )
    with open(os.path.join(sub_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"=== Harmonic {wanted_harmonic} ===")
    sweep_data = []
    for d in disp_n_max_values:
        print(f"  disp_n_max={d:3d} / {DISP_N_MAX_MAX} ...", end="", flush=True)
        res = optimize_ssb(
            wanted_harmonic = wanted_harmonic,
            beta_max        = BETA_MAX,
            n_max           = N_MAX,
            n_stages        = N_STAGES,
            poly_order      = POLY_ORDER,
            n_iter          = N_ITER,
            seed            = SEED,
            objective       = OBJECTIVE,
            phi_max         = PHI_MAX,
            disp_n_max      = d,
        )
        power        = np.abs(res["amplitudes"]) ** 2
        power_others = power.copy()
        wanted_idx   = np.where(res["harmonics"] == wanted_harmonic)[0]
        if len(wanted_idx):
            power_others[wanted_idx[0]] = 0.0
        best_other = float(np.max(power_others))

        sweep_data.append({
            "disp_n_max":       d,
            "wanted_power":     float(res["wanted_power"]),
            "best_other_power": best_other,
            "ratio":            float(res["ratio"]),
        })
        print(f"  wanted_power={res['wanted_power']:.6f}  best_other={best_other:.6f}"
              f"  ratio={res['ratio']:.4f}")

    with open(os.path.join(sub_folder, "sweep_disp_n_max.json"), "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"  Saved to {sub_folder}\n")

print(f"All done. Point plot_sweep_comparison.py at:\n  {batch_folder}")
