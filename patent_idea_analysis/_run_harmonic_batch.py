"""Temporary batch runner: run_optimization.py config for WANTED_HARMONIC 1-8."""
import json, os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ssb_spectrum import optimize_ssb, plot_optical_spectrum

from path_utils import local_path

BETA_MAX   = 5.0
N_MAX      = 20
N_STAGES   = 3
POLY_ORDER = None
N_ITER     = 100
SEED       = 1
OBJECTIVE  = "power"
PHI_MAX    = None
DISP_N_MAX = 8
OUT_DIR    = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3stage_ssbm")

for WANTED_HARMONIC in range(1, 9):
    print(f"\n=== WANTED_HARMONIC = {WANTED_HARMONIC} ===")
    config = dict(
        wanted_harmonic       = WANTED_HARMONIC,
        beta_max              = BETA_MAX,
        n_max                 = N_MAX,
        n_stages              = N_STAGES,
        poly_order            = POLY_ORDER,
        n_iter                = N_ITER,
        seed                  = SEED,
        objective             = OBJECTIVE,
        phi_max               = PHI_MAX,
        disp_n_max            = DISP_N_MAX,
        power_split_harmonics = None,
        arbitrary_targets     = None,
        resumed_from          = None,
    )

    res = optimize_ssb(
        wanted_harmonic  = WANTED_HARMONIC,
        beta_max         = BETA_MAX,
        n_max            = N_MAX,
        n_stages         = N_STAGES,
        poly_order       = POLY_ORDER,
        n_iter           = N_ITER,
        seed             = SEED,
        objective        = OBJECTIVE,
        phi_max          = PHI_MAX,
        disp_n_max       = DISP_N_MAX,
    )

    total_power = float(np.sum(np.abs(res["amplitudes"]) ** 2))
    betas_str   = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(res["betas"]))
    print(f"Done.  {betas_str}  ratio={res['ratio']:.6f}  "
          f"wanted power={res['wanted_power']:.6f}  total power={total_power:.6f}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(OUT_DIR, timestamp)
    os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(out_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    result_data = dict(
        betas        = [float(b) for b in res["betas"]],
        ratio        = float(res["ratio"]),
        wanted_power = float(res["wanted_power"]),
        nit          = int(res["opt_result"].nit),
        nfev         = int(res["opt_result"].nfev),
        message      = str(res["opt_result"].message),
    )
    with open(os.path.join(out_folder, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    phi_arr = np.empty(len(res["phi_params_list"]), dtype=object)
    for i, p in enumerate(res["phi_params_list"]):
        phi_arr[i] = p
    np.save(os.path.join(out_folder, "phi_params.npy"), phi_arr, allow_pickle=True)

    fig, _ = plot_optical_spectrum(res["harmonics"], res["amplitudes"])
    fig.savefig(os.path.join(out_folder, "spectrum.png"), dpi=150)
    plt.close(fig)

    print(f"Saved to {out_folder}")

print("\nAll 8 runs complete.")
