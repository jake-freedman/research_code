"""
Run PM1(multi-tone) -> Dispersion -> PM2(multi-tone) optimisation.

Saved files
-----------
  <OUT_DIR>/<timestamp>/
    config.json   - hyperparameters
    result.json   - betas1/2, thetas1/2, ratio, wanted_power, convergence
    phi_params.npy - raw dispersion parameters (poly coeffs or free phases)
    spectrum.png  - optimised spectrum

Warm-starting
-------------
  Set RESUME to a previous result folder.  All parameters are reconstructed
  and used as x0.  The resumed run saves to a new timestamped folder.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ssb_multitone_cascaded import optimize_cascaded_multitone
from ssb_spectrum import plot_optical_spectrum

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------
WANTED_HARMONIC = 4       # target sideband index
BETA_MAX        = 5.0     # upper bound on each individual beta_k (rad)
N_MAX           = 10      # harmonic truncation order
N_TONES         = 2       # number of RF tones per PM (k = 1 .. N_TONES)
POLY_ORDER      = None       # dispersion: None = free phase; integer = polynomial order
N_ITER          = 100     # basin-hopping iterations
SEED            = 1    # integer for reproducibility, or None for random
OBJECTIVE       = "power" # "power": maximise power at wanted harmonic
                          # "ratio": minimise unwanted/wanted power ratio
METHOD          = "conv"   # "fft" : PM transfer via time-domain FFT
                          # "conv": PM transfer via Jacobi-Anger convolution
N_FFT           = 8192    # FFT size (only used when METHOD = "fft")

OUT_DIR = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data")

RESUME = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\20260421_163348")   # e.g. local_path(r"C:\path\to\results\20260417_153012")

# ---------------------------------------------------------------------------


def load_resume(folder: str) -> np.ndarray:
    """Reconstruct the optimiser parameter vector from a previous result folder."""
    result_path = os.path.join(folder, "result.json")
    phi_path    = os.path.join(folder, "phi_params.npy")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Cannot find result.json in {folder}")
    if not os.path.exists(phi_path):
        raise FileNotFoundError(f"Cannot find phi_params.npy in {folder}")

    with open(result_path) as f:
        prev = json.load(f)

    for key in ("betas1", "betas2", "thetas1", "thetas2"):
        if len(prev[key]) != N_TONES:
            raise ValueError(
                f"Resumed run has {len(prev[key])} tones for '{key}' but "
                f"current config has N_TONES={N_TONES}."
            )

    phi_raw = np.load(phi_path, allow_pickle=False)

    n_phi_expected = (POLY_ORDER + 1) if POLY_ORDER is not None else 2 * N_MAX + 1
    if len(phi_raw) != n_phi_expected:
        raise ValueError(
            f"Resumed phi_params has {len(phi_raw)} values but current config "
            f"needs {n_phi_expected}.  Check POLY_ORDER and N_MAX."
        )

    return np.concatenate([
        prev["betas1"], prev["betas2"],
        prev["thetas1"], prev["thetas2"],
        phi_raw,
    ])


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------
x0 = None
if RESUME:
    x0 = load_resume(RESUME)
    with open(os.path.join(RESUME, "result.json")) as f:
        prev = json.load(f)
    print(f"Warm-starting from {RESUME}  "
          f"(previous ratio: {prev['ratio']:.6f}  wanted power: {prev['wanted_power']:.6f})")

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------
config = dict(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_tones         = N_TONES,
    poly_order      = POLY_ORDER,
    n_iter          = N_ITER,
    seed            = SEED,
    objective       = OBJECTIVE,
    method          = METHOD,
    n_fft           = N_FFT,
    resumed_from    = RESUME,
)
print("Running optimisation with config:", config)

res = optimize_cascaded_multitone(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_tones         = N_TONES,
    poly_order      = POLY_ORDER,
    n_iter          = N_ITER,
    seed            = SEED,
    x0              = x0,
    objective       = OBJECTIVE,
    method          = METHOD,
    n_fft           = N_FFT,
)

total_power  = float(np.sum(np.abs(res["amplitudes"]) ** 2))
b1_str  = "  ".join(f"b1_{k+1}={b:.4f}"  for k, b  in enumerate(res["betas1"]))
b2_str  = "  ".join(f"b2_{k+1}={b:.4f}"  for k, b  in enumerate(res["betas2"]))
th1_str = "  ".join(f"th1_{k+1}={t:.4f}" for k, t  in enumerate(res["thetas1"]))
th2_str = "  ".join(f"th2_{k+1}={t:.4f}" for k, t  in enumerate(res["thetas2"]))
print(f"Done.")
print(f"  PM1:  {b1_str}  {th1_str}")
print(f"  PM2:  {b2_str}  {th2_str}")
print(f"  ratio={res['ratio']:.6f}  wanted power={res['wanted_power']:.6f}  "
      f"total power={total_power:.6f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_folder = os.path.join(OUT_DIR, timestamp)
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

result_data = dict(
    betas1       = [float(b)  for b  in res["betas1"]],
    thetas1      = [float(th) for th in res["thetas1"]],
    betas2       = [float(b)  for b  in res["betas2"]],
    thetas2      = [float(th) for th in res["thetas2"]],
    ratio        = float(res["ratio"]),
    wanted_power = float(res["wanted_power"]),
    total_power  = total_power,
    nit          = int(res["opt_result"].nit),
    nfev         = int(res["opt_result"].nfev),
    message      = str(res["opt_result"].message),
)
with open(os.path.join(out_folder, "result.json"), "w") as f:
    json.dump(result_data, f, indent=2)

np.save(os.path.join(out_folder, "phi_params.npy"), res["phi_params"], allow_pickle=False)

fig, _ = plot_optical_spectrum(res["harmonics"], res["amplitudes"])
fig.savefig(os.path.join(out_folder, "spectrum.png"), dpi=150)
plt.show()

print(f"Results saved to {out_folder}")
