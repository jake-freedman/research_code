"""
Run single-PM multi-tone SSB optimisation and save results.

Physical model
--------------
  E(t) = exp(i * sum_{k=1}^{K} beta_k * sin(k*Omega*t + theta_k))

Tunable parameters: modulation depth beta_k and RF phase theta_k per tone.

Saved files
-----------
  <OUT_DIR>/<timestamp>/
    config.json   - hyperparameters used for this run
    result.json   - betas, thetas, ratio, wanted_power, convergence info
    spectrum.png  - plot of the optimised spectrum

Warm-starting
-------------
  Set RESUME to a previous result folder.  betas and thetas are read from
  result.json and used as x0.  The resumed run saves to a new timestamped
  folder so prior results are never overwritten.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ssb_multitone import optimize_multitone
from ssb_spectrum import plot_optical_spectrum

# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------
WANTED_HARMONIC = 1       # target sideband index
BETA_MAX        = 5.0     # upper bound on each beta_k (rad)
N_MAX           = 30      # harmonic truncation order
N_TONES         = 3       # number of RF drive tones (k = 1 .. N_TONES)
N_ITER          = 200     # basin-hopping iterations
SEED            = None    # integer for reproducibility, or None for random
OBJECTIVE       = "power" # "power": maximise power at wanted harmonic
                          # "ratio": minimise unwanted/wanted power ratio
METHOD          = "fft"   # "fft" : time-domain FFT (fast, ~1e-10 accuracy)
                          # "conv": Jacobi-Anger convolution (exact, slower)
N_FFT           = 8192    # FFT size (only used when METHOD = "fft")

OUT_DIR = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"

RESUME = None   # e.g. r"C:\path\to\results\20260417_153012"

# ---------------------------------------------------------------------------


def load_resume(folder: str) -> np.ndarray:
    """Reconstruct the optimiser parameter vector from a previous result folder."""
    result_path = os.path.join(folder, "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Cannot find result.json in {folder}")

    with open(result_path) as f:
        prev = json.load(f)

    betas  = prev["betas"]
    thetas = prev["thetas"]

    if len(betas) != N_TONES or len(thetas) != N_TONES:
        raise ValueError(
            f"Resumed run has {len(betas)} tones but current config has N_TONES={N_TONES}. "
            "Make sure N_TONES matches the original run."
        )
    return np.concatenate([betas, thetas])


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
    n_iter          = N_ITER,
    seed            = SEED,
    objective       = OBJECTIVE,
    method          = METHOD,
    n_fft           = N_FFT,
    resumed_from    = RESUME,
)
print("Running optimisation with config:", config)

res = optimize_multitone(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_tones         = N_TONES,
    n_iter          = N_ITER,
    seed            = SEED,
    x0              = x0,
    objective       = OBJECTIVE,
    method          = METHOD,
    n_fft           = N_FFT,
)

total_power = float(np.sum(np.abs(res["amplitudes"]) ** 2))
betas_str   = "  ".join(f"beta{k+1}={b:.4f}"  for k, b  in enumerate(res["betas"]))
thetas_str  = "  ".join(f"theta{k+1}={th:.4f}" for k, th in enumerate(res["thetas"]))
print(f"Done.  {betas_str}  {thetas_str}  "
      f"ratio={res['ratio']:.6f}  wanted power={res['wanted_power']:.6f}  "
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
    betas        = [float(b)  for b  in res["betas"]],
    thetas       = [float(th) for th in res["thetas"]],
    ratio        = float(res["ratio"]),
    wanted_power = float(res["wanted_power"]),
    total_power  = total_power,
    nit          = int(res["opt_result"].nit),
    nfev         = int(res["opt_result"].nfev),
    message      = str(res["opt_result"].message),
)
with open(os.path.join(out_folder, "result.json"), "w") as f:
    json.dump(result_data, f, indent=2)

fig, _ = plot_optical_spectrum(res["harmonics"], res["amplitudes"])
fig.savefig(os.path.join(out_folder, "spectrum.png"), dpi=150)
plt.show()

print(f"Results saved to {out_folder}")
