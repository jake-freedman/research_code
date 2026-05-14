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

from ssb_multitone import (
    optimize_multitone,
    compute_multitone_spectrum_fft,
    compute_multitone_spectrum_conv,
)
from ssb_spectrum import plot_optical_spectrum

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------
WANTED_HARMONIC = 1       # target sideband index
BETA_MAX        = 10.0     # upper bound on each beta_k (rad)
N_MAX           = 30      # harmonic truncation order
N_TONES         = 10      # number of RF drive tones (k = 1 .. N_TONES)
N_ITER          = 100     # basin-hopping iterations
SEED            = 1    # integer for reproducibility, or None for random
OBJECTIVE       = "power" # "power": maximise power at wanted harmonic
                          # "ratio": minimise unwanted/wanted power ratio
METHOD          = "fft"   # "fft" : time-domain FFT (fast, ~1e-10 accuracy)
                          # "conv": Jacobi-Anger convolution (exact, slower)
N_FFT           = 8192    # FFT size (only used when METHOD = "fft")

OUT_DIR = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data")

RESUME = None   # e.g. local_path(r"C:\path\to\results\20260417_153012")

# Sawtooth initial guess
# When True, uses the K-tone Fourier approximation of a sawtooth wave as the
# starting point.  For wanted harmonic n: beta_k = clip(2n/k, 0, BETA_MAX),
# theta_k = 0 (k odd) or π (k even).  RESUME takes precedence if both are set.
SAWTOOTH_INIT = True

# ---------------------------------------------------------------------------


def _make_sawtooth_x0(wanted_harmonic: int, n_tones: int, beta_max: float) -> np.ndarray:
    """K-tone Fourier approximation of a rising sawtooth with amplitude n·π.

    A sawtooth from -n·π to +n·π concentrates optical power at harmonic n.
    Fourier coefficients: beta_k = 2n/k (clipped to beta_max),
                          theta_k = 0 (k odd) or π (k even).
    """
    ks     = np.arange(1, n_tones + 1, dtype=float)
    betas  = np.minimum(2.0 * wanted_harmonic / ks, beta_max)
    thetas = np.where(ks % 2 == 0, np.pi, 0.0)
    return np.concatenate([betas, thetas])


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
x0           = None
sawtooth_x0  = None
sawtooth_amps = None
sawtooth_harmonics = None

if SAWTOOTH_INIT:
    sawtooth_x0 = _make_sawtooth_x0(WANTED_HARMONIC, N_TONES, BETA_MAX)
    x0 = sawtooth_x0
    _betas_s  = sawtooth_x0[:N_TONES]
    _thetas_s = sawtooth_x0[N_TONES:]
    if METHOD == "fft":
        sawtooth_harmonics, sawtooth_amps = compute_multitone_spectrum_fft(
            _betas_s, _thetas_s, n_max=N_MAX, n_fft=N_FFT,
        )
    else:
        sawtooth_harmonics, sawtooth_amps = compute_multitone_spectrum_conv(
            _betas_s, _thetas_s, n_max=N_MAX,
        )
    _saw_idx    = np.where(sawtooth_harmonics == WANTED_HARMONIC)[0]
    _saw_power  = float(np.abs(sawtooth_amps[_saw_idx[0]]) ** 2) if len(_saw_idx) else 0.0
    print(f"Sawtooth x0: betas={_betas_s.tolist()}")
    print(f"  → wanted_power={_saw_power:.6f}")

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

if sawtooth_x0 is not None:
    _saw_idx   = np.where(sawtooth_harmonics == WANTED_HARMONIC)[0]
    _saw_want  = float(np.abs(sawtooth_amps[_saw_idx[0]]) ** 2) if len(_saw_idx) else 0.0
    _saw_total = float(np.sum(np.abs(sawtooth_amps) ** 2))
    _saw_ratio = (_saw_total - _saw_want) / _saw_want if _saw_want > 1e-30 else np.inf
    with open(os.path.join(out_folder, "sawtooth_result.json"), "w") as f:
        json.dump(dict(
            betas        = sawtooth_x0[:N_TONES].tolist(),
            thetas       = sawtooth_x0[N_TONES:].tolist(),
            wanted_power = _saw_want,
            ratio        = _saw_ratio,
        ), f, indent=2)
    fig_saw, _ = plot_optical_spectrum(sawtooth_harmonics, sawtooth_amps)
    fig_saw.savefig(os.path.join(out_folder, "sawtooth_spectrum.png"), dpi=150)

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
