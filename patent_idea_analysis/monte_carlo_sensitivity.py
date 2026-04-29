"""
Monte Carlo sensitivity analysis for a saved optimisation result.

Each parameter (betas, phi values, and for multi-tone: thetas) is independently
perturbed by a Gaussian with std = FRAC_STD * |nominal value|.  The resulting
conversion efficiency to the wanted harmonic is recorded for N_SAMPLES draws,
and a histogram is saved to SAVE_DIR.

Supports both result formats (auto-detected):
  - run_optimization.py                    (single-tone cascaded PMs)
  - run_multitone_cascaded_optimization.py (multi-tone cascaded PMs)
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import compute_optical_spectrum_general
from ssb_multitone_cascaded import compute_cascaded_multitone_spectrum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_FOLDER = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3pm\20260421_182654"

N_SAMPLES     = 10000   # number of Monte Carlo draws
FRAC_STD_BETA = 0.01   # fractional std for modulation depths  (std = FRAC_STD_BETA * |beta|)
ABS_STD_PHI   = 0.1    # absolute std for all phase values [rad]  (100 mrad default)

USE_DB   = True          # if True, histogram x-axis is in dB
DB_FLOOR = -40.0          # dB floor (only used when USE_DB = True)
SEED     = 42             # random seed for reproducibility (None for random)

SAVE_DIR  = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media"   # folder to save the histogram PNG, or "" to skip
SAVE_NAME = ""   # filename stem (no extension); "" = auto-generate from harmonic + scale
SAVE_DPI  = 150

# Figure dimensions
AX_W_MM   = 110.0
AX_H_MM   =  60.0
LEFT_IN   = 0.65
RIGHT_IN  = 0.20
BOTTOM_IN = 0.45
TOP_IN    = 0.4

# ---------------------------------------------------------------------------
# Load result
# ---------------------------------------------------------------------------
with open(os.path.join(RESULT_FOLDER, "config.json")) as f:
    config = json.load(f)

with open(os.path.join(RESULT_FOLDER, "result.json")) as f:
    result = json.load(f)

IS_MULTITONE = "betas1" in result

n_max      = config["n_max"]
poly_order = config.get("poly_order", None)
disp_n_max = config.get("disp_n_max", None)
wanted_h   = config["wanted_harmonic"]
stage_range = np.arange(-n_max, n_max + 1)

rng = np.random.default_rng(SEED)

def _perturb_beta(arr: np.ndarray) -> np.ndarray:
    return arr * (1.0 + rng.normal(0.0, FRAC_STD_BETA, size=arr.shape))

def _perturb_phi(arr: np.ndarray) -> np.ndarray:
    return arr + rng.normal(0.0, ABS_STD_PHI, size=arr.shape)

# ---------------------------------------------------------------------------
# Format-specific parameter reconstruction
# ---------------------------------------------------------------------------
if IS_MULTITONE:
    method = config.get("method", "fft")
    n_fft  = config.get("n_fft", 8192)

    nom_betas1  = np.array(result["betas1"])
    nom_betas2  = np.array(result["betas2"])
    nom_thetas1 = np.array(result["thetas1"])
    nom_thetas2 = np.array(result["thetas2"])

    phi_raw = np.load(os.path.join(RESULT_FOLDER, "phi_params.npy"), allow_pickle=False)
    nom_phi = phi_raw if poly_order is None else np.polyval(phi_raw[::-1], stage_range)

    # nominal efficiency
    h_nom, a_nom = compute_cascaded_multitone_spectrum(
        nom_betas1, nom_thetas1, nom_phi, nom_betas2, nom_thetas2,
        n_max=n_max, method=method, n_fft=n_fft,
    )
    wanted_idx_nom = int(np.where(h_nom == wanted_h)[0][0])
    nominal_ce     = float(np.abs(a_nom[wanted_idx_nom]) ** 2)

    print(f"Format: multi-tone cascaded  (method={method})")
    print(f"Nominal CE to harmonic {wanted_h}: {nominal_ce:.6f}")

    def _sample_ce() -> float:
        b1  = _perturb_beta(nom_betas1)
        b2  = _perturb_beta(nom_betas2)
        th1 = _perturb_phi(nom_thetas1)
        th2 = _perturb_phi(nom_thetas2)
        phi = _perturb_phi(nom_phi)
        _, a = compute_cascaded_multitone_spectrum(
            b1, th1, phi, b2, th2, n_max=n_max, method=method, n_fft=n_fft,
        )
        return float(np.abs(a[wanted_idx_nom]) ** 2)

else:
    n_stages = config.get("n_stages", 2)
    n_disp   = n_stages - 1
    truncate = n_stages > 2

    nom_betas = np.array(result["betas"])
    phi_params_raw = np.load(
        os.path.join(RESULT_FOLDER, "phi_params.npy"), allow_pickle=True
    )

    nom_phi_arrs = []
    for s in range(n_disp):
        raw = phi_params_raw[s]
        if poly_order is None:
            if disp_n_max is not None:
                phi_arr = np.zeros(len(stage_range))
                mask = (stage_range >= -disp_n_max) & (stage_range <= disp_n_max)
                phi_arr[mask] = raw
            else:
                phi_arr = raw.copy()
        else:
            phi_arr = np.polyval(raw[::-1], stage_range)
            if disp_n_max is not None:
                phi_arr[np.abs(stage_range) > disp_n_max] = 0.0
        nom_phi_arrs.append(phi_arr)

    h_nom, a_nom = compute_optical_spectrum_general(
        nom_betas, nom_phi_arrs, n_max=n_max, truncate=truncate
    )
    wanted_idx_nom = int(np.where(h_nom == wanted_h)[0][0])
    nominal_ce     = float(np.abs(a_nom[wanted_idx_nom]) ** 2)

    print(f"Format: single-tone cascaded  (n_stages={n_stages})")
    print(f"Nominal CE to harmonic {wanted_h}: {nominal_ce:.6f}")

    def _sample_ce() -> float:
        betas    = _perturb_beta(nom_betas)
        phi_arrs = [_perturb_phi(phi) for phi in nom_phi_arrs]
        _, a = compute_optical_spectrum_general(
            betas, phi_arrs, n_max=n_max, truncate=truncate
        )
        return float(np.abs(a[wanted_idx_nom]) ** 2)

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
print(f"Running {N_SAMPLES} Monte Carlo samples  "
      f"(beta: {FRAC_STD_BETA:.0%} fractional,  phi: {ABS_STD_PHI*1000:.0f} mrad absolute) ...")
samples = np.array([_sample_ce() for _ in range(N_SAMPLES)])
print(f"Done.  mean={samples.mean():.6f}  std={samples.std():.6f}  "
      f"min={samples.min():.6f}  max={samples.max():.6f}")

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
font_props = {"family": "Arial"}
mm_per_inch = 25.4

if USE_DB:
    plot_samples  = 10.0 * np.log10(np.maximum(samples,    10 ** (DB_FLOOR / 10)))
    plot_nominal  = 10.0 * np.log10(max(nominal_ce,        10 ** (DB_FLOOR / 10)))
    xlabel = "Conversion efficiency [dB]"
else:
    plot_samples = samples
    plot_nominal = nominal_ce
    xlabel = "Conversion efficiency"

ax_w_in = AX_W_MM / mm_per_inch
ax_h_in = AX_H_MM / mm_per_inch
fig_w   = LEFT_IN + ax_w_in + RIGHT_IN
fig_h   = BOTTOM_IN + ax_h_in + TOP_IN
fig     = plt.figure(figsize=(fig_w, fig_h))
ax      = fig.add_axes([
    LEFT_IN   / fig_w,
    BOTTOM_IN / fig_h,
    ax_w_in   / fig_w,
    ax_h_in   / fig_h,
])

ax.hist(plot_samples, bins=40, color="#4C72B0", edgecolor="white",
        linewidth=0.4, density=True, zorder=2)

ax.axvline(plot_nominal, color="black",  linewidth=1.5, linestyle="--",
           label=f"Nominal  ({plot_nominal:.3f})", zorder=3)
ax.axvline(np.mean(plot_samples), color="#e05c5c", linewidth=1.5, linestyle="-",
           label=f"Mean  ({np.mean(plot_samples):.3f})", zorder=3)

ax.set_xlabel(xlabel, fontsize=10, **font_props)
ax.set_ylabel("Probability density", fontsize=10, **font_props)
ax.set_title(
    f"Monte Carlo  |  harmonic {wanted_h}  |  N={N_SAMPLES}  |  "
    f"β σ/μ={FRAC_STD_BETA:.0%}  φ σ={ABS_STD_PHI*1000:.0f} mrad",
    fontsize=9, fontfamily="Arial",
)
ax.legend(fontsize=8, frameon=False)
ax.set_xlim([-0.5, 0.1])

for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Arial")

if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    if SAVE_NAME:
        filename = f"{SAVE_NAME}.png"
    else:
        scale_tag = "dB" if USE_DB else "linear"
        filename  = f"monte_carlo_harmonic{wanted_h}_{scale_tag}.png"
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=SAVE_DPI)
    print(f"Saved {path}")

plt.show()
