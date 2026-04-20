"""
Load a previous optimisation result and visualize the phase profiles and spectrum.

Set RESULT_FOLDER to the timestamped sub-folder created by run_optimization.py.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import compute_optical_spectrum_general, plot_optical_spectrum, plot_phase_profile

# ---------------------------------------------------------------------------
# Paste the path to the result folder here
# ---------------------------------------------------------------------------
RESULT_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\20260417_155421"
USE_DB        = True   # set True to show spectrum in dB
DB_FLOOR      = -30.0  # dB floor (only used when USE_DB = True)
PHASE_MOD_2PI  = True  # set True to wrap phase profiles to [0, 2*pi)
MINIMIZE_PHASE = False  # set True to also show the minimum-total-phase equivalent profile
# ---------------------------------------------------------------------------

if not RESULT_FOLDER:
    raise ValueError("Set RESULT_FOLDER to the path of a saved result.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(os.path.join(RESULT_FOLDER, "config.json")) as f:
    config = json.load(f)

with open(os.path.join(RESULT_FOLDER, "result.json")) as f:
    result = json.load(f)

phi_params_raw = np.load(
    os.path.join(RESULT_FOLDER, "phi_params.npy"), allow_pickle=True
)

n_max      = config["n_max"]
n_stages   = config.get("n_stages", 2)       # default 2 for older saved results
poly_order = config["poly_order"]
n_disp     = n_stages - 1
betas      = result["betas"]

# ---------------------------------------------------------------------------
# Reconstruct per-stage phase profiles
# ---------------------------------------------------------------------------
# n_stages > 2 runs use truncation: all dispersion profiles span ±n_max.
# n_stages == 2 runs (and older saves): the single profile spans ±n_max too
# (after 1 PM the range is exactly ±n_max), so uniform is always correct.
truncate     = n_stages > 2
stage_ranges = [np.arange(-n_max, n_max + 1)] * n_disp
phi_profiles   = []   # evaluated arrays, one per dispersion stage
phi_arrs_input = []   # same, passed to compute_optical_spectrum_general

for s in range(n_disp):
    raw = phi_params_raw[s]
    if poly_order is None:
        phi_arr = raw
    else:
        phi_arr = np.polyval(raw[::-1], stage_ranges[s])
    phi_profiles.append(phi_arr)
    phi_arrs_input.append(phi_arr)

# ---------------------------------------------------------------------------
# Recompute spectrum
# ---------------------------------------------------------------------------
harmonics, amplitudes = compute_optical_spectrum_general(betas, phi_arrs_input, n_max=n_max,
                                                          truncate=truncate)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
for i, b in enumerate(betas):
    print(f"beta{i+1} = {b:.4f} rad")
print(f"ratio  = {result['ratio']:.6f}  (unwanted / wanted power)")
print(f"wanted harmonic: {config['wanted_harmonic']}")

# ---------------------------------------------------------------------------
# Minimum-total-phase helper
# ---------------------------------------------------------------------------
def find_min_phase(phi_arr: np.ndarray) -> np.ndarray:
    """
    Find the global 2*pi*k shift (k integer) of phi_arr that minimises the
    RMS phase, preserving the profile shape (polynomial structure, smoothness).

    Minimising sum((phi_arr + 2*pi*k)^2) over k gives:
        k = -round(mean(phi_arr) / (2*pi))
    """
    k_opt = int(np.round(-np.mean(phi_arr) / (2 * np.pi)))
    return phi_arr + 2 * np.pi * k_opt


# ---------------------------------------------------------------------------
# Plot phase profiles
# ---------------------------------------------------------------------------
for s, (phi_arr, n_range_s) in enumerate(zip(phi_profiles, stage_ranges)):
    phi_plot = phi_arr % (2 * np.pi) if PHASE_MOD_2PI else phi_arr
    fig, ax  = plot_phase_profile(n_range_s, phi_plot)
    ax.set_title(f"Dispersion stage {s + 1}", fontsize=10, fontfamily="Arial")
    if PHASE_MOD_2PI:
        ax.set_ylabel("Phase  $\\phi_n$  (rad)  mod $2\\pi$", fontsize=10,
                      fontfamily="Arial")

    if MINIMIZE_PHASE:
        phi_min = find_min_phase(phi_arr)
        rms_orig = float(np.sqrt(np.mean(phi_arr ** 2)))
        rms_min  = float(np.sqrt(np.mean(phi_min ** 2)))
        print(f"Dispersion stage {s + 1}:  RMS phase original={rms_orig:.4f} rad"
              f"  ->  min-phase={rms_min:.4f} rad"
              + ("  (reduction)" if rms_min < rms_orig else "  (already near-minimal)"))
        fig2, ax2 = plot_phase_profile(n_range_s, phi_min)
        ax2.set_title(f"Dispersion stage {s + 1}  (min total phase)",
                      fontsize=10, fontfamily="Arial")
        ax2.set_ylabel("Phase  $\\phi_n$  (rad)  [min $\\sum\\phi^2$]",
                       fontsize=10, fontfamily="Arial")

# ---------------------------------------------------------------------------
# Plot spectrum
# ---------------------------------------------------------------------------
plot_optical_spectrum(harmonics, amplitudes, db=USE_DB, db_floor=DB_FLOOR)

plt.show()
