"""
Run the ring-resonator approximation of an optimizer result and plot the comparison.

Workflow
--------
1. Load an optimizer result folder (config.json, result.json, phi_params.npy).
2. For each dispersion stage and each harmonic, find a ring resonance frequency
   that reproduces the optimizer's phase shift at that sideband frequency.
3. Recompute the optical spectrum using the complex ring transfer functions
   (which include both amplitude attenuation and phase shift).
4. Optionally save the result and plot the comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

from ring_approximation import (
    load_optimizer_result,
    compute_ring_spectrum,
    save_ring_result,
    plot_ring_vs_optimizer,
    plot_combined_ring_transmission,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to a folder produced by run_optimization.py (contains config.json etc.)
OPT_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\20260429_155817"

# Ring Q factors (same for every ring in every stage)
Q_I = 5e6   # intrinsic Q
Q_E = Q_I / 80   # external (waveguide) Q — set Q_E < Q_I for overcoupled rings

# Sideband / carrier geometry
F_CARRIER = 193e12   # optical carrier frequency [Hz]
F_MOD     = 100e9     # modulation frequency [Hz]

# Ring solver search radius [Hz] — must be >> ring linewidth
F_SEARCH_RADIUS = F_MOD

# Plot settings
N_DISPLAY = 10        # show harmonics -N_DISPLAY … +N_DISPLAY
USE_DB    = False     # True: power in dB; False: linear
DB_FLOOR  = -5.0     # dB floor (only used when USE_DB=True)
AX_W_MM   = 90.0
AX_H_MM   = 70.0

# Set to a folder path to save results, or None to skip saving
SAVE_FOLDER = None   # e.g. OPT_FOLDER + "_rings"

# ---------------------------------------------------------------------------

opt = load_optimizer_result(OPT_FOLDER)

print(f"Loaded optimizer result from: {OPT_FOLDER}")
print(f"  n_stages    = {opt['config']['n_stages']}")
print(f"  n_max       = {opt['config']['n_max']}")
print(f"  n_disp_stgs = {len(opt['phi_params_full'])}")
print(f"  betas       = {[f'{b:.4f}' for b in opt['betas']]}")

ring_res = compute_ring_spectrum(
    opt,
    Q_i=Q_I,
    Q_e=Q_E,
    f_carrier=F_CARRIER,
    f_mod=F_MOD,
)

print("Ring approximation done.")
print(f"  Output harmonics span: [{ring_res['harmonics'][0]}, {ring_res['harmonics'][-1]}]")

if SAVE_FOLDER:
    save_ring_result(ring_res, opt, SAVE_FOLDER, Q_i=Q_I, Q_e=Q_E,
                     f_carrier=F_CARRIER, f_mod=F_MOD)
    print(f"Saved ring result to: {SAVE_FOLDER}")

fig = plot_ring_vs_optimizer(
    opt_result=opt,
    ring_result=ring_res,
    n_display=N_DISPLAY,
    ax_w_mm=AX_W_MM,
    ax_h_mm=AX_H_MM,
    use_db=USE_DB,
    db_floor=DB_FLOOR,
)

fig_rings = plot_combined_ring_transmission(
    rings_per_stage=ring_res["rings_per_stage"],
    f_carrier=F_CARRIER,
    f_mod=F_MOD,
    n_max=opt["config"]["n_max"],
    power_db=USE_DB,
    db_floor=DB_FLOOR,
    phase_min=0
)

plt.show()
