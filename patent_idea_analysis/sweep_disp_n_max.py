"""
Sweep DISP_N_MAX from 1 to N_MAX and record the maximum conversion efficiency
achieved by the optimizer at each value.

Saved files
-----------
  <OUT_DIR>/<timestamp>/
    config.json              - sweep hyperparameters
    sweep_disp_n_max.json    - per-point results (disp_n_max, wanted_power, ratio)
    sweep_linear.png         - conversion efficiency vs DISP_N_MAX (linear)
    sweep_dB.png             - conversion efficiency vs DISP_N_MAX (dB)
"""

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import optimize_ssb

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration — keep consistent with run_optimization.py
# ---------------------------------------------------------------------------
WANTED_HARMONIC  = 3
BETA_MAX         = 5.0
N_MAX            = 20     # harmonic truncation order — fixed for all optimizer calls
N_STAGES         = 3
POLY_ORDER       = None
N_ITER           = 100
SEED             = None
OBJECTIVE        = "power"
PHI_MAX          = None
DISP_N_MAX_MAX   = 8  # sweep DISP_N_MAX from 1 up to this value (≤ N_MAX)

OUT_DIR = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data")
# ---------------------------------------------------------------------------

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_folder = os.path.join(OUT_DIR, f"sweep_disp_n_max_{timestamp}")
os.makedirs(out_folder, exist_ok=True)

config = dict(
    wanted_harmonic  = WANTED_HARMONIC,
    beta_max         = BETA_MAX,
    n_max            = N_MAX,
    n_stages         = N_STAGES,
    poly_order       = POLY_ORDER,
    n_iter           = N_ITER,
    seed             = SEED,
    objective        = OBJECTIVE,
    phi_max          = PHI_MAX,
    disp_n_max_max   = DISP_N_MAX_MAX,
)
with open(os.path.join(out_folder, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
disp_n_max_values = list(range(1, DISP_N_MAX_MAX + 1))
wanted_powers     = []
best_other_powers = []
ratios            = []

for d in disp_n_max_values:
    print(f"  disp_n_max={d:3d} / {DISP_N_MAX_MAX} ...", end="", flush=True)
    res = optimize_ssb(
        wanted_harmonic = WANTED_HARMONIC,
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
    wanted_idx   = np.where(res["harmonics"] == WANTED_HARMONIC)[0]
    if len(wanted_idx):
        power_others[wanted_idx[0]] = 0.0
    best_other = float(np.max(power_others))

    wanted_powers.append(float(res["wanted_power"]))
    best_other_powers.append(best_other)
    ratios.append(float(res["ratio"]))
    print(f"  wanted_power={res['wanted_power']:.6f}  best_other={best_other:.6f}"
          f"  ratio={res['ratio']:.4f}")

# ---------------------------------------------------------------------------
# Save sweep data
# ---------------------------------------------------------------------------
sweep_data = [
    {"disp_n_max": d, "wanted_power": wp, "best_other_power": bo, "ratio": r}
    for d, wp, bo, r in zip(disp_n_max_values, wanted_powers, best_other_powers, ratios)
]
with open(os.path.join(out_folder, "sweep_disp_n_max.json"), "w") as f:
    json.dump(sweep_data, f, indent=2)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
font_props  = {"family": "Arial"}
mm_per_inch = 25.4

def _make_ax(ylabel: str):
    left_in, right_in, bottom_in, top_in = 0.65, 0.20, 0.45, 0.15
    ax_w_in = 120.0 / mm_per_inch
    ax_h_in =  70.0 / mm_per_inch
    fig = plt.figure(figsize=(left_in + ax_w_in + right_in,
                               bottom_in + ax_h_in + top_in))
    ax = fig.add_axes([
        left_in / (left_in + ax_w_in + right_in),
        bottom_in / (bottom_in + ax_h_in + top_in),
        ax_w_in  / (left_in + ax_w_in + right_in),
        ax_h_in  / (bottom_in + ax_h_in + top_in),
    ])
    ax.set_xlabel(r"$N_{\mathrm{disp}}$  (max dispersed harmonic)", fontsize=10, **font_props)
    ax.set_ylabel(ylabel, fontsize=10, **font_props)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    return fig, ax

# ---------------------------------------------------------------------------
# Linear plot
# ---------------------------------------------------------------------------
fig_lin, ax_lin = _make_ax("Conversion efficiency")
ax_lin.plot(disp_n_max_values, wanted_powers,
            color="#4C72B0", linewidth=1.5, marker="o", markersize=4, zorder=3)
ax_lin.set_xlim(0.5, DISP_N_MAX_MAX + 0.5)
ax_lin.set_ylim(0, 1.1)
ax_lin.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

fig_lin.savefig(os.path.join(out_folder, "sweep_linear.png"), dpi=150)
print(f"Saved sweep_linear.png")

# ---------------------------------------------------------------------------
# dB plot
# ---------------------------------------------------------------------------
db_floor = -40.0
wanted_db = 10.0 * np.log10(np.maximum(wanted_powers, 10 ** (db_floor / 10)))

fig_db, ax_db = _make_ax("Conversion efficiency [dB]")
ax_db.plot(disp_n_max_values, wanted_db,
           color="#4C72B0", linewidth=1.5, marker="o", markersize=4, zorder=3)
ax_db.set_xlim(0.5, DISP_N_MAX_MAX + 0.5)
ax_db.set_ylim(db_floor, 2.0)
ax_db.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

fig_db.savefig(os.path.join(out_folder, "sweep_dB.png"), dpi=150)
print(f"Saved sweep_dB.png")

plt.show()
print(f"Results saved to {out_folder}")
