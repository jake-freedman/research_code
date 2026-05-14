"""
Maximum power conversion efficiency to each of the first N_SIDEBANDS sidebands
for a single phase modulator: max_beta |J_k(beta)|^2.

Also plots the optimal spectrum for a user-specified target sideband.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from ssb_spectrum import plot_optical_spectrum

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS    = 8
BETA_MAX       = 10.0
N_POINTS       = 200
N_MAX          = 10    # harmonics shown either side in spectrum plot

TARGET_SIDEBAND = 2    # sideband for which to show the optimal spectrum
USE_DB          = True
DB_FLOOR        = -60.0
N_DISPLAY       = 8    # harmonics shown either side in spectrum plot

SAVE_PATH          = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\1_pm_ssbm_summary")
SPECTRUM_SAVE_PATH = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\1_pm_spectrum_sb6")

PUBLISHED_PLOT = True

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
betas     = np.linspace(0, BETA_MAX, N_POINTS)
sidebands = np.arange(1, N_SIDEBANDS + 1)
max_powers = np.array([float(np.max(jv(k, betas) ** 2)) for k in sidebands])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
mm_per_inch = 25.4
width_mm, height_mm = 55, 45

left_in   = 0.60
bottom_in = 0.50
top_in    = 0.20
right_in  = 0.20

ax_w_in = width_mm / mm_per_inch
ax_h_in = height_mm / mm_per_inch
fig_w   = left_in + ax_w_in + right_in
fig_h   = bottom_in + ax_h_in + top_in

matplotlib.rcParams["font.family"] = "Arial"
font_props = {"family": "Arial"}

fig = plt.figure(figsize=(fig_w, fig_h))
ax  = fig.add_axes([
    left_in / fig_w,
    bottom_in / fig_h,
    ax_w_in / fig_w,
    ax_h_in / fig_h,
])

cmap   = matplotlib.colormaps["rainbow_r"]
colors = [cmap((k + 9) / 18) for k in sidebands]

bar_width = 0.5

for k, h, c in zip(sidebands, max_powers, colors):
    ax.bar(k, h, width=bar_width, color=c, edgecolor="black",
           linewidth=1.5, zorder=3)

ax.set_xticks(sidebands)
ax.set_xlim(0.5, N_SIDEBANDS + 0.5)
ax.grid(axis="y", zorder=0)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_zorder(10)

if PUBLISHED_PLOT:
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", direction="in", width=2,
                   labelbottom=False, labelleft=False)
else:
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Sideband order [dimensionless]", fontsize=10, **font_props)
    ax.set_ylabel(r"Max conversion efficiency  [dimensionless]", fontsize=10, **font_props)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

import os

if SAVE_PATH is not None:
    save_path = os.path.splitext(SAVE_PATH)[0] + ".svg" if PUBLISHED_PLOT else SAVE_PATH
    fig.savefig(save_path, dpi=200)
    print(f"Figure saved to {save_path}")

# ---------------------------------------------------------------------------
# Optimal spectrum for TARGET_SIDEBAND
# ---------------------------------------------------------------------------
beta_opt = betas[np.argmax(jv(TARGET_SIDEBAND, betas) ** 2)]
harmonics  = np.arange(-N_MAX, N_MAX + 1)
amplitudes = jv(harmonics, beta_opt).astype(complex)

fig_spec, ax_spec = plot_optical_spectrum(
    harmonics, amplitudes,
    db=USE_DB, db_floor=DB_FLOOR,
    width_mm=55, height_mm = 45, n_display=N_DISPLAY,
)

if PUBLISHED_PLOT:
    ax_spec.set_xlabel("")
    ax_spec.set_ylabel("")
    ax_spec.set_title("")
    ax_spec.tick_params(axis="both", which="both", direction="in", width=2,
                        labelbottom=False, labelleft=False)
    legend = ax_spec.get_legend()
    if legend is not None:
        legend.remove()
    for spine in ax_spec.spines.values():
        spine.set_linewidth(2)
        spine.set_zorder(10)
    if SPECTRUM_SAVE_PATH is not None:
        spec_path = os.path.splitext(SPECTRUM_SAVE_PATH)[0] + ".svg"
        fig_spec.savefig(spec_path, dpi=200)
        print(f"Figure saved to {spec_path}")
else:
    if SPECTRUM_SAVE_PATH is not None:
        fig_spec.savefig(SPECTRUM_SAVE_PATH, dpi=200)
        print(f"Figure saved to {SPECTRUM_SAVE_PATH}")

plt.show()
