"""
Maximum power conversion efficiency to each of the first N_SIDEBANDS sidebands
for two phase modulators separated by one dispersive element (N_STAGES=2).

For each target harmonic k the upper bound is:
    E_k = max_{beta1, beta2}  sum_n |J_n(beta1)| * |J_{k-n}(beta2)|
and the maximum power efficiency is E_k^2.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from ssb_spectrum import plot_optical_spectrum

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS = 8       # compute max efficiency for sidebands 1 .. N_SIDEBANDS
BETA_MAX    = 10.0    # search range for each beta
N_POINTS    = 50      # grid points per beta axis (increase for higher accuracy)
N_TERMS     = 50      # harmonic truncation (n = -N_TERMS .. N_TERMS)

TARGET_SIDEBAND    = 2     # sideband for which to show the optimal spectrum
N_MAX              = 10   # harmonic truncation for spectrum computation
USE_DB             = True
DB_FLOOR           = -60.0
N_DISPLAY          = 8    # harmonics shown either side in spectrum plot

SAVE_PATH          = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\2_pm_ssbm_summary")
SPECTRUM_SAVE_PATH = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\2_pm_spectrum_sb2")

# If True: strip all axis labels and tick labels, raise spines above content,
# set y-max to 1.1, no legend, and save as .svg regardless of SAVE_PATH extension.
PUBLISHED_PLOT = True

# ---------------------------------------------------------------------------
# Compute max efficiency for each sideband
# ---------------------------------------------------------------------------
betas   = np.linspace(0, BETA_MAX, N_POINTS)
n_range = np.arange(-N_TERMS, N_TERMS + 1)

b1, b2 = np.meshgrid(betas, betas, indexing="ij")
Jn_b1  = jv(n_range, b1[..., np.newaxis])   # shape (N_POINTS, N_POINTS, 2*N_TERMS+1)

sidebands       = np.arange(1, N_SIDEBANDS + 1)
max_powers      = np.empty(N_SIDEBANDS)   # two-PM upper bound
max_powers_1pm  = np.empty(N_SIDEBANDS)   # single-PM max: max_beta |J_k(beta)|^2

for i, k in enumerate(sidebands):
    Jm_b2          = jv(k - n_range, b2[..., np.newaxis])
    E_grid         = np.sum(np.abs(Jn_b1) * np.abs(Jm_b2), axis=-1)
    max_powers[i]  = float(np.max(E_grid) ** 2)
    max_powers_1pm[i] = float(np.max(jv(k, betas) ** 2))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
mm_per_inch = 25.4
width_mm, height_mm = 55.0, 45.0

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
bar_half  = bar_width / 2

for k, h, c in zip(sidebands, max_powers, colors):
    ax.bar(k, h, width=bar_width, color=c, edgecolor="black",
           linewidth=1.5, zorder=3)

ax.hlines(
    max_powers_1pm,
    sidebands - bar_half, sidebands + bar_half,
    colors="black", linewidths=1.5, linestyles="-", zorder=4,
    label="1 PM",
)
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
    ax.legend(fontsize=8, frameon=False, prop={"family": "Arial"})
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Sideband order [dimensionless]", fontsize=10, **font_props)
    ax.set_ylabel(r"Max conversion efficiency  [dimensionless]", fontsize=10, **font_props)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

if SAVE_PATH is not None:
    save_path = os.path.splitext(SAVE_PATH)[0] + ".svg" if PUBLISHED_PLOT else SAVE_PATH
    fig.savefig(save_path, dpi=200)
    print(f"Figure saved to {save_path}")

# ---------------------------------------------------------------------------
# Optimal spectrum for TARGET_SIDEBAND
# ---------------------------------------------------------------------------
# Find beta1, beta2 on the grid that maximise conversion to TARGET_SIDEBAND.
k_t = TARGET_SIDEBAND
Jm_target = jv(k_t - n_range, b2[..., np.newaxis])
E_target  = np.sum(np.abs(Jn_b1) * np.abs(Jm_target), axis=-1)
i1_opt, i2_opt = np.unravel_index(np.argmax(E_target), E_target.shape)
beta1_opt, beta2_opt = betas[i1_opt], betas[i2_opt]

h_pm   = np.arange(-N_MAX, N_MAX + 1)
a1     = jv(h_pm, beta1_opt).astype(complex)
a2     = jv(h_pm, beta2_opt).astype(complex)
a_conv = np.convolve(a1, a2)
h_conv = np.arange(-2 * N_MAX, 2 * N_MAX + 1)

fig_spec, ax_spec = plot_optical_spectrum(
    h_conv, a_conv,
    db=USE_DB, db_floor=DB_FLOOR,
    width_mm=55, height_mm=45, n_display=N_DISPLAY,
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
