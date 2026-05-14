"""
Power at the first sideband of a single phase modulator vs modulation depth.

For a CW input through one PM:  A_k = J_k(beta)
First sideband power:           |J_1(beta)|^2
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BETA_MAX  = 10.0   # maximum modulation depth to plot
N_POINTS  = 500    # number of points along beta axis

SAVE_PATH = local_path(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\best1_pm_only")

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
beta  = np.linspace(0, BETA_MAX, N_POINTS)
power = jv(1, beta) ** 2

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
mm_per_inch = 25.4
width_mm, height_mm = 130.0, 90.0

left_in   = 0.60
bottom_in = 0.50
top_in    = 0.15
right_in  = 0.15

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

ax.plot(beta, power, color="#b2cbf2", linewidth=3.00)

ax.set_xlabel(r"Modulation depth  $\beta$  [rad]", fontsize=10, **font_props)
ax.set_ylabel(r"Conversion efficiency [dimensionless]", fontsize=10, **font_props)
ax.set_xlim(0, BETA_MAX)
ax.set_ylim(0, 1.0)
ax.grid()

for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily("Arial")

if SAVE_PATH is not None:
    fig.savefig(SAVE_PATH, dpi=200)
    print(f"Figure saved to {SAVE_PATH}")
plt.show()
