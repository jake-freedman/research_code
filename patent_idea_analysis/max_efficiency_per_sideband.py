"""
Maximum power conversion efficiency to each of the first N_SIDEBANDS sidebands
for two phase modulators separated by one dispersive element (N_STAGES=2).

For each target harmonic k the upper bound is:
    E_k = max_{beta1, beta2}  sum_n |J_n(beta1)| * |J_{k-n}(beta2)|
and the maximum power efficiency is E_k^2.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
from scipy.special import jv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS = 8       # compute max efficiency for sidebands 1 .. N_SIDEBANDS
BETA_MAX    = 5.0    # search range for each beta
N_POINTS    = 50      # grid points per beta axis (increase for higher accuracy)
N_TERMS     = 50      # harmonic truncation (n = -N_TERMS .. N_TERMS)

SAVE_PATH = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\low_beta"

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
width_mm, height_mm = 100.0, 90.0

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
colors = [cmap(i / (N_SIDEBANDS - 1)) for i in range(N_SIDEBANDS)]

bar_width = 0.6
bar_half  = bar_width / 2

# Corner radius in mm, converted to data units for each axis separately so
# the corners look circular regardless of the axes aspect ratio.
r_mm    = 2.5
x_range = N_SIDEBANDS          # xlim: 0.5 to N_SIDEBANDS+0.5
y_range = 1.0                  # ylim: 0 to 1
rx = r_mm * x_range / width_mm
ry = r_mm * y_range / height_mm

kappa = 0.5523  # cubic Bezier approximation of a quarter-circle

def rounded_bar_path(x0, y0, w, h, rx, ry):
    """Rectangle with rounded top corners and sharp bottom corners."""
    x1, x2 = x0, x0 + w
    y1, y2 = y0, y0 + h
    verts = [
        (x1, y1),                                                          # bottom-left
        (x2, y1),                                                          # bottom-right
        (x2, y2 - ry),                                                     # up to top-right curve
        (x2, y2 - ry + kappa*ry), (x2 - rx + kappa*rx, y2), (x2 - rx, y2),  # top-right corner
        (x1 + rx, y2),                                                     # top edge
        (x1 + rx - kappa*rx, y2), (x1, y2 - ry + kappa*ry), (x1, y2 - ry),  # top-left corner
        (x1, y1),                                                          # back to bottom-left
    ]
    codes = [
        Path.MOVETO,   Path.LINETO,
        Path.LINETO,
        Path.CURVE4,   Path.CURVE4,   Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,   Path.CURVE4,   Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)

for k, h, c in zip(sidebands, max_powers, colors):
    path  = rounded_bar_path(k - bar_half, 0, bar_width, h, rx, ry)
    patch = mpatches.PathPatch(path, facecolor=c, edgecolor="none", zorder=3)
    ax.add_patch(patch)

# legend proxy
ax.add_patch(mpatches.PathPatch(
    rounded_bar_path(0, 0, 0, 0, rx, ry),
    facecolor="#888888", edgecolor="none", label="2 PMs",
))

bar_half = 0.35
ax.hlines(
    max_powers_1pm,
    sidebands - bar_half, sidebands + bar_half,
    colors="black", linewidths=3.0, linestyles="-", zorder=4,
    label="1 PM", capstyle = 'round'
)
ax.legend(fontsize=8, frameon=False, prop={"family": "Arial"})

ax.set_xlabel(r"Sideband order [dimensionless]", fontsize=10, **font_props)
ax.set_ylabel(r"Max conversion efficiency  [dimensionless]", fontsize=10, **font_props)
ax.set_xticks(sidebands)
ax.set_xlim(0.5, N_SIDEBANDS + 0.5)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", zorder=0)

for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily("Arial")

if SAVE_PATH is not None:
    fig.savefig(SAVE_PATH, dpi=200)
    print(f"Figure saved to {SAVE_PATH}")
plt.show()
