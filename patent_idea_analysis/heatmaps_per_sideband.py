"""
Save a max-efficiency heatmap (beta1 vs beta2) for each of the first
N_SIDEBANDS sidebands in a single 4-column x 2-row figure.

Uses the exact analytical upper bound (N_STAGES=2):
    E_k = max_{beta1,beta2}  sum_n |J_n(beta1)| * |J_{k-n}(beta2)|
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS = 4       # must be <= N_ROWS * N_COLS
BETA_MAX    = 10.0
N_POINTS    = 200     # grid points per beta axis
N_TERMS     = 30      # harmonic truncation

USE_DB   = False      # show in dB instead of linear
DB_FLOOR = -30.0      # dB floor for colour scale

SAVE_PATH = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\heatmaps\all_sidebands2.png"
# None = don't save

# ---------------------------------------------------------------------------
# Shared grid
# ---------------------------------------------------------------------------
betas   = np.linspace(0, BETA_MAX, N_POINTS)
n_range = np.arange(-N_TERMS, N_TERMS + 1)

b1, b2 = np.meshgrid(betas, betas, indexing="ij")
Jn_b1  = jv(n_range, b1[..., np.newaxis])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
N_COLS, N_ROWS = 2, 2
mm_per_inch = 25.4

panel_mm   = 45.0    # width = height of each square panel
gap_h_mm   = 10.0    # horizontal gap between panels
gap_v_mm   = 14.0    # vertical gap between rows (room for title)
left_mm    = 14.0    # left margin (y-label on leftmost column)
right_mm   = 18.0    # right margin (shared colorbar)
bottom_mm  = 12.0    # bottom margin (x-label on bottom row)
top_mm     = 8.0     # top margin
cbar_w_mm  = 5.0     # colorbar width
cbar_gap_mm= 4.0     # gap between last panel and colorbar

total_w_mm = left_mm + N_COLS * panel_mm + (N_COLS - 1) * gap_h_mm + cbar_gap_mm + cbar_w_mm + right_mm
total_h_mm = bottom_mm + N_ROWS * panel_mm + (N_ROWS - 1) * gap_v_mm + top_mm

fig_w = total_w_mm / mm_per_inch
fig_h = total_h_mm / mm_per_inch

matplotlib.rcParams["font.family"] = "Arial"
font_props = {"family": "Arial"}

fig = plt.figure(figsize=(fig_w, fig_h))

# Build axes positions manually so we can place the colorbar precisely.
axes = []
for row in range(N_ROWS):
    for col in range(N_COLS):
        x0 = (left_mm + col * (panel_mm + gap_h_mm)) / total_w_mm
        y0 = (bottom_mm + (N_ROWS - 1 - row) * (panel_mm + gap_v_mm)) / total_h_mm
        w  = panel_mm / total_w_mm
        h  = panel_mm / total_h_mm
        axes.append(fig.add_axes([x0, y0, w, h]))

cbar_x0 = (left_mm + N_COLS * panel_mm + (N_COLS - 1) * gap_h_mm + cbar_gap_mm) / total_w_mm
cbar_y0 = bottom_mm / total_h_mm
cbar_w  = cbar_w_mm / total_w_mm
cbar_h  = (N_ROWS * panel_mm + (N_ROWS - 1) * gap_v_mm) / total_h_mm
cbar_ax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, cbar_h])

# ---------------------------------------------------------------------------
# Fill each panel
# ---------------------------------------------------------------------------
im_ref = None   # keep one imshow handle for the colorbar

for idx, k in enumerate(range(1, N_SIDEBANDS + 1)):
    row, col = divmod(idx, N_COLS)
    ax = axes[idx]

    Jm_b2  = jv(k - n_range, b2[..., np.newaxis])
    E_grid = np.sum(np.abs(Jn_b1) * np.abs(Jm_b2), axis=-1)
    E_sq   = E_grid ** 2

    flat_idx  = np.unravel_index(np.argmax(E_sq), E_sq.shape)
    eff_best  = float(E_sq[flat_idx])
    best_b1, best_b2 = betas[flat_idx[0]], betas[flat_idx[1]]
    print(f"sideband {k}: max efficiency = {eff_best:.4f} ({10*np.log10(eff_best):.2f} dB)"
          f"  beta1={best_b1:.3f}  beta2={best_b2:.3f}")

    if USE_DB:
        plot_data  = 10.0 * np.log10(np.maximum(E_sq, 10 ** (DB_FLOOR / 10)))
        vmin, vmax = DB_FLOOR, 0.0
        cbar_label = r"Max power efficiency  [dB]"
    else:
        plot_data  = E_sq
        vmin, vmax = 0.0, 1.0
        cbar_label = r"Max power efficiency  $|E_k|^2$"

    im = ax.imshow(
        plot_data.T,
        origin="lower",
        extent=[0, BETA_MAX, 0, BETA_MAX],
        aspect="equal",
        vmin=vmin, vmax=vmax,
        cmap="viridis",
        interpolation="bilinear",
    )
    if im_ref is None:
        im_ref = im

    ax.plot(best_b1, best_b2, "o", color="orange",
            markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    ax.set_title(f"$k={k}$", fontsize=9, **font_props, pad=3)

    # x-label only on bottom row
    if row == N_ROWS - 1:
        ax.set_xlabel(r"$\beta_1$  [rad]", fontsize=8, **font_props)
    else:
        ax.set_xticklabels([])

    # y-label only on leftmost column
    if col == 0:
        ax.set_ylabel(r"$\beta_2$  [rad]", fontsize=8, **font_props)
    else:
        ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="both", direction="in", width=1.5, labelsize=7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

# ---------------------------------------------------------------------------
# Shared colorbar
# ---------------------------------------------------------------------------
cb = fig.colorbar(im_ref, cax=cbar_ax)
cb.set_label(cbar_label, fontsize=8, **font_props)
cb.ax.tick_params(labelsize=7, width=1.5, direction="in")
cb.outline.set_linewidth(1.5)
for label in cb.ax.get_yticklabels():
    label.set_fontfamily("Arial")

# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------
if SAVE_PATH is not None:
    fig.savefig(SAVE_PATH, dpi=200)
    print(f"Figure saved to {SAVE_PATH}")
plt.show()
