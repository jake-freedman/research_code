"""
Theoretical maximum power conversion efficiency to a target sideband (n_stages=2 only).

For two phase modulators separated by one dispersive element, the maximum
amplitude at harmonic k — achieved when the dispersion phases are chosen for
perfect constructive interference — is:

    E_k = sum_n  |J_n(beta1)| * |J_{k-n}(beta2)|

By Cauchy-Schwarz and the Bessel identity sum_n J_n(x)^2 = 1, E_k <= 1, so
the colour scale [0, 1] is exact.

NOTE: for n_stages > 2 this absolute-value convolution formula gives values
that can exceed 1 and is therefore physically wrong.  The reason is that with
two separate dispersion elements the phase constraints for all (n, m) pairs to
interfere constructively simultaneously are overdetermined — the tight bound
only holds for n_stages = 2.  For n_stages > 2 use the basin-hopping
optimiser (run_optimization.py) to find the achievable efficiency numerically.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.special import jv

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_STAGES        = 2       # number of phase modulators
WANTED_HARMONIC = 1       # target sideband index
BETA_MAX        = 10.0
N_POINTS        = 30      # grid points per beta axis
N_TERMS         = 50      # harmonic truncation (n = -N_TERMS .. N_TERMS)

USE_DB   = False          # show heatmap (n=2 only) in dB
DB_FLOOR = -30.0          # dB floor for colour scale

SAVE_PATH = local_path(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\best1")          # e.g. r"C:\path\to\figure.png"; None = don't save

# ---------------------------------------------------------------------------
# Helper: max-efficiency amplitude for one beta combination via iterative
# absolute-value convolution
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
if N_STAGES != 2:
    raise ValueError(
        f"N_STAGES={N_STAGES} is not supported. The absolute-value convolution "
        "bound is only valid for N_STAGES=2; for more stages it exceeds 1 and "
        "is physically wrong. Use run_optimization.py to find the achievable "
        "efficiency numerically."
    )

betas   = np.linspace(0, BETA_MAX, N_POINTS)
n_range = np.arange(-N_TERMS, N_TERMS + 1)

b1, b2 = np.meshgrid(betas, betas, indexing="ij")
Jn_b1  = jv(n_range, b1[..., np.newaxis])
Jm_b2  = jv(WANTED_HARMONIC - n_range, b2[..., np.newaxis])
E_grid = np.sum(np.abs(Jn_b1) * np.abs(Jm_b2), axis=-1)

E_sq = E_grid ** 2

# ---------------------------------------------------------------------------
# Print result
# ---------------------------------------------------------------------------
flat_idx  = np.unravel_index(np.argmax(E_sq), E_sq.shape)
best_betas = [betas[i] for i in flat_idx]
eff_best   = float(E_sq[flat_idx])
amp_best   = float(E_grid[flat_idx])

betas_str = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(best_betas))
print(f"n_stages={N_STAGES}  wanted harmonic={WANTED_HARMONIC}")
print(f"Maximum power efficiency : {eff_best:.6f}  ({10*np.log10(eff_best):.2f} dB)")
print(f"  {betas_str}")
print(f"  E amplitude = {amp_best:.6f}")

# ---------------------------------------------------------------------------
# Plot (n_stages == 2 only)
# ---------------------------------------------------------------------------
if True:  # always runs (N_STAGES==2 guaranteed above)
    mm_per_inch = 25.4
    width_mm, height_mm = 130.0, 90.0
    cbar_pad_mm = 12.0

    left_in   = 0.60
    bottom_in = 0.50
    top_in    = 0.15
    right_in  = (cbar_pad_mm + 10) / mm_per_inch

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

    ax.set_xlabel(r"$\beta_1$  [rad]", fontsize=10, **font_props)
    ax.set_ylabel(r"$\beta_2$  [rad]", fontsize=10, **font_props)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

    cbar_ax = fig.add_axes([
        (left_in + ax_w_in + 4 / mm_per_inch) / fig_w,
        bottom_in / fig_h,
        (6 / mm_per_inch) / fig_w,
        ax_h_in / fig_h,
    ])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=10, **font_props)
    cb.ax.tick_params(labelsize=8, width=2, direction="in")
    cb.outline.set_linewidth(2)
    for label in cb.ax.get_yticklabels():
        label.set_fontfamily("Arial")

    if SAVE_PATH is not None:
        fig.savefig(SAVE_PATH, dpi=200)
        print(f"Figure saved to {SAVE_PATH}")
    plt.show()
