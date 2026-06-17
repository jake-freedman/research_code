import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
BETA1 = 1.9
BETA2 = 0.88

# Harmonic orders to plot. Each p gives the beat component at p*f,
# where ch1 drives at f (β1, φ1) and ch2 drives at 2f (β2, φ2).
ORDERS = [-3, -2, -1, 0, 1, 2, 3]

N_GRID  = 400   # grid points along each axis
K_TRUNC = int(2 * max(BETA1, BETA2)) + 20

# If True, plot |S_p|^2 vs φ2 as 1D lines (φ1 fixed at 0), one per order.
# If False, plot a 2D heatmap over (φ1, φ2) for each order in separate subplots.
PLOT_1D = True

# If True, add a trace/subplot showing Σ_p |S_p|^2 summed over all ORDERS.
# In 1D mode this is an extra dashed line on the same axes.
# In 2D mode this is an extra subplot to the right of the individual-order panels.
SHOW_SUM = True

# 2D mode only: overlay lines of constant φ2 − φ1 (in units of π).
SHOW_DELTA_PHI_CONTOURS = False
DELTA_PHI_VALUES = [0, 0.5, 1.0, 1.5]
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm  = 150.0
axes_height_mm = 80.0
left_mm, right_mm  = 20.0, 15.0
bottom_mm, top_mm  = 15.0,  8.0
spine_linewidth   = 2.0
tick_width        = 2.0
tick_direction    = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize =  8.0

_ORDER_COLORS = ['#2522d4', '#d43a22', '#22a44a', '#b022d4', '#d4a022', '#22ccd4']
# ─────────────────────────────────────────────────────────────────────────────


def bessel_sum_grid(beta1: float, beta2: float, p: int,
                    phi1_grid: np.ndarray, phi2_grid: np.ndarray) -> np.ndarray:
    """
    |A_p|^2 at harmonic p over a meshgrid of (phi1, phi2).

        A_p = Σ_k  J_{p-2k}(β1) · J_k(β2) · exp(i[(p-2k)·φ1 + k·φ2])
    """
    k = np.arange(-K_TRUNC, K_TRUNC + 1)
    coeffs = jv(p - 2 * k, beta1) * jv(k, beta2)
    phase_k = (p - 2 * k)[None, None, :] * phi1_grid[:, :, None] \
            + k[None, None, :]            * phi2_grid[:, :, None]
    S = np.sum(coeffs[None, None, :] * np.exp(1j * phase_k), axis=-1)
    return np.abs(S) ** 2


phi = np.linspace(0, 2 * np.pi, N_GRID)


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))


if PLOT_1D:
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )
    z1d_arrays = []
    for idx, p in enumerate(ORDERS):
        Z1d = bessel_sum_grid(BETA1, BETA2, p,
                              np.zeros((1, N_GRID)),
                              phi[np.newaxis, :]).squeeze()
        z1d_arrays.append(Z1d)
        ax.plot(phi / np.pi, Z1d,
                color=_ORDER_COLORS[idx % len(_ORDER_COLORS)],
                linewidth=1.5, label=rf'$p={p}$')
    if SHOW_SUM and len(ORDERS) > 1:
        ax.plot(phi / np.pi, sum(z1d_arrays),
                color='#333333', linewidth=1.5, linestyle='--',
                label=r'$\Sigma\,|S_p|^2$')
    ax.set_xlabel(r'$\phi_2\,/\,\pi$', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$|S_p|^2$', fontsize=axis_label_fontsize)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_ax(ax)

else:
    n = len(ORDERS)
    n_plots = n + (1 if SHOW_SUM and n > 1 else 0)
    gap_mm = 12.0
    fig_w = left_mm + n_plots * axes_width_mm + (n_plots - 1) * gap_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_w / 25.4, fig_h / 25.4), squeeze=False)

    PHI1, PHI2 = np.meshgrid(phi, phi, indexing='ij')
    z2d_arrays = []
    for col, p in enumerate(ORDERS):
        ax = axes[0, col]
        Z = bessel_sum_grid(BETA1, BETA2, p, PHI1, PHI2)
        z2d_arrays.append(Z)
        im = ax.pcolormesh(PHI1 / np.pi, PHI2 / np.pi, Z, cmap='viridis', shading='auto')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(r'$|S_p|^2$', fontsize=axis_label_fontsize)
        cb.ax.tick_params(labelsize=tick_label_fontsize)
        ax.set_xlabel(r'$\phi_1\,/\,\pi$', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\phi_2\,/\,\pi$', fontsize=axis_label_fontsize)
        ax.set_title(rf'$p={p}$,  $\beta_1={BETA1}$,  $\beta_2={BETA2}$',
                     fontsize=axis_label_fontsize)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        _style_ax(ax)

        if SHOW_DELTA_PHI_CONTOURS and DELTA_PHI_VALUES:
            phi1_vals = np.array([0, 2])
            for dphi in DELTA_PHI_VALUES:
                for shift in [0, -2, 2]:
                    ax.plot(phi1_vals, phi1_vals + dphi + shift,
                            color='white', linewidth=0.8, linestyle='--', alpha=0.7,
                            label=(rf'$\Delta\phi/\pi={dphi}$'
                                   if shift == 0 and col == 0 else None))
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            if col == 0:
                ax.legend(fontsize=tick_label_fontsize, frameon=False,
                          loc='upper right', labelcolor='white')

    if SHOW_SUM and n > 1:
        ax = axes[0, n]
        Z_sum = sum(z2d_arrays)
        im = ax.pcolormesh(PHI1 / np.pi, PHI2 / np.pi, Z_sum, cmap='viridis', shading='auto')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(r'$\Sigma\,|S_p|^2$', fontsize=axis_label_fontsize)
        cb.ax.tick_params(labelsize=tick_label_fontsize)
        ax.set_xlabel(r'$\phi_1\,/\,\pi$', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\phi_2\,/\,\pi$', fontsize=axis_label_fontsize)
        ax.set_title(rf'Sum,  $\beta_1={BETA1}$,  $\beta_2={BETA2}$',
                     fontsize=axis_label_fontsize)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        _style_ax(ax)

out_path = Path(__file__).parent / 'bessel_sum_phase_map.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
