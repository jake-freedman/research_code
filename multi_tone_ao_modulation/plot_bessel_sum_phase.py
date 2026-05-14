import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
BETA1 = 1.94
BETA2 = 0.88

# False: sum_k J_{2k+1}(b1) * J_{-k}(b2)   * exp(i[(2k+1)*phi1 - k*phi2])
# True:  sum_k J_{2k}(b1)   * J_{1-k}(b2)  * exp(i[2k*phi1 + (1-k)*phi2])
is_second_harmonic = False

N_GRID  = 400   # grid points along each axis
K_TRUNC = int(2 * max(BETA1, BETA2)) + 20

# If True, plot |S|^2 vs phi2 as a 1D line (phi1 fixed at 0).
# If False, plot the full 2D heatmap over (phi1, phi2).
PLOT_1D = True

# Overlay lines of constant phi2 - phi1 (in units of pi, e.g. [0, 0.5, 1.0])
# Set to None or [] to disable.
SHOW_DELTA_PHI_CONTOURS = False
DELTA_PHI_VALUES = [0, 0.5, 1.0, 1.5]   # (phi2 - phi1) / pi
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm  = 120.0
axes_height_mm = 100.0
left_mm, right_mm  = 20.0, 15.0
bottom_mm, top_mm  = 15.0,  8.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth   = 2.0
tick_width        = 2.0
tick_direction    = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize =  8.0
# ─────────────────────────────────────────────────────────────────────────────


def bessel_sum_grid(beta1, beta2, phi1_grid, phi2_grid) -> np.ndarray:
    """Evaluate |S|^2 over a meshgrid of phi1, phi2."""
    k = np.arange(-K_TRUNC, K_TRUNC + 1)   # (2K+1,)

    if is_second_harmonic:
        bj1 = jv(2 * k, beta1)        # (2K+1,)
        bj2 = jv(1 - k, beta2)
        # phase[i,j,m] = 2k[m]*phi1[i,j] + (1-k[m])*phi2[i,j]
        phase_k = (2 * k)[None, None, :] * phi1_grid[:, :, None] \
                + (1 - k)[None, None, :] * phi2_grid[:, :, None]
    else:
        bj1 = jv(2 * k + 1, beta1)
        bj2 = jv(-k, beta2)
        phase_k = (2 * k + 1)[None, None, :] * phi1_grid[:, :, None] \
                - k[None, None, :]           * phi2_grid[:, :, None]

    coeffs = bj1 * bj2                          # (2K+1,)
    S = np.sum(coeffs[None, None, :] * np.exp(1j * phase_k), axis=-1)
    return np.abs(S) ** 2


phi = np.linspace(0, 2 * np.pi, N_GRID)
mode = 'second harmonic' if is_second_harmonic else 'fundamental'

fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

if PLOT_1D:
    Z1d = bessel_sum_grid(BETA1, BETA2,
                          np.zeros((1, N_GRID)),
                          phi[np.newaxis, :]).squeeze()
    ax.plot(phi / np.pi, Z1d, color='#2522d4', linewidth=1.5)
    ax.set_xlabel(r'$\phi_2 / \pi$', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$|S|^2$', fontsize=axis_label_fontsize)
    ax.set_title(
        rf'$|S|^2$  ({mode},  $\phi_1=0$,  $\beta_1={BETA1}$,  $\beta_2={BETA2}$)',
        fontsize=axis_label_fontsize,
    )
    ax.set_xlim(0, 2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
else:
    PHI1, PHI2 = np.meshgrid(phi, phi, indexing='ij')
    Z = bessel_sum_grid(BETA1, BETA2, PHI1, PHI2)
    im = ax.pcolormesh(PHI1 / np.pi, PHI2 / np.pi, Z, cmap='viridis', shading='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$|S|^2$', fontsize=axis_label_fontsize)
    cb.ax.tick_params(labelsize=tick_label_fontsize)
    ax.set_xlabel(r'$\phi_1 / \pi$', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$\phi_2 / \pi$', fontsize=axis_label_fontsize)
    ax.set_title(
        rf'$|S|^2$  ({mode},  $\beta_1={BETA1}$,  $\beta_2={BETA2}$)',
        fontsize=axis_label_fontsize,
    )
    if SHOW_DELTA_PHI_CONTOURS and DELTA_PHI_VALUES:
        phi1_vals = np.array([0, 2])
        for dphi in DELTA_PHI_VALUES:
            for shift in [0, -2, 2]:
                ax.plot(phi1_vals, phi1_vals + dphi + shift,
                        color='white', linewidth=0.8, linestyle='--', alpha=0.7,
                        label=rf'$\Delta\phi/\pi={dphi}$' if shift == 0 else None)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.legend(fontsize=tick_label_fontsize, frameon=False,
                  loc='upper right', labelcolor='white')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = Path(__file__).parent / 'bessel_sum_phase_map.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
