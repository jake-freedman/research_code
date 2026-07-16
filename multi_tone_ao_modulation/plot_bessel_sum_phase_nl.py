import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
# Modulation phase:
#   beta1*sin(Omega*t) + beta2_nl*sin(2*Omega*t) + beta2*sin(2*Omega*t + phi2)
#
# beta1 (ch1, at f) and beta2_nl (an unintentional 2f term, e.g. from AOM
# nonlinearity) carry no independent phase reference; only phi2, the phase of
# the intentional 2f drive relative to beta2_nl, is a free parameter.
BETA1    = 1.94
BETA2_NL = -0.2
BETA2    = 0.2

# Harmonic orders to plot. Each p gives the beat component at p*f.
ORDERS = [-2, -1, 0, 1, 2]
ORDERS = [-1, 1]

N_GRID  = 400   # grid points along phi2
K_TRUNC = int(2 * max(BETA1, BETA2, BETA2_NL)) + 20

# Add a trace showing Sigma_p |S_p|^2 summed over all ORDERS.
SHOW_SUM = False
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
LINEWIDTH = 2.00

# Same harmonic color scheme as dual_tone_sweep_analysis.py / dual_tone_sweep_data.py
_HARMONIC_COLORS = {
    -3: '#bf7362',
    -2: '#e5a3a3',   # RED2
    -1: '#FBD8A2',   # ORANGE2
     0: '#93C572',   # GREEN2
     1: '#b2cbf2',   # LIGHTBLUE2
     2: '#5c70aa',
     3: '#C2B7E9',   # VIOLET2
}
_EXTRA_COLORS = ['#2522d4', '#e6b8d0', '#EED7A1', '#8DB591', '#475c6c', '#777777']
# ─────────────────────────────────────────────────────────────────────────────


def bessel_sum_phi2(beta1: float, beta2_nl: float, beta2: float, p: int,
                     phi2_grid: np.ndarray) -> np.ndarray:
    """
    |S_p|^2 vs phi2, where

        S_p = Sum_{m,k} J_{p-2m-2k}(beta1) * J_m(beta2_nl) * J_k(beta2) * exp(i*k*phi2)

    Expanding each of the three exp(i*beta*sin(...)) factors as a Bessel
    series and collecting the coefficient of exp(i*p*Omega*t) gives a double
    sum over the beta1 and beta2_nl harmonic indices (n, m) constrained by
    n + 2m + 2k = p. Since only phi2 (attached to k) is a free phase, the
    m-sum can be done first, collapsing this to a single sum over k -- the
    same structure as the two-tone case.
    """
    m = np.arange(-K_TRUNC, K_TRUNC + 1)
    k = np.arange(-K_TRUNC, K_TRUNC + 1)
    order_n = p - 2 * m[:, None] - 2 * k[None, :]                       # (Nm, Nk)
    inner = np.sum(jv(order_n, beta1) * jv(m, beta2_nl)[:, None], axis=0)  # (Nk,)
    C_k = inner * jv(k, beta2)                                          # (Nk,)
    S = np.sum(C_k[None, :] * np.exp(1j * k[None, :] * phi2_grid[:, None]), axis=-1)
    return np.abs(S) ** 2


phi2 = np.linspace(0, 2 * np.pi, N_GRID)


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))


fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

extra_iter = iter(_EXTRA_COLORS)
z1d_arrays = []
for p in ORDERS:
    Z = bessel_sum_phi2(BETA1, BETA2_NL, BETA2, p, phi2)
    z1d_arrays.append(Z)
    color = _HARMONIC_COLORS.get(p, next(extra_iter, '#000000'))
    ax.plot(phi2 / np.pi, Z, color=color, linewidth=LINEWIDTH, label=rf'$p={p}$')

if SHOW_SUM and len(ORDERS) > 1:
    ax.plot(phi2 / np.pi, sum(z1d_arrays),
            color='#333333', linewidth=1.5, linestyle='--',
            label=r'$\Sigma\,|S_p|^2$')

ax.set_xlabel(r'$\phi_2\,/\,\pi$', fontsize=axis_label_fontsize)
ax.set_ylabel(r'$|S_p|^2$', fontsize=axis_label_fontsize)
ax.set_xlim(0, 2)
ax.set_ylim(0, None)
ax.legend(fontsize=tick_label_fontsize, frameon=False)
_style_ax(ax)

out_path = Path(__file__).parent / 'bessel_sum_phase_nl_map.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
