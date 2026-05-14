import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
OMEGA     = 1.0   # angular frequency (rad/s)
AMPLITUDE = 1.0   # peak amplitude

# List of truncation orders to overlay.  Include 0 to also show the exact wave.
N_TERMS_LIST = [0, 10]

N_POINTS = 2000   # number of points per plot
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
BLUE2      = '#2522d4'
GREEN2     = '#93C572'
RED2       = '#e5a3a3'
VIOLET2    = '#C2B7E9'
DARKGREEN2 = '#8DB591'
PINK2      = '#e6b8d0'
BEIGE2     = '#d9b99b'
DARKGRAY2  = '#777777'

SERIES_COLORS = [BLUE2, GREEN2, RED2, VIOLET2, DARKGREEN2, PINK2, BEIGE2]

axes_width_mm  = 150.0
axes_height_mm = 80.0
left_mm, right_mm  = 20.0, 5.0
bottom_mm, top_mm  = 12.0, 5.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

T      = 2 * np.pi / OMEGA
t      = np.linspace(0, 3 * T, N_POINTS)
# exact sawtooth: linear ramp from -A to +A over each period
sawtooth_exact = AMPLITUDE * (2 * ((t / T + 0.5) % 1) - 1)


def fourier_sawtooth(t, n_terms, omega, amplitude):
    """Sum_{n=1}^{n_terms} of the sawtooth Fourier series."""
    s = np.zeros_like(t, dtype=float)
    for n in range(1, n_terms + 1):
        s += ((-1) ** (n + 1)) / n * np.sin(n * omega * t)
    return (2 * amplitude / np.pi) * s


fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

color_idx = 0
for n in N_TERMS_LIST:
    if n == 0:
        ax.plot(t * OMEGA / np.pi, sawtooth_exact,
                color=DARKGRAY2, linewidth=1.2, linestyle='--',
                label='exact')
    else:
        y = fourier_sawtooth(t, n, OMEGA, AMPLITUDE)
        color = SERIES_COLORS[color_idx % len(SERIES_COLORS)]
        ax.plot(t * OMEGA / np.pi, y,
                color=color, linewidth=1.5,
                label=rf'$N={n}$')
        color_idx += 1

ax.axhline(0, color='black', linewidth=0.5, zorder=0)
ax.set_xlabel(r'$\Omega t \, / \, \pi$', fontsize=axis_label_fontsize)
ax.set_ylabel(r'$f(t)$', fontsize=axis_label_fontsize)
ax.set_xlim(t[0] * OMEGA / np.pi, t[-1] * OMEGA / np.pi)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.legend(fontsize=tick_label_fontsize, frameon=False)
for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = Path(__file__).parent / 'sawtooth_fourier.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
