import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
OMEGA    = 1.0      # angular frequency of the sawtooth (rad/s)
BETA     = np.pi    # sawtooth amplitude — pi gives perfect SSB for ideal wave
N_MAX    = 10       # maximum truncation order to evaluate
N_POINTS = 20000    # time-domain samples per period (must be >> N_MAX)
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
BLUE2      = '#2522d4'
GREEN2     = '#93C572'
RED2       = '#e5a3a3'
DARKGRAY2  = '#777777'

axes_width_mm  = 150.0
axes_height_mm = 55.0    # per subplot
left_mm, right_mm  = 22.0, 5.0
between_mm         = 8.0
bottom_mm, top_mm  = 12.0, 5.0
n_rows = 3
fig_w  = left_mm + axes_width_mm + right_mm
fig_h  = bottom_mm + n_rows * axes_height_mm + (n_rows - 1) * between_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

T = 2 * np.pi / OMEGA
t = np.linspace(0, T, N_POINTS, endpoint=False)

# Precompute sin(n*Omega*t) for all n up to N_MAX — shape (N_MAX, N_POINTS)
ns = np.arange(1, N_MAX + 1)
sin_terms = np.sin(np.outer(ns, OMEGA * t))           # (N_MAX, N_POINTS)
# Fourier coefficient for the nth term of the sawtooth:  2*beta/pi * (-1)^{n+1}/n
coeffs_n = (2 * BETA / np.pi) * ((-1) ** (ns + 1)) / ns  # (N_MAX,)

# Cumulative phi(t) as we add terms one by one
phi_cumulative = np.zeros(N_POINTS)

efficiency = np.zeros(N_MAX)
carrier_dB = np.zeros(N_MAX)
minus1_dB  = np.zeros(N_MAX)

orders      = np.arange(-4, 5)   # sideband orders to show in 3D plot
fft_indices = np.where(orders >= 0, orders, N_POINTS + orders)
even_spectra = {}   # N -> power array for each order

for idx in range(N_MAX):
    phi_cumulative += coeffs_n[idx] * sin_terms[idx]   # add nth term

    # Fourier coefficients of exp{i*phi(t)} via FFT
    C = np.fft.fft(np.exp(1j * phi_cumulative)) / N_POINTS

    p0  = np.abs(C[0])  ** 2   # carrier power
    p1  = np.abs(C[1])  ** 2   # +1 sideband
    pm1 = np.abs(C[-1]) ** 2   # -1 sideband

    efficiency[idx] = p1
    carrier_dB[idx] = 10 * np.log10(np.maximum(p0,  1e-20) / np.maximum(p1, 1e-20))
    minus1_dB[idx]  = 10 * np.log10(np.maximum(pm1, 1e-20) / np.maximum(p1, 1e-20))

    N = idx + 1
    if N % 2 == 0:
        even_spectra[N] = np.abs(C[fft_indices]) ** 2

N_vals = np.arange(1, N_MAX + 1)

# ── plot ──────────────────────────────────────────────────────────────────────
fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, axes = plt.subplots(3, 1, figsize=(fig_w_in, fig_h_in))

# compute subplot positions manually to match graphics style
ax_h_frac  = axes_height_mm / fig_h
gap_frac   = between_mm     / fig_h
left_frac  = left_mm        / fig_w
right_frac = 1 - right_mm   / fig_w
bot_frac   = bottom_mm      / fig_h
top_frac   = 1 - top_mm     / fig_h

# bottom edges of the three axes (bottom to top in figure coords)
bot_edges = [
    bot_frac,
    bot_frac + ax_h_frac + gap_frac,
    bot_frac + 2 * (ax_h_frac + gap_frac),
]

for ax, bot in zip(axes, reversed(bot_edges)):
    ax.set_position([left_frac, bot, right_frac - left_frac, ax_h_frac])

marker_kw = dict(marker='o', markersize=3, linewidth=1.5)

axes[0].plot(N_vals, efficiency,       color=BLUE2,     **marker_kw)
for n, eff in zip(N_vals, efficiency):
    if n % 2 == 0:
        axes[0].annotate(f'{eff*100:.1f}%',
                         xy=(n, eff), xytext=(4, 4),
                         textcoords='offset points',
                         fontsize=10, color=BLUE2)
axes[1].plot(N_vals, carrier_dB,       color=GREEN2,    **marker_kw)
axes[2].plot(N_vals, minus1_dB,        color=RED2,      **marker_kw)

axes[0].set_ylabel(r'$|c_1|^2$',              fontsize=axis_label_fontsize)
axes[1].set_ylabel(r'Carrier suppression (dB)',    fontsize=axis_label_fontsize)
axes[2].set_ylabel(r'$-1$ sideband suppression (dB)', fontsize=axis_label_fontsize)
axes[2].set_xlabel('Truncation order $N$',     fontsize=axis_label_fontsize)

axes[0].set_ylim(0, 1.05)
axes[0].axhline(1.0, color=DARKGRAY2, linewidth=0.8, linestyle='--')

for ax in axes:
    ax.set_xlim(0, N_MAX+1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

# hide x tick labels on top two axes
for ax in axes[:2]:
    ax.tick_params(labelbottom=False)

out_path = Path(__file__).parent / 'ssb_efficiency.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')

# ── single 3D stem figure (all even N) ───────────────────────────────────────
from matplotlib import cm
from matplotlib.colors import Normalize

norm  = Normalize(vmin=orders.min(), vmax=orders.max())
cmap  = cm.rainbow

fig3d = plt.figure(figsize=(fig_w_in, fig_w_in * 0.85))
ax3d  = fig3d.add_subplot(111, projection='3d')

for N, spectrum in even_spectra.items():
    for order, power in zip(orders, spectrum):
        color = cmap(norm(order))
        ax3d.plot([order, order], [N, N], [0, float(power)],
                  color=color, linewidth=1.5)
        ax3d.scatter([order], [N], [float(power)],
                     color=color, s=25, depthshade=False, zorder=5)

ax3d.set_xlabel('Sideband order', fontsize=axis_label_fontsize, labelpad=8)
ax3d.set_ylabel('Truncation $N$', fontsize=axis_label_fontsize, labelpad=8)
ax3d.set_zlabel('Power',          fontsize=axis_label_fontsize, labelpad=6)
ax3d.set_xticks(orders)
ax3d.set_yticks(sorted(even_spectra.keys()))
ax3d.set_zlim(0, 1)
ax3d.tick_params(labelsize=tick_label_fontsize)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, pad=0.1)
cbar.set_label('Sideband order', fontsize=axis_label_fontsize)
cbar.set_ticks(orders)
cbar.ax.tick_params(labelsize=tick_label_fontsize)

out3d = Path(__file__).parent / 'ssb_spectrum_3d.png'
fig3d.savefig(out3d, dpi=200, bbox_inches='tight')
print(f'Saved: {out3d}')

plt.show()
