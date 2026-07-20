import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
)

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ── configuration ─────────────────────────────────────────────────────────────

BETA1    = 2.46     # ch1 modulation depth (rad), drive at f
BETA2    = 1.10     # ch2 modulation depth (rad), drive at 2f
PHI1_DEG = 0.0      # ch1 phase (deg)
PHI2_DEG = 179.5     # ch2 phase (deg)

# Harmonic orders to plot and print. Both the spectrum and the console table
# show exactly these orders.
PRINT_ORDERS = [-3, -2, -1, 0, 1, 2, 3]

# Power display mode:
#   'percent' → |A_p|² as % of total optical power
#   'dB'      → 10·log10(|A_p|²) in dBc  (0 dBc = all power in one line)
DISPLAY_MODE = 'dB'

# dB mode only: floor of the y-axis in dBc. None = auto.
FLOOR_DBc = -80.0

# Bessel series truncation (increase for large β)
K_TRUNC = int(2 * max(BETA1, BETA2)) + 20
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm   = 100.0
axes_height_mm  =  40.0
left_mm         =  20.0
right_mm        =  10.0
bottom_mm       =  15.0
top_mm          =   8.0
spine_linewidth =   2.0
tick_width      =   2.0
tick_direction  = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize  =  8.0
stem_linewidth  =   3
markersize      =   9.0
# ─────────────────────────────────────────────────────────────────────────────


def dual_tone_amplitudes(
    beta1: float, beta2: float,
    phi1: float, phi2: float,
    orders: list[int], k_trunc: int,
) -> dict[int, complex]:
    """
    Complex amplitude A_p at harmonic p·f for a dual-tone drive.

        A_p = Σ_k  J_{p-2k}(β1) · J_k(β2) · exp(i[(p-2k)·φ1 + k·φ2])

    ch1 drives at f  with depth β1 and phase φ1.
    ch2 drives at 2f with depth β2 and phase φ2.
    """
    k = np.arange(-k_trunc, k_trunc + 1)
    return {
        p: complex(np.sum(
            jv(p - 2 * k, beta1) * jv(k, beta2)
            * np.exp(1j * ((p - 2 * k) * phi1 + k * phi2))
        ))
        for p in orders
    }


# ── compute ───────────────────────────────────────────────────────────────────
phi1 = np.deg2rad(PHI1_DEG)
phi2 = np.deg2rad(PHI2_DEG)

amps = dual_tone_amplitudes(BETA1, BETA2, phi1, phi2, sorted(set(PRINT_ORDERS)), K_TRUNC)

powers_lin   = {p: abs(A) ** 2             for p, A in amps.items()}
phases_deg_d = {p: np.degrees(np.angle(A)) for p, A in amps.items()}

plot_orders = np.array(sorted(PRINT_ORDERS))
plot_powers = np.array([powers_lin[p] for p in plot_orders])

if DISPLAY_MODE == 'percent':
    y          = plot_powers * 100.0
    y_baseline = 0.0
    ylabel     = r'$|A_p|^2$ [% of total power]'
else:
    y          = 10.0 * np.log10(np.maximum(plot_powers, 1e-30))
    y_baseline = FLOOR_DBc if FLOOR_DBc is not None else y.min() - 3.0
    ylabel     = r'$|A_p|^2$ [dBc]'

# ── console output ────────────────────────────────────────────────────────────
unit = '%' if DISPLAY_MODE == 'percent' else 'dBc'
print(f"β1={BETA1},  β2={BETA2},  φ1={PHI1_DEG:.1f}°,  φ2={PHI2_DEG:.1f}°")
print(f"  {'order':>5}   {'power':>12}   {'phase [deg]':>12}")
print(f"  {'─'*5}   {'─'*12}   {'─'*12}")
for p in sorted(PRINT_ORDERS):
    if p not in powers_lin:
        continue
    pwr = (powers_lin[p] * 100.0
           if DISPLAY_MODE == 'percent'
           else 10.0 * np.log10(max(powers_lin[p], 1e-30)))
    print(f"  p={p:+d}   {pwr:>10.4f} {unit}   {phases_deg_d[p]:>+10.2f}°")

total = sum(powers_lin[p] for p in plot_orders)
print(f"\n  Total power in plotted orders: {total * 100:.2f}%")

# ── figure ────────────────────────────────────────────────────────────────────
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

# Stems and balls, one color per combline
for idx, (xi, yi) in enumerate(zip(plot_orders, y)):
    c = _COLORS[idx % len(_COLORS)]
    ax.plot([xi, xi], [y_baseline, yi],
            color=c, linewidth=stem_linewidth,
            solid_capstyle='butt', zorder=2)
    ax.plot(xi, yi, 'o', color=c, markersize=markersize,
            markeredgewidth=0, zorder=3)

# Baseline
ax.axhline(y_baseline, color='#333333', linewidth=0.8, linestyle='-', zorder=1)

ax.set_xlabel(r'Harmonic order $p$', fontsize=axis_label_fontsize)
ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
ax.set_xlim(plot_orders[0] - 0.8, plot_orders[-1] + 0.8)
if DISPLAY_MODE == 'percent':
    ax.set_ylim(bottom=0)
else:
    ax.set_ylim(bottom=y_baseline, top = 0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

title = (rf'$\beta_1={BETA1}$,  $\beta_2={BETA2}$,'
         rf'  $\phi_1={PHI1_DEG:.0f}°$,  $\phi_2={PHI2_DEG:.0f}°$')
ax.set_title(title, fontsize=axis_label_fontsize)
ax.grid()

out_path = Path(__file__).parent / 'dual_tone_spectrum.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.show()
