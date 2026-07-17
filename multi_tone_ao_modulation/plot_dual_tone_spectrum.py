import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
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
PHI2_DEG = 180.0    # ch2 phase (deg)

# Harmonic orders to plot and print. Both the spectrum and the console table
# show exactly these orders.
PRINT_ORDERS = [-3, -2, -1, 0, 1, 2, 3]

# Power display mode:
#   'percent' → |A_p|² as % of total optical power
#   'dB'      → 10·log10(|A_p|²) in dBc  (0 dBc = all power in one line)
DISPLAY_MODE = 'dB'

# Plot style: 'stem' (ball-and-stick) or 'bar' (filled bar with a vertical
# opacity gradient from the baseline to each harmonic's value).
PLOT_STYLE = 'stem'

SHOW_GRID   = True
SHOW_LEGEND = False

# dB mode only: floor of the y-axis in dBc. None = auto.
FLOOR_DBc = -50.0

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

# ── bar style (PLOT_STYLE = 'bar') ────────────────────────────────────────────
# Each harmonic drawn as a bar from the baseline to its value, filled with a
# vertical opacity gradient (BAR_ALPHA_MIN at the baseline, BAR_ALPHA_MAX at
# the bar's value) rather than a flat fill.
BAR_WIDTH      = 0.6
BAR_LINESTYLE  = '-'
BAR_LINEWIDTH  = 1.5
BAR_EDGE_COLOR = None   # None = match the harmonic's color
BAR_EDGE_ALPHA = 1.0
BAR_FACE_COLOR = None   # None = match the harmonic's color

BAR_ALPHA_MIN = 0.15   # opacity at the baseline (bar's bottom)
BAR_ALPHA_MAX = 0.95   # opacity at the bar's value (bar's top)
# Shape of the ramp from BAR_ALPHA_MIN to BAR_ALPHA_MAX: opacity follows
# t**BAR_GRADIENT_ORDER, t going 0 (baseline) to 1 (bar's value). 1 = linear;
# >1 = stays near ALPHA_MIN longer, then ramps up faster near the top;
# <1 (e.g. 0.5) = ramps up faster near the bottom, then levels off near ALPHA_MAX.
BAR_GRADIENT_ORDER = 1.0
BAR_GRADIENT_RESOLUTION = 200   # vertical samples in the opacity gradient

# Optional marker at the bar's value; None = no marker.
BAR_MARKER            = None
BAR_MARKERSIZE        = 6
BAR_MARKER_FACECOLOR  = None   # None = match the harmonic's color
BAR_MARKER_FACE_ALPHA = 1.0
BAR_MARKER_EDGECOLOR  = None   # None = match the harmonic's color
BAR_MARKER_EDGE_ALPHA = 1.0

BAR_ZORDER         = 2   # gradient fill
BAR_OUTLINE_ZORDER = 3   # bar outline
BAR_MARKER_ZORDER  = 4
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels and the legend (from the same figure
# that's shown and PNG-saved below), and additionally saves an SVG.
FOR_PUBLICATION = False
PUBLICATION_SVG_NAME = 'dual_tone_spectrum_pub.svg'
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


def _draw_gradient_bar(ax, x_center: float, width: float, y_base: float, y_val: float,
                        color, alpha_min: float, alpha_max: float, order: float,
                        n_pts: int, zorder: float):
    """
    Fill a bar from y_base to y_val with a vertical opacity gradient: alpha_min
    at y_base, alpha_max at y_val -- i.e. increasing opacity from the
    baseline (the bar's bottom) to the harmonic's value (the bar's top).
    The ramp follows t**order (t: 0 at y_base, 1 at y_val); order=1 is linear.
    """
    y_lo, y_hi = (y_base, y_val) if y_val >= y_base else (y_val, y_base)
    y_grid = np.linspace(y_lo, y_hi, n_pts)
    if y_val == y_base:
        t = np.zeros(n_pts)
    else:
        t = np.clip((y_grid - y_base) / (y_val - y_base), 0.0, 1.0)
    alpha = alpha_min + (t ** order) * (alpha_max - alpha_min)

    rgba = np.zeros((n_pts, 1, 4))
    rgba[:, 0, :3] = mcolors.to_rgb(color)
    rgba[:, 0, 3] = alpha

    ax.imshow(rgba, extent=(x_center - width / 2, x_center + width / 2, y_lo, y_hi),
              origin='lower', aspect='auto', interpolation='bilinear', zorder=zorder)


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

if PLOT_STYLE == 'bar':
    for idx, (xi, yi) in enumerate(zip(plot_orders, y)):
        n = int(xi)
        color = _COLORS[idx % len(_COLORS)]
        edge_color = BAR_EDGE_COLOR if BAR_EDGE_COLOR is not None else color
        face_color = BAR_FACE_COLOR if BAR_FACE_COLOR is not None else color
        marker_face = BAR_MARKER_FACECOLOR if BAR_MARKER_FACECOLOR is not None else color
        marker_edge = BAR_MARKER_EDGECOLOR if BAR_MARKER_EDGECOLOR is not None else color
        label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n:+d}'

        _draw_gradient_bar(ax, xi, BAR_WIDTH, y_baseline, yi, face_color,
                            BAR_ALPHA_MIN, BAR_ALPHA_MAX, BAR_GRADIENT_ORDER,
                            BAR_GRADIENT_RESOLUTION, BAR_ZORDER)

        bars = ax.bar(xi, yi - y_baseline, width=BAR_WIDTH, bottom=y_baseline,
                       facecolor='none',
                       edgecolor=mcolors.to_rgba(edge_color, BAR_EDGE_ALPHA),
                       linewidth=BAR_LINEWIDTH, linestyle=BAR_LINESTYLE,
                       zorder=BAR_OUTLINE_ZORDER, label=label)
        for patch in bars:
            patch.set_capstyle('round')

        if BAR_MARKER is not None:
            ax.plot([xi], [yi], marker=BAR_MARKER, markersize=BAR_MARKERSIZE,
                    markerfacecolor=mcolors.to_rgba(marker_face, BAR_MARKER_FACE_ALPHA),
                    markeredgecolor=mcolors.to_rgba(marker_edge, BAR_MARKER_EDGE_ALPHA),
                    linestyle='none', zorder=BAR_MARKER_ZORDER)
else:
    # Stems and balls, one color per combline
    for idx, (xi, yi) in enumerate(zip(plot_orders, y)):
        n = int(xi)
        c = _COLORS[idx % len(_COLORS)]
        label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n:+d}'
        ax.plot([xi, xi], [y_baseline, yi],
                color=c, linewidth=stem_linewidth,
                solid_capstyle='butt', zorder=2, label=label)
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
if SHOW_GRID:
    ax.grid()
if SHOW_LEGEND:
    ax.legend(fontsize=tick_label_fontsize, frameon=False)

if FOR_PUBLICATION:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False, labelleft=False)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    pub_path = Path(__file__).parent / PUBLICATION_SVG_NAME
    fig.savefig(pub_path, format='svg', bbox_inches='tight')
    print(f'Saved: {pub_path}')

out_path = Path(__file__).parent / 'dual_tone_spectrum.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.show()
