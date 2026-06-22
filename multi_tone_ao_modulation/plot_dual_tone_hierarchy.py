"""
Hierarchical dual-tone spectrum — one figure per panel, signed amplitudes.

Figure 1 (main): Re[J_p(β1)·exp(i·p·φ1)] for each harmonic order p.
                 Positive stems go up, negative stems go down.
Figures 2+     : for each combline p in ZOOM_COMBLINES, the β2 (2f)
                 sideband amplitudes Re[J_p(β1)·e^{ipφ1}·J_k(β2)·e^{ikφ2}]
                 at relative positions 2k, using the parent combline colour.
All figures: no axes outline, no tick marks or labels, no y axis.
"""

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from pathlib import Path
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
)

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ── configuration ─────────────────────────────────────────────────────────────
BETA1    = 1.9
BETA2    = 0.88
PHI1_DEG = 0.0
PHI2_DEG = 180.0

# Harmonic orders shown in the main (β1 only) figure
MAIN_ORDERS = list(range(-3, 4))      # [-3, -2, -1, 0, 1, 2, 3]

# Which of those comblines to expand into separate β2 figures
ZOOM_COMBLINES = [-1, 0, 1]

# β2 sideband orders k: sidebands land at relative position 2k (units of f)
SIDEBAND_K = list(range(-2, 3))       # [-2, -1, 0, 1, 2]
# ─────────────────────────────────────────────────────────────────────────────

# ── layout (mm) ───────────────────────────────────────────────────────────────
# Main figure
MAIN_AXES_W_MM = 40.0
MAIN_AXES_H_MM =  20.0

# Mini figures (one per ZOOM_COMBLINE)
MINI_AXES_W_MM =  40.0
MINI_AXES_H_MM =  40.0

# Shared margins
LEFT_MM   =  5.0
RIGHT_MM  =  5.0
TOP_MM    =  5.0
BOTTOM_MM =  5.0

stem_linewidth = 3
markersize     = 8.0
baseline_color = '#333333'
baseline_lw    = 0.8
# ─────────────────────────────────────────────────────────────────────────────


def _strip_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _make_fig_ax(axes_w_mm, axes_h_mm):
    fig_w = LEFT_MM + axes_w_mm + RIGHT_MM
    fig_h = BOTTOM_MM + axes_h_mm + TOP_MM
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = LEFT_MM   / fig_w,
        right  = 1 - RIGHT_MM   / fig_w,
        bottom = BOTTOM_MM / fig_h,
        top    = 1 - TOP_MM    / fig_h,
    )
    return fig, ax


def _draw_signed_stems(ax, xs, signed_amps, colors):
    """Signed stems: positive up, negative down. colors: list or single value."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    for xi, yi, c in zip(xs, signed_amps, colors):
        ax.plot([xi, xi], [0.0, yi], color=c, linewidth=stem_linewidth,
                solid_capstyle='butt', zorder=2)
        ax.plot(xi, yi, 'o', color=c, markersize=markersize,
                markeredgewidth=0, zorder=3)
    ax.axhline(0.0, color=baseline_color, linewidth=baseline_lw, zorder=1)


def _sym_ylim(ax, amps, pad=1.15):
    y_max = float(np.abs(amps).max()) * pad
    ax.set_ylim(-y_max, y_max)


# ── compute ───────────────────────────────────────────────────────────────────
phi1 = np.deg2rad(PHI1_DEG)
phi2 = np.deg2rad(PHI2_DEG)

main_orders = np.array(sorted(MAIN_ORDERS))
sideband_k  = np.array(sorted(SIDEBAND_K))
sb_xs_rel   = 2 * sideband_k            # relative x positions (units of f)

# Signed β1 amplitudes: Re[J_p(β1)·exp(i·p·φ1)]
beta1_signed = np.real(jv(main_orders, BETA1) * np.exp(1j * main_orders * phi1))

# Colour per combline by index in MAIN_ORDERS
order_color = {int(p): _COLORS[i % len(_COLORS)] for i, p in enumerate(main_orders)}

# ── main figure (β1 spectrum) ─────────────────────────────────────────────────
fig_main, ax_main = _make_fig_ax(MAIN_AXES_W_MM, MAIN_AXES_H_MM)

colors_main = [order_color[int(p)] for p in main_orders]
_draw_signed_stems(ax_main, main_orders, beta1_signed, colors_main)
ax_main.set_xlim(float(main_orders[0]) - 0.8, float(main_orders[-1]) + 0.8)
_sym_ylim(ax_main, beta1_signed)
_strip_axes(ax_main)

out_main = Path(__file__).parent / 'dual_tone_hierarchy_main.png'
fig_main.savefig(out_main, dpi=200, bbox_inches='tight')
print(f'Saved: {out_main}')

# ── mini figures (β2 sidebands per ZOOM_COMBLINE) ─────────────────────────────
# Compute all sideband amplitudes first so y-limits can be shared across figures
sb_data = {}
for p in ZOOM_COMBLINES:
    if int(p) not in order_color:
        continue
    # Re[J_p(β1)·e^{ipφ1} · J_k(β2)·e^{ikφ2}] for each k
    carrier_complex = jv(p, BETA1) * np.exp(1j * p * phi1)
    sb_data[p] = np.real(
        carrier_complex * jv(sideband_k, BETA2) * np.exp(1j * sideband_k * phi2)
    )

y_max_mini = max(float(np.abs(v).max()) for v in sb_data.values()) * 1.15

sb_x_lo = float(sb_xs_rel[0])  - 0.8
sb_x_hi = float(sb_xs_rel[-1]) + 0.8

for p, sb_amps in sb_data.items():
    c = order_color[int(p)]
    fig_m, ax_m = _make_fig_ax(MINI_AXES_W_MM, MINI_AXES_H_MM)
    _draw_signed_stems(ax_m, sb_xs_rel, sb_amps, c)
    ax_m.set_xlim(sb_x_lo, sb_x_hi)
    ax_m.set_ylim(-y_max_mini, y_max_mini)
    _strip_axes(ax_m)

    out_m = Path(__file__).parent / f'dual_tone_hierarchy_p{p:+d}.png'
    fig_m.savefig(out_m, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_m}')

plt.show()
