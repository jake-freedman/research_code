"""
Hierarchical dual-tone spectrum — one figure per panel, signed amplitudes.

Stem height encodes power in dBc (re. unit CW carrier); direction (up/down)
encodes the sign of the real part of the complex amplitude.

Figure 1 (CW):     Input CW carrier: single stem at order 0.
Figure 2 (main):   β1-only spectrum: J_p(β1)·exp(i·p·φ1) in dB.
Figures 3+:        For each combline p in ZOOM_COMBLINES — the comb produced
                   by β2/φ2 modulation with combline p as carrier.  Shown at
                   every position in MAIN_ORDERS; positions not reached by a
                   β2 sideband get a hollow circle at zero.
Figure N (total):  Total dual-tone output: summed A_n at each bin n.
All figures share the same physical size and dB axis limits.
No axes outline, tick marks, labels, or y-axis.
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
BETA1    = 1.94
BETA2    = 0.88
PHI1_DEG = 0.0
PHI2_DEG = 180.0

# Harmonic orders shown in every figure
MAIN_ORDERS = list(range(-3, 4))

# Which comblines to expand into separate β2 sideband figures
ZOOM_COMBLINES = [-1, 1]

# β2 sideband orders k available; sidebands land at p + 2k
SIDEBAND_K = list(range(-3, 4))

# Opacity for k ≠ 0 sidebands (non-zoom combline) in mini figures
SEMI_ALPHA = 0.5

# ── combline colors ───────────────────────────────────────────────────────────
# Map harmonic order → color used in every figure.
# None = auto-assign from _COLORS in sorted order of MAIN_ORDERS.
COMBLINE_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
# ─────────────────────────────────────────────────────────────────────────────

# ── dB scale ──────────────────────────────────────────────────────────────────
# Reference power for dBc (1.0 = unit CW carrier).
REF_POWER = 1.0
# Amplitudes below this dBc are shown with zero visual height.
FLOOR_DBC = -17.0
# ─────────────────────────────────────────────────────────────────────────────

# ── layout (mm) ───────────────────────────────────────────────────────────────
AXES_W_MM = 25.0
AXES_H_MM = 20.0

LEFT_MM   =  5.0
RIGHT_MM  =  5.0
TOP_MM    =  5.0
BOTTOM_MM =  5.0

stem_linewidth  = 2
markersize      = 5.0
zero_markersize = 5.0
baseline_color  = '#000000'
baseline_lw     = 2

# ── y-axis display ────────────────────────────────────────────────────────────
# When True, expose the left spine with tick marks labeled in dBc.
SHOW_Y_AXIS = False
Y_AXIS_TICK_STEP_DBC = 10  # dBc spacing between tick marks (e.g. 5 or 10)

# Top of every axes in dBc.  0 = CW carrier exactly at the edge.
# Increase for headroom above the carrier (e.g. 5 → 5 dB gap at the top).
YMAX_DBC = 3

# Bottom of the labeled y-axis / spine in dBc (≥ FLOOR_DBC).
# Equal to FLOOR_DBC → spine starts at the x-axis (default).
# Higher (e.g. −20) → spine and ticks only span −20 dBc to YMAX_DBC.
YMIN_DBC = FLOOR_DBC
# ─────────────────────────────────────────────────────────────────────────────

# ── amplitude scale ───────────────────────────────────────────────────────────
# 'dB'     → signed dBc heights; FLOOR_DBC / YMAX_DBC / YMIN_DBC control range
# 'linear' → signed |amplitude|² / REF_POWER (0 – 1 for unit CW input)
SCALE = 'dB'
YMAX_LINEAR = 1.0             # top of y-axis in linear power (1.0 = CW carrier)
Y_AXIS_TICK_STEP_LINEAR = 0.2 # linear tick spacing
# ─────────────────────────────────────────────────────────────────────────────

# ── plot style ────────────────────────────────────────────────────────────────
# 'stem'      → lollipop stems + balls (default)
# 'lorentzian'→ each combline rendered as a narrow Lorentzian peak; the
#               baseline is still drawn and zero circles still mark absent
#               positions in mini figures.
PLOT_STYLE = 'stem'
# FWHM of each Lorentzian in combline-number units (e.g. 0.1 = 1/10 of spacing)
LORENTZIAN_FWHM = 0.1
# x-range plotted around each peak in combline units (should be >> FWHM)
LORENTZIAN_PLOT_WIDTH = 0.6
LORENTZIAN_NPTS = 500         # points per individual peak
# ─────────────────────────────────────────────────────────────────────────────

# ── SVG export ────────────────────────────────────────────────────────────────
SVG_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\hierarchy"
# ─────────────────────────────────────────────────────────────────────────────


def _strip_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if SHOW_Y_AXIS:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['left'].set_color(baseline_color)
        if SCALE == 'dB':
            y_spine_lo = YMIN_DBC - FLOOR_DBC
            ax.spines['left'].set_bounds(y_spine_lo, y_max)
            dbc_vals   = np.arange(YMIN_DBC, YMAX_DBC + 0.1, Y_AXIS_TICK_STEP_DBC)
            y_tick_pos = dbc_vals - FLOOR_DBC
            ax.set_yticks(y_tick_pos)
            ax.set_yticklabels([f'{int(d)}' for d in dbc_vals])
        else:
            ax.spines['left'].set_bounds(0, y_max)
            step      = Y_AXIS_TICK_STEP_LINEAR
            lin_vals  = np.arange(0, YMAX_LINEAR + 0.5 * step, step)
            ax.set_yticks(lin_vals)
            ax.set_yticklabels([f'{v:.2g}' for v in lin_vals])
        ax.tick_params(left=True, labelleft=True, labelsize=6)


def _make_fig_ax():
    fig_w = LEFT_MM + AXES_W_MM + RIGHT_MM
    fig_h = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = LEFT_MM   / fig_w,
        right  = 1 - RIGHT_MM  / fig_w,
        bottom = BOTTOM_MM / fig_h,
        top    = 1 - TOP_MM   / fig_h,
    )
    return fig, ax


def _to_signed_linear(complex_amps):
    """Signed linear power heights: direction × |amp|² / REF_POWER."""
    amps = np.asarray(complex_amps, dtype=complex)
    power = np.abs(amps) ** 2 / REF_POWER
    direction = np.where(np.real(amps) >= 0, 1.0, -1.0)
    return direction * power


def _to_signed_dB(complex_amps):
    """
    Signed dBc heights.
      magnitude = max(0, 10·log10(|amp|²/REF_POWER) − FLOOR_DBC)
      direction = +1 if Re[amp] ≥ 0, −1 otherwise
    """
    amps = np.asarray(complex_amps, dtype=complex)
    power = np.abs(amps) ** 2
    with np.errstate(divide='ignore'):
        power_dBc = np.where(power > 0,
                             10.0 * np.log10(power / REF_POWER),
                             FLOOR_DBC - 1.0)
    visual_mag = np.maximum(power_dBc - FLOOR_DBC, 0.0)
    direction = np.where(np.real(amps) >= 0, 1.0, -1.0)
    return direction * visual_mag


def _to_signed_heights(complex_amps):
    return _to_signed_dB(complex_amps) if SCALE == 'dB' else _to_signed_linear(complex_amps)


def _lorentzian(x, x0, height, fwhm):
    """Signed Lorentzian centred at x0 with peak value height."""
    gamma = fwhm / 2.0
    return height * gamma**2 / ((x - x0)**2 + gamma**2)


def _draw_signed_stems(ax, xs, heights, colors, alphas=None, baseline=True):
    """Draw stems + balls (stem mode) or Lorentzian curves (lorentzian mode)."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    if alphas is None:
        alphas = [1.0] * len(xs)
    if PLOT_STYLE == 'lorentzian' and len(xs) > 0:
        hw = LORENTZIAN_PLOT_WIDTH / 2.0
        for xi, yi, c, a in zip(xs, heights, colors, alphas):
            x_local = np.linspace(xi - hw, xi + hw, LORENTZIAN_NPTS)
            y_curve = _lorentzian(x_local, xi, yi, LORENTZIAN_FWHM)
            ax.plot(x_local, y_curve, color=c, linewidth=stem_linewidth,
                    solid_capstyle='round', zorder=2, alpha=a)
    else:
        for xi, yi, c, a in zip(xs, heights, colors, alphas):
            ax.plot([xi, xi], [0.0, yi], color=c, linewidth=stem_linewidth,
                    solid_capstyle='round', zorder=2, alpha=a)
            ax.plot(xi, yi, 'o', color=c, markersize=markersize,
                    markeredgewidth=0, zorder=3, alpha=a)
    if baseline:
        ax.axhline(0.0, color=baseline_color, linewidth=baseline_lw, zorder=1)


def _draw_zero_circles(ax, xs, colors, alpha=1.0):
    """Filled circles at y = 0 for positions where no sideband lands."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    for xi, c in zip(xs, colors):
        ax.plot(xi, 0, 'o', color=c, markersize=zero_markersize,
                markeredgewidth=0, zorder=3, alpha=alpha)


def _save(fig, stem):
    png_path = Path(__file__).parent / f'{stem}.png'
    fig.savefig(png_path, dpi=200)
    print(f'Saved: {png_path}')
    if SVG_FOLDER:
        svg_dir = Path(SVG_FOLDER)
        svg_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_dir / f'{stem}.svg'
        fig.savefig(svg_path, format='svg')
        print(f'Saved: {svg_path}')


# ── compute ───────────────────────────────────────────────────────────────────
phi1 = np.deg2rad(PHI1_DEG)
phi2 = np.deg2rad(PHI2_DEG)

main_orders    = np.array(sorted(MAIN_ORDERS))
sideband_k     = np.array(sorted(SIDEBAND_K))
sideband_k_set = set(int(k) for k in sideband_k)

# Complex β1 amplitudes and their dB heights
beta1_complex = jv(main_orders, BETA1) * np.exp(1j * main_orders * phi1)
beta1_heights = _to_signed_heights(beta1_complex)

# Colour per combline — user dict takes priority; fall back to auto-assignment
if COMBLINE_COLORS is not None:
    order_color = {
        int(p): COMBLINE_COLORS.get(int(p), _COLORS[i % len(_COLORS)])
        for i, p in enumerate(main_orders)
    }
else:
    order_color = {int(p): _COLORS[i % len(_COLORS)] for i, p in enumerate(main_orders)}

main_colors = [order_color[int(p)] for p in main_orders]

# Total dual-tone output: A_n = Σ_k J_{n-2k}(β1)·e^{i(n-2k)φ1}·J_k(β2)·e^{ikφ2}
_k_trunc = max(int(np.ceil(max(BETA1, BETA2))) + 15, 20)
_k_sum = np.arange(-_k_trunc, _k_trunc + 1)
total_complex = np.array([
    np.sum(
        jv(n - 2 * _k_sum, BETA1) * np.exp(1j * (n - 2 * _k_sum) * phi1) *
        jv(_k_sum, BETA2) * np.exp(1j * _k_sum * phi2)
    )
    for n in main_orders
])
total_heights = _to_signed_heights(total_complex)

y_max = (YMAX_DBC - FLOOR_DBC) if SCALE == 'dB' else YMAX_LINEAR
x_lo  = float(main_orders[0]) - 0.8
x_hi  = float(main_orders[-1]) + 0.8


# ── Figure: CW input ──────────────────────────────────────────────────────────
fig_cw, ax_cw = _make_fig_ax()
ax_cw.set_xlim(x_lo, x_hi)
ax_cw.set_ylim(-y_max, y_max)
_draw_signed_stems(ax_cw, [0], _to_signed_heights(np.array([1.0 + 0j])),
                   [order_color.get(0, _COLORS[0])])
_strip_axes(ax_cw)
_save(fig_cw, 'dual_tone_hierarchy_cw')


# ── Figure: β1 spectrum (main) ────────────────────────────────────────────────
fig_main, ax_main = _make_fig_ax()
ax_main.set_xlim(x_lo, x_hi)
ax_main.set_ylim(-y_max, y_max)
_draw_signed_stems(ax_main, main_orders, beta1_heights, main_colors)
_strip_axes(ax_main)
_save(fig_main, 'dual_tone_hierarchy_main')


# ── Figures: β2 expansion of each ZOOM_COMBLINE ───────────────────────────────
for p in ZOOM_COMBLINES:
    if int(p) not in order_color:
        continue

    fig_m, ax_m = _make_fig_ax()

    # For each position n in MAIN_ORDERS, check whether a β2 sideband lands there:
    # n = p + 2k  →  k = (n−p)/2; valid only when (n−p) is even and k ∈ SIDEBAND_K.
    mini_complex = np.zeros(len(main_orders), dtype=complex)
    mini_k_vals  = np.full(len(main_orders), np.nan)   # nan = no sideband
    for i, n in enumerate(main_orders):
        diff = int(n) - int(p)
        if diff % 2 == 0:
            k = diff // 2
            if k in sideband_k_set:
                mini_complex[i] = (jv(p, BETA1) * np.exp(1j * p * phi1) *
                                   jv(k, BETA2) * np.exp(1j * k * phi2))
                mini_k_vals[i] = k

    mini_heights = _to_signed_heights(mini_complex)
    zero_mask = np.abs(mini_complex) < 1e-15

    # k=0 is the zoom combline itself → opaque; k≠0 sidebands → semi-transparent
    alphas = np.where(mini_k_vals == 0, 1.0, SEMI_ALPHA)

    # Stems where sidebands exist
    nz_xs      = main_orders[~zero_mask].tolist()
    nz_heights = mini_heights[~zero_mask].tolist()
    nz_colors  = [order_color.get(int(n), _COLORS[0]) for n in main_orders[~zero_mask]]
    nz_alphas  = alphas[~zero_mask].tolist()
    ax_m.set_xlim(x_lo, x_hi)
    ax_m.set_ylim(-y_max, y_max)
    _draw_signed_stems(ax_m, nz_xs, nz_heights, nz_colors, alphas=nz_alphas, baseline=True)

    # Filled circles at zero — semi-transparent to match surrounding sidebands
    z_xs     = main_orders[zero_mask].tolist()
    z_colors = [order_color.get(int(n), _COLORS[0]) for n in main_orders[zero_mask]]
    _draw_zero_circles(ax_m, z_xs, z_colors, alpha=SEMI_ALPHA)

    _strip_axes(ax_m)
    _save(fig_m, f'dual_tone_hierarchy_p{p:+d}')


# ── Figure: total dual-tone output ────────────────────────────────────────────
fig_tot, ax_tot = _make_fig_ax()
ax_tot.set_xlim(x_lo, x_hi)
ax_tot.set_ylim(-y_max, y_max)
_draw_signed_stems(ax_tot, main_orders, total_heights, main_colors)
_strip_axes(ax_tot)
_save(fig_tot, 'dual_tone_hierarchy_total')


plt.show()
