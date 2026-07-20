"""
comb_displayer.py

Display one dual-tone frequency comb, either:
  - a pure theoretical prediction (Jacobi-Anger amplitudes from user-supplied
    BETA1, BETA2, PHI1_DEG, PHI2_DEG), or
  - a measured comb pulled from recorded data: a single
    dual_tone_step_spectrum.py-style sweep file (.npz), or a
    bnc_2d_power_phase_script.py 2D power x phase grid folder.

Measured sources are loaded via comb_finder.load_grid(), which already
unifies both file layouts into grid_data[(i, j)] entries (a single sweep
file is always internally grid point (0, 0)). Point DATA_PATH at either a
single .npz or a grid folder, then pick GRID_I/GRID_J (only meaningful for a
real grid folder) and STEP_INDEX (the phase/sweep step within that grid
point) to select the exact comb to display.

The comb itself is displayed the same way, with the same controllable
parameters, as plot_dual_tone_spectrum.py (stems + balls, _COLORS cycling,
DISPLAY_MODE percent/dB, FLOOR_dBc/CEILING_dBc, same graphics-style knobs).
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from comb_finder import load_grid
from path_utils import local_path
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
)

# Same harmonic-order color scheme as comb_finder.py: each harmonic order
# gets a fixed color; orders outside this table fall back to _EXTRA_COLORS,
# cycled in the order the harmonics appear.
_HARMONIC_COLORS = {
    -4: '#583d2b',
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
     4: '#754cae'
}
_EXTRA_COLORS = [BLUE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2, DARKGRAY2, BEIGE2]

# ── configuration ─────────────────────────────────────────────────────────────

# Data source:
#   'theory'   -> pure theoretical comb from BETA1/BETA2/PHI1_DEG/PHI2_DEG below
#   'measured' -> a recorded comb from DATA_PATH (single .npz or 2D grid folder)
#   'combined' -> the measured comb from DATA_PATH, with the theoretical
#                 prediction (from BETA1/BETA2/PHI1_DEG/PHI2_DEG) overlaid:
#                 a black horizontal line at each bar's value (PLOT_STYLE =
#                 'bar'), or a black open circle (PLOT_STYLE = 'stem').
SOURCE = 'measured'

# ── theory mode (SOURCE = 'theory' or 'combined') ─────────────────────────────
BETA1    = 2.44     # ch1 modulation depth (rad), drive at f
BETA2    = 1.10     # ch2 modulation depth (rad), drive at 2f
PHI1_DEG = 0.0      # ch1 phase (deg)
PHI2_DEG = 180.0    # ch2 phase (deg)

# ── measured mode (SOURCE = 'measured') ───────────────────────────────────────
# A single dual_tone_step_spectrum.py-style .npz, or a bnc_2d_power_phase_script.py
# grid folder (containing grid_meta.npz + grid_ii_jj.npz files) -- load_grid()
# tells these apart automatically (file vs. directory).
DATA_PATH = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\comb_finding\2d_power_phase_2026-07-19-15-36-40"

# Which grid point to read. Only meaningful for a 2D grid folder; a single
# sweep file is always internally treated as the sole grid point (0, 0).
GRID_I = 0
GRID_J = 4

# Which phase/sweep step within that grid point to display as the comb.
STEP_INDEX = 11

# Harmonic orders to plot and print. Both the spectrum and the console table
# show exactly these orders (measured mode: orders not recorded are skipped).
PRINT_ORDERS = list(range(-1, 2))

# Power display mode:
#   'percent' -> |A_p|^2 as % of total optical power
#   'dB'      -> 10*log10(|A_p|^2) in dBc  (0 dBc = all power in one line)
DISPLAY_MODE = 'dB'

# Plot style: 'stem' (ball-and-stick) or 'bar' (filled bar with a vertical
# opacity gradient from the baseline to each harmonic's value).
PLOT_STYLE = 'bar'

SHOW_GRID   = False
SHOW_LEGEND = False

# If False, the x tick marks are hidden (zero length/width) while the ticks
# themselves -- and their labels -- still exist at the same positions.
SHOW_XTICKS = False

# dB mode only: floor/ceiling of the y-axis in dBc. None = auto (data
# min/max +- 3 dB margin); CEILING_dBc's usual value is 0 (0 dBc = all
# power in one line, the physical ceiling for this normalization).
FLOOR_dBc = -50.0 #37
CEILING_dBc = 2.0 # 4
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm   = 45.0
axes_height_mm  = 57.0
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
markersize      =   8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── opacity gradient (both PLOT_STYLE = 'bar' and 'stem') ─────────────────────
# Each harmonic's bar fill / stem line goes from GRADIENT_ALPHA_MIN at the
# baseline to GRADIENT_ALPHA_MAX at its value, rather than a flat opacity.
GRADIENT_ALPHA_MIN = 0.15   # opacity at the baseline (bottom)
GRADIENT_ALPHA_MAX = 1.00   # opacity at the harmonic's value (top)
# Shape of the ramp from GRADIENT_ALPHA_MIN to GRADIENT_ALPHA_MAX: opacity
# follows t**GRADIENT_ORDER, t going 0 (baseline) to 1 (value). 1 = linear;
# >1 = stays near ALPHA_MIN longer, then ramps up faster near the top;
# <1 (e.g. 0.5) = ramps up faster near the bottom, then levels off near ALPHA_MAX.
GRADIENT_ORDER = 1.0
GRADIENT_RESOLUTION = 200   # vertical samples in the opacity gradient
# ─────────────────────────────────────────────────────────────────────────────

# ── bar style (PLOT_STYLE = 'bar') ────────────────────────────────────────────
# Each harmonic drawn as a bar from the baseline to its value, filled with the
# opacity gradient above rather than a flat fill.
BAR_WIDTH      = 0.5
BAR_LINESTYLE  = '-'
BAR_LINEWIDTH  = 1.0
BAR_EDGE_COLOR = '#000000'   # None = match the harmonic's color
BAR_EDGE_ALPHA = 1.0
BAR_FACE_COLOR = None   # None = match the harmonic's color

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

# ── stem style (PLOT_STYLE = 'stem') ──────────────────────────────────────────
# The vertical stem AND the ball at its tip are filled with one continuous
# opacity gradient (the same ramp as above): alpha_min at the baseline, all
# the way up through the ball, to alpha_max at the ball's far edge. The stem
# itself stops where it meets the ball's near edge, rather than running to
# the ball's center.
STEM_ZORDER = 2
BALL_ZORDER = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── theory overlay (SOURCE = 'combined') ──────────────────────────────────────
# The theoretical prediction (BETA1/BETA2/PHI1_DEG/PHI2_DEG above) drawn on
# top of the measured comb: a black horizontal line at each bar's value
# (PLOT_STYLE = 'bar'), or a black open circle (PLOT_STYLE = 'stem').
THEORY_COLOR  = '#000000'
THEORY_ALPHA  = 1.0
THEORY_ZORDER = 10

# PLOT_STYLE = 'bar': horizontal line width (data units) and linewidth (points).
THEORY_LINE_WIDTH     = None   # None = match BAR_WIDTH
THEORY_LINE_LINEWIDTH = 2.0

# PLOT_STYLE = 'stem': open-circle marker.
THEORY_MARKER           = 'o'
THEORY_MARKERSIZE       = 10
THEORY_MARKER_EDGEWIDTH = 1.5
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels, the title, and the legend (from the
# same figure that's shown and PNG-saved below), and additionally saves an
# SVG to SAVE_FOLDER.
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUBLICATION_SVG_NAME = 'comb_displayer_high_supp.svg'
# ─────────────────────────────────────────────────────────────────────────────


def dual_tone_amplitudes(
    beta1: float, beta2: float,
    phi1: float, phi2: float,
    orders: list[int], k_trunc: int,
) -> dict[int, complex]:
    """
    Complex amplitude A_p at harmonic p*f for a dual-tone drive.

        A_p = Sum_k  J_{p-2k}(beta1) * J_k(beta2) * exp(i[(p-2k)*phi1 + k*phi2])

    ch1 drives at f  with depth beta1 and phase phi1.
    ch2 drives at 2f with depth beta2 and phase phi2.
    """
    k = np.arange(-k_trunc, k_trunc + 1)
    return {
        p: complex(np.sum(
            jv(p - 2 * k, beta1) * jv(k, beta2)
            * np.exp(1j * ((p - 2 * k) * phi1 + k * phi2))
        ))
        for p in orders
    }


def _load_theory_powers(orders: list[int]):
    """Returns ({order: |A_p|^2}, header_string). Always has a value for
    every requested order (Jacobi-Anger is defined everywhere)."""
    k_trunc = int(2 * max(BETA1, BETA2)) + 20
    amps = dual_tone_amplitudes(
        BETA1, BETA2, np.deg2rad(PHI1_DEG), np.deg2rad(PHI2_DEG), orders, k_trunc)
    powers_lin = {p: abs(A) ** 2 for p, A in amps.items()}
    header = (f"Theory:  beta1={BETA1},  beta2={BETA2},  "
              f"phi1={PHI1_DEG:.1f} deg,  phi2={PHI2_DEG:.1f} deg")
    return powers_lin, header


def _load_measured_powers(orders: list[int]):
    """
    Returns ({order: |A_p|^2}, header_string, (i, j) key actually used) for
    STEP_INDEX within DATA_PATH. Orders in `orders` that aren't recorded are
    skipped (with a console warning), same as dual_tone_step_spectrum.py.

    load_grid() always maps a single sweep file to grid point (0, 0), so
    GRID_I/GRID_J are ignored for that case (likely stale settings left over
    from a previous grid-folder DATA_PATH) rather than erroring on an index
    that can't apply here.
    """
    resolved_path = local_path(DATA_PATH)
    meta, grid_data = load_grid(resolved_path)
    key = (0, 0) if os.path.isfile(resolved_path) else (GRID_I, GRID_J)

    if key not in grid_data:
        raise KeyError(
            f"Grid point {key} not found in {DATA_PATH!r}. "
            f"Available: {sorted(grid_data.keys())}"
        )
    d = grid_data[key]
    harmonics   = d['harmonics'].astype(int)
    spectra     = d['spectra']                  # (P, N, K)
    cal_spectra = d.get('cal_spectra', None)     # (P, K) or None
    phases      = d.get('ch2_phases_deg', None)  # (P,) or None

    P = spectra.shape[0]
    if not (0 <= STEP_INDEX < P):
        raise ValueError(
            f"STEP_INDEX={STEP_INDEX} out of range (grid point {key} has {P} step(s))."
        )
    if cal_spectra is None:
        raise RuntimeError(
            f"Grid point {key} has no cal_spectra; DISPLAY_MODE={DISPLAY_MODE!r} "
            "requires per-step calibration."
        )
    cal_peak_dbm = float(cal_spectra[STEP_INDEX].max())

    powers_lin = {}
    for p in orders:
        idx = np.where(harmonics == p)[0]
        if len(idx) == 0:
            print(f"  Warning: order {p} not recorded; skipping.")
            continue
        peak_dbm = float(spectra[STEP_INDEX, int(idx[0])].max())
        powers_lin[p] = 10.0 ** ((peak_dbm - cal_peak_dbm) / 10.0)

    ch1_pwr = float(meta['ch1_powers_dbm'][key[0]])
    ch2_pwr = float(meta['ch2_powers_dbm'][key[1]])
    phase_str = f", ch2 phase={float(phases[STEP_INDEX]):.1f} deg" if phases is not None else ""
    header = (f"Measured: {DATA_PATH}\n"
              f"  Grid point {key} (ch1={ch1_pwr:+.1f} dBm, ch2={ch2_pwr:+.1f} dBm), "
              f"step {STEP_INDEX}/{P - 1}{phase_str}")
    return powers_lin, header, key


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


def _draw_gradient_stem(ax, x_center: float, y_base: float, y_stop: float, y_ref_top: float,
                         color, linewidth: float, alpha_min: float, alpha_max: float,
                         order: float, n_segments: int, zorder: float):
    """
    Draw the vertical stem from y_base to y_stop (the ball's near edge -- not
    all the way to the harmonic's value) as many short line segments, opacity
    ramping from alpha_min at y_base toward alpha_max at y_ref_top (the
    ball's far edge). y_ref_top, not y_stop, anchors the ramp's t=1 point, so
    the stem's opacity continues seamlessly into the ball drawn above it
    (see _draw_gradient_ball). The ramp follows t**order (t: 0 at y_base, 1
    at y_ref_top).
    """
    if y_ref_top == y_base:
        ax.plot([x_center, x_center], [y_base, y_stop], color=color, linewidth=linewidth,
                alpha=alpha_max, solid_capstyle='butt', zorder=zorder)
        return

    y_edges = np.linspace(y_base, y_stop, n_segments + 1)
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    t = np.clip((y_mid - y_base) / (y_ref_top - y_base), 0.0, 1.0)
    alphas = alpha_min + (t ** order) * (alpha_max - alpha_min)

    rgba = np.tile(mcolors.to_rgba(color), (n_segments, 1))
    rgba[:, 3] = alphas
    segments = np.stack(
        [np.column_stack([np.full(n_segments, x_center), y_edges[:-1]]),
         np.column_stack([np.full(n_segments, x_center), y_edges[1:]])],
        axis=1,
    )
    lc = LineCollection(segments, colors=rgba, linewidths=linewidth,
                         capstyle='butt', zorder=zorder)
    ax.add_collection(lc)


def _points_to_data_scale(ax):
    """
    Return (data units per point in x, data units per point in y) at the
    axes' current limits/figure size -- constant across a linear axes, used
    to size a points-based marker radius in data units for _draw_gradient_ball.
    """
    dpi = ax.figure.dpi
    pts_to_px = dpi / 72.0
    inv = ax.transData.inverted()
    p0_disp = ax.transData.transform((0.0, 0.0))
    px_disp = p0_disp + np.array([pts_to_px, 0.0])
    py_disp = p0_disp + np.array([0.0, pts_to_px])
    p0_data = inv.transform(p0_disp)
    px_data = inv.transform(px_disp)
    py_data = inv.transform(py_disp)
    return abs(px_data[0] - p0_data[0]), abs(py_data[1] - p0_data[1])


def _draw_gradient_ball(ax, x_center: float, y_center: float, radius_x: float, radius_y: float,
                         y_base: float, y_far: float, color, alpha_min: float, alpha_max: float,
                         order: float, n_pts: int, zorder: float):
    """
    Fill a circular marker centered at (x_center, y_center) -- an ellipse in
    data units, but a true circle on screen since radius_x/radius_y already
    account for the axes' pixel scale -- with the same vertical opacity
    gradient as the stem beneath it: alpha_min at y_base, alpha_max at y_far
    (the point of the circle farthest from the baseline), continuing the
    ramp seamlessly from where the stem left off.
    """
    xs = np.linspace(-radius_x, radius_x, n_pts)
    ys = np.linspace(-radius_y, radius_y, n_pts)
    xx, yy = np.meshgrid(xs, ys)
    inside = (xx / radius_x) ** 2 + (yy / radius_y) ** 2 <= 1.0

    y_abs = y_center + yy
    if y_far == y_base:
        t = np.ones_like(y_abs)
    else:
        t = np.clip((y_abs - y_base) / (y_far - y_base), 0.0, 1.0)
    alpha = alpha_min + (t ** order) * (alpha_max - alpha_min)
    alpha = np.where(inside, alpha, 0.0)

    rgba = np.zeros((n_pts, n_pts, 4))
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = alpha

    ax.imshow(rgba, extent=(x_center - radius_x, x_center + radius_x,
                            y_center - radius_y, y_center + radius_y),
              origin='lower', aspect='auto', interpolation='bilinear', zorder=zorder)


def _build_figure(plot_orders, y, y_baseline, y_ceiling, ylabel, title, show_labels: bool,
                   y_theory=None):
    """
    Build one comb figure (bar or stem, per PLOT_STYLE). show_labels=True
    gives the normal, fully-labeled figure; False gives the stripped
    publication variant -- built as its own separate figure rather than a
    mutation of the labeled one, so both can be shown side by side.

    y_theory (SOURCE = 'combined' only): the theoretical value at each of
    plot_orders, overlaid as a black horizontal line at the bar's value
    (PLOT_STYLE = 'bar') or a black open circle (PLOT_STYLE = 'stem').
    """
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    # Limits are set before any drawing: the gradient ball (stem mode) needs
    # the axes' final data-per-point scale to size itself correctly.
    ax.set_xlim(plot_orders[0] - 0.8, plot_orders[-1] + 0.8)
    if DISPLAY_MODE == 'percent':
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=y_baseline, top=y_ceiling)
    fig.canvas.draw()

    _extra_iter = iter(_EXTRA_COLORS)
    harmonic_colors = {int(n): _HARMONIC_COLORS.get(int(n), next(_extra_iter, '#000000'))
                       for n in plot_orders}

    if PLOT_STYLE == 'bar':
        for xi, yi in zip(plot_orders, y):
            n = int(xi)
            color = harmonic_colors[n]
            edge_color = BAR_EDGE_COLOR if BAR_EDGE_COLOR is not None else color
            face_color = BAR_FACE_COLOR if BAR_FACE_COLOR is not None else color
            marker_face = BAR_MARKER_FACECOLOR if BAR_MARKER_FACECOLOR is not None else color
            marker_edge = BAR_MARKER_EDGECOLOR if BAR_MARKER_EDGECOLOR is not None else color
            label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n:+d}'

            _draw_gradient_bar(ax, xi, BAR_WIDTH, y_baseline, yi, face_color,
                                GRADIENT_ALPHA_MIN, GRADIENT_ALPHA_MAX, GRADIENT_ORDER,
                                GRADIENT_RESOLUTION, BAR_ZORDER)

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

        if y_theory is not None:
            half_w = (THEORY_LINE_WIDTH if THEORY_LINE_WIDTH is not None else BAR_WIDTH) / 2.0
            for xi, yt in zip(plot_orders, y_theory):
                ax.plot([xi - half_w, xi + half_w], [yt, yt],
                        color=THEORY_COLOR, linewidth=THEORY_LINE_LINEWIDTH,
                        alpha=THEORY_ALPHA, solid_capstyle='butt', zorder=THEORY_ZORDER)
    else:
        # Stems (opacity gradient, baseline to value) and balls, one color per combline.
        # The stem stops at the ball's near edge; the gradient continues
        # seamlessly through the ball, up to its far edge.
        dx_per_pt, dy_per_pt = _points_to_data_scale(ax)
        radius_x = (markersize / 2.0) * dx_per_pt
        radius_y = (markersize / 2.0) * dy_per_pt

        for xi, yi in zip(plot_orders, y):
            n = int(xi)
            c = harmonic_colors[n]
            label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n:+d}'

            ax.plot([], [], color=c, linewidth=stem_linewidth, label=label)  # legend proxy

            direction = 1.0 if yi >= y_baseline else -1.0
            y_near = yi - direction * radius_y   # where the stem stops (ball's near edge)
            y_far  = yi + direction * radius_y   # ball's far edge (gradient's top)

            _draw_gradient_stem(ax, xi, y_baseline, y_near, y_far, c, stem_linewidth,
                                 GRADIENT_ALPHA_MIN, GRADIENT_ALPHA_MAX, GRADIENT_ORDER,
                                 GRADIENT_RESOLUTION, STEM_ZORDER)

            _draw_gradient_ball(ax, xi, yi, radius_x, radius_y, y_baseline, y_far, c,
                                 GRADIENT_ALPHA_MIN, GRADIENT_ALPHA_MAX, GRADIENT_ORDER,
                                 GRADIENT_RESOLUTION, BALL_ZORDER)

        if y_theory is not None:
            ax.plot(plot_orders, y_theory, linestyle='none',
                    marker=THEORY_MARKER, markersize=THEORY_MARKERSIZE,
                    markerfacecolor='none', markeredgecolor=THEORY_COLOR,
                    markeredgewidth=THEORY_MARKER_EDGEWIDTH, alpha=THEORY_ALPHA,
                    zorder=THEORY_ZORDER)

    # Baseline
    ax.axhline(y_baseline, color='#333333', linewidth=0.8, linestyle='-', zorder=1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    if not SHOW_XTICKS:
        ax.tick_params(axis='x', width=0, length=0)

    if show_labels:
        ax.set_xlabel(r'Harmonic order $p$', fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.set_title(title, fontsize=axis_label_fontsize)
        if SHOW_GRID:
            ax.grid()
        if SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def main():
    orders = sorted(set(PRINT_ORDERS))

    used_key = None
    theory_powers_lin = None
    if SOURCE == 'theory':
        powers_lin, header = _load_theory_powers(orders)
    elif SOURCE == 'measured':
        powers_lin, header, used_key = _load_measured_powers(orders)
    elif SOURCE == 'combined':
        powers_lin, header, used_key = _load_measured_powers(orders)
        theory_powers_lin, theory_header = _load_theory_powers(orders)
        header = header + "\n" + theory_header
    else:
        raise ValueError(f"Unknown SOURCE: {SOURCE!r}")

    valid_orders = sorted(powers_lin)
    if not valid_orders:
        raise RuntimeError("No orders available to plot.")
    plot_orders = np.array(valid_orders)
    plot_powers = np.array([powers_lin[p] for p in valid_orders])

    y_ceiling = None
    if DISPLAY_MODE == 'percent':
        y          = plot_powers * 100.0
        y_baseline = 0.0
        ylabel     = r'$|A_p|^2$ [% of total power]'
    else:
        y          = 10.0 * np.log10(np.maximum(plot_powers, 1e-30))
        y_baseline = FLOOR_dBc if FLOOR_dBc is not None else y.min() - 3.0
        y_ceiling  = CEILING_dBc if CEILING_dBc is not None else y.max() + 3.0
        ylabel     = r'$|A_p|^2$ [dBc]'

    y_theory = None
    if theory_powers_lin is not None:
        theory_powers = np.array([theory_powers_lin[p] for p in valid_orders])
        y_theory = (theory_powers * 100.0 if DISPLAY_MODE == 'percent'
                    else 10.0 * np.log10(np.maximum(theory_powers, 1e-30)))

    # ── console output ────────────────────────────────────────────────────────
    unit = '%' if DISPLAY_MODE == 'percent' else 'dBc'
    print(header)
    print(f"  {'order':>5}   {'power':>12}")
    print(f"  {'─'*5}   {'─'*12}")
    for p in valid_orders:
        pwr = (powers_lin[p] * 100.0
               if DISPLAY_MODE == 'percent'
               else 10.0 * np.log10(max(powers_lin[p], 1e-30)))
        print(f"  p={p:+d}   {pwr:>10.4f} {unit}")

    total = sum(powers_lin.values())
    print(f"\n  Total power in plotted orders: {total * 100:.2f}%")

    # ── figure ────────────────────────────────────────────────────────────────
    if SOURCE == 'theory':
        title = (rf'$\beta_1={BETA1}$,  $\beta_2={BETA2}$,'
                 rf'  $\phi_1={PHI1_DEG:.0f}°$,  $\phi_2={PHI2_DEG:.0f}°$')
    else:
        title = f'Grid {used_key}, step {STEP_INDEX}'

    # Always build the normal, fully-labeled figure.
    fig, ax = _build_figure(plot_orders, y, y_baseline, y_ceiling, ylabel, title,
                             show_labels=True, y_theory=y_theory)

    # FOR_PUBLICATION additionally builds a separate, stripped-down figure for
    # the SVG export -- it does not touch the labeled figure above.
    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_figure(plot_orders, y, y_baseline, y_ceiling, ylabel, title,
                                          show_labels=False, y_theory=y_theory)
        pub_path = Path(SAVE_FOLDER) / PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'Saved: {pub_path}')

    out_path = Path(__file__).parent / 'comb_displayer.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
