"""
optimized_search_viewer.py

Views the output of a dark-window optimization run that, once it locks onto
a (ch1 power, ch2 power, ch2 phase) setting, repeatedly re-measures the comb
some number of times in a row (e.g. 100 repeats) to characterize shot-to-shot
repeatability. This script ignores the search trajectory entirely (the
opt_*/refine_* fields) and only reads out the repeated-readout block:
harmonics, spectra (repeat, harmonic, ESA bin), cal_spectra (repeat, ESA bin),
ch1_powers_dbm/ch2_powers_dbm/ch2_phases_deg (all constant across repeats).

Three views of that data:
  1. A single-repeat comb display (same bar/stem rendering as
     comb_displayer.py) for one chosen repeat. If REPEAT_INDEX is left None,
     the repeat with the deepest (lowest) dark-window metric -- max CE across
     DARK_ORDERS -- is auto-selected.
  2. Sideband power vs. repeat (iteration) index, for a user-chosen list of
     harmonic orders.
  3. Histograms of sideband power across all repeats, for a user-chosen list
     of harmonic orders.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.optimize import differential_evolution

from comb_finder import load_grid
from path_utils import local_path
from comb_displayer import (
    _HARMONIC_COLORS, _EXTRA_COLORS,
    _draw_gradient_bar, _draw_gradient_stem, _draw_gradient_ball, _points_to_data_scale,
    dual_tone_amplitudes,
)

# ── data source ────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\comb_finding\very_good_suppression2.npz"

# Which repeat (0-indexed) to show in the comb display (Capability 1). None ->
# auto-pick the repeat with the deepest dark window, i.e. the lowest max CE
# across DARK_ORDERS.
REPEAT_INDEX = None

# Orders defining "best dark window" when REPEAT_INDEX is None. None -> use
# the dark_orders array the search itself recorded in the file.
DARK_ORDERS = None
# ─────────────────────────────────────────────────────────────────────────────

# ── capability 1: single-repeat comb display ──────────────────────────────────
PRINT_ORDERS = list(range(-1, 2))   # harmonic orders to show/print for the comb

# Power display mode, shared by all three capabilities below:
#   'percent' -> |A_p|^2 as % of total optical power
#   'dB'      -> 10*log10(|A_p|^2) in dBc  (0 dBc = all power in one line)
DISPLAY_MODE = 'dB'

# Plot style: 'stem' (ball-and-stick) or 'bar' (filled bar with a vertical
# opacity gradient from the baseline to each harmonic's value).
PLOT_STYLE = 'bar'

COMB_SHOW_GRID   = False
COMB_SHOW_LEGEND = False

# If False, the x tick marks are hidden (zero length/width) while the ticks
# themselves -- and their labels -- still exist at the same positions.
COMB_SHOW_XTICKS = False

# dB mode only: floor/ceiling of the y-axis in dBc. None = auto (data
# min/max +- 3 dB margin).
FLOOR_dBc   = -65.0
CEILING_dBc =   2.0
# ─────────────────────────────────────────────────────────────────────────────

# ── comb graphics style ────────────────────────────────────────────────────────
comb_axes_width_mm   = 45.0
comb_axes_height_mm  = 57.0
comb_left_mm         = 20.0
comb_right_mm        = 10.0
comb_bottom_mm       = 15.0
comb_top_mm          =  8.0
comb_spine_linewidth =  2.0
comb_tick_width      =  2.0
comb_tick_direction  = 'in'
comb_axis_label_fontsize = 10.0
comb_tick_label_fontsize =  8.0
comb_stem_linewidth  =  3
comb_markersize      =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── comb opacity gradient (both PLOT_STYLE = 'bar' and 'stem') ────────────────
GRADIENT_ALPHA_MIN   = 0.15   # opacity at the baseline (bottom)
GRADIENT_ALPHA_MAX   = 1.00   # opacity at the harmonic's value (top)
GRADIENT_ORDER       = 1.0    # ramp shape, t**GRADIENT_ORDER; 1 = linear
GRADIENT_RESOLUTION  = 200    # vertical samples in the opacity gradient
# ─────────────────────────────────────────────────────────────────────────────

# ── comb bar style (PLOT_STYLE = 'bar') ───────────────────────────────────────
BAR_WIDTH      = 0.5
BAR_LINESTYLE  = '-'
BAR_LINEWIDTH  = 1.0
BAR_EDGE_COLOR = '#000000'   # None = match the harmonic's color
BAR_EDGE_ALPHA = 1.0
BAR_FACE_COLOR = None        # None = match the harmonic's color

BAR_MARKER            = None
BAR_MARKERSIZE        = 6
BAR_MARKER_FACECOLOR  = None
BAR_MARKER_FACE_ALPHA = 1.0
BAR_MARKER_EDGECOLOR  = None
BAR_MARKER_EDGE_ALPHA = 1.0

BAR_ZORDER         = 2
BAR_OUTLINE_ZORDER = 3
BAR_MARKER_ZORDER  = 4
# ─────────────────────────────────────────────────────────────────────────────

# ── comb stem style (PLOT_STYLE = 'stem') ─────────────────────────────────────
STEM_ZORDER = 2
BALL_ZORDER = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── fit best comb: theoretical (beta1, beta2, phi2) match to displayed comb ──
# Finds the Jacobi-Anger theory (beta1, beta2, phi2) that best matches the
# measured comb currently being displayed (repeat_idx, PRINT_ORDERS), holding
# phi1 = FIT_PHI1_DEG fixed as the phase reference. The fit always minimizes
# squared error in dB (equal weight across orders spanning a wide dynamic
# range), independent of DISPLAY_MODE.
FIT_BEST_COMB = True

# Orders to fit against. None -> the orders actually shown in the comb plot
# (PRINT_ORDERS, restricted to those recorded).
FIT_ORDERS = None

FIT_PHI1_DEG = 0.0                # ch1 phase, held fixed during the fit
FIT_BETA1_BOUNDS = (0.0, 3.0)     # search bounds for beta1 (rad)
FIT_BETA2_BOUNDS = (0.0, 3.0)     # search bounds for beta2 (rad)
FIT_SEED = None                   # differential_evolution seed; None = nondeterministic

# Optional initial guess seeding the optimizer's population (still a global
# search around it, not a strict local refinement -- useful to steer it
# toward the intended solution when the comb has multiple near-degenerate
# fits). Any left None falls back to the midpoint of its search bounds.
FIT_GUESS_BETA1    = 2.46
FIT_GUESS_BETA2    = 1.10
FIT_GUESS_PHI2_DEG = 180

SHOW_FIT_ON_PLOT = True           # overlay the fit on the comb figure

FIT_LINESTYLE  = '-'
FIT_LINEWIDTH  = 2.0
FIT_LINE_COLOR = '#000000'
FIT_LINE_ALPHA = 1.0
# Horizontal dash width in data units at each order. None = match BAR_WIDTH.
FIT_LINE_WIDTH = None
FIT_ZORDER = 11
# ─────────────────────────────────────────────────────────────────────────────

# ── capability 2: sideband power vs iteration ─────────────────────────────────
SHOW_ITERATION_PLOT = True
ITERATION_ORDERS = [0, -1, 1]   # harmonic orders to trace across repeats

# Restrict to a range of repeat indices [ITERATION_START, ITERATION_STOP).
# None -> 0 / the last repeat, respectively (i.e. the full range).
ITERATION_START = 0
ITERATION_STOP  = 20

ITER_LINESTYLE = '-'
ITER_LINEWIDTH = 1.0 # 1.5
ITER_MARKER    = 'o'
ITER_MARKERSIZE = 2 # 4.0

ITER_LINE_COLOR  = None   # None = each order's harmonic color
ITER_LINE_ALPHA  = 1.0
ITER_MARKER_FACECOLOR  = None   # None = match line color
ITER_MARKER_FACE_ALPHA = 1.0
ITER_MARKER_EDGECOLOR  = None   # None = match line color
ITER_MARKER_EDGE_ALPHA = 1.0

ITER_ZORDER = 2

ITER_SHOW_GRID   = False
ITER_SHOW_LEGEND = True
ITER_XLIM = None
ITER_YMIN = -65   # None = autoscale
ITER_YMAX = 3   # None = autoscale

iter_axes_width_mm  = 25.0
iter_axes_height_mm =  14.0
iter_left_mm        =  20.0
iter_right_mm       =  10.0
iter_bottom_mm      =  16.0
iter_top_mm         =   8.0
iter_spine_linewidth =  2.0
iter_tick_width      =  2.0
iter_tick_direction  = 'in'
iter_axis_label_fontsize = 10.0
iter_tick_label_fontsize =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── capability 3: sideband power histograms ───────────────────────────────────
SHOW_HISTOGRAMS = True
HIST_ORDERS  = [0, -1, 1]   # harmonic orders to histogram
HIST_OVERLAY = True      # True: all orders on one axes; False: one figure per order

# Restrict to a range of repeat indices [HIST_START, HIST_STOP). None -> 0 /
# the last repeat, respectively (i.e. the full range).
HIST_START = 0
HIST_STOP  = 20

HIST_BINS      = 5
HIST_LINESTYLE = '-'
HIST_LINEWIDTH = 1.0
HIST_EDGECOLOR = None   # None = match each order's harmonic color
HIST_EDGE_ALPHA = 1.0
HIST_FACECOLOR = None   # None = match each order's harmonic color
HIST_FACE_ALPHA = 0.6
HIST_ZORDER     = 2

HIST_SHOW_GRID   = False
HIST_SHOW_LEGEND = True
HIST_XLIM = None
HIST_YLIM = None

hist_axes_width_mm  = 100.0
hist_axes_height_mm =  40.0
hist_left_mm        =  20.0
hist_right_mm       =  10.0
hist_bottom_mm      =  16.0
hist_top_mm         =   8.0
hist_spine_linewidth =  2.0
hist_tick_width      =  2.0
hist_tick_direction  = 'in'
hist_axis_label_fontsize = 10.0
hist_tick_label_fontsize =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels, the title, and the legend (from a
# separate figure built alongside the normal one shown/PNG-saved below), and
# additionally saves an SVG of each to SAVE_FOLDER.
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
COMB_PUBLICATION_SVG_NAME = 'optimized_search_viewer_comb.svg'
ITER_PUBLICATION_SVG_NAME = 'optimized_search_viewer_iteration.svg'
HIST_PUBLICATION_SVG_NAME = 'optimized_search_viewer_histogram.svg'
# ─────────────────────────────────────────────────────────────────────────────


def _build_harmonic_colors(orders):
    extra_iter = iter(_EXTRA_COLORS)
    return {int(n): _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000')) for n in orders}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_repeats():
    """
    Returns (harmonics, spectra, cal_spectra, ch1_pwr, ch2_pwr, ch2_phase_deg,
    file_dark_orders). spectra: (P, N, K); cal_spectra: (P, K); P = number of
    repeats, N = number of recorded harmonics, K = ESA bins per sweep.
    """
    resolved_path = local_path(DATA_PATH)
    meta, grid_data = load_grid(resolved_path)
    d = grid_data[(0, 0)]

    harmonics   = d['harmonics'].astype(int)
    spectra     = d['spectra']
    cal_spectra = d['cal_spectra']
    ch1_pwr     = float(meta['ch1_powers_dbm'][0])
    ch2_pwr     = float(meta['ch2_powers_dbm'][0])
    ch2_phase   = float(d['ch2_phases_deg'][0])

    with np.load(resolved_path, allow_pickle=True) as raw:
        file_dark_orders = raw['dark_orders'].astype(int).tolist() if 'dark_orders' in raw.files else None

    return harmonics, spectra, cal_spectra, ch1_pwr, ch2_pwr, ch2_phase, file_dark_orders


def _dark_metric_db(harmonics, spectra, cal_spectra, dark_orders):
    """Per-repeat max CE [dBc] across dark_orders. Returns (P,) array."""
    idxs = [int(np.where(harmonics == o)[0][0]) for o in dark_orders if o in harmonics]
    if not idxs:
        raise ValueError(f"None of dark_orders={dark_orders} are recorded (have {list(harmonics)}).")
    peak_dbm = spectra[:, idxs, :].max(axis=-1)        # (P, len(idxs))
    cal_dbm  = cal_spectra.max(axis=-1)[:, np.newaxis]  # (P, 1)
    dbc = peak_dbm - cal_dbm
    return dbc.max(axis=-1)   # (P,)


def _powers_at_repeat(harmonics, spectra, cal_spectra, idx, orders):
    """{order: |A_p|^2 (linear fraction)} for a single repeat, skipping any
    order not recorded (with a console warning)."""
    cal_peak_dbm = float(cal_spectra[idx].max())
    powers_lin = {}
    for p in orders:
        where = np.where(harmonics == p)[0]
        if len(where) == 0:
            print(f"  Warning: order {p} not recorded; skipping.")
            continue
        peak_dbm = float(spectra[idx, int(where[0])].max())
        powers_lin[p] = 10.0 ** ((peak_dbm - cal_peak_dbm) / 10.0)
    return powers_lin


def _power_series(harmonics, spectra, cal_spectra, order):
    """|A_p|^2 (linear fraction) across all repeats for one order, or None if
    that order isn't recorded."""
    where = np.where(harmonics == order)[0]
    if len(where) == 0:
        return None
    h_idx = int(where[0])
    peak_dbm = spectra[:, h_idx, :].max(axis=-1)   # (P,)
    cal_dbm  = cal_spectra.max(axis=-1)            # (P,)
    return 10.0 ** ((peak_dbm - cal_dbm) / 10.0)


def _resolve_repeat_range(P, start, stop):
    """Clip [start, stop) (either may be None) to a valid [lo, hi) within
    [0, P)."""
    lo = 0 if start is None else max(0, int(start))
    hi = P if stop is None else min(P, int(stop))
    if lo >= hi:
        raise ValueError(f"Empty repeat range: start={start}, stop={stop} (P={P}).")
    return lo, hi


def _to_display(power_lin):
    if DISPLAY_MODE == 'percent':
        return power_lin * 100.0
    return 10.0 * np.log10(np.maximum(power_lin, 1e-30))


# ─────────────────────────────────────────────────────────────────────────────
# Capability 1: single-repeat comb display (mirrors comb_displayer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_comb_figure(plot_orders, y, y_baseline, y_ceiling, ylabel, title, show_labels: bool,
                        y_fit=None):
    fig_w = comb_left_mm + comb_axes_width_mm + comb_right_mm
    fig_h = comb_bottom_mm + comb_axes_height_mm + comb_top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = comb_left_mm   / fig_w,
        right  = 1 - comb_right_mm  / fig_w,
        bottom = comb_bottom_mm / fig_h,
        top    = 1 - comb_top_mm    / fig_h,
    )

    ax.set_xlim(plot_orders[0] - 0.8, plot_orders[-1] + 0.8)
    if DISPLAY_MODE == 'percent':
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=y_baseline, top=y_ceiling)
    fig.canvas.draw()

    harmonic_colors = _build_harmonic_colors(plot_orders)

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
    else:
        dx_per_pt, dy_per_pt = _points_to_data_scale(ax)
        radius_x = (comb_markersize / 2.0) * dx_per_pt
        radius_y = (comb_markersize / 2.0) * dy_per_pt

        for xi, yi in zip(plot_orders, y):
            n = int(xi)
            c = harmonic_colors[n]
            label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n:+d}'

            ax.plot([], [], color=c, linewidth=comb_stem_linewidth, label=label)

            direction = 1.0 if yi >= y_baseline else -1.0
            y_near = yi - direction * radius_y
            y_far  = yi + direction * radius_y

            _draw_gradient_stem(ax, xi, y_baseline, y_near, y_far, c, comb_stem_linewidth,
                                 GRADIENT_ALPHA_MIN, GRADIENT_ALPHA_MAX, GRADIENT_ORDER,
                                 GRADIENT_RESOLUTION, STEM_ZORDER)
            _draw_gradient_ball(ax, xi, yi, radius_x, radius_y, y_baseline, y_far, c,
                                 GRADIENT_ALPHA_MIN, GRADIENT_ALPHA_MAX, GRADIENT_ORDER,
                                 GRADIENT_RESOLUTION, BALL_ZORDER)

    if y_fit is not None:
        half_w = (FIT_LINE_WIDTH if FIT_LINE_WIDTH is not None else BAR_WIDTH) / 2.0
        for i, (xi, yt) in enumerate(zip(plot_orders, y_fit)):
            ax.plot([xi - half_w, xi + half_w], [yt, yt],
                    color=mcolors.to_rgba(FIT_LINE_COLOR, FIT_LINE_ALPHA),
                    linewidth=FIT_LINEWIDTH, linestyle=FIT_LINESTYLE,
                    solid_capstyle='round', dash_capstyle='round',
                    zorder=FIT_ZORDER, label=('Best fit' if i == 0 else None))

    ax.axhline(y_baseline, color='#333333', linewidth=0.8, linestyle='-', zorder=1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for spine in ax.spines.values():
        spine.set_linewidth(comb_spine_linewidth)
    ax.tick_params(axis='both', direction=comb_tick_direction,
                   width=comb_tick_width, labelsize=comb_tick_label_fontsize)
    if not COMB_SHOW_XTICKS:
        ax.tick_params(axis='x', width=0, length=0)

    if show_labels:
        ax.set_xlabel(r'Harmonic order $p$', fontsize=comb_axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=comb_axis_label_fontsize)
        ax.set_title(title, fontsize=comb_axis_label_fontsize)
        if COMB_SHOW_GRID:
            ax.grid()
        if COMB_SHOW_LEGEND:
            ax.legend(fontsize=comb_tick_label_fontsize, frameon=False)
    else:
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def _theory_powers_lin(beta1, beta2, phi1_deg, phi2_deg, orders):
    """{order: |A_p|^2} from Jacobi-Anger theory (reuses comb_displayer's
    dual_tone_amplitudes)."""
    k_trunc = int(2 * max(beta1, beta2)) + 20
    amps = dual_tone_amplitudes(
        beta1, beta2, np.deg2rad(phi1_deg), np.deg2rad(phi2_deg), orders, k_trunc)
    return {p: abs(A) ** 2 for p, A in amps.items()}


def _fit_cost(params, orders, measured_db):
    beta1, beta2, phi2_deg = params
    theory_lin = _theory_powers_lin(beta1, beta2, FIT_PHI1_DEG, phi2_deg, orders)
    theory_db = np.array([10.0 * np.log10(max(theory_lin[p], 1e-30)) for p in orders])
    return float(np.sum((measured_db - theory_db) ** 2))


def _fit_best_comb(powers_lin, orders):
    """
    Finds (beta1, beta2, phi2_deg) minimizing squared dB error against the
    measured powers_lin at `orders`, holding phi1 = FIT_PHI1_DEG fixed. Uses a
    global optimizer (differential_evolution) since the Bessel-function comb
    can have multiple local optima in phi2.
    """
    measured_db = np.array([10.0 * np.log10(max(powers_lin[p], 1e-30)) for p in orders])
    bounds = [FIT_BETA1_BOUNDS, FIT_BETA2_BOUNDS, (0.0, 360.0)]

    guess = [FIT_GUESS_BETA1, FIT_GUESS_BETA2, FIT_GUESS_PHI2_DEG]
    x0 = None
    if any(g is not None for g in guess):
        x0 = [float(np.clip(g if g is not None else 0.5 * (lo + hi), lo, hi))
              for g, (lo, hi) in zip(guess, bounds)]

    result = differential_evolution(_fit_cost, bounds, args=(orders, measured_db),
                                     seed=FIT_SEED, tol=1e-12, polish=True, x0=x0)
    beta1, beta2, phi2_deg = result.x
    phi2_deg = float(phi2_deg % 360.0)
    theory_lin = _theory_powers_lin(beta1, beta2, FIT_PHI1_DEG, phi2_deg, orders)
    return {
        'beta1':             float(beta1),
        'beta2':             float(beta2),
        'phi1_deg':          FIT_PHI1_DEG,
        'phi2_deg':          phi2_deg,
        'cost':              float(result.fun),
        'theory_powers_lin': theory_lin,
    }


def _run_comb_display(harmonics, spectra, cal_spectra, repeat_idx, ch1_pwr, ch2_pwr, ch2_phase):
    orders = sorted(set(PRINT_ORDERS))
    powers_lin = _powers_at_repeat(harmonics, spectra, cal_spectra, repeat_idx, orders)

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

    unit = '%' if DISPLAY_MODE == 'percent' else 'dBc'
    print(f"Repeat {repeat_idx} (ch1={ch1_pwr:+.2f} dBm, ch2={ch2_pwr:+.2f} dBm, "
          f"ch2 phase={ch2_phase:.2f} deg):")
    print(f"  {'order':>5}   {'power':>12}")
    print(f"  {'-'*5}   {'-'*12}")
    for p in valid_orders:
        pwr = (powers_lin[p] * 100.0 if DISPLAY_MODE == 'percent'
               else 10.0 * np.log10(max(powers_lin[p], 1e-30)))
        print(f"  p={p:+d}   {pwr:>10.4f} {unit}")
    total = sum(powers_lin.values())
    print(f"\n  Total power in plotted orders: {total * 100:.2f}%")

    y_fit = None
    if FIT_BEST_COMB:
        fit_orders = sorted(set(FIT_ORDERS)) if FIT_ORDERS is not None else valid_orders
        fit_powers_lin = (powers_lin if fit_orders == valid_orders else
                           _powers_at_repeat(harmonics, spectra, cal_spectra, repeat_idx, fit_orders))
        fit_orders = sorted(fit_powers_lin)

        if not fit_orders:
            print("\n  Warning: no FIT_ORDERS recorded; skipping fit.")
        else:
            fit = _fit_best_comb(fit_powers_lin, fit_orders)
            print(f"\nBest-fit theory (phi1={fit['phi1_deg']:.1f} deg fixed):")
            print(f"  beta1 = {fit['beta1']:.4f} rad")
            print(f"  beta2 = {fit['beta2']:.4f} rad")
            print(f"  phi2  = {fit['phi2_deg']:.2f} deg")
            print(f"  Sum of squared dB residuals: {fit['cost']:.4f}")

            theory_lin = fit['theory_powers_lin']
            if 0 in theory_lin:
                csr_db = 10.0 * np.log10(max(theory_lin[0], 1e-30))
                print(f"  CSR (carrier suppression, order 0): {csr_db:+.3f} dB")
            else:
                print("  CSR: order 0 not in FIT_ORDERS; skipping.")
            if 1 in theory_lin and -1 in theory_lin:
                ce_p1 = 10.0 * np.log10(max(theory_lin[1], 1e-30))
                ce_m1 = 10.0 * np.log10(max(theory_lin[-1], 1e-30))
                print(f"  SSR (CE[+1] - CE[-1]): {ce_p1 - ce_m1:+.3f} dB")
            else:
                print("  SSR: orders +1/-1 not both in FIT_ORDERS; skipping.")
            if 1 in theory_lin:
                print(f"  CE[+1] (conversion efficiency to order +1): {theory_lin[1] * 100.0:.4f} %")
            else:
                print("  CE[+1]: order +1 not in FIT_ORDERS; skipping.")

            print(f"  {'order':>5}   {'measured':>10}   {'theory':>10}   {'resid':>8}")
            print(f"  {'-'*5}   {'-'*10}   {'-'*10}   {'-'*8}")
            for p in fit_orders:
                m_db = 10.0 * np.log10(max(fit_powers_lin[p], 1e-30))
                t_db = 10.0 * np.log10(max(fit['theory_powers_lin'][p], 1e-30))
                print(f"  p={p:+d}   {m_db:>8.3f} dB   {t_db:>8.3f} dB   {m_db - t_db:>+6.3f} dB")

            if SHOW_FIT_ON_PLOT:
                plot_theory_lin = _theory_powers_lin(fit['beta1'], fit['beta2'],
                                                      fit['phi1_deg'], fit['phi2_deg'],
                                                      valid_orders)
                plot_theory = np.array([plot_theory_lin[p] for p in valid_orders])
                y_fit = (plot_theory * 100.0 if DISPLAY_MODE == 'percent'
                         else 10.0 * np.log10(np.maximum(plot_theory, 1e-30)))

    title = f'Repeat {repeat_idx}'
    fig, ax = _build_comb_figure(plot_orders, y, y_baseline, y_ceiling, ylabel, title,
                                  show_labels=True, y_fit=y_fit)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_comb_figure(plot_orders, y, y_baseline, y_ceiling, ylabel,
                                               title, show_labels=False, y_fit=y_fit)
        pub_path = Path(SAVE_FOLDER) / COMB_PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'Saved: {pub_path}')

    out_path = Path(__file__).parent / 'optimized_search_viewer_comb.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Capability 2: sideband power vs iteration
# ─────────────────────────────────────────────────────────────────────────────

def _build_iteration_figure(x, series_display, orders, harmonic_colors, ylabel, show_labels: bool):
    fig_w = iter_left_mm + iter_axes_width_mm + iter_right_mm
    fig_h = iter_bottom_mm + iter_axes_height_mm + iter_top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = iter_left_mm   / fig_w,
        right  = 1 - iter_right_mm  / fig_w,
        bottom = iter_bottom_mm / fig_h,
        top    = 1 - iter_top_mm    / fig_h,
    )

    for p in orders:
        y = series_display[p]
        color = ITER_LINE_COLOR if ITER_LINE_COLOR is not None else harmonic_colors[p]
        marker_face = ITER_MARKER_FACECOLOR if ITER_MARKER_FACECOLOR is not None else color
        marker_edge = ITER_MARKER_EDGECOLOR if ITER_MARKER_EDGECOLOR is not None else color
        label = 'Carrier (n=0)' if p == 0 else f'Harmonic {p:+d}'

        ax.plot(x, y, linestyle=ITER_LINESTYLE, linewidth=ITER_LINEWIDTH,
                 color=mcolors.to_rgba(color, ITER_LINE_ALPHA),
                 marker=ITER_MARKER, markersize=ITER_MARKERSIZE,
                 markerfacecolor=mcolors.to_rgba(marker_face, ITER_MARKER_FACE_ALPHA),
                 markeredgecolor=mcolors.to_rgba(marker_edge, ITER_MARKER_EDGE_ALPHA),
                 solid_capstyle='round', zorder=ITER_ZORDER, label=label)

    for spine in ax.spines.values():
        spine.set_linewidth(iter_spine_linewidth)
    ax.tick_params(axis='both', direction=iter_tick_direction,
                   width=iter_tick_width, labelsize=iter_tick_label_fontsize)

    if ITER_XLIM is not None:
        ax.set_xlim(*ITER_XLIM)
    if ITER_YMIN is not None or ITER_YMAX is not None:
        ax.set_ylim(bottom=ITER_YMIN, top=ITER_YMAX)

    if show_labels:
        ax.set_xlabel('Iteration', fontsize=iter_axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=iter_axis_label_fontsize)
        if ITER_SHOW_GRID:
            ax.grid()
        if ITER_SHOW_LEGEND:
            ax.legend(fontsize=iter_tick_label_fontsize, frameon=False)
    else:
        ax.set_title('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def _run_iteration_plot(harmonics, spectra, cal_spectra):
    orders = []
    series_lin = {}
    for p in sorted(set(ITERATION_ORDERS)):
        s = _power_series(harmonics, spectra, cal_spectra, p)
        if s is None:
            print(f"  Warning: order {p} not recorded; skipping from iteration plot.")
            continue
        orders.append(p)
        series_lin[p] = s

    if not orders:
        print("  No requested iteration orders are recorded; skipping iteration plot.")
        return

    P = len(next(iter(series_lin.values())))
    lo, hi = _resolve_repeat_range(P, ITERATION_START, ITERATION_STOP)
    print(f"  Using iterations {lo}-{hi - 1} (of {P} total).")
    x = np.arange(lo, hi)
    series_display = {p: _to_display(series_lin[p][lo:hi]) for p in orders}
    ylabel = (r'$|A_p|^2$ [% of total power]' if DISPLAY_MODE == 'percent'
              else r'$|A_p|^2$ [dBc]')
    harmonic_colors = _build_harmonic_colors(orders)

    fig, ax = _build_iteration_figure(x, series_display, orders, harmonic_colors, ylabel,
                                       show_labels=True)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_iteration_figure(x, series_display, orders, harmonic_colors,
                                                     ylabel, show_labels=False)
        pub_path = Path(SAVE_FOLDER) / ITER_PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'Saved: {pub_path}')

    out_path = Path(__file__).parent / 'optimized_search_viewer_iteration.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Capability 3: sideband power histograms
# ─────────────────────────────────────────────────────────────────────────────

def _build_histogram_figure(data_by_order, orders, harmonic_colors, xlabel, title, show_labels: bool):
    fig_w = hist_left_mm + hist_axes_width_mm + hist_right_mm
    fig_h = hist_bottom_mm + hist_axes_height_mm + hist_top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = hist_left_mm   / fig_w,
        right  = 1 - hist_right_mm  / fig_w,
        bottom = hist_bottom_mm / fig_h,
        top    = 1 - hist_top_mm    / fig_h,
    )

    for p in orders:
        color = harmonic_colors[p]
        face_color = HIST_FACECOLOR if HIST_FACECOLOR is not None else color
        edge_color = HIST_EDGECOLOR if HIST_EDGECOLOR is not None else color
        label = 'Carrier (n=0)' if p == 0 else f'Harmonic {p:+d}'

        ax.hist(data_by_order[p], bins=HIST_BINS,
                facecolor=mcolors.to_rgba(face_color, HIST_FACE_ALPHA),
                edgecolor=mcolors.to_rgba(edge_color, HIST_EDGE_ALPHA),
                linestyle=HIST_LINESTYLE, linewidth=HIST_LINEWIDTH,
                zorder=HIST_ZORDER, label=label)

    for spine in ax.spines.values():
        spine.set_linewidth(hist_spine_linewidth)
    ax.tick_params(axis='both', direction=hist_tick_direction,
                   width=hist_tick_width, labelsize=hist_tick_label_fontsize)

    if HIST_XLIM is not None:
        ax.set_xlim(*HIST_XLIM)
    if HIST_YLIM is not None:
        ax.set_ylim(*HIST_YLIM)

    if show_labels:
        ax.set_xlabel(xlabel, fontsize=hist_axis_label_fontsize)
        ax.set_ylabel('Count', fontsize=hist_axis_label_fontsize)
        ax.set_title(title, fontsize=hist_axis_label_fontsize)
        if HIST_SHOW_GRID:
            ax.grid()
        if HIST_SHOW_LEGEND:
            ax.legend(fontsize=hist_tick_label_fontsize, frameon=False)
    else:
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def _run_histograms(harmonics, spectra, cal_spectra):
    orders = []
    data_lin = {}
    for p in sorted(set(HIST_ORDERS)):
        s = _power_series(harmonics, spectra, cal_spectra, p)
        if s is None:
            print(f"  Warning: order {p} not recorded; skipping from histograms.")
            continue
        orders.append(p)
        data_lin[p] = s

    if not orders:
        print("  No requested histogram orders are recorded; skipping histograms.")
        return

    P = len(next(iter(data_lin.values())))
    lo, hi = _resolve_repeat_range(P, HIST_START, HIST_STOP)
    print(f"  Using iterations {lo}-{hi - 1} (of {P} total).")
    data_display = {p: _to_display(data_lin[p][lo:hi]) for p in orders}
    unit = '%' if DISPLAY_MODE == 'percent' else 'dBc'
    xlabel = (r'$|A_p|^2$ [% of total power]' if DISPLAY_MODE == 'percent'
              else r'$|A_p|^2$ [dBc]')
    harmonic_colors = _build_harmonic_colors(orders)

    print("Sideband power histograms:")
    for p in orders:
        vals = data_display[p]
        print(f"  p={p:+d}:  mean={vals.mean():.4f} {unit}  std={vals.std():.4f} {unit}  "
              f"min={vals.min():.4f} {unit}  max={vals.max():.4f} {unit}")

    if HIST_OVERLAY:
        title = f'Orders {orders}'
        fig, ax = _build_histogram_figure(data_display, orders, harmonic_colors, xlabel, title,
                                           show_labels=True)
        if FOR_PUBLICATION:
            fig_pub, _ax_pub = _build_histogram_figure(data_display, orders, harmonic_colors,
                                                         xlabel, title, show_labels=False)
            pub_path = Path(SAVE_FOLDER) / HIST_PUBLICATION_SVG_NAME
            fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
            print(f'Saved: {pub_path}')
        out_path = Path(__file__).parent / 'optimized_search_viewer_histogram.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {out_path}')
    else:
        pub_stem = Path(HIST_PUBLICATION_SVG_NAME).stem
        pub_suffix = Path(HIST_PUBLICATION_SVG_NAME).suffix
        for p in orders:
            title = f'Order {p:+d}'
            fig, ax = _build_histogram_figure({p: data_display[p]}, [p], harmonic_colors, xlabel,
                                               title, show_labels=True)
            if FOR_PUBLICATION:
                fig_pub, _ax_pub = _build_histogram_figure({p: data_display[p]}, [p],
                                                             harmonic_colors, xlabel, title,
                                                             show_labels=False)
                pub_path = Path(SAVE_FOLDER) / f'{pub_stem}_{p:+d}{pub_suffix}'
                fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
                print(f'Saved: {pub_path}')
            out_path = Path(__file__).parent / f'optimized_search_viewer_histogram_{p:+d}.png'
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    (harmonics, spectra, cal_spectra, ch1_pwr, ch2_pwr, ch2_phase,
     file_dark_orders) = _load_repeats()
    P = spectra.shape[0]
    print(f"Loaded: {DATA_PATH}")
    print(f"  Repeats    : {P}")
    print(f"  Harmonics  : {list(harmonics)}")
    print(f"  Settings   : ch1={ch1_pwr:+.2f} dBm, ch2={ch2_pwr:+.2f} dBm, "
          f"ch2 phase={ch2_phase:.2f} deg\n")

    dark_orders = DARK_ORDERS if DARK_ORDERS is not None else file_dark_orders
    if REPEAT_INDEX is None:
        if dark_orders is None:
            raise ValueError(
                "REPEAT_INDEX is None and no DARK_ORDERS could be determined "
                "(file has no dark_orders field); set DARK_ORDERS explicitly."
            )
        metric_db = _dark_metric_db(harmonics, spectra, cal_spectra, dark_orders)
        repeat_idx = int(np.argmin(metric_db))
        print(f"Auto-selected repeat {repeat_idx} (deepest dark window over orders "
              f"{dark_orders}: {metric_db[repeat_idx]:.2f} dBc).\n")
    else:
        repeat_idx = REPEAT_INDEX

    _run_comb_display(harmonics, spectra, cal_spectra, repeat_idx, ch1_pwr, ch2_pwr, ch2_phase)

    if SHOW_ITERATION_PLOT:
        print()
        _run_iteration_plot(harmonics, spectra, cal_spectra)

    if SHOW_HISTOGRAMS:
        print()
        _run_histograms(harmonics, spectra, cal_spectra)

    plt.show()


if __name__ == '__main__':
    main()
