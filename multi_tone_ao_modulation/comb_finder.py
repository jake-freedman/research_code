"""
Comb finder: searches a 2D power-phase grid sweep produced by
bnc_2d_power_phase_script.py for the optimal ch1 power, ch2 power,
and ch2 phase combination according to a given criterion.

Criteria implemented
--------------------
'max_harmonic' : find the (ch1 power, ch2 power, ch2 phase) triple that
                 maximises the peak ESA power of TARGET_HARMONIC.
                 Metric is raw dBm when NORMALIZE=False, fraction of
                 optical carrier [%] when NORMALIZE='percent', or dBc
                 when NORMALIZE=True.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_w, axes_height_mm as _default_h,
    left_mm as _left, right_mm as _right,
    bottom_mm as _bottom, top_mm as _top,
)

_HARMONIC_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_EXTRA_COLORS = [BLUE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2, DARKGRAY2]


def _harmonic_color(n: int) -> str:
    return _HARMONIC_COLORS.get(n, _EXTRA_COLORS[abs(n) % len(_EXTRA_COLORS)])

# ─────────────────────────────────────────────────────────────────────────────
# User settings
# ─────────────────────────────────────────────────────────────────────────────

FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\comb_finding\2d_power_phase_2026-07-19-15-36-40"

# Optimisation criterion.
CRITERION = 'dark_window'

# Which harmonic order to optimise (for 'max_harmonic').
TARGET_HARMONIC = 1

# Inclusive range of harmonic orders to evaluate for the 'flatness' criterion.
FLATNESS_ORDER_MIN = -2
FLATNESS_ORDER_MAX =  2

# Orders to suppress for the 'dark_window' criterion. The metric minimised is
# the maximum CE across all listed orders (minimax suppression).
DARK_WINDOW_ORDERS = [0, -1]

# Metric normalisation:
#   False      → raw peak power [dBm]
#   'percent'  → fraction of optical carrier [%]  (requires cal_spectra)
#   True       → relative to carrier [dBc]         (requires cal_spectra)
NORMALIZE = True

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_figure(axes_w=_default_w, axes_h=_default_h):
    mm = 1.0 / 25.4
    fig_w = _left + axes_w + _right
    fig_h = _bottom + axes_h + _top
    fig, ax = plt.subplots(figsize=(fig_w * mm, fig_h * mm))
    fig.subplots_adjust(
        left=_left / fig_w, right=1 - _right / fig_w,
        bottom=_bottom / fig_h, top=1 - _top / fig_h,
    )
    return fig, ax


def _style_ax(ax):
    ax.tick_params(direction=tick_direction, width=tick_width,
                   labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)


def _metric_label(normalize, target_harmonic):
    n_lbl = f'n={target_harmonic}'
    if normalize == 'percent':
        return f'{n_lbl} power [% of carrier]'
    if normalize:
        return f'{n_lbl} power [dBc]'
    return f'{n_lbl} power [dBm]'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_grid(folder: str):
    """
    Load a 2D power-phase grid folder produced by bnc_2d_power_phase_script.py,
    or a single .npz file produced by bnc_dual_tone_esa_script.py.

    When given a single file, it is wrapped into a 1×1 grid where the M sweep
    steps are treated as the phase axis (appropriate for phase sweeps).

    Returns
    -------
    meta : dict
        ch1_powers_dbm, ch2_powers_dbm, harmonics, ch2_phases_deg, …
    grid_data : dict of (i, j) -> dict
        Per-point data arrays keyed by npz field name.
    """
    if os.path.isfile(folder):
        return _load_single_npz(folder)

    meta_path = os.path.join(folder, 'grid_meta.npz')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"grid_meta.npz not found in {folder}")
    meta = dict(np.load(meta_path, allow_pickle=True))

    grid_data = {}
    for fpath in sorted(glob.glob(os.path.join(folder, 'grid_[0-9]*.npz'))):
        name = os.path.basename(fpath).replace('.npz', '')   # grid_ii_jj
        parts = name.split('_')
        i, j = int(parts[1]), int(parts[2])
        grid_data[(i, j)] = dict(np.load(fpath, allow_pickle=True))

    if not grid_data:
        raise FileNotFoundError(f"No grid_*.npz files found in {folder}")
    return meta, grid_data


def _load_single_npz(path: str):
    """
    Wrap a single dual-tone sweep .npz into the (meta, grid_data) format
    expected by the criterion functions. The M sweep steps are treated as
    the phase axis at a single (ch1_power, ch2_power) grid point.
    """
    d = dict(np.load(path, allow_pickle=True))
    harmonics      = d['harmonics'].astype(int)
    ch2_phases_deg = d['ch2_phases_deg']           # (M,)
    spectra        = d['spectra']                  # (M, N, K)
    cal_spectra    = d.get('cal_spectra', None)    # (M, K) or None
    ch1_pwr        = float(d['ch1_powers_dbm'][0])
    ch2_pwr        = float(d['ch2_powers_dbm'][0])

    meta = {
        'ch1_powers_dbm': np.array([ch1_pwr]),
        'ch2_powers_dbm': np.array([ch2_pwr]),
        'harmonics':      harmonics,
        'ch2_phases_deg': ch2_phases_deg,
    }
    point = {
        'spectra':        spectra,
        'cal_spectra':    cal_spectra,
        'ch2_phases_deg': ch2_phases_deg,
        'harmonics':      harmonics,
    }
    return meta, {(0, 0): point}


def _harmonic_idx(harmonics, target: int) -> int:
    idx = np.where(np.asarray(harmonics, dtype=int) == int(target))[0]
    if len(idx) == 0:
        raise ValueError(f"Harmonic {target} not in dataset. Available: {list(harmonics)}")
    return int(idx[0])


def _compute_metric(spectra, cal_spectra, h_idx: int, normalize):
    """
    Per-phase metric for a single grid point.

    Parameters
    ----------
    spectra     : (P, N, K) array
    cal_spectra : (P, K) array or None
    h_idx       : harmonic column index in axis 1

    Returns (P,) metric array.
    """
    peak_dbm = spectra[:, h_idx, :].max(axis=-1)   # (P,)

    if normalize in (True, 'percent'):
        if cal_spectra is None:
            raise RuntimeError(
                "normalize requires cal_spectra, which is absent from this grid point."
            )
        cal_dbm = cal_spectra.max(axis=-1)           # (P,)
        dbc = peak_dbm - cal_dbm
        if normalize == 'percent':
            return 10.0 ** (dbc / 10.0) * 100.0
        return dbc   # dBc

    return peak_dbm


# ─────────────────────────────────────────────────────────────────────────────
# Criterion: max_harmonic
# ─────────────────────────────────────────────────────────────────────────────


def find_max_harmonic(meta, grid_data, target_harmonic: int, normalize):
    """
    Exhaustively search every grid point and phase to find the combination
    that maximises the peak metric of `target_harmonic`.

    Returns
    -------
    dict with keys:
        best_i, best_j          – grid indices of best point
        best_phase_idx          – phase index within best grid point
        best_phase_deg          – corresponding ch2 phase in degrees
        best_metric             – metric value at optimum
        best_ch1_power          – ch1 dBm at optimum
        best_ch2_power          – ch2 dBm at optimum
        metric_grid             – (n1, n2) array, best metric per grid point
        best_phase_grid         – (n1, n2) array, optimal phase per grid point
        phase_curve             – (P,) metric vs phase at best grid point
        ch2_phases_deg          – phase array at best grid point
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    n1, n2 = len(ch1_powers), len(ch2_powers)

    metric_grid     = np.full((n1, n2), np.nan)
    best_phase_grid = np.full((n1, n2), np.nan)

    best_val = -np.inf
    best_i = best_j = best_phase_idx = None
    best_phase_curve = None
    best_phases_arr  = None

    for (i, j), d in grid_data.items():
        harmonics = d['harmonics'].astype(int)
        try:
            h_idx = _harmonic_idx(harmonics, target_harmonic)
        except ValueError as e:
            print(f"  Grid ({i},{j}): {e}")
            continue

        spectra     = d['spectra']                              # (P, N, K)
        cal_spectra = d.get('cal_spectra', None)                # (P, K) or None
        phases      = d['ch2_phases_deg']                       # (P,)

        metric = _compute_metric(spectra, cal_spectra, h_idx, normalize)   # (P,)
        best_p = int(np.argmax(metric))

        metric_grid[i, j]     = float(metric[best_p])
        best_phase_grid[i, j] = float(phases[best_p])

        if metric[best_p] > best_val:
            best_val        = float(metric[best_p])
            best_i          = i
            best_j          = j
            best_phase_idx  = best_p
            best_phase_curve = metric
            best_phases_arr  = phases
            best_d          = d

    if best_i is None:
        raise RuntimeError("No valid grid points found.")

    # Per-harmonic CE at the winning (i, j, phase).
    best_spectra = best_d['spectra'][best_phase_idx]     # (N, K)
    best_cal     = best_d.get('cal_spectra', None)
    best_peaks_dbm = best_spectra.max(axis=-1)           # (N,)
    if normalize in (True, 'percent') and best_cal is not None:
        cal_peak = float(best_cal[best_phase_idx].max())
        dbc = best_peaks_dbm - cal_peak
        best_comb_ce = 10.0 ** (dbc / 10.0) * 100.0 if normalize == 'percent' else dbc
    else:
        best_comb_ce = best_peaks_dbm

    return {
        'best_i':          best_i,
        'best_j':          best_j,
        'best_phase_idx':  best_phase_idx,
        'best_phase_deg':  float(best_phases_arr[best_phase_idx]),
        'best_metric':     best_val,
        'best_ch1_power':  float(ch1_powers[best_i]),
        'best_ch2_power':  float(ch2_powers[best_j]),
        'metric_grid':     metric_grid,
        'best_phase_grid': best_phase_grid,
        'phase_curve':     best_phase_curve,
        'ch2_phases_deg':  best_phases_arr,
        'best_comb_ce':    best_comb_ce,          # (N,) CE for all harmonics
        'best_harmonics':  best_d['harmonics'].astype(int),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_max_harmonic(result: dict, meta: dict, normalize, target_harmonic: int):
    """
    Figure 1 — heatmap of best achievable metric per grid point (maximised
                over phase). Best point marked with a red cross.
    Figure 2 — metric vs ch2 phase at the best grid point.
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    ylabel = _metric_label(normalize, target_harmonic)

    # ── Figure 1: heatmap ────────────────────────────────────────────────────
    mm = 1.0 / 25.4
    fig1, ax1 = plt.subplots(figsize=(90 * mm, 75 * mm))
    fig1.subplots_adjust(left=0.18, right=0.88, bottom=0.15, top=0.93)

    n1, n2 = len(ch1_powers), len(ch2_powers)
    if n1 > 1 and n2 > 1:
        dx = (ch1_powers[-1] - ch1_powers[0]) / (n1 - 1) / 2
        dy = (ch2_powers[-1] - ch2_powers[0]) / (n2 - 1) / 2
        extent = [ch1_powers[0] - dx, ch1_powers[-1] + dx,
                  ch2_powers[0] - dy, ch2_powers[-1] + dy]
    else:
        extent = [ch1_powers[0], ch1_powers[-1],
                  ch2_powers[0], ch2_powers[-1]]

    im = ax1.imshow(
        result['metric_grid'].T,    # transpose so x=ch1, y=ch2
        origin='lower', aspect='auto', extent=extent, cmap='viridis',
    )
    ax1.scatter(
        [result['best_ch1_power']], [result['best_ch2_power']],
        color='red', marker='x', s=80, linewidths=1.5, zorder=5,
    )
    cb = plt.colorbar(im, ax=ax1)
    cb.set_label(ylabel, fontsize=tick_label_fontsize)
    cb.ax.tick_params(labelsize=tick_label_fontsize)
    ax1.set_xlabel('Ch1 power [dBm]', fontsize=axis_label_fontsize)
    ax1.set_ylabel('Ch2 power [dBm]', fontsize=axis_label_fontsize)
    _style_ax(ax1)

    # ── Figure 2: phase curve at best grid point ──────────────────────────
    fig2, ax2 = _make_figure()
    phases = result['ch2_phases_deg']
    curve  = result['phase_curve']
    ax2.plot(phases, curve, color=BLUE2, linewidth=1.5, marker='o', markersize=3)
    ax2.scatter(
        [phases[result['best_phase_idx']]], [result['best_metric']],
        color=RED2, zorder=5, s=40,
    )
    ax2.set_xlabel('Ch2 phase [deg]', fontsize=axis_label_fontsize)
    ax2.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    _style_ax(ax2)

    return fig1, fig2


# ─────────────────────────────────────────────────────────────────────────────
# Criterion: flatness
# ─────────────────────────────────────────────────────────────────────────────

def _flatness_metric(spectra, cal_spectra, harmonic_indices, normalize):
    """
    Per-phase std of CE across the selected harmonic indices.
    Lower std = flatter comb.

    spectra          : (P, N, K)
    cal_spectra      : (P, K) or None
    harmonic_indices : list of int, column indices into axis 1

    Returns (P,) array of std values.
    """
    peak_dbm = spectra[:, harmonic_indices, :].max(axis=-1)   # (P, M)

    if normalize in (True, 'percent'):
        if cal_spectra is None:
            raise RuntimeError(
                "normalize requires cal_spectra, which is absent from this grid point."
            )
        cal_dbm = cal_spectra.max(axis=-1)[:, np.newaxis]     # (P, 1)
        dbc = peak_dbm - cal_dbm
        ce = 10.0 ** (dbc / 10.0) * 100.0 if normalize == 'percent' else dbc
    else:
        ce = peak_dbm

    return ce.std(axis=-1)   # (P,)


def find_flattest_comb(meta, grid_data, order_min: int, order_max: int, normalize):
    """
    Find the (ch1 power, ch2 power, ch2 phase) triple that minimises the
    std of CE across harmonic orders in [order_min, order_max] (inclusive).

    Returns the same dict structure as find_max_harmonic, with best_metric
    being the minimum std (lower = flatter). Also includes:
        flatness_orders  – list of harmonic orders actually evaluated
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    n1, n2 = len(ch1_powers), len(ch2_powers)

    metric_grid     = np.full((n1, n2), np.nan)
    best_phase_grid = np.full((n1, n2), np.nan)

    best_val = np.inf   # lower std = better
    best_i = best_j = best_phase_idx = None
    best_phase_curve = None
    best_phases_arr  = None
    best_d           = None
    flatness_orders  = None

    for (i, j), d in grid_data.items():
        harmonics = d['harmonics'].astype(int)
        in_range  = [k for k, n in enumerate(harmonics)
                     if order_min <= int(n) <= order_max]
        if len(in_range) < 2:
            print(f"  Grid ({i},{j}): fewer than 2 harmonics in "
                  f"[{order_min}, {order_max}]; skipping.")
            continue

        if flatness_orders is None:
            flatness_orders = [int(harmonics[k]) for k in in_range]

        spectra     = d['spectra']
        cal_spectra = d.get('cal_spectra', None)
        phases      = d['ch2_phases_deg']

        metric = _flatness_metric(spectra, cal_spectra, in_range, normalize)   # (P,)
        best_p = int(np.argmin(metric))   # minimum std = flattest

        metric_grid[i, j]     = float(metric[best_p])
        best_phase_grid[i, j] = float(phases[best_p])

        if metric[best_p] < best_val:
            best_val         = float(metric[best_p])
            best_i           = i
            best_j           = j
            best_phase_idx   = best_p
            best_phase_curve = metric
            best_phases_arr  = phases
            best_d           = d

    if best_i is None:
        raise RuntimeError("No valid grid points found.")

    # Per-harmonic CE at the winning point (all recorded harmonics).
    best_spectra   = best_d['spectra'][best_phase_idx]   # (N, K)
    best_cal       = best_d.get('cal_spectra', None)
    best_peaks_dbm = best_spectra.max(axis=-1)
    if normalize in (True, 'percent') and best_cal is not None:
        cal_peak = float(best_cal[best_phase_idx].max())
        dbc = best_peaks_dbm - cal_peak
        best_comb_ce = 10.0 ** (dbc / 10.0) * 100.0 if normalize == 'percent' else dbc
    else:
        best_comb_ce = best_peaks_dbm

    return {
        'best_i':          best_i,
        'best_j':          best_j,
        'best_phase_idx':  best_phase_idx,
        'best_phase_deg':  float(best_phases_arr[best_phase_idx]),
        'best_metric':     best_val,
        'best_ch1_power':  float(ch1_powers[best_i]),
        'best_ch2_power':  float(ch2_powers[best_j]),
        'metric_grid':     metric_grid,
        'best_phase_grid': best_phase_grid,
        'phase_curve':     best_phase_curve,
        'ch2_phases_deg':  best_phases_arr,
        'best_comb_ce':    best_comb_ce,
        'best_harmonics':  best_d['harmonics'].astype(int),
        'flatness_orders': flatness_orders,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Criterion: dark_window
# ─────────────────────────────────────────────────────────────────────────────

def _dark_window_metric(spectra, cal_spectra, harmonic_indices, normalize):
    """
    Per-phase maximum CE across the specified harmonic indices.
    Lower max = deeper dark window (all listed sidebands more suppressed).

    spectra          : (P, N, K)
    cal_spectra      : (P, K) or None
    harmonic_indices : list of int, column indices into axis 1

    Returns (P,) array of max values.
    """
    peak_dbm = spectra[:, harmonic_indices, :].max(axis=-1)   # (P, len(indices))

    if normalize in (True, 'percent'):
        if cal_spectra is None:
            raise RuntimeError(
                "normalize requires cal_spectra, which is absent from this grid point."
            )
        cal_dbm = cal_spectra.max(axis=-1)[:, np.newaxis]     # (P, 1)
        dbc = peak_dbm - cal_dbm
        ce = 10.0 ** (dbc / 10.0) * 100.0 if normalize == 'percent' else dbc
    else:
        ce = peak_dbm

    return ce.max(axis=-1)   # (P,) — worst sideband at each phase


def find_dark_window(meta, grid_data, dark_orders, normalize):
    """
    Find the (ch1 power, ch2 power, ch2 phase) triple that minimises the
    maximum CE across the specified harmonic orders (minimax suppression).

    Returns the same dict structure as find_max_harmonic, with best_metric
    being the minimum of the per-phase maximum CE across dark_orders.
    Also includes:
        dark_orders  – list of harmonic orders actually matched in the data
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    n1, n2 = len(ch1_powers), len(ch2_powers)

    metric_grid     = np.full((n1, n2), np.nan)
    best_phase_grid = np.full((n1, n2), np.nan)

    best_val = np.inf
    best_i = best_j = best_phase_idx = None
    best_phase_curve = None
    best_phases_arr  = None
    best_d           = None
    actual_orders    = None

    target_set = set(int(o) for o in dark_orders)

    for (i, j), d in grid_data.items():
        harmonics = d['harmonics'].astype(int)
        in_window = [k for k, n in enumerate(harmonics) if int(n) in target_set]
        if not in_window:
            print(f"  Grid ({i},{j}): none of orders {dark_orders} found; skipping.")
            continue

        if actual_orders is None:
            actual_orders = [int(harmonics[k]) for k in in_window]

        spectra     = d['spectra']
        cal_spectra = d.get('cal_spectra', None)
        phases      = d['ch2_phases_deg']

        metric = _dark_window_metric(spectra, cal_spectra, in_window, normalize)
        best_p = int(np.argmin(metric))

        metric_grid[i, j]     = float(metric[best_p])
        best_phase_grid[i, j] = float(phases[best_p])

        if metric[best_p] < best_val:
            best_val         = float(metric[best_p])
            best_i           = i
            best_j           = j
            best_phase_idx   = best_p
            best_phase_curve = metric
            best_phases_arr  = phases
            best_d           = d

    if best_i is None:
        raise RuntimeError("No valid grid points found.")

    best_spectra   = best_d['spectra'][best_phase_idx]
    best_cal       = best_d.get('cal_spectra', None)
    best_peaks_dbm = best_spectra.max(axis=-1)
    if normalize in (True, 'percent') and best_cal is not None:
        cal_peak = float(best_cal[best_phase_idx].max())
        dbc = best_peaks_dbm - cal_peak
        best_comb_ce = 10.0 ** (dbc / 10.0) * 100.0 if normalize == 'percent' else dbc
    else:
        best_comb_ce = best_peaks_dbm

    return {
        'best_i':          best_i,
        'best_j':          best_j,
        'best_phase_idx':  best_phase_idx,
        'best_phase_deg':  float(best_phases_arr[best_phase_idx]),
        'best_metric':     best_val,
        'best_ch1_power':  float(ch1_powers[best_i]),
        'best_ch2_power':  float(ch2_powers[best_j]),
        'metric_grid':     metric_grid,
        'best_phase_grid': best_phase_grid,
        'phase_curve':     best_phase_curve,
        'ch2_phases_deg':  best_phases_arr,
        'best_comb_ce':    best_comb_ce,
        'best_harmonics':  best_d['harmonics'].astype(int),
        'dark_orders':     actual_orders,
    }


def plot_dark_window(result: dict, meta: dict, normalize):
    """
    Figure 1 — heatmap of the best achievable max-CE per grid point (min over
               phase). Lower = deeper window. Best point marked with a red cross.
    Figure 2 — max-CE vs ch2 phase at the best grid point (minimum marked).
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    unit = '% of carrier' if normalize == 'percent' else ('dBc' if normalize else 'dBm')
    orders = result.get('dark_orders', [])

    mm = 1.0 / 25.4
    fig1, ax1 = plt.subplots(figsize=(90 * mm, 75 * mm))
    fig1.subplots_adjust(left=0.18, right=0.88, bottom=0.15, top=0.93)

    n1, n2 = len(ch1_powers), len(ch2_powers)
    if n1 > 1 and n2 > 1:
        dx = (ch1_powers[-1] - ch1_powers[0]) / (n1 - 1) / 2
        dy = (ch2_powers[-1] - ch2_powers[0]) / (n2 - 1) / 2
        extent = [ch1_powers[0] - dx, ch1_powers[-1] + dx,
                  ch2_powers[0] - dy, ch2_powers[-1] + dy]
    else:
        extent = [ch1_powers[0], ch1_powers[-1],
                  ch2_powers[0], ch2_powers[-1]]

    im = ax1.imshow(
        result['metric_grid'].T,
        origin='lower', aspect='auto', extent=extent, cmap='viridis_r',
    )
    ax1.scatter(
        [result['best_ch1_power']], [result['best_ch2_power']],
        color='red', marker='x', s=80, linewidths=1.5, zorder=5,
    )
    cb = plt.colorbar(im, ax=ax1)
    cb.set_label(f'Max CE in window [{unit}]  (lower = darker)', fontsize=tick_label_fontsize)
    cb.ax.tick_params(labelsize=tick_label_fontsize)
    ax1.set_xlabel('Ch1 power [dBm]', fontsize=axis_label_fontsize)
    ax1.set_ylabel('Ch2 power [dBm]', fontsize=axis_label_fontsize)
    _style_ax(ax1)

    fig2, ax2 = _make_figure()
    phases = result['ch2_phases_deg']
    curve  = result['phase_curve']
    ax2.plot(phases, curve, color=BLUE2, linewidth=1.5, marker='o', markersize=3)
    ax2.scatter(
        [phases[result['best_phase_idx']]], [result['best_metric']],
        color=RED2, zorder=5, s=40,
    )
    ax2.set_xlabel('Ch2 phase [deg]', fontsize=axis_label_fontsize)
    ax2.set_ylabel(f'Max CE in window [{unit}]  (orders {orders})', fontsize=axis_label_fontsize)
    _style_ax(ax2)

    return fig1, fig2


def plot_flattest_comb(result: dict, meta: dict, normalize):
    """
    Figure 1 — heatmap of best achievable std per grid point (min over phase).
               Lower = flatter. Best point marked with a red cross.
    Figure 2 — std vs ch2 phase at the best grid point (minimum marked).
    """
    ch1_powers = meta['ch1_powers_dbm']
    ch2_powers = meta['ch2_powers_dbm']
    unit = '% of carrier' if normalize == 'percent' else ('dBc' if normalize else 'dBm')
    orders = result.get('flatness_orders', [])

    # ── Figure 1: heatmap (reversed colormap: darker = flatter = better) ──
    mm = 1.0 / 25.4
    fig1, ax1 = plt.subplots(figsize=(90 * mm, 75 * mm))
    fig1.subplots_adjust(left=0.18, right=0.88, bottom=0.15, top=0.93)

    n1, n2 = len(ch1_powers), len(ch2_powers)
    if n1 > 1 and n2 > 1:
        dx = (ch1_powers[-1] - ch1_powers[0]) / (n1 - 1) / 2
        dy = (ch2_powers[-1] - ch2_powers[0]) / (n2 - 1) / 2
        extent = [ch1_powers[0] - dx, ch1_powers[-1] + dx,
                  ch2_powers[0] - dy, ch2_powers[-1] + dy]
    else:
        extent = [ch1_powers[0], ch1_powers[-1],
                  ch2_powers[0], ch2_powers[-1]]

    im = ax1.imshow(
        result['metric_grid'].T,
        origin='lower', aspect='auto', extent=extent, cmap='viridis_r',
    )
    ax1.scatter(
        [result['best_ch1_power']], [result['best_ch2_power']],
        color='red', marker='x', s=80, linewidths=1.5, zorder=5,
    )
    cb = plt.colorbar(im, ax=ax1)
    cb.set_label(f'CE std [{unit}]  (lower = flatter)', fontsize=tick_label_fontsize)
    cb.ax.tick_params(labelsize=tick_label_fontsize)
    ax1.set_xlabel('Ch1 power [dBm]', fontsize=axis_label_fontsize)
    ax1.set_ylabel('Ch2 power [dBm]', fontsize=axis_label_fontsize)
    _style_ax(ax1)

    # ── Figure 2: std vs phase at best grid point ─────────────────────────
    fig2, ax2 = _make_figure()
    phases = result['ch2_phases_deg']
    curve  = result['phase_curve']
    ax2.plot(phases, curve, color=BLUE2, linewidth=1.5, marker='o', markersize=3)
    ax2.scatter(
        [phases[result['best_phase_idx']]], [result['best_metric']],
        color=RED2, zorder=5, s=40,
    )
    ax2.set_xlabel('Ch2 phase [deg]', fontsize=axis_label_fontsize)
    ax2.set_ylabel(f'CE std [{unit}]  (orders {orders})', fontsize=axis_label_fontsize)
    _style_ax(ax2)

    return fig1, fig2


def plot_comb_bars(result: dict, normalize):
    """
    Bar chart of CE vs harmonic order at the best identified comb point.
    Bars are colored by harmonic order using the standard scheme.
    """
    harmonics = result['best_harmonics']
    ce        = result['best_comb_ce']
    if normalize is True:
        ce     = -ce
        ylabel = 'CE [dB below carrier]'
    elif normalize == 'percent':
        ylabel = 'CE [% of carrier]'
    else:
        ylabel = 'Peak power [dBm]'

    fig, ax = _make_figure()
    for n, val in zip(harmonics, ce):
        ax.bar(int(n), val, color=_harmonic_color(int(n)), width=0.7,
               edgecolor='none', zorder=2)

    ax.set_xlabel('Harmonic order', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xticks(harmonics)
    _style_ax(ax)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    meta, grid_data = load_grid(FOLDER)

    ch1 = meta['ch1_powers_dbm']
    ch2 = meta['ch2_powers_dbm']
    print(f"Loaded: {FOLDER}")
    print(f"  Grid size  : {len(ch1)} × {len(ch2)}  ({len(grid_data)} point(s))")
    print(f"  Ch1 powers : {ch1[0]:+.1f}" + (f" – {ch1[-1]:+.1f}" if len(ch1) > 1 else "") + " dBm")
    print(f"  Ch2 powers : {ch2[0]:+.1f}" + (f" – {ch2[-1]:+.1f}" if len(ch2) > 1 else "") + " dBm")
    print(f"  Harmonics  : {list(meta['harmonics'])}")
    print(f"  Criterion  : {CRITERION},  normalize={NORMALIZE!r}\n")

    unit = '% of carrier' if NORMALIZE == 'percent' else ('dBc' if NORMALIZE else 'dBm')

    if CRITERION == 'max_harmonic':
        result = find_max_harmonic(meta, grid_data, TARGET_HARMONIC, NORMALIZE)

        print(f"Best comb for maximum n={TARGET_HARMONIC} power:")
        print(f"  Ch1 power : {result['best_ch1_power']:+.2f} dBm  (grid i={result['best_i']})")
        print(f"  Ch2 power : {result['best_ch2_power']:+.2f} dBm  (grid j={result['best_j']})")
        print(f"  Ch2 phase : {result['best_phase_deg']:.1f} deg  (k={result['best_phase_idx']})")
        print(f"  Metric    : {result['best_metric']:.3f} {unit}")

        plot_max_harmonic(result, meta, NORMALIZE, TARGET_HARMONIC)
        plot_comb_bars(result, NORMALIZE)

    elif CRITERION == 'flatness':
        result = find_flattest_comb(
            meta, grid_data, FLATNESS_ORDER_MIN, FLATNESS_ORDER_MAX, NORMALIZE)

        print(f"Flattest comb over orders {result['flatness_orders']}:")
        print(f"  Ch1 power : {result['best_ch1_power']:+.2f} dBm  (grid i={result['best_i']})")
        print(f"  Ch2 power : {result['best_ch2_power']:+.2f} dBm  (grid j={result['best_j']})")
        print(f"  Ch2 phase : {result['best_phase_deg']:.1f} deg  (k={result['best_phase_idx']})")
        print(f"  CE std    : {result['best_metric']:.3f} {unit}")

        plot_flattest_comb(result, meta, NORMALIZE)
        plot_comb_bars(result, NORMALIZE)

    elif CRITERION == 'dark_window':
        result = find_dark_window(meta, grid_data, DARK_WINDOW_ORDERS, NORMALIZE)

        print(f"Darkest window for orders {result['dark_orders']}:")
        print(f"  Ch1 power : {result['best_ch1_power']:+.2f} dBm  (grid i={result['best_i']})")
        print(f"  Ch2 power : {result['best_ch2_power']:+.2f} dBm  (grid j={result['best_j']})")
        print(f"  Ch2 phase : {result['best_phase_deg']:.1f} deg  (k={result['best_phase_idx']})")
        print(f"  Max CE    : {result['best_metric']:.3f} {unit}")

        plot_dark_window(result, meta, NORMALIZE)
        plot_comb_bars(result, NORMALIZE)

    else:
        raise ValueError(f"Unknown criterion: {CRITERION!r}")

    plt.show()


if __name__ == '__main__':
    main()
