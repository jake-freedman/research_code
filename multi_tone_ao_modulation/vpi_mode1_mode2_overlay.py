"""
vpi_mode1_mode2_overlay.py

Per-device 2D overlay plots from a folder of bnc_cw_harmonic_esa_script.py
heterodyne-sweep output files, covering several devices (waveguide 'a' or
'b', numbered p1-p6) each measured twice: once around its mode-1 resonance,
once around its mode-2 resonance (mode-3 files, if present, are ignored).

For each device this produces one 2D figure overlaying:
  - the mode-1 curve, with its frequency axis DOUBLED
  - the mode-2 curve, as measured
so both land in the same displayed frequency window -- the same "double the
lower resonance's frequency axis" trick used for the two-peaks-in-one-sweep
insets in vpi_vs_freq_3d.py, just applied across two separate sweep files
instead of two peaks found within one continuous sweep.

Filenames are expected to look like either:
    p{N}{a|b}_mode{M}_heterodyne_sweep_<timestamp>.npz
    e.g. p3a_mode1_heterodyne_sweep_2026-07-27-15-30-58.npz
    -- no wafer/die label; all such files are treated as one shared,
    "unlabeled" wafer+die group (e.g. an original single-wafer dataset) --
or, for devices spanning multiple wafers/dies in the same folder:
    w{W}_d{D}_wg{G}_p{N}{a|b}_mode{M}_heterodyne_sweep_<timestamp>.npz
    e.g. w1_d3-2_wg5_p1a_mode1_heterodyne_sweep_2026-07-29-13-09-21.npz

Every distinct (wafer+die, waveguide) combo gets its OWN color family (a
light->dark gradient of one hue), cycling through COLOR_FAMILIES below --
GREEN2->dark-green and LIGHTBLUE2->DARKBLUE2 are the first two (so an
unlabeled/original single-wafer dataset's 'a'/'b' devices keep their
original look), followed by VIOLET2, ORANGE2, PINK2, etc. for additional
combos. Devices within one combo (different p-number/wg) shade from that
family's light anchor to its dark end. The curve STROKES are always black
regardless of waveguide/device/wafer/die: solid for mode 2, dashed for
mode 1 -- so combo (fill color) and mode (line dash pattern) are each
encoded by a single, consistent visual channel. Set SEPARATE_BY_WAVEGUIDE =
False to instead group/color by wafer+die ALONE, so a wafer+die's 'a' and
'b' devices share one row/color (still told apart by their own label).
"""

import os
import re
import sys
import colorsys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from heterodyne_sweep_data import HeterodyneSweepData
from path_utils import local_path
from graphics import (
    GREEN2, LIGHTBLUE2, DARKBLUE2, VIOLET2, ORANGE2, PINK2, RED2, TAN2, BEIGE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\yield_characterization"   # <- set this to your folder

# Filenames must match either:
#   p{N}{a|b}_mode{M}_..., e.g. 'p3a_mode1_heterodyne_sweep_...npz'
#     (no wafer/die label -- all such files are treated as one shared,
#     unlabeled wafer+die group)
#   w{W}_d{D}_wg{G}_p{N}{a|b}_mode{M}_..., e.g.
#     'w1_d3-2_wg5_p1a_mode1_heterodyne_sweep_...npz'
FILENAME_RE = re.compile(
    r'^(?:w(?P<wafer>\d+)_d(?P<die>[^_]+)_wg(?P<wg>\d+)_)?'
    r'p(?P<p_num>\d+)(?P<waveguide>[ab])_mode(?P<mode>\d+)_.*\.npz$',
    re.IGNORECASE)

# ── extraction ────────────────────────────────────────────────────────────────
HARMONIC_NUMERATOR   = 1
HARMONIC_DENOMINATOR = 0
BETA_GUESS = 1.0

# VNA/BNC drive power in dBm, used to convert beta -> Ppi. bnc_cw_harmonic_esa_script.py
# doesn't save this into the .npz, so it must be supplied here. Single value
# applied to every device for now.
DRIVE_POWER_DBM = 5.0

# 'dBm' or 'mW': plots P_pi; a resonance is a real dip, so the axis is
# flipped to display it as a peak (see _flip_axis).
# 'V': plots 1/V_pi instead; a resonance is already a real peak, no flip.
POWER_PI_UNIT = 'dBm'
# ─────────────────────────────────────────────────────────────────────────────

# ── smoothing ─────────────────────────────────────────────────────────────────
# None -> no smoothing; int -> moving average over that many points;
# float -> uniform average over all points within +-SMOOTH/2 GHz.
SMOOTH = 10
# ─────────────────────────────────────────────────────────────────────────────

# ── per-device y-limit overrides ────────────────────────────────────────────────
# Override YLIM for specific devices only. Keys are device labels in the same
# format used for titles/filenames ('w{wafer}_d{die}_wg{wg}_p{p_num}{a|b}' for
# labeled devices, or plain 'p{p_num}{a|b}' for unlabeled ones), e.g.
# 'w7_d21_wg5_p5b'. Values are (ymin, ymax) tuples, same convention as YLIM
# (auto-flipped for POWER_PI_UNIT != 'V'), or None to fall back to that
# device's own auto range (ignoring YLIM too). Devices not listed here use
# YLIM as usual.
DEVICE_YLIM_OVERRIDE = {
    'w7_d21_wg5_p1a': (29, 66),
    'w7_d21_wg5_p3a': (29, 66),
    'w7_d21_wg5_p5a': (29, 66),
    'w7_d21_wg5_p1b': (29, 66),
}
# ─────────────────────────────────────────────────────────────────────────────

# ── grouping ──────────────────────────────────────────────────────────────────
# True  -> group/color by (wafer+die, waveguide), same as always: 'a' and 'b'
#          devices at the same wafer+die get their OWN row/color each.
# False -> group/color by wafer+die ONLY: 'a' and 'b' devices at the same
#          wafer+die share one row and one color (still distinguishable from
#          each other by their label/title, and mode1 vs mode2 is still solid
#          vs dashed regardless). Devices with no wafer/die label (the
#          "unlabeled" group) become a single shared row/color too, rather
#          than one row for 'a' and one for 'b'.
SEPARATE_BY_WAVEGUIDE = False
# ─────────────────────────────────────────────────────────────────────────────

# ── device fill color scheme ──────────────────────────────────────────────────
# Every distinct row-group (wafer+die+waveguide, or just wafer+die --
# depending on SEPARATE_BY_WAVEGUIDE above) -- i.e. every row of the
# combined grid figure -- is assigned its OWN color family from this list, in
# row order (cycling back to FAMILIES[0] if there are more combos than
# entries here). Each family is a (light, dark) hex pair; devices within one
# combo (different p-number/wg) shade from that family's light anchor (first
# device in the row) to its dark end (last device in the row). dark = None
# auto-derives a darker, more saturated shade of the same hue (see
# _auto_dark). GREEN2 and LIGHTBLUE2/DARKBLUE2 are the first two families, so
# an unlabeled/original single-wafer dataset's 'a' and 'b' devices keep using
# this script's original colors.
COLOR_FAMILIES = [
    (GREEN2, '#3C5928'),
    (LIGHTBLUE2, DARKBLUE2),
    (VIOLET2, None),
    (ORANGE2, None),
    (PINK2, None),
    ('#F26767', None),
    (TAN2, None),
    (BEIGE2, None),
]

COLOR_FAMILIES = [
    (GREEN2, GREEN2),
    (LIGHTBLUE2, LIGHTBLUE2),
    (VIOLET2, VIOLET2),
    (ORANGE2, ORANGE2),
    (PINK2, PINK2),
    (RED2, RED2),
    (TAN2, None),
    (BEIGE2, None),
]
# ─────────────────────────────────────────────────────────────────────────────

# ── curve style ───────────────────────────────────────────────────────────────
# Curve strokes are always black, distinguished by linestyle: solid for
# mode 2, dashed for mode 1.
MODE1_LINE_COLOR = '#000000'
MODE1_LINESTYLE  = '--'
MODE2_LINE_COLOR = '#000000'
MODE2_LINESTYLE  = '-'
LINEWIDTH         = 2.0
LINE_ALPHA        = 1.0
MARKER             = None
MARKERSIZE         = 4.0
MARKER_FACE_ALPHA  = 1.0
MARKER_EDGE_COLOR  = 'same'   # 'same' = match that curve's own stroke color
MARKER_EDGE_ALPHA  = 1.0

SHOW_FILL           = True
FILL_ALPHA          = 0.35    # where only one curve (mode1 or mode2) covers a point
FILL_ALPHA_OVERLAP  = 0.8     # where both curves cover the same point
FILL_N_COLS         = 300     # horizontal resolution of the fill raster
FILL_ZORDER_OFFSET  = -1      # relative to the curve zorder

ZORDER_BASE  = 2
SHOW_GRID    = False
SHOW_LEGEND  = True

XLIM = (2.11, 2.41)   # (xmin, xmax) or None = auto per device (union of both curves)
YLIM = (14, 51)   # (ymin, ymax) or None = auto per device
# ─────────────────────────────────────────────────────────────────────────────

# ── figure size (mm) ──────────────────────────────────────────────────────────
axes_width_mm  = 21.5
axes_height_mm = 14.5
left_mm        = 20.0
right_mm       = 8.0
bottom_mm      = 16.0
top_mm         = 8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes title/axis/tick labels and the legend, and saves each
# device's figure as an SVG into SAVE_FOLDER (one file per device, unaffected
# by GRID_ROWS_BY_COLOR below).
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
# ─────────────────────────────────────────────────────────────────────────────

# ── combined (non-publication) grid figure ────────────────────────────────────
# The regular, labeled per-device plots (everything EXCEPT the "for
# publication" SVGs above) are combined into a single figure laid out as a
# grid: one ROW per distinct wafer+die+waveguide color group, one COLUMN per
# device within that group (sorted by wg then p-number). Rows/groups with
# fewer devices than the widest row simply leave the remaining columns blank.
SAVE_GRID_PNG  = True     # also save the combined grid as a PNG into SAVE_FOLDER
GRID_PNG_NAME  = 'vpi_mode1_mode2_overlay_grid.png'
GRID_PNG_DPI   = 300
# ─────────────────────────────────────────────────────────────────────────────

# ── best-Ppi frequency-alignment scatter ────────────────────────────────────────
# For every device, find the frequency (within its own mode1 sweep, and
# separately its own mode2 sweep) at which Ppi is minimized -- i.e. its
# resonance -- then plot one point per device, each colored exactly like
# that device's own line plot. Since the rest of this script assumes a
# device's mode2 resonance sits at twice its mode1 resonance, this is a QC
# check for that assumption.
SHOW_BEST_FREQ_SCATTER = True

# True  -> x = 2*f1_best (mode1 frequency doubled, same convention as every
#          other plot in this script), y = f2_best. Perfect alignment is the
#          y = x line -- x and y are on the same numeric footing, so
#          BEST_FREQ_SHOW_DIAGONAL/_BAND force a shared square axis range
#          (see BEST_FREQ_AXES_WIDTH_MM/HEIGHT_MM below) so that line
#          actually renders at 45 degrees.
# False -> x = f1_best RAW (not doubled), y = f2_best. Perfect alignment is
#          now the y = 2x line instead -- x and y are on genuinely different
#          numeric scales (y ~ 2x), so the axes are NOT forced square; each
#          keeps its own independent range (BEST_FREQ_XLIM/YLIM if set, else
#          auto).
BEST_FREQ_DOUBLE_MODE1 = False

# When True: skip every per-device figure/SVG AND the combined grid entirely
# -- only the scatter above (and its publication SVG, if FOR_PUBLICATION) is
# built. Per-device data is still loaded as usual (the scatter needs it).
SCATTER_ONLY = False

BEST_FREQ_MARKER            = 'o'
BEST_FREQ_MARKERSIZE        = 8.0
BEST_FREQ_MARKER_FACE_ALPHA = 0.7
BEST_FREQ_MARKER_EDGE_COLOR = '#000000'   # 'same' = match that point's own face color
BEST_FREQ_MARKER_EDGE_ALPHA = 1.0
BEST_FREQ_MARKER_EDGE_WIDTH = 1.00        # points

BEST_FREQ_SHOW_DIAGONAL  = True    # reference y = x line (perfect mode1/mode2 alignment)
BEST_FREQ_DIAGONAL_COLOR = '#000000'
BEST_FREQ_DIAGONAL_ALPHA = 1.00
BEST_FREQ_DIAGONAL_STYLE = '--'
BEST_FREQ_DIAGONAL_WIDTH = 2.0
BEST_FREQ_DIAGONAL_ZORDER = 1

# Shaded tolerance band around the reference line itself (y=x, or y=2x when
# BEST_FREQ_DOUBLE_MODE1=False): {(x,y) : |y - slope*x| <= half-width}, i.e.
# +/- half-width MHz in y at every x.
BEST_FREQ_SHOW_DIAGONAL_BAND        = True
BEST_FREQ_DIAGONAL_BAND_HALFWIDTH_MHZ = 40.0
BEST_FREQ_DIAGONAL_BAND_COLOR        = '#777777'
BEST_FREQ_DIAGONAL_BAND_ALPHA        = 0.2
BEST_FREQ_DIAGONAL_BAND_ZORDER       = 0   # below the diagonal line (1) and points (3)

BEST_FREQ_ZORDER     = 3    # scatter points; diagonal is BEST_FREQ_DIAGONAL_ZORDER
BEST_FREQ_SHOW_GRID   = False
BEST_FREQ_SHOW_LEGEND = False   # off by default -- one entry per device gets cluttered fast

BEST_FREQ_XLIM = (0.5*2.22, 0.5*2.37)   # (xmin, xmax) or None = auto (union of all points + diagonal)
BEST_FREQ_YLIM = (2.22, 2.37)   # (ymin, ymax) or None = auto

# Square by default (unlike the 100x40 default elsewhere) so a perfect
# alignment diagonal actually renders at 45 degrees.
BEST_FREQ_AXES_WIDTH_MM  = 40.0
BEST_FREQ_AXES_HEIGHT_MM = 48.261
BEST_FREQ_LEFT_MM   = 20.0
BEST_FREQ_RIGHT_MM  = 8.0
BEST_FREQ_BOTTOM_MM = 16.0
BEST_FREQ_TOP_MM    = 8.0

BEST_FREQ_SVG_NAME = 'best_ppi_freq_alignment.svg'   # saved into SAVE_FOLDER when FOR_PUBLICATION
# ─────────────────────────────────────────────────────────────────────────────

_LOG20 = 10.0 * np.log10(20.0)


def _smooth(freqs_ghz, y, smooth):
    if smooth is None:
        return y.copy()
    if isinstance(smooth, int):
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(y, size=smooth, mode='nearest')
    hw = float(smooth) / 2.0
    out = np.empty_like(y)
    for i, fc in enumerate(freqs_ghz):
        mask = np.abs(freqs_ghz - fc) <= hw
        out[i] = y[mask].mean()
    return out


def _flip_axis():
    return POWER_PI_UNIT != 'V'


def _zlabel():
    if POWER_PI_UNIT == 'dBm':
        return r'$P_\pi$ [dBm]'
    elif POWER_PI_UNIT == 'mW':
        return r'$P_\pi$ [mW]'
    elif POWER_PI_UNIT == 'V':
        return r'$1/V_\pi$ [1/V]'
    raise ValueError(f"POWER_PI_UNIT must be 'dBm', 'mW', or 'V', got {POWER_PI_UNIT!r}.")


def _betas_to_y(betas, drive_power_dbm):
    v_rms = 10.0 ** ((drive_power_dbm - _LOG20) / 20.0)
    vpi = np.pi * v_rms / betas
    if POWER_PI_UNIT == 'dBm':
        return 20.0 * np.log10(vpi) + _LOG20
    elif POWER_PI_UNIT == 'mW':
        return vpi ** 2 * 20.0
    elif POWER_PI_UNIT == 'V':
        return 1.0 / vpi
    raise ValueError(f"POWER_PI_UNIT must be 'dBm', 'mW', or 'V', got {POWER_PI_UNIT!r}.")


def _load_mode_data(path):
    """Load one heterodyne sweep file -> (freqs_ghz, y) in POWER_PI_UNIT,
    smoothed per SMOOTH."""
    data = HeterodyneSweepData.from_file(local_path(path))
    betas = data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR, BETA_GUESS)
    freqs_ghz = data.cw_freqs / 1e9
    y = _betas_to_y(betas, DRIVE_POWER_DBM)
    return freqs_ghz, _smooth(freqs_ghz, y, SMOOTH)


def _best_ppi_freq(freqs_ghz, y):
    """Frequency (from freqs_ghz) at which Ppi is minimized. y IS Ppi for
    POWER_PI_UNIT 'dBm'/'mW' (lower = better -> argmin); y = 1/Vpi for 'V'
    (higher = better -> argmax) -- same sense as _flip_axis()."""
    idx = np.argmin(y) if _flip_axis() else np.argmax(y)
    return float(freqs_ghz[idx])


def _discover_devices(folder):
    """
    Scan folder for files matching FILENAME_RE. Returns
    {(wafer, die, wg, p_num, waveguide): {1: path, 2: path}}, keeping only
    mode 1 and mode 2 (mode 3, if present, is dropped). wafer/die/wg are
    None for files with no wafer/die label.
    """
    devices = {}
    for p in sorted(Path(folder).glob('*.npz')):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        mode = int(m.group('mode'))
        if mode not in (1, 2):
            continue
        p_num = int(m.group('p_num'))
        waveguide = m.group('waveguide').lower()
        wafer = m.group('wafer')
        die = m.group('die')
        wg = m.group('wg')
        devices.setdefault((wafer, die, wg, p_num, waveguide), {})[mode] = str(p)
    return devices


def _row_key(key):
    """(wafer_die, waveguide) grouping key for one device key tuple --
    every device sharing a row_key is drawn in the same color family. When
    SEPARATE_BY_WAVEGUIDE is False, waveguide is replaced with a constant
    placeholder so 'a' and 'b' devices at the same wafer+die collapse into
    one shared row_key (and therefore one shared row/color)."""
    wafer, die, wg, p_num, waveguide = key
    wafer_die = (wafer, die) if wafer is not None else None
    return (wafer_die, waveguide if SEPARATE_BY_WAVEGUIDE else None)


def _row_sort_key(row_key):
    wafer_die, waveguide = row_key
    wafer, die = wafer_die if wafer_die is not None else (None, None)
    return (wafer is not None, wafer or '', die or '', waveguide)


def _row_order(devices):
    """Sorted list of distinct (wafer_die, waveguide) row keys -- the
    unlabeled group's rows always sort first, so an original single-wafer
    dataset keeps using the first two entries of COLOR_FAMILIES."""
    return sorted({_row_key(k) for k in devices}, key=_row_sort_key)


def _auto_dark(hex_color):
    """Derive a darker, more saturated shade of the same hue -- used for any
    COLOR_FAMILIES entry that doesn't specify its own dark endpoint."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, min(1.0, max(s, 0.5)), max(0.2, v * 0.45))


_FAMILY_CMAPS = [
    mcolors.LinearSegmentedColormap.from_list(
        f'family{i}', [light, dark if dark is not None else _auto_dark(light)])
    for i, (light, dark) in enumerate(COLOR_FAMILIES)
]


def _device_color(row_index, idx_in_row, n_in_row):
    """idx_in_row=0 -> that row's family's light anchor; idx_in_row=n_in_row-1
    -> its dark end (single-device rows get the light anchor exactly)."""
    cmap = _FAMILY_CMAPS[row_index % len(_FAMILY_CMAPS)]
    frac = idx_in_row / max(1, n_in_row - 1)
    return cmap(frac)


def _make_fig():
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )
    return fig, ax


def _make_grid_fig(n_rows, n_cols):
    """One figure containing an n_rows x n_cols array of axes, each sized
    and margined exactly like a standalone _make_fig() panel (so combining
    panels into a grid doesn't change any individual panel's look). Row 0
    is placed at the top."""
    cell_w_mm = left_mm + axes_width_mm + right_mm
    cell_h_mm = bottom_mm + axes_height_mm + top_mm
    fig_w_mm = cell_w_mm * n_cols
    fig_h_mm = cell_h_mm * n_rows
    mm = 1.0 / 25.4
    fig = plt.figure(figsize=(fig_w_mm * mm, fig_h_mm * mm))
    axes = [[None] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            x0 = (j * cell_w_mm + left_mm) / fig_w_mm
            y0 = 1.0 - (i * cell_h_mm + top_mm + axes_height_mm) / fig_h_mm
            w = axes_width_mm / fig_w_mm
            h = axes_height_mm / fig_h_mm
            axes[i][j] = fig.add_axes([x0, y0, w, h])
    return fig, axes


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)


def _draw_fill_overlap(ax, f1, y1, f2, y2, fill_color, x_lo, x_hi, y_bot, y_top):
    """Fill toward the baseline (off-resonance) end of the y-axis for both
    curves, in one flat fill_color -- FILL_ALPHA where only one curve covers
    a given point, FILL_ALPHA_OVERLAP where both do (same recipe as
    vpi_vs_freq_3d.py's _draw_inset_fill_overlap)."""
    n_rows = 256
    x_grid = np.linspace(x_lo, x_hi, FILL_N_COLS)
    y_vals = np.linspace(y_bot, y_top, n_rows)

    y1_grid = np.interp(x_grid, f1, y1)
    in_domain1 = (x_grid >= f1.min()) & (x_grid <= f1.max())
    y2_grid = np.interp(x_grid, f2, y2)
    in_domain2 = (x_grid >= f2.min()) & (x_grid <= f2.max())

    if _flip_axis():
        covers1 = y_vals[:, None] >= y1_grid[None, :]
        covers2 = y_vals[:, None] >= y2_grid[None, :]
    else:
        covers1 = y_vals[:, None] <= y1_grid[None, :]
        covers2 = y_vals[:, None] <= y2_grid[None, :]
    above1 = in_domain1[None, :] & covers1
    above2 = in_domain2[None, :] & covers2
    n_covering = above1.astype(int) + above2.astype(int)

    fill_rgba = np.zeros((n_rows, FILL_N_COLS, 4))
    fill_rgba[:, :, :3] = mcolors.to_rgb(fill_color)
    fill_rgba[:, :, 3] = np.select(
        [n_covering == 2, n_covering == 1],
        [FILL_ALPHA_OVERLAP, FILL_ALPHA],
        default=0.0,
    )
    ax.imshow(fill_rgba, aspect='auto', extent=[x_lo, x_hi, y_bot, y_top],
              origin='lower', interpolation='nearest', zorder=ZORDER_BASE + FILL_ZORDER_OFFSET)


def _draw_device_curves(ax, freqs1_ghz, y1, freqs2_ghz, y2, color):
    """Draw the fill + mode1/mode2 curves for one device onto ax. Returns
    the device's own natural (x_lo, x_hi, y_bot, y_top) bounds."""
    freqs1_doubled = freqs1_ghz * 2.0

    x_lo = min(freqs1_doubled.min(), freqs2_ghz.min())
    x_hi = max(freqs1_doubled.max(), freqs2_ghz.max())
    y_all = np.concatenate([y1, y2])
    y_lo = float(y_all.min())
    y_hi = float(y_all.max())
    y_pad = (y_hi - y_lo) * 0.05 if y_hi != y_lo else 0.1
    y_bot, y_top = y_lo - y_pad, y_hi + y_pad

    if SHOW_FILL:
        _draw_fill_overlap(ax, freqs1_doubled, y1, freqs2_ghz, y2, color, x_lo, x_hi, y_bot, y_top)

    def _marker_kwargs(stroke_color):
        if MARKER is None:
            return {}
        edge_color = stroke_color if MARKER_EDGE_COLOR == 'same' else MARKER_EDGE_COLOR
        return {
            'marker': MARKER, 'markersize': MARKERSIZE,
            'markerfacecolor': mcolors.to_rgba(stroke_color, MARKER_FACE_ALPHA),
            'markeredgecolor': mcolors.to_rgba(edge_color, MARKER_EDGE_ALPHA),
        }

    mode1_rgba = mcolors.to_rgba(MODE1_LINE_COLOR, LINE_ALPHA)
    mode2_rgba = mcolors.to_rgba(MODE2_LINE_COLOR, LINE_ALPHA)
    ax.plot(freqs1_doubled, y1, color=mode1_rgba, linestyle=MODE1_LINESTYLE, linewidth=LINEWIDTH,
            solid_capstyle='round', dash_capstyle='round', zorder=ZORDER_BASE,
            label='mode 1 (x2 freq)', **_marker_kwargs(MODE1_LINE_COLOR))
    ax.plot(freqs2_ghz, y2, color=mode2_rgba, linestyle=MODE2_LINESTYLE, linewidth=LINEWIDTH,
            solid_capstyle='round', zorder=ZORDER_BASE,
            label='mode 2', **_marker_kwargs(MODE2_LINE_COLOR))

    return x_lo, x_hi, y_bot, y_top


def _finish_device_axes(ax, label, x_lo, x_hi, y_bot, y_top, show_labels):
    if XLIM is not None:
        ax.set_xlim(XLIM)
    else:
        ax.set_xlim(x_lo, x_hi)

    ylim = DEVICE_YLIM_OVERRIDE.get(label, YLIM)
    if ylim is not None:
        ax.set_ylim(ylim if not _flip_axis() else (ylim[1], ylim[0]))
    else:
        ax.set_ylim((y_top, y_bot) if _flip_axis() else (y_bot, y_top))

    ax.grid(SHOW_GRID)
    _style_ax(ax)

    if show_labels:
        ax.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel(_zlabel(), fontsize=axis_label_fontsize)
        ax.set_title(label, fontsize=axis_label_fontsize)
        if SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)


def _build_device_figure(label, freqs1_ghz, y1, freqs2_ghz, y2, color, show_labels):
    fig, ax = _make_fig()
    bounds = _draw_device_curves(ax, freqs1_ghz, y1, freqs2_ghz, y2, color)
    _finish_device_axes(ax, label, *bounds, show_labels)
    return fig, ax


def _build_grid_figure(loaded, row_order):
    """Combine every loaded (non-publication) device panel into one figure:
    one row per distinct (wafer_die, waveguide) color group (in row_order),
    one column per device within that group. Rows narrower than the widest
    row leave their remaining columns blank."""
    rows = {rk: [] for rk in row_order}
    for d in loaded:
        rows[d['row_key']].append(d)
    for devs in rows.values():
        devs.sort(key=lambda d: d['sort_key'])
    row_keys = [rk for rk in row_order if rows[rk]]

    n_rows = len(row_keys)
    n_cols = max(len(rows[rk]) for rk in row_keys)
    fig, axes = _make_grid_fig(n_rows, n_cols)

    for i, rk in enumerate(row_keys):
        row_devices = rows[rk]
        for j in range(n_cols):
            ax = axes[i][j]
            if j < len(row_devices):
                d = row_devices[j]
                bounds = _draw_device_curves(ax, d['freqs1_ghz'], d['y1'],
                                              d['freqs2_ghz'], d['y2'], d['color'])
                _finish_device_axes(ax, d['label'], *bounds, show_labels=True)
            else:
                ax.set_visible(False)

    return fig


def _build_best_freq_scatter(entries, show_labels):
    """One point per device at (2*f1_best, f2_best) -- or (f1_best, f2_best)
    if BEST_FREQ_DOUBLE_MODE1 is False -- each in that device's own color.
    See SHOW_BEST_FREQ_SCATTER."""
    fig_w = BEST_FREQ_LEFT_MM + BEST_FREQ_AXES_WIDTH_MM + BEST_FREQ_RIGHT_MM
    fig_h = BEST_FREQ_BOTTOM_MM + BEST_FREQ_AXES_HEIGHT_MM + BEST_FREQ_TOP_MM
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = BEST_FREQ_LEFT_MM   / fig_w,
        right  = 1 - BEST_FREQ_RIGHT_MM  / fig_w,
        bottom = BEST_FREQ_BOTTOM_MM / fig_h,
        top    = 1 - BEST_FREQ_TOP_MM    / fig_h,
    )

    for e in entries:
        face = e['color']
        edge = face if BEST_FREQ_MARKER_EDGE_COLOR == 'same' else BEST_FREQ_MARKER_EDGE_COLOR
        ax.plot(e['x'], e['y'], marker=BEST_FREQ_MARKER, linestyle='none',
                markersize=BEST_FREQ_MARKERSIZE,
                markerfacecolor=mcolors.to_rgba(face, BEST_FREQ_MARKER_FACE_ALPHA),
                markeredgecolor=mcolors.to_rgba(edge, BEST_FREQ_MARKER_EDGE_ALPHA),
                markeredgewidth=BEST_FREQ_MARKER_EDGE_WIDTH,
                zorder=BEST_FREQ_ZORDER, label=e['label'])

    if BEST_FREQ_XLIM is not None:
        ax.set_xlim(BEST_FREQ_XLIM)
    if BEST_FREQ_YLIM is not None:
        ax.set_ylim(BEST_FREQ_YLIM)

    slope = 1.0 if BEST_FREQ_DOUBLE_MODE1 else 2.0

    if BEST_FREQ_SHOW_DIAGONAL or BEST_FREQ_SHOW_DIAGONAL_BAND:
        if BEST_FREQ_DOUBLE_MODE1:
            # x (2*f1_best) and y (f2_best) are on the same numeric footing
            # -- force a shared square range so a perfect-alignment y=x line
            # actually renders at 45 degrees.
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            x_line = np.array([lo, hi])
        else:
            # x (raw f1_best) and y (f2_best) are on genuinely different
            # numeric scales (y ~ 2x) -- keep each axis's own range.
            x_line = np.array(ax.get_xlim())
        y_line = slope * x_line

        if BEST_FREQ_SHOW_DIAGONAL_BAND:
            band_hw = BEST_FREQ_DIAGONAL_BAND_HALFWIDTH_MHZ / 1000.0   # MHz -> GHz
            ax.fill_between(x_line, y_line - band_hw, y_line + band_hw,
                             facecolor=mcolors.to_rgba(BEST_FREQ_DIAGONAL_BAND_COLOR, BEST_FREQ_DIAGONAL_BAND_ALPHA),
                             edgecolor='none', zorder=BEST_FREQ_DIAGONAL_BAND_ZORDER)

        if BEST_FREQ_SHOW_DIAGONAL:
            ax.plot(x_line, y_line,
                    color=mcolors.to_rgba(BEST_FREQ_DIAGONAL_COLOR, BEST_FREQ_DIAGONAL_ALPHA),
                    linestyle=BEST_FREQ_DIAGONAL_STYLE, linewidth=BEST_FREQ_DIAGONAL_WIDTH,
                    solid_capstyle='round', dash_capstyle='round', zorder=BEST_FREQ_DIAGONAL_ZORDER)

        if BEST_FREQ_DOUBLE_MODE1:
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

    ax.grid(BEST_FREQ_SHOW_GRID)
    _style_ax(ax)

    if show_labels:
        xlabel = ('2 x mode1 best-Ppi frequency [GHz]' if BEST_FREQ_DOUBLE_MODE1
                  else 'mode1 best-Ppi frequency [GHz]')
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel('mode2 best-Ppi frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_title('Mode alignment check', fontsize=axis_label_fontsize)
        if BEST_FREQ_SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def main():
    folder = local_path(DATA_FOLDER)
    devices = _discover_devices(folder)
    if not devices:
        raise FileNotFoundError(
            f"No files matching p{{N}}{{a|b}}_mode{{M}}_...npz found in {folder!r}."
        )

    print(f"Found {len(devices)} device(s) in {folder}")
    row_order = _row_order(devices)
    n_families = len(COLOR_FAMILIES)
    grouping_desc = ("wafer+die+waveguide" if SEPARATE_BY_WAVEGUIDE else "wafer+die")
    print(f"  {len(row_order)} distinct {grouping_desc} color group(s) "
          f"-> {min(len(row_order), n_families)} of {n_families} color families used"
          + (" (cycling back to the start)" if len(row_order) > n_families else ""))

    keys_by_row = {rk: [] for rk in row_order}
    for key in devices:
        keys_by_row[_row_key(key)].append(key)
    for keys_in_row in keys_by_row.values():
        # waveguide, wg, p_num -- waveguide first so a merged (wafer+die-only)
        # row still shows all 'a' devices grouped before all 'b' devices.
        keys_in_row.sort(key=lambda k: (k[4], k[2] or '0', k[3]))

    loaded = []
    all_labels = set()
    for row_index, rk in enumerate(row_order):
        keys_in_row = keys_by_row[rk]
        n_in_row = len(keys_in_row)
        for idx, key in enumerate(keys_in_row):
            wafer, die, wg, p_num, waveguide = key
            modes = devices[key]
            if wafer is not None:
                label = f'w{wafer}_d{die}_wg{wg}_p{p_num}{waveguide}'
            else:
                label = f'p{p_num}{waveguide}'
            if 1 not in modes or 2 not in modes:
                missing = [m for m in (1, 2) if m not in modes]
                print(f"  Warning: {label} is missing mode{missing}; skipping.")
                continue

            all_labels.add(label)
            freqs1_ghz, y1 = _load_mode_data(modes[1])
            freqs2_ghz, y2 = _load_mode_data(modes[2])
            color = _device_color(row_index, idx, n_in_row)
            f1_best = _best_ppi_freq(freqs1_ghz, y1)
            f2_best = _best_ppi_freq(freqs2_ghz, y2)

            print(f"  {label}: mode1 {freqs1_ghz[0]:.4f}-{freqs1_ghz[-1]:.4f} GHz "
                  f"(x2 -> {2*freqs1_ghz[0]:.4f}-{2*freqs1_ghz[-1]:.4f} GHz), "
                  f"mode2 {freqs2_ghz[0]:.4f}-{freqs2_ghz[-1]:.4f} GHz")
            print(f"    Best-Ppi: mode1 {f1_best:.4f} GHz (x2 -> {2*f1_best:.4f} GHz), "
                  f"mode2 {f2_best:.4f} GHz")
            if label in DEVICE_YLIM_OVERRIDE:
                print(f"    Using YLIM override: {DEVICE_YLIM_OVERRIDE[label]}")

            if FOR_PUBLICATION and not SCATTER_ONLY:
                fig_pub, _ax_pub = _build_device_figure(label, freqs1_ghz, y1,
                                                          freqs2_ghz, y2, color, show_labels=False)
                svg_path = os.path.join(SAVE_FOLDER, f'{label}_mode_overlay.svg')
                fig_pub.savefig(svg_path, format='svg', bbox_inches='tight')
                print(f"    Saved: {svg_path}")
                plt.close(fig_pub)

            loaded.append({
                'row_key': rk,
                'sort_key': (waveguide, wg or '0', p_num),
                'label': label,
                'freqs1_ghz': freqs1_ghz, 'y1': y1,
                'freqs2_ghz': freqs2_ghz, 'y2': y2,
                'color': color,
                'f1_best_ghz': f1_best, 'f2_best_ghz': f2_best,
            })

    unmatched = set(DEVICE_YLIM_OVERRIDE) - all_labels
    if unmatched:
        print(f"  Warning: DEVICE_YLIM_OVERRIDE has label(s) with no matching discovered "
              f"device (check for typos): {sorted(unmatched)}")

    if loaded and not SCATTER_ONLY:
        grid_fig = _build_grid_figure(loaded, row_order)
        if SAVE_GRID_PNG:
            png_path = os.path.join(SAVE_FOLDER, GRID_PNG_NAME)
            grid_fig.savefig(png_path, format='png', dpi=GRID_PNG_DPI, bbox_inches='tight')
            print(f"  Saved combined grid: {png_path}")

    if loaded and SHOW_BEST_FREQ_SCATTER:
        entries = [
            {'x': (2.0 * d['f1_best_ghz'] if BEST_FREQ_DOUBLE_MODE1 else d['f1_best_ghz']),
             'y': d['f2_best_ghz'],
             'color': d['color'], 'label': d['label']}
            for d in loaded
        ]
        _build_best_freq_scatter(entries, show_labels=True)
        if FOR_PUBLICATION:
            fig_bf_pub, _ax_bf_pub = _build_best_freq_scatter(entries, show_labels=False)
            svg_path = os.path.join(SAVE_FOLDER, BEST_FREQ_SVG_NAME)
            fig_bf_pub.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"  Saved best-Ppi frequency alignment plot: {svg_path}")
            plt.close(fig_bf_pub)

    plt.show()


if __name__ == '__main__':
    main()
