"""
vpi_vs_freq_3d.py

3D companion to heterodyne_modulation_plot.py: shows multiple devices' half-
wave RF power (P_pi) vs. drive frequency curves on the same 3D axes --
frequency on x, device number on y (mostly into the page), P_pi on z.

For now there is only one measured heterodyne sweep file. To preview what
the multi-device comparison will look like once real per-device data exists,
that single file is copied N_DEVICES times, each copy's frequency axis
shifted by an independent random offset drawn uniformly from
+-FREQ_SHIFT_RANGE_MHZ/2 (the modulation-depth values themselves are
untouched -- only their frequency labels move).

Curves further back (higher device index, or reversed via REVERSE_DEPTH) are
drawn with continuously decreasing opacity relative to the front-most curve,
as a simple depth cue.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from heterodyne_sweep_data import HeterodyneSweepData
from path_utils import local_path
from graphics import (
    RED2, VIOLET2, GREEN2, BEIGE2, LIGHTBLUE2, ORANGE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

# ── data ────────────────────────────────────────────────────────────────────────
# A single file path -> the synthetic multi-device preview below (one file
# copied N_DEVICES times with a random frequency shift each).
# A folder path, or a list/tuple of file paths -> each .npz is treated as one
# real device's own sweep (folder contents sorted by filename), and the
# synthetic generation is skipped entirely. Files need not share the same
# frequency grid or number of points.
DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\paper_data\long_frequency_vpi"
# DATA_FILE = r"...\folder_of_device_sweeps"
# DATA_FILE = [
#     r"...\device1_heterodyne_sweep.npz",
#     r"...\device2_heterodyne_sweep.npz",
# ]

# ── synthetic multi-device generation (DATA_FILE = a single path only) ───────
# Temporary stand-in until real per-device sweep files exist: the single
# DATA_FILE sweep is copied N_DEVICES times, each copy's frequency axis
# shifted by its own random offset drawn uniformly from
# +-FREQ_SHIFT_RANGE_MHZ/2 (the beta/Ppi values themselves are untouched).
N_DEVICES            = 4
FREQ_SHIFT_RANGE_MHZ = 20.0
RANDOM_SEED          = 0     # None = nondeterministic
# ─────────────────────────────────────────────────────────────────────────────

# ── extraction ────────────────────────────────────────────────────────────────
HARMONIC_NUMERATOR   = 1
HARMONIC_DENOMINATOR = 0
BETA_GUESS = 1.0

# VNA drive power in dBm, used to convert beta -> Vpi/Ppi. Either a single
# value (used for every device), or a list -- one value per device, in the
# same order as the *reordered* display curves (same convention as
# DEVICE_LABELS: DRIVE_POWER_DBM[0] is the drive power of whichever sweep
# DEVICE_ORDER[0] selects, etc; if DEVICE_ORDER is None, list order is load
# order instead). Real per-device measurements (DATA_FILE = a folder/list)
# may genuinely have been taken at different drive powers; for the
# single-file synthetic preview, a list instead previews what varying drive
# power (rather than frequency shift) would do to the same measured curve.
DRIVE_POWER_DBM = [5,5,10]
# 'dBm' or 'mW': plots P_pi (RF power for a pi shift); a resonance is a real
# dip, so the value axis is flipped to display it as a peak (see
# _flip_axis).
# 'V': plots 1/V_pi (inverse RMS voltage) instead of V_pi itself, so a
# resonance is already a real peak -- the value axis is NOT flipped.
POWER_PI_UNIT   = 'dBm'
# ─────────────────────────────────────────────────────────────────────────────

# ── smoothing ─────────────────────────────────────────────────────────────────
# None -> no smoothing; int -> moving average over that many points;
# float -> uniform average over all points within +-SMOOTH/2 GHz.
SMOOTH = 20
# ─────────────────────────────────────────────────────────────────────────────

# ── axis limits ───────────────────────────────────────────────────────────────
XMIN = 0.5   # GHz, None = auto
XMAX = 3.5
ZMIN = None   # power_pi units, None = auto
ZMAX = None

DEVICE_LABELS = None   # list of N_DEVICES strings; None = "Device 1", "Device 2", ...

# Selects and reorders which loaded sweeps appear along the y (device) axis,
# without touching how each was loaded. A list of indices into the loaded
# devices: DEVICE_ORDER[0] is shown at y=0 (front, unless REVERSE_DEPTH),
# DEVICE_ORDER[1] at y=1, etc. Any loaded index NOT listed is simply omitted
# from the plot entirely -- it need not be a full permutation of 0..N-1.
# None = keep everything, in load order. E.g. [2, 0] shows only the 3rd- and
# 1st-loaded sweeps (in that order), dropping the rest.
DEVICE_ORDER = [1, 4, 0]
# ─────────────────────────────────────────────────────────────────────────────

# ── line style / coloring (same scheme as heterodyne_modulation_plot.py) ─────
# The line is drawn in COLOR everywhere, but smoothly transitions to a
# per-band highlight colour within ±HIGHLIGHT_BWS[i]/2 of each
# HIGHLIGHT_CENTERS[i] (cosine rolloff back to COLOR at the band edges).
# Applied per device using that device's own (frequency-shifted) x-axis, so
# each device's highlight dip lands wherever its shifted curve crosses that
# absolute frequency. Pass empty lists to disable and use a flat COLOR.
COLOR = '#cccccc'
HIGHLIGHT_CENTERS = [1.146, 2.290, 2.613, 2.990, 3.265]      # GHz
HIGHLIGHT_BWS     = [0.15, 0.15, 0.15, 0.15, 0.15]           # GHz full-width
HIGHLIGHT_COLORS  = [RED2, LIGHTBLUE2, VIOLET2, GREEN2, BEIGE2]

LINESTYLE  = '-'
LINEWIDTH  = 2.0

# When True, the curve line itself is drawn as a uniform solid color at
# UNIFORM_LINE_WIDTH (device depth-alpha still applies), ignoring
# COLOR/HIGHLIGHT_* entirely for the line -- the fill below is unaffected
# either way, and keeps using the blended per-x coloring above.
SHOW_UNIFORM_LINE  = True
UNIFORM_LINE_COLOR = '#000000'
UNIFORM_LINE_WIDTH = 1.0

MARKER            = None
MARKERSIZE        = 4.0
MARKER_FACECOLOR  = None        # None = match each point's blended color
MARKER_FACE_ALPHA = 1.0
MARKER_EDGECOLOR  = None        # None = match each point's blended color
MARKER_EDGE_ALPHA = 1.0
# ─────────────────────────────────────────────────────────────────────────────

# ── fill ───────────────────────────────────────────────────────────────────────
# COLORLESS_PLOT = True: fills the region between each device's curve and a
# flat baseline at the bottom of the (flipped) z-axis (i.e. toward high P_pi)
# with a single flat color + alpha -- exactly one vector polygon
# (Poly3DCollection) per device, so it stays lightweight in an exported SVG
# no matter how many frequency points the sweep has (uses the flat COLOR,
# not the per-frequency highlight blend).
# COLORLESS_PLOT = False: the original per-height opacity ramp (FILL_ALPHA
# below FILL_DECAY_START, decaying to FILL_ALPHA_TOP at the top of the axes)
# and per-frequency highlight-color blend (COLOR + HIGHLIGHT_*), rendered as
# a FILL_N_ROWS x FILL_N_COLS mesh -- much heavier in an exported SVG (one
# polygon per mesh cell), but with the full gradient/color look.
# Either way, scaled by each device's own depth alpha.
COLORLESS_PLOT = True

SHOW_FILL  = True
FILL_ALPHA = 0.5   # opacity of the flat fill (COLORLESS_PLOT=True), before depth-alpha
FILL_ZORDER_OFFSET = -1   # relative to each device's own line zorder

# COLORLESS_PLOT = False only:
FILL_ALPHA_TOP   = 0.2    # alpha at the top of the axes
FILL_DECAY_START = 40     # z-value (actual P_pi units) at which alpha begins to decay
FILL_N_ROWS      = 32     # vertical resolution of the fill mesh
FILL_N_COLS      = 150    # horizontal (frequency) resolution of the fill mesh
# ─────────────────────────────────────────────────────────────────────────────

# ── depth fade ────────────────────────────────────────────────────────────────
# Curves further back are drawn more transparent than curves in front. Front
# = device index 0 unless REVERSE_DEPTH is True (useful if your VIEW_AZIM
# happens to put index 0 away from the viewer instead).
FRONT_ALPHA   = 1.00
BACK_ALPHA    = 0.15
DEPTH_ORDER   = 0.8     # ramp shape t**DEPTH_ORDER, t: 0 (front) to 1 (back); 1 = linear
REVERSE_DEPTH = False

# 3D rendering order is mostly auto depth-sorted by matplotlib itself; this
# only breaks ties between curves that end up at very similar depth.
ZORDER_BASE = 2
# ─────────────────────────────────────────────────────────────────────────────

# ── 3D view ───────────────────────────────────────────────────────────────────
VIEW_ELEV, VIEW_AZIM, VIEW_ROLL = 29, -70, 0 # 32 -85 5
BOX_ASPECT = (4, 3, 1.0)   # relative (x, y, z) box proportions

# True -> orthographic (no perspective foreshortening, parallel projection);
# False -> matplotlib's default perspective projection.
ORTHOGRAPHIC = True

SHOW_GRID   = False
SHOW_LEGEND = False

# Text on the axes: off by default since with many devices they mostly just
# clutter the plot. Independent of FOR_PUBLICATION, which always strips both
# regardless of these (same convention as every other script here).
SHOW_TICK_LABELS = False   # the numbers/device names on x/y/z ticks
SHOW_AXIS_LABELS = False   # "Drive frequency [GHz]", the z-axis label

# Axes "walls" (panes): leave unfilled (just the edges) rather than the
# default light-gray fill.
FILL_PANES          = False
PANE_EDGE_COLOR     = '#888888'
PANE_EDGE_LINEWIDTH = 0.8
# ─────────────────────────────────────────────────────────────────────────────

# ── figure size (mm) ──────────────────────────────────────────────────────────
axes_width_mm  = 130.0
axes_height_mm = 90.0
left_mm        = 15.0
right_mm       = 15.0
bottom_mm      = 15.0
top_mm         = 10.0
# ─────────────────────────────────────────────────────────────────────────────

# ── per-device peak insets ────────────────────────────────────────────────────
# For each device, finds its first two prominent resonance peaks (a "peak" =
# local minimum in real P_pi, since a deep resonance reads as a peak on the
# flipped 3D display) and builds one small standalone 2D figure per device
# containing both: the 2nd peak plotted at its own frequency, and the 1st
# peak overlaid with its frequency axis DOUBLED so it lands in the same
# window as the 2nd -- useful for comparing a fundamental resonance against
# its second-order counterpart (same "doubled" trick as
# heterodyne_modulation_plot.py's zoom plot). Both curves within one inset
# are solid black, and share a single fill color -- but that color differs
# device to device: INSET_FILL_COLORS[i] is used for device i's inset (i in
# display order), cycling if there are more devices than colors. Devices
# with fewer than 2 detected peaks are skipped (with a console warning).
SHOW_PEAK_INSETS = True

INSET_PEAK_PROMINENCE          = 10.0    # scipy find_peaks prominence, P_pi axis units
INSET_MIN_PEAK_SEPARATION_MHZ  = 100.0   # minimum spacing enforced between accepted peaks
INSET_HALF_WIDTH_MHZ           = 40.0   # inset window = peak freq +- this (doubled for the 1st peak)

# If both are set, this fixed [INSET_OVERRIDE_XMIN, INSET_OVERRIDE_XMAX] GHz
# span replaces the per-device auto-computed one (from each device's own
# peak positions +- INSET_HALF_WIDTH_MHZ) for *every* inset -- useful for
# comparing insets across devices on one shared frequency scale. The two
# curves are still windowed/doubled so they each fill this span entirely
# (same "fill the whole width" behavior as the auto-computed case). Peak
# detection itself is unaffected; only the displayed span is overridden, so
# a poorly chosen span may not contain the peaks at all. None (either) =
# auto per-device, as before.
INSET_OVERRIDE_XMIN = 2.15   # GHz
INSET_OVERRIDE_XMAX = 2.4   # GHz

# A single color -> both curves (1st peak, 2nd peak) drawn in that same
# color. A 2-element list/tuple -> INSET_LINE_COLOR[0] for the 1st-peak
# curve, INSET_LINE_COLOR[1] for the 2nd-peak curve, per inset.
INSET_LINE_COLOR = '#000000'
INSET_LINEWIDTH  = 2.0
INSET_LINESTYLE  = '-'

INSET_FILL_COLORS = [LIGHTBLUE2, GREEN2, VIOLET2, ORANGE2]   # one color per device, cycling
# Both curves' fills share one flat color (INSET_FILL_COLORS[i]) but use two
# different opacities depending on whether only one curve covers a given
# point or both do -- not the natural alpha-compositing result of stacking
# two semi-transparent layers, but an explicit override for each case (so
# the overlap can be made to stand out more, or less, than plain compositing
# would give).
INSET_FILL_ALPHA         = 0.5   # where only one curve's fill covers a point
INSET_FILL_ALPHA_OVERLAP = 0.9   # where both curves' fills cover the same point
INSET_FILL_N_COLS        = 300   # horizontal resolution of the combined-fill raster

INSET_YMIN = 16   # P_pi units, None = auto per device
INSET_YMAX = 42
INSET_SHOW_GRID   = False
INSET_SHOW_LEGEND = True

INSET_AXES_WIDTH_MM  = 23.0
INSET_AXES_HEIGHT_MM = 12.0
INSET_LEFT_MM   = 20.0
INSET_RIGHT_MM  = 10.0
INSET_BOTTOM_MM = 16.0
INSET_TOP_MM    = 8.0

INSET_PUBLICATION_SVG_NAME = 'vpi_vs_freq_3d_inset.svg'   # a device-name suffix is appended
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels and the legend, and saves an SVG to
# SAVE_FOLDER (in addition to the normal figure that's still shown/PNG-saved).
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUBLICATION_SVG_NAME = 'vpi_vs_freq_3d.svg'
# ─────────────────────────────────────────────────────────────────────────────

_LOG20 = 10.0 * np.log10(20.0)


def _smooth(freqs_ghz, y, smooth):
    if smooth is None:
        return y.copy()
    if isinstance(smooth, int):
        return uniform_filter1d(y, size=smooth, mode='nearest')
    # float: frequency window in GHz -- average all points within ±smooth/2
    hw = float(smooth) / 2.0
    out = np.empty_like(y)
    for i, fc in enumerate(freqs_ghz):
        mask = np.abs(freqs_ghz - fc) <= hw
        out[i] = y[mask].mean()
    return out


def _point_colors(freqs_ghz, base_color, centers, bws, h_colors):
    """
    Return an (N, 3) RGB array blending base_color toward each highlight
    colour using a cosine-like rolloff inside each band (same recipe as
    heterodyne_modulation_plot.py). Overlapping bands are resolved by a
    weighted average of the highlight colours, clamped so total blend <= 1.
    """
    base_rgb = np.array(mcolors.to_rgb(base_color))
    N = len(freqs_ghz)

    if not centers:
        return np.tile(base_rgb, (N, 1))

    B = len(centers)
    weights = np.zeros((N, B))
    for b, (c, bw) in enumerate(zip(centers, bws)):
        df = np.abs(freqs_ghz - c)
        weights[:, b] = 1.0 / (1.0 + (2.0 * df / bw) ** 2)

    band_rgb = np.array([mcolors.to_rgb(c) for c in h_colors])  # (B, 3)
    w_sum = weights.sum(axis=1)                                   # (N,)
    total_w = np.clip(w_sum, 0.0, 1.0)

    safe_sum = np.where(w_sum > 0, w_sum, 1.0)
    highlight_rgb = (weights[:, :, None] * band_rgb[None, :, :]).sum(axis=1) / safe_sum[:, None]

    return (1.0 - total_w[:, None]) * base_rgb + total_w[:, None] * highlight_rgb


def _make_fig_ax_3d(w_mm, h_mm):
    fig_w = left_mm + w_mm + right_mm
    fig_h = bottom_mm + h_mm + top_mm
    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))
    ax = fig.add_subplot(111, projection='3d')
    # By default Axes3D computes each artist's draw order from its actual 3D
    # depth, which ignores our explicit zorder and can put the (thin) line at
    # the same depth as the (broad) fill surface beneath it -- sometimes
    # drawing the fill on top. Disabling this makes 3D artists draw in plain
    # zorder order instead, like a normal 2D axes, so fill-behind-line via
    # zorder actually holds.
    ax.computed_zorder = False
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )
    return fig, ax


def _style_panes(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if not FILL_PANES:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(PANE_EDGE_COLOR)
        axis.pane.set_linewidth(PANE_EDGE_LINEWIDTH)


def _depth_alpha(n_devices):
    """(n_devices,) per-device alpha: front (index 0) -> FRONT_ALPHA, back
    (index n_devices-1) -> BACK_ALPHA, ramping as t**DEPTH_ORDER."""
    if n_devices == 1:
        return np.array([FRONT_ALPHA])
    t = np.linspace(0.0, 1.0, n_devices)
    if REVERSE_DEPTH:
        t = 1.0 - t
    return FRONT_ALPHA + (t ** DEPTH_ORDER) * (BACK_ALPHA - FRONT_ALPHA)


def _draw_device_fill_flat(ax, freqs, curve_y, device_index, z_lo_axis, z_hi_axis,
                            alpha_device, zorder):
    """
    COLORLESS_PLOT = True. Fill the region between the curve and a flat
    baseline at the far (off-resonance) end of the value axis, with a single
    flat color + alpha. When _flip_axis() (P_pi: resonance is a dip), that's
    z_hi_axis, plotted negated so it lands at the bottom of the flipped axis;
    otherwise (1/V_pi: resonance is already a real peak, no flip) it's
    z_lo_axis, plotted as-is. Exactly one polygon (Poly3DCollection), so it
    stays cheap in an exported SVG regardless of point count; the curve's own
    full frequency resolution is still used for the polygon's top edge, so
    the fill hugs the real curve shape exactly.
    """
    flip = _flip_axis()
    sign = -1.0 if flip else 1.0
    z_baseline = z_hi_axis if flip else z_lo_axis

    y_val = float(device_index)
    top = np.column_stack([freqs, np.full_like(freqs, y_val), sign * curve_y])
    bottom = np.array([
        [freqs[-1], y_val, sign * z_baseline],
        [freqs[0],  y_val, sign * z_baseline],
    ])
    verts = np.vstack([top, bottom])

    poly = Poly3DCollection([verts], facecolor=mcolors.to_rgba(COLOR, FILL_ALPHA * alpha_device),
                             edgecolor='none', zorder=zorder)
    ax.add_collection3d(poly)


def _draw_device_fill_gradient(ax, freqs, curve_y, device_index, z_lo_axis, z_hi_axis,
                                alpha_device, zorder):
    """
    COLORLESS_PLOT = False. Fill the region between the curve and the far
    (off-resonance) end of the value axis with a per-height opacity ramp,
    colored the same per-x highlight blend as the line. This is a
    FILL_N_ROWS x FILL_N_COLS mesh (one flat-shaded quad per cell, since
    mplot3d can't interpolate color across a 3D face smoothly), so it's much
    heavier than _draw_device_fill_flat in an exported SVG.

    Everything here is computed in actual (non-negated) z units, then
    negated only at the very end for plotting when _flip_axis() -- see the
    z_plot note in _build_figure for why the axis itself is never inverted
    via set_zlim. FILL_DECAY_START/FILL_ALPHA_TOP always ramp away from the
    resonance end of the axis toward the far/baseline end, whichever end
    that is for the current unit.
    """
    flip = _flip_axis()
    col_idx = np.linspace(0, len(freqs) - 1, FILL_N_COLS).round().astype(int)
    fill_x = freqs[col_idx]
    curve_cols = curve_y[col_idx]

    row_z = np.linspace(z_lo_axis, z_hi_axis, FILL_N_ROWS)   # actual z units, low -> high
    if flip:
        # Baseline (off-resonance) end is z_hi_axis -- ramp toward it.
        y_span = max(z_hi_axis - FILL_DECAY_START, 1e-9)
        row_frac = np.clip((row_z - FILL_DECAY_START) / y_span, 0.0, 1.0)
    else:
        # Baseline (off-resonance) end is z_lo_axis -- ramp toward it instead.
        y_span = max(FILL_DECAY_START - z_lo_axis, 1e-9)
        row_frac = np.clip((FILL_DECAY_START - row_z) / y_span, 0.0, 1.0)
    row_alpha = FILL_ALPHA + (FILL_ALPHA_TOP - FILL_ALPHA) * row_frac

    row_z_c = 0.5 * (row_z[:-1] + row_z[1:])
    row_alpha_c = 0.5 * (row_alpha[:-1] + row_alpha[1:])
    curve_c = 0.5 * (curve_cols[:-1] + curve_cols[1:])
    # Filled from the curve toward the baseline end of the axis (z_hi_axis if
    # flipped, z_lo_axis otherwise), matching heterodyne_modulation_plot.py's
    # `y_vals >= y` convention when flipped (in real z units).
    above = (row_z_c[:, None] >= curve_c[None, :] if flip
             else row_z_c[:, None] <= curve_c[None, :])

    pt_colors = _point_colors(fill_x, COLOR, HIGHLIGHT_CENTERS, HIGHLIGHT_BWS, HIGHLIGHT_COLORS)
    rgb_c = 0.5 * (pt_colors[:-1] + pt_colors[1:])

    facecolors = np.zeros((FILL_N_ROWS - 1, FILL_N_COLS - 1, 4))
    facecolors[:, :, :3] = rgb_c[None, :, :]
    facecolors[:, :, 3] = np.where(above, row_alpha_c[:, None] * alpha_device, 0.0)

    sign = -1.0 if flip else 1.0
    X = np.tile(fill_x, (FILL_N_ROWS, 1))
    Y = np.full_like(X, float(device_index))
    Z = np.tile(sign * row_z[:, None], (1, FILL_N_COLS))   # see _build_figure

    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                     linewidth=0, edgecolor='none', antialiased=True,
                     shade=False, zorder=zorder)


def _draw_device_fill(ax, freqs, curve_y, device_index, z_lo_axis, z_hi_axis,
                       alpha_device, zorder):
    """Dispatch to the flat single-polygon fill (COLORLESS_PLOT=True) or the
    original gradient/highlight-color mesh fill (COLORLESS_PLOT=False)."""
    if COLORLESS_PLOT:
        _draw_device_fill_flat(ax, freqs, curve_y, device_index, z_lo_axis, z_hi_axis,
                                alpha_device, zorder)
    else:
        _draw_device_fill_gradient(ax, freqs, curve_y, device_index, z_lo_axis, z_hi_axis,
                                    alpha_device, zorder)


def _generate_synthetic_devices(freqs_ghz, betas, drive_powers):
    """
    Copy the single measured curve N_DEVICES times, each with its own random
    frequency-axis shift drawn from +-FREQ_SHIFT_RANGE_MHZ/2 (converted to
    GHz) and its own drive power (drive_powers[i], from _resolve_drive_powers
    -- identical for every copy unless DRIVE_POWER_DBM is a list). Returns
    (device_freqs, device_y, shifts_ghz): the first two are lists of
    N_DEVICES 1D arrays (same convention as _load_device_data, so both code
    paths in main() feed _build_figure identically).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    half_range_ghz = (FREQ_SHIFT_RANGE_MHZ / 1e3) / 2.0
    shifts_ghz = rng.uniform(-half_range_ghz, half_range_ghz, size=N_DEVICES)
    device_freqs = [freqs_ghz + shift for shift in shifts_ghz]
    device_y = [_smooth(freqs_ghz, _betas_to_y(betas, drive_powers[i]), SMOOTH)
                for i in range(N_DEVICES)]
    return device_freqs, device_y, shifts_ghz


def _flip_axis():
    """
    Whether the value axis (z in the main 3D plot, y in each inset) should be
    flipped so that resonances display as visual peaks.

    For 'dBm'/'mW' (P_pi), a resonance is a real local MINIMUM (P_pi dips at
    resonance), so the axis is flipped (data negated, axis limits left
    normal/ascending -- see the note in _build_figure) to display that dip as
    a peak. For 'V', 1/V_pi is plotted instead of V_pi itself, so a
    resonance is already a real local MAXIMUM (1/V_pi is large exactly where
    V_pi is small) -- no flip is needed, plotting it plainly already puts
    resonances at the top.
    """
    return POWER_PI_UNIT != 'V'


def _zlabel_and_name():
    """The z/y-axis label (LaTeX) and short quantity name (for print output)
    for the current POWER_PI_UNIT."""
    if POWER_PI_UNIT == 'dBm':
        return r'$P_\pi$ [dBm]', 'Ppi'
    elif POWER_PI_UNIT == 'mW':
        return r'$P_\pi$ [mW]', 'Ppi'
    elif POWER_PI_UNIT == 'V':
        return r'$1/V_\pi$ [1/V]', 'invVpi'
    raise ValueError(f"POWER_PI_UNIT must be 'dBm', 'mW', or 'V', got {POWER_PI_UNIT!r}.")


def _resolve_drive_powers(n_raw):
    """
    Resolve DRIVE_POWER_DBM into a list of n_raw values, one per raw-loaded
    device (in raw load order) -- i.e. the order _load_device_data/
    _generate_synthetic_devices are called in, *before* DEVICE_ORDER
    reorders or drops anything.

    A single DRIVE_POWER_DBM value is just repeated n_raw times. A list is
    given in *displayed* order (same convention as DEVICE_LABELS), so it's
    inverted through DEVICE_ORDER here: DRIVE_POWER_DBM[k] belongs to
    whichever raw index DEVICE_ORDER[k] selects. Raw indices DEVICE_ORDER
    doesn't list are dropped later by _apply_device_order anyway, so their
    placeholder value here is never actually used.
    """
    if not isinstance(DRIVE_POWER_DBM, (list, tuple)):
        return [DRIVE_POWER_DBM] * n_raw

    if DEVICE_ORDER is None:
        if len(DRIVE_POWER_DBM) != n_raw:
            raise ValueError(
                f"DRIVE_POWER_DBM has {len(DRIVE_POWER_DBM)} entries but "
                f"{n_raw} device(s) were loaded (DEVICE_ORDER is None, so "
                f"they must match 1:1 in load order)."
            )
        return list(DRIVE_POWER_DBM)

    order = list(DEVICE_ORDER)
    if len(DRIVE_POWER_DBM) != len(order):
        raise ValueError(
            f"DRIVE_POWER_DBM has {len(DRIVE_POWER_DBM)} entries but "
            f"DEVICE_ORDER has {len(order)} -- they must match 1:1, both in "
            f"the displayed curves' order."
        )
    raw_to_power = {raw_idx: DRIVE_POWER_DBM[k] for k, raw_idx in enumerate(order)}
    return [raw_to_power.get(i, DRIVE_POWER_DBM[0]) for i in range(n_raw)]


def _betas_from_file(path):
    """Load one heterodyne sweep file and return (freqs_ghz, betas) -- the
    raw modulation depth, before any drive-power conversion or smoothing."""
    data = HeterodyneSweepData.from_file(local_path(path))
    betas = data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR, BETA_GUESS)
    freqs_ghz = data.cw_freqs / 1e9
    return freqs_ghz, betas


def _betas_to_y(betas, drive_power_dbm):
    """Convert modulation-depth betas -> y in the current POWER_PI_UNIT,
    using the given VNA drive power (dBm)."""
    v_rms = 10.0 ** ((drive_power_dbm - _LOG20) / 20.0)
    vpi = np.pi * v_rms / betas
    if POWER_PI_UNIT == 'dBm':
        return 20.0 * np.log10(vpi) + _LOG20
    elif POWER_PI_UNIT == 'mW':
        return vpi ** 2 * 20.0
    elif POWER_PI_UNIT == 'V':
        return 1.0 / vpi
    raise ValueError(f"POWER_PI_UNIT must be 'dBm', 'mW', or 'V', got {POWER_PI_UNIT!r}.")


def _load_device_data(path, drive_power_dbm):
    """Load one heterodyne sweep file and return (freqs_ghz, y) in the
    current POWER_PI_UNIT, converted using drive_power_dbm and smoothed per
    SMOOTH -- one real device's data."""
    freqs_ghz, betas = _betas_from_file(path)
    y = _betas_to_y(betas, drive_power_dbm)
    return freqs_ghz, _smooth(freqs_ghz, y, SMOOTH)


def _apply_device_order(device_freqs, device_y, default_labels):
    """Select and reorder the per-device lists per DEVICE_ORDER (indices into
    the loaded devices; any index not listed is dropped). default_labels
    (auto filenames, if any) is carried along so each label still matches
    its own data after reordering/dropping."""
    if DEVICE_ORDER is None:
        return device_freqs, device_y, default_labels
    n = len(device_freqs)
    order = list(DEVICE_ORDER)
    if len(set(order)) != len(order):
        raise ValueError(f"DEVICE_ORDER has duplicate indices: {order!r}.")
    bad = [i for i in order if not (0 <= i < n)]
    if bad:
        raise ValueError(
            f"DEVICE_ORDER index/indices {bad!r} out of range "
            f"({n} loaded sweep(s): valid indices are 0..{n - 1})."
        )
    new_freqs = [device_freqs[i] for i in order]
    new_y = [device_y[i] for i in order]
    new_labels = [default_labels[i] for i in order] if default_labels is not None else None
    return new_freqs, new_y, new_labels


def _clip_devices_to_xlim(device_freqs, device_y):
    """Drop points outside [XMIN, XMAX] (whichever are set) from every
    device's data before anything is plotted or auto-scaled from it."""
    if XMIN is None and XMAX is None:
        return device_freqs, device_y
    lo = XMIN if XMIN is not None else -np.inf
    hi = XMAX if XMAX is not None else np.inf
    clipped_freqs, clipped_y = [], []
    for freqs, y in zip(device_freqs, device_y):
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            raise ValueError(
                f"XMIN={XMIN}, XMAX={XMAX} excludes all of a device's data "
                f"(its range is {freqs.min():.4f}-{freqs.max():.4f} GHz)."
            )
        clipped_freqs.append(freqs[mask])
        clipped_y.append(y[mask])
    return clipped_freqs, clipped_y


def _resolve_device_labels(n_devices, default_labels=None):
    return (DEVICE_LABELS if DEVICE_LABELS is not None
            else default_labels if default_labels is not None
            else [f'Device {i + 1}' for i in range(n_devices)])


def _build_figure(device_freqs, device_y, zlabel, show_labels: bool, default_labels=None):
    fig, ax = _make_fig_ax_3d(axes_width_mm, axes_height_mm)

    n_devices = len(device_freqs)
    alphas = _depth_alpha(n_devices)
    device_labels = _resolve_device_labels(n_devices, default_labels)

    z_lo = ZMIN if ZMIN is not None else min(a.min() for a in device_y)
    z_hi = ZMAX if ZMAX is not None else max(a.max() for a in device_y)

    # When _flip_axis() (P_pi: resonance is a real dip), the z-axis is
    # flipped (low P_pi at the top) by negating the plotted z data and using
    # a *normal* (non-inverted) set_zlim below, rather than by passing
    # set_zlim(z_hi, z_lo). Explicitly inverting the limits instead confuses
    # mplot3d's automatic pane placement (the x-axis/ticks jump from the
    # bottom edge to the middle of the plot) -- negating the data instead
    # keeps a normal zlim, so that placement stays put. Not user-visible: all
    # z tick labels are hidden anyway (SHOW_TICK_LABELS). When not
    # _flip_axis() (1/V_pi: resonance is already a real peak), the data and
    # zlim are both plotted plainly -- no flip needed.
    flip = _flip_axis()
    sign = -1.0 if flip else 1.0
    for i in range(n_devices):
        y_pos = np.full_like(device_freqs[i], float(i))
        # Tie-break only: draw the frontmost (least transparent) curves last.
        zorder = ZORDER_BASE + (i if REVERSE_DEPTH else (n_devices - i))

        if SHOW_FILL:
            _draw_device_fill(ax, device_freqs[i], device_y[i], i, z_lo, z_hi,
                               alphas[i], zorder + FILL_ZORDER_OFFSET)

        z_plot = sign * device_y[i]
        pt_colors = _point_colors(device_freqs[i], COLOR,
                                   HIGHLIGHT_CENTERS, HIGHLIGHT_BWS, HIGHLIGHT_COLORS)

        if SHOW_UNIFORM_LINE:
            # A single continuous line (one SVG path on export) instead of a
            # Line3DCollection of per-segment 2-point paths: every segment
            # would get the same color anyway here, so the per-segment
            # structure below is pure bloat -- thousands of separate <path>
            # elements per device in the FOR_PUBLICATION SVG, for a curve
            # that's visually just one solid line. Illustrator (and other
            # vector editors) have to import/hold each as a distinct object.
            ax.plot(device_freqs[i], y_pos, z_plot,
                    color=mcolors.to_rgba(UNIFORM_LINE_COLOR, alphas[i]),
                    linestyle=LINESTYLE, linewidth=UNIFORM_LINE_WIDTH,
                    solid_capstyle='round', solid_joinstyle='round',
                    zorder=zorder, label=device_labels[i])
        else:
            # Per-point highlight-blend coloring genuinely varies along the
            # curve, so it does need one segment per color change -- a
            # Line3DCollection (many small paths) is unavoidable here.
            points = np.column_stack([device_freqs[i], y_pos, z_plot])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            seg_rgb = 0.5 * (pt_colors[:-1] + pt_colors[1:])
            seg_rgba = np.concatenate([seg_rgb, np.full((len(seg_rgb), 1), alphas[i])], axis=1)
            lc = Line3DCollection(segments, colors=seg_rgba, linewidths=LINEWIDTH,
                                  linestyles=LINESTYLE, capstyle='round',
                                  joinstyle='round', zorder=zorder, label=device_labels[i])
            ax.add_collection3d(lc)

        if MARKER is not None:
            face_rgba = np.concatenate(
                [pt_colors, np.full((len(pt_colors), 1), MARKER_FACE_ALPHA * alphas[i])], axis=1)
            if MARKER_FACECOLOR is not None:
                face_rgba = np.tile(mcolors.to_rgba(MARKER_FACECOLOR, MARKER_FACE_ALPHA * alphas[i]),
                                     (len(pt_colors), 1))
            edge_rgba = (mcolors.to_rgba(MARKER_EDGECOLOR, MARKER_EDGE_ALPHA * alphas[i])
                         if MARKER_EDGECOLOR is not None else 'none')
            ax.scatter(device_freqs[i], y_pos, z_plot, marker=MARKER, s=MARKERSIZE ** 2,
                       facecolor=face_rgba, edgecolor=edge_rgba, linewidths=0.5,
                       zorder=zorder + 0.5, depthshade=False)

    if XMIN is not None or XMAX is not None:
        ax.set_xlim(XMIN if XMIN is not None else min(f.min() for f in device_freqs),
                    XMAX if XMAX is not None else max(f.max() for f in device_freqs))
    if flip:
        ax.set_zlim(-z_hi, -z_lo)
    else:
        ax.set_zlim(z_lo, z_hi)
    ax.set_ylim(-0.5, n_devices - 0.5)
    ax.set_yticks(np.arange(n_devices))

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)
    ax.set_proj_type('ortho' if ORTHOGRAPHIC else 'persp')
    ax.set_box_aspect(BOX_ASPECT)
    _style_panes(ax)
    ax.grid(SHOW_GRID)

    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(spine_linewidth)

    if show_labels and SHOW_AXIS_LABELS:
        ax.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_zlabel(zlabel, fontsize=axis_label_fontsize)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    if show_labels and SHOW_TICK_LABELS:
        ax.set_yticklabels(device_labels, fontsize=tick_label_fontsize)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    if show_labels and SHOW_LEGEND:
        ax.legend(fontsize=tick_label_fontsize, frameon=False)

    return fig, ax


def _find_first_two_peaks(freqs_ghz, y):
    """
    Find the first two (lowest-frequency) prominent peaks in y. A "peak" is a
    real local maximum in y for POWER_PI_UNIT='V' (1/V_pi genuinely peaks at
    resonance), or a real local minimum otherwise (P_pi dips at resonance,
    read as a peak once the display axis is flipped -- see _flip_axis).
    Peaks must clear INSET_PEAK_PROMINENCE and be spaced at least
    INSET_MIN_PEAK_SEPARATION_MHZ apart. Returns a list of 0, 1, or 2 peak
    frequencies (GHz), in frequency order.
    """
    d_ghz = float(np.median(np.diff(freqs_ghz)))
    distance = max(1, int(round((INSET_MIN_PEAK_SEPARATION_MHZ / 1e3) / d_ghz)))
    sign = -1.0 if _flip_axis() else 1.0
    idx, _ = find_peaks(sign * y, prominence=INSET_PEAK_PROMINENCE, distance=distance)
    return freqs_ghz[np.sort(idx)[:2]].tolist()


def _value_at_peak(freqs_ghz, y, peak_ghz):
    """The curve's own y value (in the current POWER_PI_UNIT) at its nearest
    sample to peak_ghz -- i.e. the P_pi/V_pi extremum found at that peak."""
    idx_peak = int(np.argmin(np.abs(freqs_ghz - peak_ghz)))
    return float(y[idx_peak])


def _peak_3db_linewidth_mhz(freqs_ghz, y, peak_ghz):
    """
    3-dB (half-power) linewidth of the resonance at peak_ghz, in MHz.

    The resonance's actual power response is ~beta^2 (beta = modulation
    depth); P_pi/V_pi are both ~1/beta (not beta^2 itself), so a genuine 3 dB
    (2x) drop in resonance power corresponds to a specific, unit-dependent
    move in y away from its value at the peak:
      'dBm': P_pi (dBm) rises by 10*log10(2) ~= 3.01 dB from its dip minimum
             (Vpi ~ 1/beta increases by sqrt(2) -> Ppi_mW = Vpi^2*const
             doubles -> +3.01 dB).
      'mW':  P_pi (mW) doubles from its dip minimum (same relation, linear).
      'V':   1/V_pi falls by a factor of sqrt(2) from its peak maximum
             (1/V_pi ~ beta directly, so it drops by sqrt(2) when beta does).
    Finds the nearest frequency crossing that threshold on each side of
    peak_ghz (linear interpolation between the bracketing samples) and
    returns their spacing. Returns None if the curve never re-crosses the
    threshold before running out of data on either side (e.g. a peak right
    at the edge of the sweep).
    """
    idx_peak = int(np.argmin(np.abs(freqs_ghz - peak_ghz)))
    y_extremum = y[idx_peak]

    if POWER_PI_UNIT == 'dBm':
        threshold = y_extremum + 10.0 * np.log10(2.0)
        inside = y <= threshold
    elif POWER_PI_UNIT == 'mW':
        threshold = y_extremum * 2.0
        inside = y <= threshold
    elif POWER_PI_UNIT == 'V':
        threshold = y_extremum / np.sqrt(2.0)
        inside = y >= threshold
    else:
        raise ValueError(f"POWER_PI_UNIT must be 'dBm', 'mW', or 'V', got {POWER_PI_UNIT!r}.")

    def _cross(step):
        i = idx_peak
        while inside[i]:
            i += step
            if i < 0 or i >= len(freqs_ghz):
                return None
        f0, f1 = freqs_ghz[i - step], freqs_ghz[i]
        y0, y1 = y[i - step], y[i]
        frac = (threshold - y0) / (y1 - y0)
        return f0 + frac * (f1 - f0)

    f_lo, f_hi = _cross(-1), _cross(+1)
    if f_lo is None or f_hi is None:
        return None
    return (f_hi - f_lo) * 1e3   # GHz -> MHz


def _make_inset_fig(w_mm, h_mm):
    fig_w = INSET_LEFT_MM + w_mm + INSET_RIGHT_MM
    fig_h = INSET_BOTTOM_MM + h_mm + INSET_TOP_MM
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = INSET_LEFT_MM   / fig_w,
        right  = 1 - INSET_RIGHT_MM  / fig_w,
        bottom = INSET_BOTTOM_MM / fig_h,
        top    = 1 - INSET_TOP_MM    / fig_h,
    )
    return fig, ax


def _resolve_inset_line_colors():
    """INSET_LINE_COLOR is either one color (both curves) or a 2-element
    list/tuple (1st-peak curve, 2nd-peak curve). Returns (color1, color2)."""
    if isinstance(INSET_LINE_COLOR, (list, tuple)):
        if len(INSET_LINE_COLOR) != 2:
            raise ValueError(
                f"INSET_LINE_COLOR list/tuple must have exactly 2 entries, "
                f"got {len(INSET_LINE_COLOR)}: {INSET_LINE_COLOR!r}."
            )
        return INSET_LINE_COLOR[0], INSET_LINE_COLOR[1]
    return INSET_LINE_COLOR, INSET_LINE_COLOR


def _draw_inset_line(ax, freqs_ghz, y, color, label):
    """Solid line for one curve (fill is handled separately -- see
    _draw_inset_fill_overlap -- since it needs both curves at once to know
    where they overlap)."""
    ax.plot(freqs_ghz, y, color=color, linewidth=INSET_LINEWIDTH,
             linestyle=INSET_LINESTYLE, solid_capstyle='round', zorder=2, label=label)


def _draw_inset_fill_overlap(ax, f1, y1, f2, y2, fill_color, x_lo, x_hi, y_bot, y_top):
    """
    Fill toward the baseline (off-resonance) end of the y-axis for both
    curves -- toward y_top if _flip_axis() (P_pi, matching the "above"
    convention in _draw_device_fill_gradient), toward y_bot otherwise
    (1/V_pi, resonance is already a real peak so the baseline is the low
    end) -- in one flat fill_color, at INSET_FILL_ALPHA where only one curve
    covers a given point and INSET_FILL_ALPHA_OVERLAP where both do. Both
    curves are resampled onto one shared x-grid (since they don't generally
    share the same x-samples -- one is frequency-doubled) so "covered by
    both" can be evaluated pointwise; a curve only counts as covering points
    within its own actual frequency range.
    """
    n_rows = 256
    x_grid = np.linspace(x_lo, x_hi, INSET_FILL_N_COLS)
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

    fill_rgba = np.zeros((n_rows, INSET_FILL_N_COLS, 4))
    fill_rgba[:, :, :3] = mcolors.to_rgb(fill_color)
    fill_rgba[:, :, 3] = np.select(
        [n_covering == 2, n_covering == 1],
        [INSET_FILL_ALPHA_OVERLAP, INSET_FILL_ALPHA],
        default=0.0,
    )
    ax.imshow(fill_rgba, aspect='auto', extent=[x_lo, x_hi, y_bot, y_top],
              origin='lower', interpolation='nearest', zorder=1)


def _build_peak_inset_figure(freqs_ghz, y, device_label, show_labels: bool, zlabel, fill_color):
    """
    One device's peak-comparison inset: its 1st peak (frequency doubled) and
    2nd peak (as-is), overlaid on shared axes, both filled in fill_color.
    Returns (fig, ax, peak1_ghz, peak2_ghz), or None if fewer than 2 peaks
    (or too few points near one) were found. peak1_ghz/peak2_ghz are None
    when INSET_OVERRIDE_XMIN/XMAX are both set -- see below, no peak search
    is needed (or performed) in that case.

    # Peak 1 is shown with its frequency doubled (so its outer half-width,
    # taken before doubling, becomes 2*half_ghz once displayed); peak 2's own
    # outer half-width is likewise doubled to match. The two curves share one
    # combined displayed range (the union of each peak's own outer extent);
    # both are windowed to span that *entire* combined range -- peak 1's real
    # (undoubled) window is half of it (so doubling stretches it back out to
    # the full range), peak 2's is used directly -- so both curves (and
    # fills) cover the whole inset width, not just their own half of it.
    """
    use_override = INSET_OVERRIDE_XMIN is not None and INSET_OVERRIDE_XMAX is not None
    if use_override:
        # The displayed span is fixed regardless of where (or whether) any
        # peak actually falls, so there's nothing for a peak search to
        # determine here -- skip it entirely rather than requiring 2 peaks
        # to be found just to throw their frequencies away.
        disp_lo, disp_hi = INSET_OVERRIDE_XMIN, INSET_OVERRIDE_XMAX
        peak1_ghz, peak2_ghz = None, None
    else:
        peaks_ghz = _find_first_two_peaks(freqs_ghz, y)
        if len(peaks_ghz) < 2:
            print(f"  Warning: {device_label}: found only {len(peaks_ghz)} peak(s) "
                  f"(need 2); skipping inset.")
            return None
        peak1_ghz, peak2_ghz = peaks_ghz[0], peaks_ghz[1]
        half_ghz = INSET_HALF_WIDTH_MHZ / 1e3
        disp_lo = min(2.0 * (peak1_ghz - half_ghz), peak2_ghz - 2.0 * half_ghz)
        disp_hi = max(2.0 * (peak1_ghz + half_ghz), peak2_ghz + 2.0 * half_ghz)

    mask1 = (freqs_ghz >= disp_lo / 2.0) & (freqs_ghz <= disp_hi / 2.0)
    mask2 = (freqs_ghz >= disp_lo) & (freqs_ghz <= disp_hi)
    f1, y1 = freqs_ghz[mask1], y[mask1]
    f2, y2 = freqs_ghz[mask2], y[mask2]
    if len(f1) < 2 or len(f2) < 2:
        print(f"  Warning: {device_label}: not enough points near a peak for "
              f"the inset window; skipping.")
        return None

    f1_doubled = f1 * 2.0

    y_all = np.concatenate([y1, y2])
    y_lo = INSET_YMIN if INSET_YMIN is not None else float(y_all.min())
    y_hi = INSET_YMAX if INSET_YMAX is not None else float(y_all.max())
    y_pad = (y_hi - y_lo) * 0.05 if y_hi != y_lo else 0.1
    y_bot, y_top = y_lo - y_pad, y_hi + y_pad

    x_lo = min(f1_doubled.min(), f2.min())
    x_hi = max(f1_doubled.max(), f2.max())

    fig, ax = _make_inset_fig(INSET_AXES_WIDTH_MM, INSET_AXES_HEIGHT_MM)
    ax.set_xlim(x_lo, x_hi)
    if _flip_axis():
        ax.set_ylim(y_top, y_bot)   # inverted: low P_pi at the top, matching the main 3D plot
    else:
        ax.set_ylim(y_bot, y_top)   # normal: 1/V_pi already peaks at resonance, no flip needed

    line_color1, line_color2 = _resolve_inset_line_colors()
    _draw_inset_fill_overlap(ax, f1_doubled, y1, f2, y2, fill_color, x_lo, x_hi, y_bot, y_top)
    _draw_inset_line(ax, f1_doubled, y1, line_color1, label='1st peak (x2 freq)')
    _draw_inset_line(ax, f2, y2, line_color2, label='2nd peak')

    if INSET_SHOW_GRID:
        ax.grid(linewidth=0.4, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    if show_labels:
        ax.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel(zlabel, fontsize=axis_label_fontsize)
        ax.set_title(device_label, fontsize=axis_label_fontsize)
        if INSET_SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax, peak1_ghz, peak2_ghz


def _resolve_data_paths(data_file):
    """
    Turn DATA_FILE into (paths, is_multi_source):
      - list/tuple of paths -> that list, is_multi_source=True
      - a folder            -> its *.npz files sorted by filename, True
      - a single file       -> [that file], False
    is_multi_source distinguishes "real per-device files" (even a folder or
    list containing just one file) from "single file -> synthetic preview".
    """
    if isinstance(data_file, (list, tuple)):
        return list(data_file), True
    resolved = local_path(data_file)
    if os.path.isdir(resolved):
        return sorted(str(p) for p in Path(resolved).glob('*.npz')), True
    return [data_file], False


def main():
    zlabel, qty_name = _zlabel_and_name()
    paths, is_multi_source = _resolve_data_paths(DATA_FILE)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in DATA_FILE: {DATA_FILE!r}")

    if is_multi_source:
        # Real per-device sweeps: one file per device, no synthetic shift.
        drive_powers = _resolve_drive_powers(len(paths))
        device_freqs, device_y = [], []
        for p, drive_power_dbm in zip(paths, drive_powers):
            freqs_ghz, y = _load_device_data(p, drive_power_dbm)
            device_freqs.append(freqs_ghz)
            device_y.append(y)
            print(f"Loaded: {p}")
            print(f"  {len(freqs_ghz)} CW steps: {freqs_ghz[0]:.4f} - {freqs_ghz[-1]:.4f} GHz")
            print(f"  Drive power: {drive_power_dbm:.2f} dBm")
            print(f"  {qty_name} range: {y.min():.4f} - {y.max():.4f} {POWER_PI_UNIT}")
        default_labels = [Path(p).stem for p in paths]
    else:
        drive_powers = _resolve_drive_powers(N_DEVICES)
        freqs_ghz, betas = _betas_from_file(paths[0])
        y = _smooth(freqs_ghz, _betas_to_y(betas, drive_powers[0]), SMOOTH)
        print(f"Loaded: {paths[0]}")
        print(f"  {len(freqs_ghz)} CW steps: {freqs_ghz[0]:.4f} - {freqs_ghz[-1]:.4f} GHz")
        print(f"  {qty_name} range: {y.min():.4f} - {y.max():.4f} {POWER_PI_UNIT}")

        device_freqs, device_y, shifts_ghz = _generate_synthetic_devices(freqs_ghz, betas, drive_powers)
        print(f"\nSynthetic devices: {N_DEVICES} "
              f"(frequency shift range +-{FREQ_SHIFT_RANGE_MHZ / 2:.1f} MHz)")
        for i, shift in enumerate(shifts_ghz):
            print(f"  Device {i + 1}: shift {shift * 1e3:+.2f} MHz, "
                  f"drive power {drive_powers[i]:.2f} dBm")
        default_labels = None

    device_freqs, device_y, default_labels = _apply_device_order(
        device_freqs, device_y, default_labels)
    device_freqs, device_y = _clip_devices_to_xlim(device_freqs, device_y)

    fig, ax = _build_figure(device_freqs, device_y, zlabel, show_labels=True,
                             default_labels=default_labels)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_figure(device_freqs, device_y, zlabel, show_labels=False,
                                          default_labels=default_labels)
        pub_path = Path(SAVE_FOLDER) / PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'\nSaved: {pub_path}')

    out_path = Path(__file__).parent / 'vpi_vs_freq_3d.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

    if SHOW_PEAK_INSETS:
        print()
        device_labels = _resolve_device_labels(len(device_freqs), default_labels)
        pub_stem = Path(INSET_PUBLICATION_SVG_NAME).stem
        pub_suffix = Path(INSET_PUBLICATION_SVG_NAME).suffix
        linewidths_mhz_by_peak = {1: [], 2: []}
        peak_freqs_ghz_by_peak = {1: [], 2: []}
        peak_values_by_peak = {1: [], 2: []}
        for i, label in enumerate(device_labels):
            safe_label = label.replace(' ', '_')
            fill_color = INSET_FILL_COLORS[i % len(INSET_FILL_COLORS)]
            print(f"Peak insets: {label}")
            result = _build_peak_inset_figure(device_freqs[i], device_y[i], label,
                                               show_labels=True, zlabel=zlabel,
                                               fill_color=fill_color)
            if result is None:
                continue
            fig_i, _ax_i, peak1_ghz, peak2_ghz = result
            if peak1_ghz is not None:
                for peak_num, peak_ghz in enumerate((peak1_ghz, peak2_ghz), start=1):
                    peak_freqs_ghz_by_peak[peak_num].append(peak_ghz)
                    peak_value = _value_at_peak(device_freqs[i], device_y[i], peak_ghz)
                    peak_values_by_peak[peak_num].append(peak_value)
                    lw_mhz = _peak_3db_linewidth_mhz(device_freqs[i], device_y[i], peak_ghz)
                    doubled = ' (shown doubled, at {:.4f} GHz)'.format(2 * peak_ghz) if peak_num == 1 else ''
                    if lw_mhz is not None:
                        linewidths_mhz_by_peak[peak_num].append(lw_mhz)
                        print(f"  {peak_num}{'st' if peak_num == 1 else 'nd'} peak: {peak_ghz:.4f} GHz{doubled}, "
                              f"{qty_name} {peak_value:.4f} {POWER_PI_UNIT}, "
                              f"3-dB linewidth {lw_mhz:.2f} MHz")
                    else:
                        print(f"  {peak_num}{'st' if peak_num == 1 else 'nd'} peak: {peak_ghz:.4f} GHz{doubled}, "
                              f"{qty_name} {peak_value:.4f} {POWER_PI_UNIT}, "
                              f"3-dB linewidth: N/A (doesn't re-cross within the data)")
            else:
                print(f"  Fixed span (INSET_OVERRIDE_XMIN/XMAX): "
                      f"{INSET_OVERRIDE_XMIN:.4f} - {INSET_OVERRIDE_XMAX:.4f} GHz; no peak search performed.")

            if FOR_PUBLICATION:
                result_pub = _build_peak_inset_figure(device_freqs[i], device_y[i], label,
                                                       show_labels=False, zlabel=zlabel,
                                                       fill_color=fill_color)
                if result_pub is not None:
                    fig_pub_i, _ax_pub_i, _p1, _p2 = result_pub
                    pub_path_i = Path(SAVE_FOLDER) / f'{pub_stem}_{safe_label}{pub_suffix}'
                    fig_pub_i.savefig(pub_path_i, format='svg', bbox_inches='tight')
                    print(f'  Saved: {pub_path_i}')

            out_path_i = Path(__file__).parent / f'vpi_vs_freq_3d_inset_{safe_label}.png'
            fig_i.savefig(out_path_i, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out_path_i}')

        if peak_freqs_ghz_by_peak[1] or peak_freqs_ghz_by_peak[2]:
            print()
            for peak_num, ordinal in ((1, '1st'), (2, '2nd')):
                freqs = peak_freqs_ghz_by_peak[peak_num]
                if freqs:
                    print(f"{ordinal} peak frequency over {len(freqs)} device(s): "
                          f"mean {np.mean(freqs):.4f} GHz, std {np.std(freqs) * 1e3:.2f} MHz")
                values = peak_values_by_peak[peak_num]
                if values:
                    print(f"{ordinal} peak {qty_name} over {len(values)} device(s): "
                          f"mean {np.mean(values):.4f} {POWER_PI_UNIT}, std {np.std(values):.4f} {POWER_PI_UNIT}")
                lws = linewidths_mhz_by_peak[peak_num]
                if lws:
                    print(f"{ordinal} peak 3-dB linewidth over {len(lws)} device(s): "
                          f"mean {np.mean(lws):.2f} MHz, std {np.std(lws):.2f} MHz")
                elif freqs:
                    print(f"{ordinal} peak 3-dB linewidth: no device produced a valid value.")

    plt.show()


if __name__ == '__main__':
    main()
