"""
ce_and_suppression_contour.py

For dual-tone phase modulation (ch1 at f with depth beta1/phase phi1, ch2 at
2f with depth beta2/phase phi2 -- see dual_tone_amplitudes in
comb_displayer.py), maps out, on the (beta1, beta2) plane, which (beta1,
beta2) pairs can reach each of two families of user-specified targets, at
*some* choice of phi2 (phi1 fixed at PHI1_DEG as an overall phase reference --
it doesn't affect any |A_p|^2, so fixing it loses no generality):

  CE_LIST_PERCENT:     conversion efficiency into WANTED_ORDER, |A_{+1}|^2 (%)
  SUPPRESSION_LIST_DB: the worse of carrier suppression (|A_0|^2) and
                       sideband suppression (|A_-1|^2 / |A_+1|^2), i.e. the
                       same minimax objective as ce_suppression_tradeoff.py

For each (beta1, beta2), "achievable at some phi2" is defined via the BEST
case over phi2: CE_max(beta1, beta2) = max over phi2 of CE, and
best_minimax(beta1, beta2) = min over phi2 of max(carrier_supp_dB,
sideband_supp_dB). A target CE is achievable wherever CE_max >= target (you
can always back off from the best case); a target suppression level is
achievable wherever best_minimax <= target. Setting FIXED_PHI2_DEG instead
skips this per-metric optimization entirely: every field is evaluated at
that one fixed phi2 (same value for every beta1, beta2 point), showing what
a single real phase setting actually delivers across the plane rather than
each metric's own best-case ceiling. Each requested value becomes one
opaque contour line, colored by where it falls within its own family's
gradient (CE: pale -> CE_COLOR_HIGH as CE increases; suppression: pale ->
SUPP_COLOR_HIGH as the suppression requirement gets stricter/more negative).
Both families are drawn on the same (beta1, beta2) axes; two small vertical
colorbars alongside it (rather than a legend) show which color corresponds
to which requested value.

The (beta1, beta2, phi2) grid is evaluated in one vectorized pass per order
(a matrix product over the shared Jacobi-Anger k-sum), not a per-point
optimizer call, since phi2 only enters through a finite Fourier series -- see
_amplitude_grid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import TextBox
from scipy.special import jv

from graphics import (
    DARKBLUE2, LIGHTBLUE2, RED2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

# ── target values ──────────────────────────────────────────────────────────────
CE_LIST_PERCENT     = [6, 20, 35, 47, 54]      # conversion efficiency into WANTED_ORDER
SUPPRESSION_LIST_DB = [-10, -14, -20, -30, -40]      # worse of carrier/sideband suppression
# CE_LIST_PERCENT     = [41, 43, 45, 47, 48.5, 50]      # conversion efficiency into WANTED_ORDER
# SUPPRESSION_LIST_DB = [-46, -40, -33.5, -30]      # worse of carrier/sideband suppression
CE_LIST_PERCENT     = [6, 20, 35, 47, 54, 10, 13, 0.6]      # conversion efficiency into WANTED_ORDER
SUPPRESSION_LIST_DB = [-5, -10, -14, -20, -40] 
CE_LIST_PERCENT     = [10, 20, 30, 40, 50, 23, 27, 13]      # conversion efficiency into WANTED_ORDER
SUPPRESSION_LIST_DB = [-20, -30, -15, -10] 
# ─────────────────────────────────────────────────────────────────────────────

# ── orders / phase convention ─────────────────────────────────────────────────
WANTED_ORDER   =  1    # CE = |A_{WANTED_ORDER}|^2 (fraction of total input power)
UNWANTED_ORDER = -1    # sideband suppression = |A_{UNWANTED_ORDER}|^2 / |A_{WANTED_ORDER}|^2
CARRIER_ORDER  =  0    # carrier suppression  = |A_{CARRIER_ORDER}|^2
PHI1_DEG = 0.0         # ch1 phase, fixed reference (see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

# ── beta1/beta2 grid ───────────────────────────────────────────────────────────
BETA1_BOUNDS = (0.0, 5)   # rad
BETA2_BOUNDS = (0.0, 5)   # rad
# BETA1_BOUNDS = (2.3, 2.6)   # rad
# BETA2_BOUNDS = (1.0, 1.2)   # rad
GRID_N1 = 300               # beta1 samples
GRID_N2 = 300               # beta2 samples
PHI2_N  = 361               # phi2 samples swept per (beta1, beta2) point (1 deg resolution)

# None (default): each (beta1, beta2) point's fields are the BEST achievable
# value over a PHI2_N-point phi2 sweep, independently per metric (see
# _compute_fields) -- "can this beta pair reach X at *some* phi2".
# A number instead: skip the sweep/optimization entirely and evaluate every
# field at that ONE fixed phi2 (the same value for every (beta1, beta2)
# point) -- "does this beta pair reach X at *this specific* phi2", i.e. what
# a single real hardware phase setting actually delivers across the plane,
# rather than each metric's own best-case ceiling.
FIXED_PHI2_DEG = None
# ─────────────────────────────────────────────────────────────────────────────

# ── CE contours ────────────────────────────────────────────────────────────────
# Each requested CE level is drawn fully opaque, colored along a gradient from
# CE_COLOR_LOW (smallest requested CE) to CE_COLOR_HIGH (largest).
CE_COLOR_LOW  = LIGHTBLUE2
CE_COLOR_HIGH = DARKBLUE2
CE_ALPHA      = 1.0
CE_LINESTYLE  = '-'
CE_LINEWIDTH  = 2.0
CE_ZORDER = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── suppression contours ──────────────────────────────────────────────────────
# SHOW_SEPARATE_SUPPRESSION = False: one "combined" field per (beta1, beta2)
# -- the best-over-phi2 minimax of carrier and sideband suppression (the same
# metric ce_suppression_tradeoff.py optimizes) -- contoured at
# SUPPRESSION_LIST_DB in SUPP_COLOR_LOW -> SUPP_COLOR_HIGH (red).
# SHOW_SEPARATE_SUPPRESSION = True: skip that combined field and instead
# contour carrier suppression (order CARRIER_ORDER, i.e. order 0) and
# sideband suppression (order UNWANTED_ORDER, i.e. order -1) SEPARATELY, each
# still best-over-phi2 but optimized independently of the other -- both at
# the same SUPPRESSION_LIST_DB levels, carrier in CARRIER_SUPP_COLOR_LOW ->
# CARRIER_SUPP_COLOR_HIGH (green), sideband in SUPP_COLOR_LOW -> SUPP_COLOR_HIGH
# (the same red as the combined case).
SHOW_SEPARATE_SUPPRESSION = False

# Each requested suppression level is drawn fully opaque, colored along a
# gradient from *_COLOR_LOW (weakest/smallest-magnitude requested
# suppression) to *_COLOR_HIGH (strictest/largest-magnitude).
SUPP_COLOR_LOW  = '#FBD8A2'   # sideband suppression (order -1) -- red
SUPP_COLOR_HIGH = '#6F3D00'
SUPP_ALPHA      = 1.0
SUPP_LINESTYLE  = '-'
SUPP_LINEWIDTH  = 2.0
SUPP_ZORDER = 4

CARRIER_SUPP_COLOR_LOW  = '#cce5cc'   # carrier suppression (order 0) -- green, SHOW_SEPARATE_SUPPRESSION only
CARRIER_SUPP_COLOR_HIGH = '#004c00'
CARRIER_SUPP_ALPHA      = 1.0
CARRIER_SUPP_LINESTYLE  = '-'
CARRIER_SUPP_LINEWIDTH  = 3.0
CARRIER_SUPP_ZORDER = 5
# ─────────────────────────────────────────────────────────────────────────────

# ── plot layout ────────────────────────────────────────────────────────────────
SHOW_GRID      = False
SHOW_COLORBARS = True   # two vertically stacked colorbars (CE above suppression) instead of a legend
XLIM = None   # None = BETA1_BOUNDS
YLIM = None   # None = BETA2_BOUNDS

axes_width_mm  = 60.0
axes_height_mm = 60.0
left_mm    = 20.0
right_mm   = 22.0   # wide enough for the colorbars' tick labels + rotated ylabel
bottom_mm  = 16.0
top_mm     =  8.0

# Colorbar strip to the right of the main axes: 2 (combined suppression) or
# 3 (SHOW_SEPARATE_SUPPRESSION) bars stacked top to bottom, CE always on top.
# Each bar gets a FIXED height (colorbar_height_mm) regardless of how many
# are stacked -- rather than dividing axes_height_mm evenly -- so a label
# like "Carrier suppression [dB]" always has the same room whether there are
# 2 or 3 bars. The colorbar stack and the main axes share the same top edge;
# whichever of the two (stack height vs axes_height_mm) is taller sets the
# plot's total vertical extent, so the shorter one just doesn't reach as far
# down.
colorbar_width_mm  = 5.0
colorbar_height_mm = 48.0   # tall enough that the longest label ("Carrier suppression [dB]") fits at axis_label_fontsize
colorbar_gap_mm    = 12.0   # horizontal gap between the main axes and the colorbars
colorbar_vgap_mm   = 10.0   # vertical gap between stacked colorbars
colorbars_fig_left_mm = 3.0   # small left buffer for the standalone colorbars-only figure below
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels and the legend, and saves an SVG to
# SAVE_FOLDER (in addition to the normal figure that's still shown/PNG-saved).
# When SHOW_COLORBARS is also True, a second, entirely separate SVG
# (COLORBARS_PUBLICATION_SVG_NAME) is saved containing ONLY the colorbar
# stack (with its labels/ticks -- unlike the main plot, blank colorbars would
# be useless), meant to be composited alongside the label-stripped main
# figure in whatever tool assembles the final publication figure.
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUBLICATION_SVG_NAME = 'ce_and_suppression_contour.svg'
COLORBARS_PUBLICATION_SVG_NAME = 'ce_and_suppression_contour_colorbars.svg'
# ─────────────────────────────────────────────────────────────────────────────

# ── interactive mode ───────────────────────────────────────────────────────────
# If True, main() skips the static save/show path entirely and instead opens a
# window with the contour plot plus two text boxes (CE levels, suppression
# levels, comma/whitespace separated). Editing a box and pressing Enter
# reparses that list and redraws immediately. The (beta1, beta2, phi2) field
# evaluation -- the expensive part -- runs once at startup; every redraw only
# recomputes the (cheap) contour lines and colorbars from the cached fields.
INTERACTIVE = False
INTERACTIVE_TEXTBOX_HEIGHT_MM = 10.0
INTERACTIVE_TEXTBOX_GAP_MM    =  6.0   # vertical gap: plot -> boxes, and between the two boxes
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Theory / fields
# ─────────────────────────────────────────────────────────────────────────────

def _k_trunc():
    return int(2 * max(BETA1_BOUNDS[1], BETA2_BOUNDS[1])) + 20


def _amplitude_grid(beta1_vals, beta2_vals, phi2_vals_rad, order):
    """
    A_order[i, j, m] = complex amplitude at harmonic `order`, for every
    combination of beta1_vals[i], beta2_vals[j], phi2_vals_rad[m].

        A_p = Sum_k J_{p-2k}(beta1) * J_k(beta2) * exp(i[(p-2k)*phi1 + k*phi2])

    phi2 only enters through the exp(i*k*phi2) factor, so for fixed
    (beta1, beta2) this is a finite Fourier series in phi2 -- computed here
    for every (beta1, beta2, phi2) at once as a single matrix product
    (contracting over k), rather than looping per grid point.
    """
    k_trunc = _k_trunc()
    k = np.arange(-k_trunc, k_trunc + 1)
    phi1_rad = np.deg2rad(PHI1_DEG)

    X = jv(order - 2 * k, beta1_vals[:, None]) * np.exp(1j * (order - 2 * k) * phi1_rad)  # (n1, nk)
    Y = jv(k, beta2_vals[:, None])                                                          # (n2, nk)
    Z = np.exp(1j * np.outer(k, phi2_vals_rad))                                             # (nk, nphi2)

    C = X[:, None, :] * Y[None, :, :]        # (n1, n2, nk)
    n1, n2, nk = C.shape
    return (C.reshape(n1 * n2, nk) @ Z).reshape(n1, n2, -1)


def _compute_fields(beta1_vals, beta2_vals):
    """
    Returns (ce_field, minimax_field, carrier_field, sideband_field), each
    shape (len(beta1_vals), len(beta2_vals)).

    FIXED_PHI2_DEG is None (default): each field is the BEST value of that
    metric over a PHI2_N-point phi2 sweep, independently per metric -- the
    phi2 achieving carrier_field needn't be the same one achieving
    sideband_field or minimax_field.

    FIXED_PHI2_DEG is a number: no sweep/optimization at all -- every field
    is just that metric's actual value at that one fixed phi2 (the same
    phi2 for every (beta1, beta2) point).
    """
    if FIXED_PHI2_DEG is not None:
        phi2_vals_rad = np.deg2rad(np.array([FIXED_PHI2_DEG]))
    else:
        phi2_vals_rad = np.deg2rad(np.linspace(0.0, 360.0, PHI2_N, endpoint=False))

    p_carrier = np.abs(_amplitude_grid(beta1_vals, beta2_vals, phi2_vals_rad, CARRIER_ORDER)) ** 2
    p_wanted  = np.abs(_amplitude_grid(beta1_vals, beta2_vals, phi2_vals_rad, WANTED_ORDER)) ** 2
    p_unwanted = np.abs(_amplitude_grid(beta1_vals, beta2_vals, phi2_vals_rad, UNWANTED_ORDER)) ** 2

    ce_percent = p_wanted * 100.0
    carrier_supp_db = 10.0 * np.log10(np.maximum(p_carrier, 1e-30))
    sideband_supp_db = (10.0 * np.log10(np.maximum(p_unwanted, 1e-30))
                         - 10.0 * np.log10(np.maximum(p_wanted, 1e-30)))
    minimax_db = np.maximum(carrier_supp_db, sideband_supp_db)

    if FIXED_PHI2_DEG is not None:
        ce_field = ce_percent[:, :, 0]
        minimax_field = minimax_db[:, :, 0]
        carrier_field = carrier_supp_db[:, :, 0]
        sideband_field = sideband_supp_db[:, :, 0]
    else:
        ce_field = ce_percent.max(axis=-1)
        minimax_field = minimax_db.min(axis=-1)
        carrier_field = carrier_supp_db.min(axis=-1)
        sideband_field = sideband_supp_db.min(axis=-1)
    return ce_field, minimax_field, carrier_field, sideband_field


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _level_colormap(levels, color_low, color_high):
    """LinearSegmentedColormap running from color_low (at min(levels)) to
    color_high (at max(levels)), plus a Normalize spanning that same range."""
    cmap = mcolors.LinearSegmentedColormap.from_list('grad', [color_low, color_high])
    lo, hi = min(levels), max(levels)
    norm = mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
    return cmap, norm


def _field_source_phrase():
    """A noun phrase describing a field's extremum, for warning/print
    messages -- best-over-phi2 (default) or one fixed phi2 (FIXED_PHI2_DEG).
    Reads naturally after both "exceeds ..." and "is better than ...":
    "exceeds the best achievable value anywhere on the grid" / "exceeds the
    value anywhere on the grid at the fixed phi2=X deg"."""
    if FIXED_PHI2_DEG is not None:
        return f"the value anywhere on the grid at the fixed phi2={FIXED_PHI2_DEG:g} deg"
    return "the best achievable value anywhere on the grid"


def _draw_ce_contours(ax, B1, B2, ce_max, cmap, norm):
    levels = sorted(set(CE_LIST_PERCENT))
    field_max = ce_max.max()
    for level in levels:
        if level > field_max:
            print(f"  Warning: CE={level:g}% exceeds {_field_source_phrase()} "
                  f"({field_max:.2f}%); that contour won't appear.")
            continue
        color = mcolors.to_rgba(cmap(norm(level)), CE_ALPHA)
        ax.contour(B1, B2, ce_max, levels=[level], colors=[color],
                   linewidths=CE_LINEWIDTH, linestyles=CE_LINESTYLE, zorder=CE_ZORDER)


def _draw_suppression_like_contours(ax, B1, B2, field, cmap, norm, linestyle, linewidth, alpha,
                                     zorder, name):
    """
    Draws SUPPRESSION_LIST_DB as contours of `field` (a suppression-style
    metric in dB, more negative = better -- the combined minimax, or
    carrier/sideband suppression alone; either best-over-phi2 or at one
    fixed phi2, see _compute_fields/FIXED_PHI2_DEG). `name` only affects the
    console warning when a requested level isn't reached anywhere on the grid.
    """
    levels = sorted(set(SUPPRESSION_LIST_DB))
    field_min = field.min()
    for level in levels:
        if level < field_min:
            print(f"  Warning: {name} suppression={level:g} dB is better than {_field_source_phrase()} "
                  f"({field_min:.2f} dB); that contour won't appear.")
            continue
        color = mcolors.to_rgba(cmap(norm(level)), alpha)
        ax.contour(B1, B2, field, levels=[level], colors=[color],
                   linewidths=linewidth, linestyles=linestyle, zorder=zorder)


def _compute_colorbar_specs():
    """
    Returns the (ylabel, cmap, norm, levels, invert_yaxis) list -- CE first,
    then either the combined suppression or the separate carrier/sideband
    pair (per SHOW_SEPARATE_SUPPRESSION), topmost first. Shared by
    _draw_all_contours (which pulls each spec's cmap/norm to color its
    matching contours) and _build_colorbars_figure (the standalone
    colorbars-only publication figure), so the two never drift apart.
    """
    ce_levels = sorted(set(CE_LIST_PERCENT))
    ce_cmap, ce_norm = _level_colormap(ce_levels, CE_COLOR_LOW, CE_COLOR_HIGH)
    specs = [('CE [%]', ce_cmap, ce_norm, ce_levels, False)]

    # Reversed color order (*_COLOR_HIGH first): levels are negative dB, so
    # the most negative (strictest) requested value sits at norm's low end
    # but should still map to the "strong"/dark end of the gradient.
    supp_levels = sorted(set(SUPPRESSION_LIST_DB))
    if SHOW_SEPARATE_SUPPRESSION:
        carrier_cmap, carrier_norm = _level_colormap(supp_levels, CARRIER_SUPP_COLOR_HIGH,
                                                       CARRIER_SUPP_COLOR_LOW)
        specs.append(('Carrier suppression [dB]', carrier_cmap, carrier_norm, supp_levels, True))

        sideband_cmap, sideband_norm = _level_colormap(supp_levels, SUPP_COLOR_HIGH, SUPP_COLOR_LOW)
        specs.append(('Sideband suppression [dB]', sideband_cmap, sideband_norm, supp_levels, True))
    else:
        supp_cmap, supp_norm = _level_colormap(supp_levels, SUPP_COLOR_HIGH, SUPP_COLOR_LOW)
        specs.append(('Suppression [dB]', supp_cmap, supp_norm, supp_levels, True))

    return specs


def _draw_all_contours(ax, B1, B2, ce_max, best_minimax, best_carrier_supp, best_sideband_supp):
    """
    Draws CE contours plus either the combined suppression contours or the
    separate carrier/sideband contours (per SHOW_SEPARATE_SUPPRESSION) onto
    ax, using the colors from _compute_colorbar_specs(). Returns that same
    specs list.
    """
    specs = _compute_colorbar_specs()
    _draw_ce_contours(ax, B1, B2, ce_max, specs[0][1], specs[0][2])

    if SHOW_SEPARATE_SUPPRESSION:
        _, carrier_cmap, carrier_norm, _, _ = specs[1]
        _draw_suppression_like_contours(ax, B1, B2, best_carrier_supp, carrier_cmap, carrier_norm,
                                         CARRIER_SUPP_LINESTYLE, CARRIER_SUPP_LINEWIDTH,
                                         CARRIER_SUPP_ALPHA, CARRIER_SUPP_ZORDER, 'carrier')
        _, sideband_cmap, sideband_norm, _, _ = specs[2]
        _draw_suppression_like_contours(ax, B1, B2, best_sideband_supp, sideband_cmap, sideband_norm,
                                         SUPP_LINESTYLE, SUPP_LINEWIDTH, SUPP_ALPHA, SUPP_ZORDER,
                                         'sideband')
    else:
        _, supp_cmap, supp_norm, _, _ = specs[1]
        _draw_suppression_like_contours(ax, B1, B2, best_minimax, supp_cmap, supp_norm,
                                         SUPP_LINESTYLE, SUPP_LINEWIDTH, SUPP_ALPHA, SUPP_ZORDER,
                                         'combined')

    return specs


def _style_colorbar(cb, ylabel, show_labels: bool = True):
    if show_labels:
        cb.ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        cb.ax.tick_params(direction=tick_direction, width=tick_width, labelsize=tick_label_fontsize)
    else:
        cb.ax.set_ylabel('')
        cb.ax.tick_params(direction=tick_direction, width=tick_width,
                           labelleft=False, labelright=False)
    cb.outline.set_linewidth(spine_linewidth)


def _colorbar_stack_height_mm(n):
    return n * colorbar_height_mm + max(n - 1, 0) * colorbar_vgap_mm


def _make_colorbar_axes(fig, n, cbar_left, cbar_w, plot_top_mm, fig_h):
    """Creates n empty axes stacked vertically, each a fixed colorbar_height_mm
    tall regardless of n, top-aligned with plot_top_mm (the shared top edge
    with the main axes) -- to be filled (once, or repeatedly) by
    _fill_colorbars."""
    caxes = []
    for i in range(n):
        top_mm_i = plot_top_mm - i * (colorbar_height_mm + colorbar_vgap_mm)
        bottom_mm_i = top_mm_i - colorbar_height_mm
        caxes.append(fig.add_axes([cbar_left, bottom_mm_i / fig_h, cbar_w, colorbar_height_mm / fig_h]))
    return caxes


def _fill_colorbars(fig, caxes, specs, show_labels: bool = True):
    """(Re-)draws one colorbar per (cax, spec) pair, clearing each cax first
    so this is safe to call repeatedly (e.g. on every interactive redraw)."""
    for cax, (label, cmap, norm, levels, invert) in zip(caxes, specs):
        cax.clear()
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=levels)
        _style_colorbar(cb, label, show_labels)
        if invert:
            cb.ax.invert_yaxis()


def _draw_stacked_colorbars(fig, specs, cbar_left, cbar_w, plot_top_mm, fig_h, show_labels: bool = True):
    """One-shot version for the static figure: makes fresh axes and fills them."""
    caxes = _make_colorbar_axes(fig, len(specs), cbar_left, cbar_w, plot_top_mm, fig_h)
    _fill_colorbars(fig, caxes, specs, show_labels)


def _build_colorbars_figure(specs, show_labels: bool):
    """
    Standalone figure containing ONLY the colorbar stack. show_labels=False
    (the FOR_PUBLICATION case) strips the ylabel and tick labels -- matching
    this project's usual "for publication" convention -- leaving just the
    bare colored bars (plus tick marks) to be composited as the color key
    alongside the label-stripped main plot.
    """
    stack_mm = _colorbar_stack_height_mm(len(specs))
    fig_w = colorbars_fig_left_mm + colorbar_width_mm + right_mm
    fig_h = bottom_mm + stack_mm + top_mm
    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))

    cbar_left = colorbars_fig_left_mm / fig_w
    cbar_w = colorbar_width_mm / fig_w
    plot_top_mm = bottom_mm + stack_mm
    _draw_stacked_colorbars(fig, specs, cbar_left, cbar_w, plot_top_mm, fig_h, show_labels)

    return fig


def _build_contour_figure(beta1_vals, beta2_vals, ce_max, best_minimax, best_carrier_supp,
                           best_sideband_supp, show_labels: bool):
    show_colorbars = show_labels and SHOW_COLORBARS
    n_colorbars = (3 if SHOW_SEPARATE_SUPPRESSION else 2) if show_colorbars else 0
    extra_mm = (colorbar_gap_mm + colorbar_width_mm) if show_colorbars else 0.0

    # The main axes and the colorbar stack share the same top edge; whichever
    # is taller (axes_height_mm vs the colorbar stack) sets the plot's total
    # vertical extent, so the shorter one just doesn't reach as far down.
    plot_region_mm = max(axes_height_mm, _colorbar_stack_height_mm(n_colorbars))

    fig_w = left_mm + axes_width_mm + extra_mm + right_mm
    fig_h = bottom_mm + plot_region_mm + top_mm
    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))

    plot_top_mm = bottom_mm + plot_region_mm
    axes_bottom_mm = plot_top_mm - axes_height_mm
    ax = fig.add_axes([left_mm / fig_w, axes_bottom_mm / fig_h,
                        axes_width_mm / fig_w, axes_height_mm / fig_h])

    B1, B2 = np.meshgrid(beta1_vals, beta2_vals, indexing='ij')
    specs = _draw_all_contours(ax, B1, B2, ce_max, best_minimax, best_carrier_supp, best_sideband_supp)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    ax.set_xlim(*(XLIM if XLIM is not None else BETA1_BOUNDS))
    ax.set_ylim(*(YLIM if YLIM is not None else BETA2_BOUNDS))

    if show_labels:
        ax.set_xlabel(r'$\beta_1$ [rad]', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\beta_2$ [rad]', fontsize=axis_label_fontsize)
        if SHOW_GRID:
            ax.grid()
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

    if show_colorbars:
        cbar_left = (left_mm + axes_width_mm + colorbar_gap_mm) / fig_w
        cbar_w = colorbar_width_mm / fig_w
        _draw_stacked_colorbars(fig, specs, cbar_left, cbar_w, plot_top_mm, fig_h)

    return fig, ax


def _parse_number_list(text):
    """Parses a comma- and/or whitespace-separated list of numbers, e.g.
    '10, 20, 30' or '10 20 30'."""
    return [float(tok) for tok in text.replace(',', ' ').split()]


def _format_number_list(values):
    return ', '.join(f'{v:g}' for v in values)


def _run_interactive(beta1_vals, beta2_vals, ce_max, best_minimax, best_carrier_supp,
                      best_sideband_supp):
    """
    Opens an interactive window: the contour plot + colorbars on top, two
    text boxes (CE levels, suppression levels) below. Submitting a box (press
    Enter) reparses CE_LIST_PERCENT/SUPPRESSION_LIST_DB from its text and
    redraws -- the fields (already computed once by the caller) are reused
    as-is, so only the cheap contour/colorbar drawing reruns. The number of
    colorbars (2 combined, or 3 with SHOW_SEPARATE_SUPPRESSION) is fixed for
    the session at whatever SHOW_SEPARATE_SUPPRESSION is when this opens.
    """
    global CE_LIST_PERCENT, SUPPRESSION_LIST_DB

    box_h_mm = INTERACTIVE_TEXTBOX_HEIGHT_MM
    box_gap_mm = INTERACTIVE_TEXTBOX_GAP_MM
    extra_right_mm = colorbar_gap_mm + colorbar_width_mm
    n_colorbars = 3 if SHOW_SEPARATE_SUPPRESSION else 2

    # Bottom-to-top: [box_gap][supp box][box_gap][CE box][bottom_mm, for the
    # main axes' own x-ticks/x-label][axes and colorbar stack, sharing a top
    # edge][top_mm]. bottom_mm is reused unchanged from the static layout --
    # it's still just "room below the axes for its ticks and x-label" -- the
    # text boxes stack BELOW that, not inside it.
    plot_region_mm = max(axes_height_mm, _colorbar_stack_height_mm(n_colorbars))

    fig_w = left_mm + axes_width_mm + extra_right_mm + right_mm
    boxes_mm = box_gap_mm + box_h_mm + box_gap_mm + box_h_mm + box_gap_mm
    plot_bottom_mm = bottom_mm + boxes_mm
    plot_top_mm = plot_bottom_mm + plot_region_mm
    axes_bottom_mm = plot_top_mm - axes_height_mm
    fig_h = plot_top_mm + top_mm

    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))
    ax = fig.add_axes([left_mm / fig_w, axes_bottom_mm / fig_h,
                        axes_width_mm / fig_w, axes_height_mm / fig_h])

    cbar_left = (left_mm + axes_width_mm + colorbar_gap_mm) / fig_w
    cbar_w = colorbar_width_mm / fig_w
    caxes = _make_colorbar_axes(fig, n_colorbars, cbar_left, cbar_w, plot_top_mm, fig_h)

    box_w_frac = axes_width_mm / fig_w
    ce_box_ax = fig.add_axes([left_mm / fig_w, (box_gap_mm + box_h_mm + box_gap_mm) / fig_h,
                               box_w_frac, box_h_mm / fig_h])
    supp_box_ax = fig.add_axes([left_mm / fig_w, box_gap_mm / fig_h, box_w_frac, box_h_mm / fig_h])

    B1, B2 = np.meshgrid(beta1_vals, beta2_vals, indexing='ij')

    def redraw():
        ax.clear()

        specs = _draw_all_contours(ax, B1, B2, ce_max, best_minimax, best_carrier_supp,
                                    best_sideband_supp)

        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax.tick_params(axis='both', direction=tick_direction,
                       width=tick_width, labelsize=tick_label_fontsize)
        ax.set_xlim(*(XLIM if XLIM is not None else BETA1_BOUNDS))
        ax.set_ylim(*(YLIM if YLIM is not None else BETA2_BOUNDS))
        ax.set_xlabel(r'$\beta_1$ [rad]', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\beta_2$ [rad]', fontsize=axis_label_fontsize)
        if SHOW_GRID:
            ax.grid()

        _fill_colorbars(fig, caxes, specs)

        fig.canvas.draw_idle()

    def on_submit_ce(text):
        global CE_LIST_PERCENT
        try:
            CE_LIST_PERCENT = _parse_number_list(text)
        except ValueError:
            print(f"Could not parse CE list {text!r} as numbers; leaving it unchanged.")
            return
        redraw()

    def on_submit_supp(text):
        global SUPPRESSION_LIST_DB
        try:
            SUPPRESSION_LIST_DB = _parse_number_list(text)
        except ValueError:
            print(f"Could not parse suppression list {text!r} as numbers; leaving it unchanged.")
            return
        redraw()

    # Title-above-box instead of TextBox's own left-of-box label: that label
    # renders outside (to the left of) the box axes, which would need a much
    # wider left_mm than the main plot's y-axis label ever does.
    ce_box_ax.set_title('CE [%]', fontsize=tick_label_fontsize, loc='left')
    ce_box = TextBox(ce_box_ax, '', initial=_format_number_list(CE_LIST_PERCENT))
    ce_box.on_submit(on_submit_ce)

    supp_box_ax.set_title('Suppression [dB]', fontsize=tick_label_fontsize, loc='left')
    supp_box = TextBox(supp_box_ax, '', initial=_format_number_list(SUPPRESSION_LIST_DB))
    supp_box.on_submit(on_submit_supp)

    redraw()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    beta1_vals = np.linspace(*BETA1_BOUNDS, GRID_N1)
    beta2_vals = np.linspace(*BETA2_BOUNDS, GRID_N2)

    if FIXED_PHI2_DEG is not None:
        print(f"Evaluating {GRID_N1}x{GRID_N2} (beta1, beta2) grid at the fixed phi2={FIXED_PHI2_DEG:g} deg "
              f"(phi1={PHI1_DEG:.1f} deg fixed)...")
    else:
        print(f"Evaluating {GRID_N1}x{GRID_N2} (beta1, beta2) grid over {PHI2_N} phi2 samples "
              f"(phi1={PHI1_DEG:.1f} deg fixed)...")
    ce_max, best_minimax, best_carrier_supp, best_sideband_supp = _compute_fields(beta1_vals, beta2_vals)
    print(f"  CE (order {WANTED_ORDER:+d}), {_field_source_phrase()}: {ce_max.max():.2f}%")
    print(f"  Minimax suppression, {_field_source_phrase()}: {best_minimax.min():.2f} dB")
    print(f"  Carrier suppression, {_field_source_phrase()}: {best_carrier_supp.min():.2f} dB")
    print(f"  Sideband suppression, {_field_source_phrase()}: {best_sideband_supp.min():.2f} dB\n")

    if INTERACTIVE:
        print("Interactive mode: edit a text box and press Enter to redraw.")
        _run_interactive(beta1_vals, beta2_vals, ce_max, best_minimax, best_carrier_supp,
                          best_sideband_supp)
        return

    print("CE contours requested:", CE_LIST_PERCENT)
    print("Suppression contours requested (dB):", SUPPRESSION_LIST_DB, "\n")

    fig, ax = _build_contour_figure(beta1_vals, beta2_vals, ce_max, best_minimax, best_carrier_supp,
                                     best_sideband_supp, show_labels=True)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_contour_figure(beta1_vals, beta2_vals, ce_max, best_minimax,
                                                   best_carrier_supp, best_sideband_supp,
                                                   show_labels=False)
        pub_path = Path(SAVE_FOLDER) / PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'Saved: {pub_path}')

        if SHOW_COLORBARS:
            fig_cbars = _build_colorbars_figure(_compute_colorbar_specs(), show_labels=False)
            cbars_path = Path(SAVE_FOLDER) / COLORBARS_PUBLICATION_SVG_NAME
            fig_cbars.savefig(cbars_path, format='svg', bbox_inches='tight')
            print(f'Saved: {cbars_path}')

    out_path = Path(__file__).parent / 'ce_and_suppression_contour.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

    plt.show()


if __name__ == '__main__':
    main()
