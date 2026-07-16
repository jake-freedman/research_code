"""
Analyse a 2D power-phase grid sweep produced by bnc_2d_power_phase_script.py.

Loads all per-grid-point .npz files from a folder, extracts the maximum
sideband power (over the phase sweep) at a chosen harmonic, and plots a
contour map with Ch1 and Ch2 RMS voltages on the axes.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
from scipy.special import jv
from dual_tone_sweep_data import DualToneSweepData
from graphics import (
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

# Colors the heatmap colormap linearly interpolates between (low value ->
# high value). Any matplotlib color spec works (hex string, name, ...).
# Defaults match this codebase's -1/+1 sideband colors (see
# dual_tone_sweep_data.py's _HARMONIC_COLORS: ORANGE2 / LIGHTBLUE2).
CMAP_COLOR_LOW  = '#b2cbf2'   # LIGHTBLUE2
CMAP_COLOR_HIGH = '#FBD8A2'   # ORANGE2

SIDEBAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'sideband_pm1', [CMAP_COLOR_LOW, CMAP_COLOR_HIGH]
)

FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\2d_power_phase_2026-06-24-14-33-24"
# FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\2d_power_phase_2026-06-23-14-30-45"

# Harmonic order to plot (must be in the harmonics list recorded by the script)
HARMONIC = -1

# Normalization for the colour axis:
#   False     → dBm  (raw ESA peak power)
#   True      → dBc  (relative to per-grid-point RF-off calibration)
#   'percent' → %    (fraction of carrier power, linear)
NORMALIZE = 'percent'

# Colour axis limits. None = auto-scale.
CMIN = None
CMAX = None

# Number of filled contour levels
N_LEVELS = 30

# ── figure size ───────────────────────────────────────────────────
axes_width_mm  = 48
axes_height_mm = 38

# When n_grid_repeats > 1: which repeat to display in the contour map.
#   None → use the mean across all repeats (default)
#   0, 1, … → use only that repeat index (0-based)
GRID_REPEAT_INDEX = 0

# ── 3D surface plot ─────────────────────────────────────────────────
# When True, render the heatmap(s) as 3D surfaces (Z = value) instead of the
# 2D filled contour -- colored with the same colormap/levels, with a sparse
# wireframe overlay and an orthographic (non-perspective) projection.
PLOT_3D_SURFACE = True
# Wireframe grid-line stride (every Nth line, in each direction), separate
# for the experimental and theory surfaces since they may have very
# different grid resolutions (see THEORY_N1/THEORY_N2).
WIREFRAME_STRIDE_EXPERIMENT = 1
WIREFRAME_STRIDE_THEORY     = 12
WIREFRAME_COLOR      = "#eeeeee"
WIREFRAME_LINEWIDTH  = 0.5
WIREFRAME_ALPHA      = 1
SURFACE_ALPHA        = 1.0
# View angle in degrees: ELEV tilts up/down, AZIM rotates the view about the
# vertical (z) axis -- e.g. add/subtract 90 to spin the surface a quarter
# turn without touching the data.
SURFACE_VIEW_ELEV, SURFACE_VIEW_AZIM = 50, 120 + 90 + 20
# Relative box aspect (x, y, z): stretches/compresses the plotted box without
# touching the data or tick labels. z=1 matches the x/y box size; z>1 makes
# the surface look taller, z<1 flatter.
SURFACE_BOX_ASPECT = (4, 4, 3)

# ── publication export ────────────────────────────────────────────
# When True: removes tick labels, axis labels, colorbar labels, and title,
# and saves SVGs to SAVE_FOLDER.
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"

# Contour lines drawn on top of the filled map when FOR_PUBLICATION = True.
# Set PUB_CONTOUR_COLOR = None to skip contour lines entirely.
PUB_CONTOUR_COLOR     = '#000000'   # any matplotlib color, or None to skip
PUB_CONTOUR_WIDTH     = 0       # linewidth in points

# ── theoretical figure ────────────────────────────────────────────
SHOW_THEORETICAL = True

# Half-wave voltage [V_rms] for each channel.
VPI1 = 5   # ch1, drives at f
VPI2 = 2.3   # ch2, drives at 2f

# Grid resolution for the theory map along each axis. None = same resolution
# as the measurement grid (n1, n2) -- set these to get a smoother theory
# surface/contour independent of how coarse the actual sweep was.
THEORY_N1 = 100   # ch1 axis
THEORY_N2 = 100   # ch2 axis

# Voltage range [V_rms] for the theory map along each axis. None = same
# range as the measurement grid (v1.min()..v1.max() / v2.min()..v2.max()) --
# set these to extrapolate beyond, or zoom into a sub-range of, the swept
# voltages.
THEORY_V1_RANGE = (0,4.2)   # (vmin, vmax), ch1
THEORY_V2_RANGE = (0,1.3)   # ch2

# On the theory surface (3D only), outline the rectangle in (V1, V2) that
# the actual experimental sweep covered -- useful when THEORY_V1_RANGE/
# THEORY_V2_RANGE extrapolate beyond the measured voltages, to mark where
# the surface is backed by data vs. pure extrapolation.
SHOW_EXPERIMENT_BOUNDARY = True
EXPERIMENT_BOUNDARY_COLOR = 'black'
EXPERIMENT_BOUNDARY_LINEWIDTH = 2.0
EXPERIMENT_BOUNDARY_ZORDER = 10
EXPERIMENT_BOUNDARY_N_POINTS = 100   # points used to trace each edge

# Phase resolution for the theoretical maximum search (higher = more accurate).
N_PHASE_THEORY = 36

# Bessel series truncation for theory. None = auto (derived from max beta).
K_TRUNC_THEORY = None

# ------------------------------------------------------------------

_LOG20 = 10.0 * np.log10(20.0)


def _dbm_to_vrms(dbm: np.ndarray) -> np.ndarray:
    return 10.0 ** ((dbm - _LOG20) / 20.0)


def _theory_peak_fraction(beta1: float, beta2: float, harmonic: int,
                           n_phase: int, k_trunc: int) -> float:
    """Max |A_p|² over φ2 ∈ [0, 2π) with φ1 = 0 for the given harmonic."""
    k = np.arange(-k_trunc, k_trunc + 1)
    coeffs = jv(harmonic - 2 * k, beta1) * jv(k, beta2)   # (2K+1,)
    phi2 = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)
    A = np.exp(1j * np.outer(phi2, k)) @ coeffs            # (n_phase,)
    return float(np.max(np.abs(A) ** 2))


def _theory_z(v1_vals: np.ndarray, v2_vals: np.ndarray) -> np.ndarray:
    """Theory map value (in the current NORMALIZE units) at arbitrary
    (v1, v2) points, evaluated directly rather than read off a grid."""
    z = np.empty_like(v1_vals, dtype=float)
    for idx, (vi, vj) in enumerate(zip(v1_vals, v2_vals)):
        beta1 = np.pi * vi / VPI1
        beta2 = np.pi * vj / VPI2
        k_trunc = (K_TRUNC_THEORY if K_TRUNC_THEORY is not None
                   else int(2 * max(beta1, beta2)) + 20)
        frac = _theory_peak_fraction(beta1, beta2, HARMONIC, N_PHASE_THEORY, k_trunc)
        if NORMALIZE == 'percent':
            z[idx] = frac * 100.0
        elif NORMALIZE:
            z[idx] = 10.0 * np.log10(max(frac, 1e-30))
        else:
            z[idx] = frac * 100.0
    return z


def _plot_experiment_boundary_3d(ax, v1_range, v2_range):
    """Outline, on the theory surface, the rectangle in (V1, V2) spanned by
    v1_range/v2_range (the actual experimental sweep), tracing each edge at
    the theory surface's own height so it hugs the surface."""
    v1_lo, v1_hi = v1_range
    v2_lo, v2_hi = v2_range
    t = np.linspace(0.0, 1.0, EXPERIMENT_BOUNDARY_N_POINTS)

    edges = [
        (np.full_like(t, v1_lo), v2_lo + t * (v2_hi - v2_lo)),
        (np.full_like(t, v1_hi), v2_lo + t * (v2_hi - v2_lo)),
        (v1_lo + t * (v1_hi - v1_lo), np.full_like(t, v2_lo)),
        (v1_lo + t * (v1_hi - v1_lo), np.full_like(t, v2_hi)),
    ]
    for v1e, v2e in edges:
        ze = _theory_z(v1e, v2e)
        ax.plot(v1e, v2e, ze, color=EXPERIMENT_BOUNDARY_COLOR,
                linewidth=EXPERIMENT_BOUNDARY_LINEWIDTH,
                zorder=EXPERIMENT_BOUNDARY_ZORDER)


def _make_contour_fig(axes_w, axes_h):
    mm = 1.0 / 25.4
    cb_mm = 20.0
    fig, ax = plt.subplots(figsize=(
        (_left_mm + axes_w + _right_mm + cb_mm) * mm,
        (_bottom_mm + axes_h + _top_mm) * mm,
    ))
    total_w = _left_mm + axes_w + _right_mm + cb_mm
    fig.subplots_adjust(
        left   = _left_mm / total_w,
        right  = (_left_mm + axes_w) / total_w,
        bottom = _bottom_mm / (_bottom_mm + axes_h + _top_mm),
        top    = (_bottom_mm + axes_h) / (_bottom_mm + axes_h + _top_mm),
    )
    return fig, ax


def _pub_strip(ax, cb):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.tick_params(labelbottom=False, labelleft=False)
    cb.set_label('')
    cb.ax.tick_params(labelsize=0)


def _plot_surface_fig(V1, V2, Z, levels, cb_label, wireframe_stride, cmap=SIDEBAND_CMAP):
    """
    3D surface colored like the 2D contourf (same colormap/levels), with a
    sparse wireframe overlay and an orthographic projection.
    """
    if isinstance(levels, int):
        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    else:
        vmin, vmax = float(levels[0]), float(levels[-1])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    mm = 1.0 / 25.4
    fig = plt.figure(figsize=((axes_width_mm + 50) * mm, (axes_height_mm + 40) * mm))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')

    # antialiased=False + edgecolor='none' avoids the light/gray seams
    # matplotlib otherwise draws between adjacent surface quads.
    ax.plot_surface(V1, V2, Z, facecolors=cmap(norm(Z)),
                     rstride=1, cstride=1, linewidth=0, edgecolor='none',
                     antialiased=False, shade=False, alpha=SURFACE_ALPHA,
                     zorder=1)
    # mplot3d's z-order is an approximate whole-object depth sort, not true
    # per-pixel depth testing, so an opaque surface can end up drawn in
    # front of a wireframe traced along the same points -- nudge the
    # wireframe slightly toward the viewer (in data units) so it always wins.
    z_range = (vmax - vmin) or 1.0
    ax.plot_wireframe(V1, V2, Z + 1e-3 * z_range,
                       rstride=wireframe_stride, cstride=wireframe_stride,
                       color=WIREFRAME_COLOR, linewidth=WIREFRAME_LINEWIDTH,
                       alpha=WIREFRAME_ALPHA, zorder=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cb.ax.tick_params(labelsize=tick_label_fontsize)
    cb.set_label(cb_label, fontsize=axis_label_fontsize)

    ax.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax.set_zlabel(cb_label, fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_label_fontsize)
    ax.view_init(elev=SURFACE_VIEW_ELEV, azim=SURFACE_VIEW_AZIM)
    ax.set_box_aspect(SURFACE_BOX_ASPECT)
    # Pin the z-axis to the same (vmin, vmax) used for the color normalization
    # (shared `levels` from main()) so the experiment and theory surfaces are
    # directly comparable, rather than each auto-scaling to its own data range.
    ax.set_zlim(vmin, vmax)
    return fig, ax, cb


def _pub_strip_3d(ax, cb):
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    cb.set_label('')
    cb.ax.tick_params(labelsize=0)


def main():
    import os as _os
    meta = np.load(os.path.join(FOLDER, 'grid_meta.npz'))
    ch1_powers_dbm = meta['ch1_powers_dbm']
    ch2_powers_dbm = meta['ch2_powers_dbm']
    n1, n2 = len(ch1_powers_dbm), len(ch2_powers_dbm)

    v1 = _dbm_to_vrms(ch1_powers_dbm)
    v2 = _dbm_to_vrms(ch2_powers_dbm)

    peak_map = np.full((n1, n2), np.nan)
    for i in range(n1):
        for j in range(n2):
            fpath = os.path.join(FOLDER, f'grid_{i:02d}_{j:02d}.npz')
            if not os.path.exists(fpath):
                print(f"Missing: grid_{i:02d}_{j:02d}.npz")
                continue
            try:
                data = DualToneSweepData.from_file(fpath)
                if GRID_REPEAT_INDEX is not None and data.spectra_all is not None:
                    n_reps = data.spectra_all.shape[0]
                    if not (0 <= GRID_REPEAT_INDEX < n_reps):
                        raise ValueError(
                            f"GRID_REPEAT_INDEX {GRID_REPEAT_INDEX} out of range "
                            f"(file has {n_reps} repeats)."
                        )
                    data.spectra = data.spectra_all[GRID_REPEAT_INDEX]
                    if data.cal_spectra_all is not None:
                        data.cal_spectra = data.cal_spectra_all[GRID_REPEAT_INDEX]
                harm_idx = int(np.where(data.harmonics == HARMONIC)[0][0])
                if NORMALIZE == 'percent':
                    vals = (
                        10.0 ** (data.normalized_peak_powers_dbm()[:, harm_idx] / 10.0)
                        * 100.0
                    )
                elif NORMALIZE:
                    vals = data.normalized_peak_powers_dbm()[:, harm_idx]
                else:
                    vals = data.peak_powers_dbm()[:, harm_idx]
                peak_map[i, j] = float(vals.max())
            except Exception as exc:
                print(f"Warning: could not load grid ({i},{j}): {exc}")

    print(f"Peak map range: {np.nanmin(peak_map):.2f} – {np.nanmax(peak_map):.2f}")

    axes_w, axes_h = axes_width_mm, axes_height_mm
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')

    if NORMALIZE == 'percent':
        cb_lbl = f'Max harmonic {HARMONIC} power [% of carrier]'
    elif NORMALIZE:
        cb_lbl = f'Max harmonic {HARMONIC} power [dBc]'
    else:
        cb_lbl = f'Max harmonic {HARMONIC} power [dBm]'

    # ── theory map (computed early so both plots can share its color/z scale) ─
    peak_map_theory = V1_theory = V2_theory = None
    if SHOW_THEORETICAL:
        n1_theory = THEORY_N1 if THEORY_N1 is not None else n1
        n2_theory = THEORY_N2 if THEORY_N2 is not None else n2
        v1t_min, v1t_max = THEORY_V1_RANGE if THEORY_V1_RANGE is not None else (v1.min(), v1.max())
        v2t_min, v2t_max = THEORY_V2_RANGE if THEORY_V2_RANGE is not None else (v2.min(), v2.max())
        v1_theory = np.linspace(v1t_min, v1t_max, n1_theory)
        v2_theory = np.linspace(v2t_min, v2t_max, n2_theory)
        V1_theory, V2_theory = np.meshgrid(v1_theory, v2_theory, indexing='ij')

        peak_map_theory = np.full((n1_theory, n2_theory), np.nan)
        for i, vi in enumerate(v1_theory):
            for j, vj in enumerate(v2_theory):
                beta1 = np.pi * vi / VPI1
                beta2 = np.pi * vj / VPI2
                k_trunc = (K_TRUNC_THEORY if K_TRUNC_THEORY is not None
                           else int(2 * max(beta1, beta2)) + 20)
                frac = _theory_peak_fraction(beta1, beta2, HARMONIC,
                                             N_PHASE_THEORY, k_trunc)
                if NORMALIZE == 'percent':
                    peak_map_theory[i, j] = frac * 100.0
                elif NORMALIZE:
                    peak_map_theory[i, j] = 10.0 * np.log10(max(frac, 1e-30))
                else:
                    peak_map_theory[i, j] = frac * 100.0

        print(f"Theory map range: {np.nanmin(peak_map_theory):.2f} – "
              f"{np.nanmax(peak_map_theory):.2f}")

    # Color scale (and, for 3D, z-axis range) shared between both plots:
    # explicit CMIN/CMAX win; otherwise use the theory map's range so the
    # experiment and theory figures are always normalized/scaled the same way.
    if CMIN is not None and CMAX is not None:
        levels = np.linspace(CMIN, CMAX, N_LEVELS + 1)
    elif peak_map_theory is not None:
        levels = np.linspace(float(np.nanmin(peak_map_theory)),
                              float(np.nanmax(peak_map_theory)), N_LEVELS + 1)
    else:
        levels = N_LEVELS

    # ── experimental figure ───────────────────────────────────────────────────
    if PLOT_3D_SURFACE:
        fig, ax, cb = _plot_surface_fig(V1, V2, peak_map, levels, cb_lbl,
                                         WIREFRAME_STRIDE_EXPERIMENT)
        if FOR_PUBLICATION:
            _pub_strip_3d(ax, cb)
            svg_path = _os.path.join(SAVE_FOLDER, '2d_power_phase_experiment_3d.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path}")
    else:
        fig, ax = _make_contour_fig(axes_w, axes_h)
        cs = ax.contourf(V1, V2, np.ma.masked_invalid(peak_map), levels=levels, cmap=SIDEBAND_CMAP)
        cb = fig.colorbar(cs, ax=ax)
        cb.ax.tick_params(labelsize=tick_label_fontsize)
        cb.set_label(cb_lbl, fontsize=axis_label_fontsize)

        ax.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', direction=tick_direction,
                       width=tick_width, labelsize=tick_label_fontsize)
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(spine_linewidth)

        if FOR_PUBLICATION:
            if PUB_CONTOUR_COLOR is not None:
                ax.contour(V1, V2, np.ma.masked_invalid(peak_map),
                           levels=levels, colors=PUB_CONTOUR_COLOR,
                           linewidths=PUB_CONTOUR_WIDTH)
            _pub_strip(ax, cb)
            svg_path = _os.path.join(SAVE_FOLDER, '2d_power_phase_experiment.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path}")

    if not SHOW_THEORETICAL:
        plt.show()
        return

    # ── theoretical figure ────────────────────────────────────────────────────
    theory_masked = np.ma.masked_invalid(peak_map_theory)
    theory_unit = ('% of carrier' if NORMALIZE == 'percent'
                   else ('dBc' if NORMALIZE else '% of carrier'))
    cb2_lbl = f'Max harmonic {HARMONIC} power [{theory_unit}]  (theory)'
    theory_title = rf'Theory:  $V_{{\pi,1}}={VPI1}$ V,  $V_{{\pi,2}}={VPI2}$ V'

    if PLOT_3D_SURFACE:
        fig2, ax2, cb2 = _plot_surface_fig(V1_theory, V2_theory, peak_map_theory, levels, cb2_lbl,
                                            WIREFRAME_STRIDE_THEORY)
        ax2.set_title(theory_title, fontsize=axis_label_fontsize)
        if SHOW_EXPERIMENT_BOUNDARY:
            _plot_experiment_boundary_3d(ax2, (v1.min(), v1.max()), (v2.min(), v2.max()))
        if FOR_PUBLICATION:
            _pub_strip_3d(ax2, cb2)
            svg_path2 = _os.path.join(SAVE_FOLDER, '2d_power_phase_theory_3d.svg')
            fig2.savefig(svg_path2, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path2}")
    else:
        fig2, ax2 = _make_contour_fig(axes_w, axes_h)
        cs2 = ax2.contourf(V1_theory, V2_theory, theory_masked, levels=levels, cmap=SIDEBAND_CMAP)
        cb2 = fig2.colorbar(cs2, ax=ax2)
        cb2.ax.tick_params(labelsize=tick_label_fontsize)
        cb2.set_label(cb2_lbl, fontsize=axis_label_fontsize)
        ax2.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
        ax2.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
        ax2.set_title(theory_title, fontsize=axis_label_fontsize)
        ax2.tick_params(axis='both', direction=tick_direction,
                        width=tick_width, labelsize=tick_label_fontsize)
        for side in ['top', 'bottom', 'left', 'right']:
            ax2.spines[side].set_linewidth(spine_linewidth)

        if FOR_PUBLICATION:
            if PUB_CONTOUR_COLOR is not None:
                ax2.contour(V1_theory, V2_theory, theory_masked,
                            levels=levels, colors=PUB_CONTOUR_COLOR,
                            linewidths=PUB_CONTOUR_WIDTH)
            _pub_strip(ax2, cb2)
            svg_path2 = _os.path.join(SAVE_FOLDER, '2d_power_phase_theory.svg')
            fig2.savefig(svg_path2, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path2}")

    plt.show()


if __name__ == '__main__':
    main()
