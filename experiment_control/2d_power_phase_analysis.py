"""
Analyse a 2D power-phase grid sweep produced by bnc_2d_power_phase_script.py.

Loads all per-grid-point .npz files from a folder, extracts the maximum
sideband power (over the phase sweep) at a chosen harmonic, and plots a
contour map with Ch1 and Ch2 RMS voltages on the axes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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

FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\2d_power_phase_2026-06-24-14-52-14"

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

# ── publication export ────────────────────────────────────────────
# When True: removes tick labels, axis labels, colorbar labels, and title,
# and saves SVGs to SAVE_FOLDER.
FOR_PUBLICATION = False
SAVE_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"

# Contour lines drawn on top of the filled map when FOR_PUBLICATION = True.
# Set PUB_CONTOUR_COLOR = None to skip contour lines entirely.
PUB_CONTOUR_COLOR     = '#000000'   # any matplotlib color, or None to skip
PUB_CONTOUR_WIDTH     = 0       # linewidth in points

# ── theoretical figure ────────────────────────────────────────────
SHOW_THEORETICAL = True

# Half-wave voltage [V_rms] for each channel.
VPI1 = 5   # ch1, drives at f
VPI2 = 2.3   # ch2, drives at 2f

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

    if CMIN is not None and CMAX is not None:
        levels = np.linspace(CMIN, CMAX, N_LEVELS + 1)
    else:
        levels = N_LEVELS

    V1, V2 = np.meshgrid(v1, v2, indexing='ij')

    # ── experimental figure ───────────────────────────────────────────────────
    fig, ax = _make_contour_fig(axes_w, axes_h)
    cs = ax.contourf(V1, V2, np.ma.masked_invalid(peak_map), levels=levels, cmap='viridis')
    cb = fig.colorbar(cs, ax=ax)
    cb.ax.tick_params(labelsize=tick_label_fontsize)

    if NORMALIZE == 'percent':
        cb_lbl = f'Max harmonic {HARMONIC} power [% of carrier]'
    elif NORMALIZE:
        cb_lbl = f'Max harmonic {HARMONIC} power [dBc]'
    else:
        cb_lbl = f'Max harmonic {HARMONIC} power [dBm]'
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
    peak_map_theory = np.full((n1, n2), np.nan)
    for i, vi in enumerate(v1):
        for j, vj in enumerate(v2):
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

    fig2, ax2 = _make_contour_fig(axes_w, axes_h)
    theory_masked = np.ma.masked_invalid(peak_map_theory)
    cs2 = ax2.contourf(V1, V2, theory_masked, levels=levels, cmap='viridis')
    cb2 = fig2.colorbar(cs2, ax=ax2)
    cb2.ax.tick_params(labelsize=tick_label_fontsize)

    theory_unit = ('% of carrier' if NORMALIZE == 'percent'
                   else ('dBc' if NORMALIZE else '% of carrier'))
    cb2.set_label(f'Max harmonic {HARMONIC} power [{theory_unit}]  (theory)',
                  fontsize=axis_label_fontsize)
    ax2.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax2.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax2.set_title(rf'Theory:  $V_{{\pi,1}}={VPI1}$ V,  $V_{{\pi,2}}={VPI2}$ V',
                  fontsize=axis_label_fontsize)
    ax2.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax2.spines[side].set_linewidth(spine_linewidth)

    if FOR_PUBLICATION:
        if PUB_CONTOUR_COLOR is not None:
            ax2.contour(V1, V2, theory_masked,
                        levels=levels, colors=PUB_CONTOUR_COLOR,
                        linewidths=PUB_CONTOUR_WIDTH)
        _pub_strip(ax2, cb2)
        svg_path2 = _os.path.join(SAVE_FOLDER, '2d_power_phase_theory.svg')
        fig2.savefig(svg_path2, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path2}")

    plt.show()


if __name__ == '__main__':
    main()
