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
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\2d_power_phase_2026-06-17-14-28-55"

# Harmonic order to plot (must be in the harmonics list recorded by the script)
HARMONIC = 1

# Normalization for the colour axis:
#   False     → dBm  (raw ESA peak power)
#   True      → dBc  (relative to per-grid-point RF-off calibration)
#   'percent' → %    (fraction of carrier power, linear)
NORMALIZE = 'percent'

# Colour axis limits. None = auto-scale.
CMIN = 0
CMAX = 60

# Number of filled contour levels
N_LEVELS = 20

# ── theoretical figure ────────────────────────────────────────────
SHOW_THEORETICAL = True

# Half-wave voltage [V_rms] for each channel.
VPI1 = 4.8   # ch1, drives at f
VPI2 = 1.77   # ch2, drives at 2f

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


def main():
    meta = np.load(os.path.join(FOLDER, 'grid_meta.npz'))
    ch1_powers_dbm = meta['ch1_powers_dbm']
    ch2_powers_dbm = meta['ch2_powers_dbm']
    n1, n2 = len(ch1_powers_dbm), len(ch2_powers_dbm)

    v1 = _dbm_to_vrms(ch1_powers_dbm)   # (n1,) V_rms
    v2 = _dbm_to_vrms(ch2_powers_dbm)   # (n2,) V_rms

    peak_map = np.full((n1, n2), np.nan)

    for i in range(n1):
        for j in range(n2):
            fpath = os.path.join(FOLDER, f'grid_{i:02d}_{j:02d}.npz')
            if not os.path.exists(fpath):
                print(f"Missing: grid_{i:02d}_{j:02d}.npz")
                continue
            try:
                data = DualToneSweepData.from_file(fpath)
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

    # --- figure ---
    mm = 1.0 / 25.4
    axes_w = _default_axes_w
    axes_h = _default_axes_h
    fig, ax = plt.subplots(figsize=(
        (_left_mm + axes_w + _right_mm + 20) * mm,   # +20 mm for colorbar
        (_bottom_mm + axes_h + _top_mm) * mm,
    ))
    fig.subplots_adjust(
        left=_left_mm / (_left_mm + axes_w + _right_mm + 20),
        right=(_left_mm + axes_w) / (_left_mm + axes_w + _right_mm + 20),
        bottom=_bottom_mm / (_bottom_mm + axes_h + _top_mm),
        top=(_bottom_mm + axes_h) / (_bottom_mm + axes_h + _top_mm),
    )

    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    peak_masked = np.ma.masked_invalid(peak_map)

    if CMIN is not None and CMAX is not None:
        levels = np.linspace(CMIN, CMAX, N_LEVELS + 1)
    else:
        levels = N_LEVELS

    cs = ax.contourf(V1, V2, peak_masked, levels=levels, cmap='viridis')
    cb = fig.colorbar(cs, ax=ax)
    cb.ax.tick_params(labelsize=tick_label_fontsize)

    if NORMALIZE == 'percent':
        cb.set_label(
            f'Max harmonic {HARMONIC} power [% of carrier]',
            fontsize=axis_label_fontsize,
        )
    elif NORMALIZE:
        cb.set_label(
            f'Max harmonic {HARMONIC} power [dBc]',
            fontsize=axis_label_fontsize,
        )
    else:
        cb.set_label(
            f'Max harmonic {HARMONIC} power [dBm]',
            fontsize=axis_label_fontsize,
        )

    ax.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax.tick_params(
        axis='both', direction=tick_direction,
        width=tick_width, labelsize=tick_label_fontsize,
    )
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)

    plt.show()

    if not SHOW_THEORETICAL:
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
                peak_map_theory[i, j] = frac * 100.0   # percent fallback for theory

    print(f"Theory map range : {np.nanmin(peak_map_theory):.2f} – "
          f"{np.nanmax(peak_map_theory):.2f}")

    fig2, ax2 = plt.subplots(figsize=(
        (_left_mm + axes_w + _right_mm + 20) * mm,
        (_bottom_mm + axes_h + _top_mm) * mm,
    ))
    fig2.subplots_adjust(
        left=_left_mm / (_left_mm + axes_w + _right_mm + 20),
        right=(_left_mm + axes_w) / (_left_mm + axes_w + _right_mm + 20),
        bottom=_bottom_mm / (_bottom_mm + axes_h + _top_mm),
        top=(_bottom_mm + axes_h) / (_bottom_mm + axes_h + _top_mm),
    )

    theory_masked = np.ma.masked_invalid(peak_map_theory)
    cs2 = ax2.contourf(V1, V2, theory_masked, levels=levels, cmap='viridis')
    cb2 = fig2.colorbar(cs2, ax=ax2)
    cb2.ax.tick_params(labelsize=tick_label_fontsize)

    theory_unit = ('% of carrier' if NORMALIZE == 'percent'
                   else ('dBc' if NORMALIZE else '% of carrier'))
    cb2.set_label(
        f'Max harmonic {HARMONIC} power [{theory_unit}]  (theory)',
        fontsize=axis_label_fontsize,
    )
    ax2.set_xlabel(r'Ch1 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax2.set_ylabel(r'Ch2 drive voltage [V$_\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax2.set_title(
        rf'Theory:  $V_{{\pi,1}}={VPI1}$ V,  $V_{{\pi,2}}={VPI2}$ V',
        fontsize=axis_label_fontsize,
    )
    ax2.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax2.spines[side].set_linewidth(spine_linewidth)

    plt.show()


if __name__ == '__main__':
    main()
