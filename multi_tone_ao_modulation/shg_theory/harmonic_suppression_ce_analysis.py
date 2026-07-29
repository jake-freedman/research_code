"""
Analyse a harmonic_suppression_test.py (experiment_control) output file:
the BNC signal generator's own fundamental/2nd-harmonic (SH) content, driven
directly into the ESA with no optical path.

Produces three figures vs. drive power (or voltage):

1. SH suppression: measured n=2 power relative to n=1, in dBc.
2. beta2: the phase-modulation depth the SG's own leaked 2nd-harmonic
   content would impart at 2*Omega, given a user-supplied Vpi at 2*Omega.
3. Sideband conversion efficiency: assuming this same drive signal is
   applied to a phase modulator with Vpi(Omega) and Vpi(2*Omega), the
   resulting comb via the two-tone Jacobi-Anger expansion

       theta(t) = beta1*sin(Omega*t) + beta2*sin(2*Omega*t + phi2)

   beta1 = pi*V1/Vpi_omega, beta2 = pi*V2/Vpi_2omega, with V1/V2 the
   measured n=1/n=2 RF voltages at each drive step. phi2 (the relative
   phase between the two tones) is NOT measured by harmonic_suppression_test
   (which only records power spectra), so it is a fixed, user-supplied
   assumption (RELATIVE_PHASE_DEG below), held constant across every step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import jv as bessel_jv
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

_ORDER_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_EXTRA_COLORS = [BLUE2, GREEN2, PINK2, TAN2, DARKGREEN2, DARKBLUE2, DARKGRAY2]


def _order_color(n):
    return _ORDER_COLORS.get(int(n), _EXTRA_COLORS[abs(int(n)) % len(_EXTRA_COLORS)])


# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

INPUT_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\signal_generator_harmonic_suppression\harmonic_suppression_2026-07-15-13-21-33.npz"

# 'dbm' or 'voltage' -- shared x-axis for all three plots below.
X_AXIS = 'voltage'

# ── modulator Vpi ──────────────────────────────────────────────────────────
VPI_OMEGA_V  = 5.7    # Vpi at the fundamental drive frequency (Omega)
VPI_2OMEGA_V = 2.5    # Vpi at 2x the fundamental drive frequency (2*Omega)

# Insertion loss (dB) between the SG output and the actual modulator input
# (e.g. cables/connectors) -- subtracted from the measured n=1/n=2 power
# before converting to voltage for beta1/beta2 (and therefore the beta2 and
# sideband-CE plots below), so those reflect the power that actually reaches
# the modulator, not the power measured directly at the SG/ESA.
IL_DB = 4.0

# Fixed relative phase (deg) between the fundamental and 2nd-harmonic tones
# in the two-tone Jacobi-Anger comb -- not measured by
# harmonic_suppression_test.py, so this is a user-supplied assumption held
# constant across every drive-power step.
RELATIVE_PHASE_DEG = 0.0

# Sideband orders to show in the conversion-efficiency plot.
CE_ORDERS = [-3, -2, -1, 1, 2, 3]

# CE y-axis units: 'percent' (% of total input power) or 'dBc'.
CE_NORMALIZE = 'percent'

# Bessel-sum truncation index. None = auto (computed from the data's actual
# beta1/beta2 range each run); set an int to override.
K_TRUNC = None

# ── figure size (shared by all three plots) ───────────────────────────────
axes_width_mm  = 100
axes_height_mm = 40

# Folder "for publication" SVGs are saved into. Must not be this script's
# own folder -- point it at a media subfolder (or set your own).
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"

# ── SH suppression plot ────────────────────────────────────────────────────
SUPPRESSION_LINESTYLE         = '-'
SUPPRESSION_LINEWIDTH         = 1.5
SUPPRESSION_LINE_COLOR        = RED2
SUPPRESSION_LINE_ALPHA        = 1.0
SUPPRESSION_MARKER            = 'o'
SUPPRESSION_MARKER_SIZE       = 5       # points
SUPPRESSION_MARKER_FACE_COLOR = 'same'  # 'same' = match SUPPRESSION_LINE_COLOR
SUPPRESSION_MARKER_FACE_ALPHA = 1.0
SUPPRESSION_MARKER_EDGE_COLOR = '#000000'
SUPPRESSION_MARKER_EDGE_ALPHA = 1.0
SUPPRESSION_MARKER_EDGE_WIDTH = 0.5     # points
SUPPRESSION_ZORDER            = 2
SUPPRESSION_SHOW_GRID         = False
SUPPRESSION_SHOW_LEGEND       = False
SUPPRESSION_XLIM = None   # (xmin, xmax) or None = auto
SUPPRESSION_YLIM = None   # (ymin, ymax) or None = auto

SUPPRESSION_FOR_PUBLICATION  = False
SUPPRESSION_PUBLICATION_SVG  = 'sh_suppression.svg'

# ── beta2 (at 2*Omega) vs. drive plot ──────────────────────────────────────
BETA2_LINESTYLE         = '-'
BETA2_LINEWIDTH         = 1.5
BETA2_LINE_COLOR        = VIOLET2
BETA2_LINE_ALPHA        = 1.0
BETA2_MARKER            = 'o'
BETA2_MARKER_SIZE       = 5       # points
BETA2_MARKER_FACE_COLOR = 'same'  # 'same' = match BETA2_LINE_COLOR
BETA2_MARKER_FACE_ALPHA = 1.0
BETA2_MARKER_EDGE_COLOR = '#000000'
BETA2_MARKER_EDGE_ALPHA = 1.0
BETA2_MARKER_EDGE_WIDTH = 0.5     # points
BETA2_ZORDER            = 2
BETA2_SHOW_GRID         = False
BETA2_SHOW_LEGEND       = False
BETA2_XLIM = None   # (xmin, xmax) or None = auto
BETA2_YLIM = None   # (ymin, ymax) or None = auto

BETA2_FOR_PUBLICATION = False
BETA2_PUBLICATION_SVG = 'beta2_vs_drive.svg'

# ── sideband conversion-efficiency plot ────────────────────────────────────
CE_LINESTYLE         = '-'
CE_LINEWIDTH         = 1.5
CE_LINE_ALPHA        = 1.0
CE_MARKER            = 'o'
CE_MARKER_SIZE       = 4       # points
CE_MARKER_FACE_ALPHA = 1.0
CE_MARKER_EDGE_COLOR = 'same'  # 'same' = match that order's line color
CE_MARKER_EDGE_ALPHA = 1.0
CE_MARKER_EDGE_WIDTH = 0.0     # points
CE_ZORDER            = 2
CE_SHOW_GRID         = False
CE_SHOW_LEGEND       = True
CE_XLIM = None   # (xmin, xmax) or None = auto
CE_YLIM = None   # (ymin, ymax) or None = auto

CE_FOR_PUBLICATION = False
CE_PUBLICATION_SVG = 'sideband_ce_vs_drive.svg'

# ------------------------------------------------------------------


def _dbm_to_vrms(power_dbm):
    return 10.0 ** ((np.asarray(power_dbm, dtype=float) - 10.0 * np.log10(20.0)) / 20.0)


def _make_fig(w_mm, h_mm):
    mm = 1.0 / 25.4
    left_mm, right_mm, bottom_mm, top_mm = 18.0, 8.0, 14.0, 8.0
    fig_w = left_mm + w_mm + right_mm
    fig_h = bottom_mm + h_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w * mm, fig_h * mm))
    fig.subplots_adjust(
        left=left_mm / fig_w, right=1 - right_mm / fig_w,
        bottom=bottom_mm / fig_h, top=1 - top_mm / fig_h,
    )
    return fig, ax


def _style_ax(ax):
    ax.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)


def _finish_plot(fig, ax, for_publication, svg_name, has_legend):
    if for_publication:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)
        if has_legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        svg_path = os.path.join(SAVE_FOLDER, svg_name)
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path}")


def _dual_tone_amplitudes(beta1_arr, beta2_arr, phi2_rad, orders, k_trunc):
    """
    Vectorized two-tone Jacobi-Anger amplitudes A_n for
        theta(t) = beta1*sin(Omega*t) + beta2*sin(2*Omega*t + phi2),
    phi1 = 0 fixed reference, phi2 a single fixed relative phase (radians).
    beta1_arr/beta2_arr are (M,) arrays (one value per drive-power step);
    phi2_rad is a scalar. Returns {order: (M,) array}.
    """
    beta1_arr = np.asarray(beta1_arr, dtype=float)
    beta2_arr = np.asarray(beta2_arr, dtype=float)
    k = np.arange(-k_trunc, k_trunc + 1)
    Jk_beta2 = bessel_jv(k[:, None], beta2_arr[None, :])          # (K, M)
    phase = np.exp(1j * k[:, None] * phi2_rad)                     # (K, 1), broadcasts

    result = {}
    for n in orders:
        Jn_beta1 = bessel_jv((n - 2 * k)[:, None], beta1_arr[None, :])   # (K, M)
        result[n] = np.sum(Jn_beta1 * Jk_beta2 * phase, axis=0)
    return result


def main():
    d = np.load(INPUT_FILE, allow_pickle=True)
    drive_powers = d['drive_powers']
    spectra = d['spectra']            # (M, N, K)
    labels = [str(l) for l in d['labels']]
    drive_freq = float(d['drive_freq'])

    if 'n=1' not in labels or 'n=2' not in labels:
        raise ValueError(
            f"Input file must include both 'n=1' and 'n=2' harmonic windows; "
            f"found {labels}."
        )
    idx1 = labels.index('n=1')
    idx2 = labels.index('n=2')

    peak_powers_dbm = spectra.max(axis=2)   # (M, N)
    p1_dbm = peak_powers_dbm[:, idx1]
    p2_dbm = peak_powers_dbm[:, idx2]
    suppression_dbc = p2_dbm - p1_dbm

    if X_AXIS == 'voltage':
        x = _dbm_to_vrms(drive_powers)
        xlabel = r'Drive voltage [V$_\mathrm{rms}$]'
    else:
        x = drive_powers
        xlabel = 'Drive power [dBm]'

    print(f"Loaded: {INPUT_FILE}")
    print(f"  Drive freq   : {drive_freq / 1e9:.4f} GHz")
    print(f"  Drive powers : {drive_powers[0]:+.1f} to {drive_powers[-1]:+.1f} dBm "
          f"({len(drive_powers)} steps)")
    print(f"  Windows      : {labels}")
    print(f"  SH suppression: {suppression_dbc.min():+.2f} to {suppression_dbc.max():+.2f} dBc")

    # ── Figure 1: SH suppression vs. drive ────────────────────────────────
    sup_line_color = mcolors.to_rgba(SUPPRESSION_LINE_COLOR, alpha=SUPPRESSION_LINE_ALPHA)
    sup_face_base = (SUPPRESSION_LINE_COLOR if SUPPRESSION_MARKER_FACE_COLOR == 'same'
                      else SUPPRESSION_MARKER_FACE_COLOR)
    sup_face_color = mcolors.to_rgba(sup_face_base, alpha=SUPPRESSION_MARKER_FACE_ALPHA)
    sup_edge_color = mcolors.to_rgba(SUPPRESSION_MARKER_EDGE_COLOR, alpha=SUPPRESSION_MARKER_EDGE_ALPHA)

    fig1, ax1 = _make_fig(axes_width_mm, axes_height_mm)
    ax1.plot(x, suppression_dbc,
              linestyle=SUPPRESSION_LINESTYLE, linewidth=SUPPRESSION_LINEWIDTH, color=sup_line_color,
              marker=SUPPRESSION_MARKER, markersize=SUPPRESSION_MARKER_SIZE,
              markerfacecolor=sup_face_color, markeredgecolor=sup_edge_color,
              markeredgewidth=SUPPRESSION_MARKER_EDGE_WIDTH,
              solid_capstyle='round', zorder=SUPPRESSION_ZORDER, label='SH suppression')
    if SUPPRESSION_XLIM is not None:
        ax1.set_xlim(SUPPRESSION_XLIM)
    if SUPPRESSION_YLIM is not None:
        ax1.set_ylim(SUPPRESSION_YLIM)
    ax1.grid(SUPPRESSION_SHOW_GRID)
    if SUPPRESSION_SHOW_LEGEND:
        ax1.legend(fontsize=tick_label_fontsize, frameon=False)
    ax1.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax1.set_ylabel('SH suppression [dBc]', fontsize=axis_label_fontsize)
    _style_ax(ax1)
    _finish_plot(fig1, ax1, SUPPRESSION_FOR_PUBLICATION, SUPPRESSION_PUBLICATION_SVG,
                 SUPPRESSION_SHOW_LEGEND)

    # ── beta1 (Omega), beta2 (2*Omega) from measured V1, V2, minus IL ─────
    v1 = _dbm_to_vrms(p1_dbm - IL_DB)
    v2 = _dbm_to_vrms(p2_dbm - IL_DB)
    beta1 = np.pi * v1 / VPI_OMEGA_V
    beta2 = np.pi * v2 / VPI_2OMEGA_V
    print(f"  IL             : {IL_DB:.2f} dB")
    print(f"  beta1 (Omega)  : {beta1.min():.4f} to {beta1.max():.4f} rad")
    print(f"  beta2 (2*Omega): {beta2.min():.4f} to {beta2.max():.4f} rad")

    # ── Figure 2: beta2 vs. drive ──────────────────────────────────────────
    b2_line_color = mcolors.to_rgba(BETA2_LINE_COLOR, alpha=BETA2_LINE_ALPHA)
    b2_face_base = BETA2_LINE_COLOR if BETA2_MARKER_FACE_COLOR == 'same' else BETA2_MARKER_FACE_COLOR
    b2_face_color = mcolors.to_rgba(b2_face_base, alpha=BETA2_MARKER_FACE_ALPHA)
    b2_edge_color = mcolors.to_rgba(BETA2_MARKER_EDGE_COLOR, alpha=BETA2_MARKER_EDGE_ALPHA)

    fig2, ax2 = _make_fig(axes_width_mm, axes_height_mm)
    ax2.plot(x, beta2,
              linestyle=BETA2_LINESTYLE, linewidth=BETA2_LINEWIDTH, color=b2_line_color,
              marker=BETA2_MARKER, markersize=BETA2_MARKER_SIZE,
              markerfacecolor=b2_face_color, markeredgecolor=b2_edge_color,
              markeredgewidth=BETA2_MARKER_EDGE_WIDTH,
              solid_capstyle='round', zorder=BETA2_ZORDER, label=r'$\beta_2$')
    if BETA2_XLIM is not None:
        ax2.set_xlim(BETA2_XLIM)
    if BETA2_YLIM is not None:
        ax2.set_ylim(BETA2_YLIM)
    ax2.grid(BETA2_SHOW_GRID)
    if BETA2_SHOW_LEGEND:
        ax2.legend(fontsize=tick_label_fontsize, frameon=False)
    ax2.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax2.set_ylabel(r'$\beta_2$ (2$\Omega$) [rad]', fontsize=axis_label_fontsize)
    _style_ax(ax2)
    _finish_plot(fig2, ax2, BETA2_FOR_PUBLICATION, BETA2_PUBLICATION_SVG, BETA2_SHOW_LEGEND)

    # ── Figure 3: sideband conversion efficiency vs. drive ─────────────────
    k_trunc = K_TRUNC if K_TRUNC is not None else int(2 * max(beta1.max(), beta2.max())) + 20
    phi2_rad = np.deg2rad(RELATIVE_PHASE_DEG)
    amps = _dual_tone_amplitudes(beta1, beta2, phi2_rad, CE_ORDERS, k_trunc)

    fig3, ax3 = _make_fig(axes_width_mm, axes_height_mm)
    print(f"  Sideband CE (at max drive):")
    for n in CE_ORDERS:
        frac = np.abs(amps[n]) ** 2
        if CE_NORMALIZE == 'percent':
            y = frac * 100.0
            unit = '%'
        else:
            y = 10.0 * np.log10(np.maximum(frac, 1e-30))
            unit = 'dBc'
        color = _order_color(n)
        edge_base = color if CE_MARKER_EDGE_COLOR == 'same' else CE_MARKER_EDGE_COLOR
        line_rgba = mcolors.to_rgba(color, alpha=CE_LINE_ALPHA)
        face_rgba = mcolors.to_rgba(color, alpha=CE_MARKER_FACE_ALPHA)
        edge_rgba = mcolors.to_rgba(edge_base, alpha=CE_MARKER_EDGE_ALPHA)
        ax3.plot(x, y,
                  linestyle=CE_LINESTYLE, linewidth=CE_LINEWIDTH, color=line_rgba,
                  marker=CE_MARKER, markersize=CE_MARKER_SIZE,
                  markerfacecolor=face_rgba, markeredgecolor=edge_rgba,
                  markeredgewidth=CE_MARKER_EDGE_WIDTH,
                  solid_capstyle='round', zorder=CE_ZORDER, label=f'n={n:+d}')
        print(f"    n={n:+d}: {y[-1]:.4f} {unit}")

    if CE_XLIM is not None:
        ax3.set_xlim(CE_XLIM)
    if CE_YLIM is not None:
        ax3.set_ylim(CE_YLIM)
    ax3.grid(CE_SHOW_GRID)
    if CE_SHOW_LEGEND:
        ax3.legend(fontsize=tick_label_fontsize, frameon=False, ncol=2)
    ax3.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax3.set_ylabel('Sideband CE [%]' if CE_NORMALIZE == 'percent' else 'Sideband CE [dBc]',
                    fontsize=axis_label_fontsize)
    _style_ax(ax3)
    _finish_plot(fig3, ax3, CE_FOR_PUBLICATION, CE_PUBLICATION_SVG, CE_SHOW_LEGEND)

    plt.show()


if __name__ == '__main__':
    main()
