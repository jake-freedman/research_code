"""
Publication-grade plot of modulation depth β or half-wave voltage Vπ vs
drive frequency from a heterodyne sweep .npz file.

Set PLOT_MODE to 'beta' or 'vpi'. For Vpi mode, provide DRIVE_POWER_DBM
so the script can convert β to Vπ = π · V_rms / β.

Frequency highlight bands: the line is drawn in COLOR everywhere, but
smoothly transitions to a per-band highlight colour within ±BW/2 of each
centre frequency, with a cosine rolloff back to the base colour at the edges.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.ndimage import uniform_filter1d
from heterodyne_sweep_data import HeterodyneSweepData
from path_utils import local_path
from graphics import (
    BLUE2,
    RED2, VIOLET2, GREEN2, BEIGE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ── data ──────────────────────────────────────────────────────────────────────
DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\test_heterodyne_sweep_2026-06-17-16-20-31.npz"

# ── extraction ────────────────────────────────────────────────────────────────
HARMONIC_NUMERATOR   = 1
HARMONIC_DENOMINATOR = 0
BETA_GUESS = 1.0

# 'beta'     → y-axis is β [rad]
# 'vpi'      → y-axis is Vπ [V_rms]; requires DRIVE_POWER_DBM
# 'power_pi' → RF power required for a π phase shift; requires DRIVE_POWER_DBM
PLOT_MODE = 'power_pi'

# VNA drive power in dBm. Only used when PLOT_MODE = 'vpi' or 'power_pi'.
DRIVE_POWER_DBM = 5

# Units for power_pi mode: 'dBm' or 'mW'  (ignored for other modes)
POWER_PI_UNIT = 'dBm'

# ── smoothing ─────────────────────────────────────────────────────────────────
# None   → no smoothing
# int    → uniform moving average over that many points
# float  → uniform average over all points within ±SMOOTH/2 GHz
SMOOTH = None # 10

# ── plot limits ───────────────────────────────────────────────────────────────
YMIN = None
YMAX = None
XMIN = None   # GHz, None = auto
XMAX = None   # GHz, None = auto

# ── frequency highlights ──────────────────────────────────────────────────────
# Lists of equal length. The line colour smoothly transitions from COLOR to
# each HIGHLIGHT_COLORS[i] at the centre frequency, with a cosine rolloff back
# to COLOR at ±HIGHLIGHT_BWS[i]/2 GHz. Pass empty lists to disable.
HIGHLIGHT_CENTERS = [1.146, 2.290, 2.613, 2.990, 3.265]      # GHz,  e.g. [1.146]
HIGHLIGHT_BWS     = [0.50, 0.50, 0.3, 0.3, 0.3]      # GHz full-width,  e.g. [0.2]
HIGHLIGHT_COLORS  = [RED2, BLUE2, VIOLET2, GREEN2, BEIGE2]      # matplotlib colour strings, e.g. ['#ff4444']

# ── graphics ──────────────────────────────────────────────────────────────────
axes_width_mm  = 190
axes_height_mm = 50
left_mm   = _left_mm
right_mm  = _right_mm
bottom_mm = _bottom_mm
top_mm    = _top_mm
linewidth  = 3
COLOR      = '#cccccc'
SHOW_MARKERS = False
MARKERSIZE   = 4.0
SHOW_GRID    = False

# ── publication export ────────────────────────────────────────────────────────
FOR_PUBLICATION = True
SAVE_PATH = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\beta_vs_freq.svg"
# ─────────────────────────────────────────────────────────────────────────────

_LOG20 = 10.0 * np.log10(20.0)


def _smooth(freqs_ghz, y, smooth):
    if smooth is None:
        return y.copy()
    if isinstance(smooth, int):
        return uniform_filter1d(y, size=smooth, mode='nearest')
    # float: frequency window in GHz — average all points within ±smooth/2
    hw = float(smooth) / 2.0
    out = np.empty_like(y)
    for i, fc in enumerate(freqs_ghz):
        mask = np.abs(freqs_ghz - fc) <= hw
        out[i] = y[mask].mean()
    return out


def _point_colors(freqs_ghz, base_color, centers, bws, h_colors):
    """
    Return an (N, 3) RGB array blending base_color toward each highlight colour
    using a cosine rolloff inside each band. Overlapping bands are resolved by
    a weighted average of the highlight colours, clamped so total blend ≤ 1.
    """
    base_rgb = np.array(mcolors.to_rgb(base_color))
    N = len(freqs_ghz)

    if not centers:
        return np.tile(base_rgb, (N, 1))

    B = len(centers)
    weights = np.zeros((N, B))
    for b, (c, bw) in enumerate(zip(centers, bws)):
        df = np.abs(freqs_ghz - c)
        inside = df <= bw / 2.0
        weights[inside, b] = 0.5 * (1.0 + np.cos(np.pi * df[inside] / (bw / 2.0)))

    band_rgb = np.array([mcolors.to_rgb(c) for c in h_colors])  # (B, 3)
    w_sum = weights.sum(axis=1)                                   # (N,)
    total_w = np.clip(w_sum, 0.0, 1.0)

    # Weighted average of highlight colours at each point
    safe_sum = np.where(w_sum > 0, w_sum, 1.0)
    highlight_rgb = (weights[:, :, None] * band_rgb[None, :, :]).sum(axis=1) / safe_sum[:, None]

    return (1.0 - total_w[:, None]) * base_rgb + total_w[:, None] * highlight_rgb


def main():
    data = HeterodyneSweepData.from_file(local_path(DATA_FILE))
    betas = data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR, BETA_GUESS)
    freqs_ghz = data.cw_freqs / 1e9

    if PLOT_MODE in ('vpi', 'power_pi'):
        v_rms = 10.0 ** ((DRIVE_POWER_DBM - _LOG20) / 20.0)
        vpi = np.pi * v_rms / betas
        if PLOT_MODE == 'vpi':
            y = vpi
            ylabel = r'$V_\pi$ [V$_\mathrm{rms}$]'
        elif POWER_PI_UNIT == 'dBm':
            y = 20.0 * np.log10(vpi) + _LOG20
            ylabel = r'$P_\pi$ [dBm]'
        else:
            y = vpi ** 2 * 20.0
            ylabel = r'$P_\pi$ [mW]'
    else:
        y = betas
        ylabel = r'Modulation depth $\beta$ [rad]'

    y = _smooth(freqs_ghz, y, SMOOTH)

    print(f"Loaded: {DATA_FILE}")
    print(f"  {len(freqs_ghz)} CW steps: "
          f"{freqs_ghz[0]:.4f} – {freqs_ghz[-1]:.4f} GHz")
    print(f"  β  range: {betas.min():.4f} – {betas.max():.4f} rad")
    if PLOT_MODE == 'vpi':
        print(f"  Vπ range: {y.min():.4f} – {y.max():.4f} V")
    elif PLOT_MODE == 'power_pi':
        unit = POWER_PI_UNIT
        print(f"  Pπ range: {y.min():.4f} – {y.max():.4f} {unit}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    # ── line with optional colour highlights ──────────────────────────────────
    pt_colors = _point_colors(freqs_ghz, COLOR,
                               HIGHLIGHT_CENTERS, HIGHLIGHT_BWS, HIGHLIGHT_COLORS)

    if HIGHLIGHT_CENTERS:
        # Per-segment colour: average of the two endpoint colours
        xy = np.column_stack([freqs_ghz, y])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        seg_colors = 0.5 * (pt_colors[:-1] + pt_colors[1:])
        lc = LineCollection(segments, colors=seg_colors, linewidth=linewidth,
                            capstyle='round', joinstyle='round')
        ax.add_collection(lc)
        # LineCollection doesn't auto-scale; set limits explicitly below
    else:
        plot_kw = dict(color=COLOR, linewidth=linewidth)
        if SHOW_MARKERS:
            plot_kw.update(marker='o', markersize=MARKERSIZE, markeredgewidth=0)
        ax.plot(freqs_ghz, y, **plot_kw)

    if SHOW_MARKERS and HIGHLIGHT_CENTERS:
        ax.scatter(freqs_ghz, y, c=pt_colors,
                   s=MARKERSIZE**2, linewidths=0, zorder=3)

    # ── axes limits ───────────────────────────────────────────────────────────
    x_lo = XMIN if XMIN is not None else freqs_ghz.min()
    x_hi = XMAX if XMAX is not None else freqs_ghz.max()
    y_lo = YMIN if YMIN is not None else y.min()
    y_hi = YMAX if YMAX is not None else y.max()
    y_pad = (y_hi - y_lo) * 0.05 if y_hi != y_lo else 0.1
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    if SHOW_GRID:
        ax.grid(linewidth=0.4, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    ax.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # ── publication export ────────────────────────────────────────────────────
    if FOR_PUBLICATION:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

        svg_path = SAVE_PATH
        if svg_path is None:
            base = os.path.splitext(os.path.abspath(local_path(DATA_FILE)))[0]
            svg_path = base + f'_{PLOT_MODE}.svg'
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path}")

    plt.show()


if __name__ == '__main__':
    main()
