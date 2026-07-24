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
    RED2, VIOLET2, GREEN2, BEIGE2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ── data ──────────────────────────────────────────────────────────────────────
DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d21_wg23b_p5\test_heterodyne_sweep_2026-07-09-10-38-22.npz"

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
YMIN = 5
YMAX = 60
XMIN = None   # GHz, None = auto
XMAX = 3.5   # GHz, None = auto

# ── frequency highlights ──────────────────────────────────────────────────────
# Lists of equal length. The line colour smoothly transitions from COLOR to
# each HIGHLIGHT_COLORS[i] at the centre frequency, with a cosine rolloff back
# to COLOR at ±HIGHLIGHT_BWS[i]/2 GHz. Pass empty lists to disable.
HIGHLIGHT_CENTERS = [1.146, 2.290, 2.613, 2.990, 3.265]      # GHz,  e.g. [1.146]
HIGHLIGHT_BWS     = [0.15, 0.15, 0.15, 0.15, 0.15]      # GHz full-width,  e.g. [0.2]
HIGHLIGHT_COLORS  = [RED2, LIGHTBLUE2, VIOLET2, GREEN2, BEIGE2]      # matplotlib colour strings, e.g. ['#ff4444']

# ── graphics ──────────────────────────────────────────────────────────────────
axes_width_mm  = 190
axes_height_mm = 50
left_mm   = _left_mm
right_mm  = _right_mm
bottom_mm = _bottom_mm
top_mm    = _top_mm
linewidth  = 3
COLOR      = '#cccccc'
FILL_ALPHA       = 0.5   # alpha below the decay start
FILL_ALPHA_TOP   = 0.1   # alpha at the top of the axes (fill decays toward this)
FILL_DECAY_START = 40    # y-value at which the alpha begins to decay (same for all x)
SHOW_MARKERS = False
MARKERSIZE   = 4.0
SHOW_GRID    = False

# ── publication export ────────────────────────────────────────────────────────
FOR_PUBLICATION = True
SAVE_PATH = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media\beta_vs_freq.svg"

# ── zoom plot ─────────────────────────────────────────────────────────────────
# Overlays the main curve in ZOOM_XMIN–ZOOM_XMAX with a second series whose
# data come from half those frequencies, plotted with their x-coordinates
# doubled (so they land in the same window) and their original colours.
SHOW_ZOOM_PLOT      = True
ZOOM_XMIN           = 2.1    # GHz
ZOOM_XMAX           = 2.5    # GHz
ZOOM_YMIN           = 5
ZOOM_YMAX           = 60
ZOOM_AXES_WIDTH_MM  = 40
ZOOM_AXES_HEIGHT_MM = 20
ZOOM_SAVE_PATH      = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media\beta_vs_freq_inset.svg"
ZOOM_DBL_LINESTYLE  = '--'   # linestyle for the doubled-frequency overlay ('-', '--', ':', '-.')
ZOOM_DBL_FILL       = False  # whether to draw the fill for the doubled-frequency overlay
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
        weights[:, b] = 1.0 / (1.0 + (2.0 * df / bw) ** 2)

    band_rgb = np.array([mcolors.to_rgb(c) for c in h_colors])  # (B, 3)
    w_sum = weights.sum(axis=1)                                   # (N,)
    total_w = np.clip(w_sum, 0.0, 1.0)

    # Weighted average of highlight colours at each point
    safe_sum = np.where(w_sum > 0, w_sum, 1.0)
    highlight_rgb = (weights[:, :, None] * band_rgb[None, :, :]).sum(axis=1) / safe_sum[:, None]

    return (1.0 - total_w[:, None]) * base_rgb + total_w[:, None] * highlight_rgb


def _make_fig_ax(w_mm, h_mm):
    fig_w = left_mm + w_mm + right_mm
    fig_h = bottom_mm + h_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )
    return fig, ax


def _draw_series(ax, freqs_ghz, y, pt_colors, y_bot, y_top,
                 linestyle='-', show_fill=True):
    """Render fill + line for one data series onto ax. ax limits must already be set."""
    if show_fill:
        n_rows = 256
        y_vals = np.linspace(y_bot, y_top, n_rows)
        fill_rgba = np.zeros((n_rows, len(freqs_ghz), 4))
        fill_rgba[:, :, :3] = pt_colors[None, :, :]
        above    = y_vals[:, None] >= y[None, :]
        y_span   = max(y_top - FILL_DECAY_START, 1e-9)
        frac     = np.clip((y_vals[:, None] - FILL_DECAY_START) / y_span, 0.0, 1.0)
        alpha    = FILL_ALPHA + (FILL_ALPHA_TOP - FILL_ALPHA) * frac
        fill_rgba[:, :, 3] = np.where(above, alpha, 0.0)
        ax.imshow(fill_rgba, aspect='auto',
                  extent=[freqs_ghz[0], freqs_ghz[-1], y_bot, y_top],
                  origin='lower', interpolation='nearest', zorder=1)

    if HIGHLIGHT_CENTERS:
        xy = np.column_stack([freqs_ghz, y])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        seg_colors = 0.5 * (pt_colors[:-1] + pt_colors[1:])
        lc = LineCollection(segments, colors=seg_colors, linewidth=linewidth,
                            linestyle=linestyle,
                            capstyle='round', joinstyle='round', zorder=2)
        ax.add_collection(lc)
    else:
        plot_kw = dict(color=COLOR, linewidth=linewidth, linestyle=linestyle, zorder=2)
        if SHOW_MARKERS:
            plot_kw.update(marker='o', markersize=MARKERSIZE, markeredgewidth=0)
        ax.plot(freqs_ghz, y, **plot_kw)

    if SHOW_MARKERS and HIGHLIGHT_CENTERS:
        ax.scatter(freqs_ghz, y, c=pt_colors,
                   s=MARKERSIZE**2, linewidths=0, zorder=3)


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

    # ── per-x point colours (shared by both plots) ───────────────────────────
    pt_colors = _point_colors(freqs_ghz, COLOR,
                               HIGHLIGHT_CENTERS, HIGHLIGHT_BWS, HIGHLIGHT_COLORS)

    # ── main figure ───────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax(axes_width_mm, axes_height_mm)

    x_lo = XMIN if XMIN is not None else freqs_ghz.min()
    x_hi = XMAX if XMAX is not None else freqs_ghz.max()
    y_lo = YMIN if YMIN is not None else y.min()
    y_hi = YMAX if YMAX is not None else y.max()
    y_pad = (y_hi - y_lo) * 0.05 if y_hi != y_lo else 0.1
    y_bot = y_lo - y_pad
    y_top = y_hi + y_pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_bot, y_top)

    _draw_series(ax, freqs_ghz, y, pt_colors, y_bot, y_top)

    if SHOW_GRID:
        ax.grid(linewidth=0.4, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    ax.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

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

    fig.canvas.draw()
    print(f"  x ticks: {ax.get_xticks().tolist()}")
    print(f"  y ticks: {ax.get_yticks().tolist()}")

    # ── zoom figure ───────────────────────────────────────────────────────────
    if SHOW_ZOOM_PLOT:
        fig_z, ax_z = _make_fig_ax(ZOOM_AXES_WIDTH_MM, ZOOM_AXES_HEIGHT_MM)

        zm = (freqs_ghz >= ZOOM_XMIN) & (freqs_ghz <= ZOOM_XMAX)
        y_lo_z = ZOOM_YMIN if ZOOM_YMIN is not None else (y[zm].min() if zm.any() else y.min())
        y_hi_z = ZOOM_YMAX if ZOOM_YMAX is not None else (y[zm].max() if zm.any() else y.max())

        # doubled series: data at f/2, plotted at 2f with colour from f/2
        hm = (freqs_ghz >= ZOOM_XMIN / 2.0) & (freqs_ghz <= ZOOM_XMAX / 2.0)
        dbl_freqs  = freqs_ghz[hm] * 2.0
        dbl_y      = y[hm]
        dbl_colors = pt_colors[hm]

        if ZOOM_YMIN is None and hm.any():
            y_lo_z = min(y_lo_z, dbl_y.min())
        if ZOOM_YMAX is None and hm.any():
            y_hi_z = max(y_hi_z, dbl_y.max())

        y_pad_z = (y_hi_z - y_lo_z) * 0.05 if y_hi_z != y_lo_z else 0.1
        y_bot_z  = y_lo_z - y_pad_z
        y_top_z  = y_hi_z + y_pad_z
        ax_z.set_xlim(ZOOM_XMIN, ZOOM_XMAX)
        ax_z.set_ylim(y_bot_z, y_top_z)

        _draw_series(ax_z, freqs_ghz, y, pt_colors, y_bot_z, y_top_z)
        if hm.any():
            _draw_series(ax_z, dbl_freqs, dbl_y, dbl_colors, y_bot_z, y_top_z,
                         linestyle=ZOOM_DBL_LINESTYLE, show_fill=ZOOM_DBL_FILL)

        if SHOW_GRID:
            ax_z.grid(linewidth=0.4, alpha=0.6)
        for spine in ax_z.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax_z.tick_params(axis='both', direction=tick_direction,
                         width=tick_width, labelsize=tick_label_fontsize)
        ax_z.set_xlabel('Drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax_z.set_ylabel(ylabel, fontsize=axis_label_fontsize)

        if FOR_PUBLICATION:
            ax_z.set_xlabel('')
            ax_z.set_ylabel('')
            ax_z.tick_params(labelbottom=False, labelleft=False)
            svg_path_z = ZOOM_SAVE_PATH
            if svg_path_z is None:
                base = os.path.splitext(os.path.abspath(local_path(DATA_FILE)))[0]
                svg_path_z = base + f'_{PLOT_MODE}_zoom.svg'
            os.makedirs(os.path.dirname(svg_path_z), exist_ok=True)
            fig_z.savefig(svg_path_z, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path_z}")

        fig_z.canvas.draw()
        print(f"  zoom x ticks: {ax_z.get_xticks().tolist()}")
        print(f"  zoom y ticks: {ax_z.get_yticks().tolist()}")

    plt.show()


if __name__ == '__main__':
    main()
