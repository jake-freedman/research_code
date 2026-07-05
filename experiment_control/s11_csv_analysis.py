"""
Load and plot a single S11 (or S11+S21) CSV file saved by VNA.save_s11()
or VNA.save_s11_s21().

The file format is auto-detected from the column count:
    4 columns → S11 only   (frequency_hz, s11_real, s11_imag, s11_db)
    7 columns → S11 + S21  (... s21_real, s21_imag, s21_db)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from vna_control import S11Data, S11S21Data
from path_utils import local_path
from graphics import (
    VIOLET2,
    TAN2,
    BEIGE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ── data ──────────────────────────────────────────────────────────────────────
# One or more CSV files.  Each is plotted on the same axes, offset vertically
# by i * Y_SHIFT dB (file 0 is unshifted).
DATA_FILES = [
    r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg9a_p1_2026-06-15-19-13-00.csv",
    r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg14a_p1_2026-06-15-19-10-09.csv",
    r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg5a_p5_2026-06-15-19-27-36.csv",
    r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg11a_p1_2026-06-15-19-11-55.csv",
    r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg6a_p1_2026-06-15-18-50-41.csv"

]
Y_SHIFT = 6.5   # dB offset applied between consecutive files
# Per-file colors; None = auto-cycle through [BEIGE2, VIOLET2, TAN2, ...]
COLORS = None

# ── plot limits ───────────────────────────────────────────────────────────────
YMIN = -8.0   # dB, None = auto
YMAX =  28   # dB, None = auto
XMIN =  0.0   # GHz, None = auto
XMAX =  3.5   # GHz, None = auto

# ── resonance search ──────────────────────────────────────────────────────────
# Print the frequency of the S11 minimum within this window (GHz). None = full range.
FREQ_MIN = None
FREQ_MAX = None

# ── graphics ──────────────────────────────────────────────────────────────────
axes_width_mm  = 90
axes_height_mm = 50
left_mm   = _left_mm
right_mm  = _right_mm
bottom_mm = _bottom_mm
top_mm    = _top_mm
linewidth = 2
SHOW_GRID = False

# ── publication export ────────────────────────────────────────────────────────
# When True: removes all axis/tick labels and saves an SVG next to DATA_FILE.
FOR_PUBLICATION = True
# Override the SVG save path. None = DATA_FILE with .svg extension.
SAVE_PATH = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\s11_fig.svg"
# ─────────────────────────────────────────────────────────────────────────────


def _count_columns(filepath: str) -> int:
    with open(filepath) as f:
        for line in f:
            if not line.startswith('#'):
                return len(line.split(','))
    return 0


def main():
    _auto_colors = [BEIGE2, VIOLET2, TAN2]
    _colors = COLORS if COLORS is not None else _auto_colors

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

    last_freqs = None
    for i, raw_path in enumerate(DATA_FILES):
        fpath  = local_path(raw_path)
        ncols  = _count_columns(fpath)
        data   = S11S21Data.from_file(fpath) if ncols >= 7 else S11Data.from_file(fpath)
        freqs  = data.freqs
        s11_db = data.s11_db + i * Y_SHIFT
        color  = _colors[i % len(_colors)]
        last_freqs = freqs

        # ── resonance dip ─────────────────────────────────────────────────────
        mask = np.ones(len(freqs), dtype=bool)
        if FREQ_MIN is not None:
            mask &= freqs >= FREQ_MIN * 1e9
        if FREQ_MAX is not None:
            mask &= freqs <= FREQ_MAX * 1e9
        res_idx  = int(np.argmin(s11_db[mask]))
        res_freq = freqs[mask][res_idx]
        res_val  = s11_db[mask][res_idx]
        label = os.path.basename(fpath)
        print(f"[{label}]  resonance: {res_freq / 1e9:.6f} GHz  ({res_val:.2f} dB)"
              + (f"  (shift {i * Y_SHIFT:+.1f} dB)" if Y_SHIFT else ""))

        ax.plot(freqs / 1e9, s11_db, color=color, linewidth=linewidth)

    if XMIN is not None or XMAX is not None:
        ax.set_xlim(
            left  = XMIN if XMIN is not None else last_freqs[0]  / 1e9,
            right = XMAX if XMAX is not None else last_freqs[-1] / 1e9,
        )
    if YMIN is not None or YMAX is not None:
        ax.set_ylim(
            bottom = YMIN if YMIN is not None else None,
            top    = YMAX if YMAX is not None else None,
        )

    if SHOW_GRID:
        ax.grid(linewidth=0.4, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    ax.set_xlabel('Frequency [GHz]', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$S_{11}$ [dB]',  fontsize=axis_label_fontsize)

    # ── publication export ────────────────────────────────────────────────────
    if FOR_PUBLICATION:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

        svg_path = SAVE_PATH
        if svg_path is None:
            base = os.path.splitext(os.path.abspath(local_path(DATA_FILES[0])))[0]
            svg_path = base + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path}")

    plt.show()


if __name__ == '__main__':
    main()
