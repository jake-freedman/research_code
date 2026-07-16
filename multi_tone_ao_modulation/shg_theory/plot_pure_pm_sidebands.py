"""
Pure phase-modulation sideband theory (no mechanical nonlinearity).

A single tone phase-modulates a shared optical mode:

    phi(t) = b*sin(Omega*t)
    E(t)   = E0 * exp(i*wL*t) * exp(i*phi(t))

By the Jacobi-Anger expansion, the sideband powers follow directly from a
single Bessel function per order -- no Bessel-sum truncation is needed here,
unlike the two-mode nonlinear case in plot_two_mode_ao_sidebands.py:

    P_n = J_n(b)^2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv as _jv
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

_COMBLINE_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Optical sidebands to track (order n -> sideband at wL + n*Omega)
sideband_orders = [-2, -1, 0, 1, 2]

# Modulation-depth sweep
b_min   = 0.0
b_max   = 5.0
n_pts   = 300
b_scale = 'linear'   # 'log' or 'linear'

# Output units:
#   'dBc' -> power relative to carrier (n=0) in dB
#   'eff' -> conversion efficiency in % (|J_n(b)|^2 of unit input)
OUTPUT = 'eff'

# Figure layout (mm)
axes_width_mm  = 120.0
axes_height_mm =  55.0
left_mm        =  18.0
right_mm       =   8.0
bottom_mm      =  14.0
top_mm         =   8.0

# Curve style
LINEWIDTH = 2.0
LINESTYLE = '-'
ALPHA     = 1.0

# Negative orders sit exactly on top of their positive counterparts (|J_-n(b)|
# = |J_n(b)|), so they're drawn last, dashed, so both remain visible.
NEGATIVE_LINESTYLE = '--'

# Save path (None = don't save)
SAVE_PNG = Path(__file__).parent / 'pure_pm_sideband_powers.png'

# ── publication export ────────────────────────────────────────────────────────
# When True: strips axis/tick labels and the legend, and saves an SVG
# (instead of the PNG above) next to this script.
FOR_PUBLICATION = False
SAVE_SVG = Path(__file__).parent / 'pure_pm_sideband_powers.svg'
# ─────────────────────────────────────────────────────────────────────────────


def _order_color(order):
    auto_idx = list(_COMBLINE_COLORS.keys()).index(order) if order in _COMBLINE_COLORS else (order + 10)
    return _COMBLINE_COLORS.get(int(order), _COLORS[auto_idx % len(_COLORS)])


def _make_fig_ax():
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm   / fig_h,
    )
    return fig, ax


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)


def run_sideband_theory():
    if b_scale == 'log':
        b_vals = np.logspace(np.log10(max(b_min, 1e-6)), np.log10(b_max), n_pts)
    else:
        b_vals = np.linspace(b_min, b_max, n_pts)

    sb_power = {n: _jv(n, b_vals) ** 2 for n in sideband_orders}

    if OUTPUT == 'dBc':
        carrier   = sb_power[0]
        plot_data = {n: 10 * np.log10(sb_power[n] / (carrier + 1e-30) + 1e-30)
                     for n in sideband_orders}
        ylabel = r'Sideband power [dBc]'
    else:
        plot_data = {n: 100 * sb_power[n] for n in sideband_orders}
        ylabel = r'Conversion efficiency [%]'

    fig, ax = _make_fig_ax()

    # Draw non-negative orders first (solid), then negative orders on top
    # (dashed) so overlapping +-n traces are both visible.
    for n in sorted(sideband_orders, key=lambda n: n < 0):
        lbl = f'n={n:+d}' if n != 0 else 'n=0 (carrier)'
        line, = ax.plot(b_vals, plot_data[n],
                         color=_order_color(n),
                         linewidth=LINEWIDTH,
                         linestyle=NEGATIVE_LINESTYLE if n < 0 else LINESTYLE,
                         alpha=ALPHA,
                         zorder=3 if n < 0 else 2,
                         label=lbl)
        if n < 0:
            line.set_dash_capstyle('butt')

    if b_scale == 'log':
        ax.set_xscale('log')
    ax.set_xlabel(r'Modulation depth $b$ [rad]', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_ax(ax)

    if FOR_PUBLICATION:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if SAVE_SVG:
            fig.savefig(SAVE_SVG, format='svg', bbox_inches='tight')
            print(f"Saved: {SAVE_SVG}")
    elif SAVE_PNG:
        fig.savefig(SAVE_PNG, dpi=200, bbox_inches='tight')
        print(f"Saved: {SAVE_PNG}")

    # Console summary at max modulation depth
    print(f"\nAt b = {b_vals[-1]:.3f}:")
    unit = 'dBc' if OUTPUT == 'dBc' else '%'
    for n in sideband_orders:
        print(f"  n={n:+d}: {plot_data[n][-1]:.3f} {unit}")

    plt.show()


if __name__ == '__main__':
    run_sideband_theory()
