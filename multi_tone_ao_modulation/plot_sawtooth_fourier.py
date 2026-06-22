import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
OMEGA     = 1.0   # angular frequency (rad/s)
AMPLITUDE = 1.0   # peak amplitude

# List of truncation orders to overlay.  Include 0 to also show the exact wave.
N_TERMS_LIST = [0, 2, 6, 10]

N_POINTS = 2000   # number of points per plot
GRID_MODE       = True   # True → 2×5 grid of N=1..10 instead of the overlay plot
GRID_SHOW_EXACT = False   # False → omit the dashed ideal-sawtooth from each panel
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
BLUE2      = '#2522d4'
GREEN2     = '#93C572'
RED2       = '#e5a3a3'
VIOLET2    = '#C2B7E9'
DARKGREEN2 = '#8DB591'
PINK2      = '#e6b8d0'
BEIGE2     = '#d9b99b'
DARKGRAY2  = '#777777'
LIGHTBLUE2 = '#b2cbf2'

SERIES_COLORS = [LIGHTBLUE2, GREEN2, PINK2, VIOLET2, DARKGREEN2, PINK2, BEIGE2]

axes_width_mm  = 150.0
axes_height_mm = 80.0
left_mm, right_mm  = 20.0, 5.0
bottom_mm, top_mm  = 12.0, 5.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

T      = 2 * np.pi / OMEGA
t      = np.linspace(0, 3 * T, N_POINTS)
# exact sawtooth: linear ramp from -A to +A over each period
sawtooth_exact = AMPLITUDE * (2 * ((t / T + 0.5) % 1) - 1)


def fourier_sawtooth(t, n_terms, omega, amplitude):
    """Sum_{n=1}^{n_terms} of the sawtooth Fourier series."""
    s = np.zeros_like(t, dtype=float)
    for n in range(1, n_terms + 1):
        s += ((-1) ** (n + 1)) / n * np.sin(n * omega * t)
    return (2 * amplitude / np.pi) * s


if not GRID_MODE:
    fig_w_in = fig_w / 25.4
    fig_h_in = fig_h / 25.4
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    color_idx = 0
    for n in N_TERMS_LIST:
        if n == 0:
            ax.plot(t * OMEGA / np.pi, sawtooth_exact,
                    color=DARKGRAY2, linewidth=1.2, linestyle='--',
                    label='exact')
        else:
            y = fourier_sawtooth(t, n, OMEGA, AMPLITUDE)
            color = SERIES_COLORS[color_idx % len(SERIES_COLORS)]
            ax.plot(t * OMEGA / np.pi, y,
                    color=color, linewidth=1.5,
                    label=rf'$N={n}$')
            color_idx += 1

    ax.axhline(0, color='black', linewidth=0.5, zorder=0)
    ax.set_xlabel(r'$\Omega t \, / \, \pi$', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$f(t)$', fontsize=axis_label_fontsize)
    ax.set_xlim(t[0] * OMEGA / np.pi, t[-1] * OMEGA / np.pi)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(fontsize=tick_label_fontsize, frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    out_path = Path(__file__).parent / 'sawtooth_fourier.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

else:
    # ── 2×5 grid of N=1..10 ──────────────────────────────────────────────────
    n_grid_rows, n_grid_cols = 2, 5
    panel_w_mm   = 52.0
    panel_h_mm   = 40.0
    hgap_mm      =  6.0
    vgap_mm      = 10.0
    gleft_mm     = 18.0
    gright_mm    =  5.0
    gbottom_mm   = 12.0
    gtop_mm      =  5.0

    gfig_w = gleft_mm + n_grid_cols * panel_w_mm + (n_grid_cols - 1) * hgap_mm + gright_mm
    gfig_h = gbottom_mm + n_grid_rows * panel_h_mm + (n_grid_rows - 1) * vgap_mm + gtop_mm

    fig_grid, axes_grid = plt.subplots(n_grid_rows, n_grid_cols,
                                       figsize=(gfig_w / 25.4, gfig_h / 25.4))

    panel_w_frac = panel_w_mm / gfig_w
    panel_h_frac = panel_h_mm / gfig_h
    hgap_frac    = hgap_mm    / gfig_w
    vgap_frac    = vgap_mm    / gfig_h
    gleft_frac   = gleft_mm   / gfig_w
    gbot_frac    = gbottom_mm / gfig_h

    _grid_cmap = LinearSegmentedColormap.from_list('', [LIGHTBLUE2, BLUE2])

    for n_idx in range(10):
        row = n_idx // n_grid_cols
        col = n_idx % n_grid_cols
        ax  = axes_grid[row, col]

        x0 = gleft_frac + col * (panel_w_frac + hgap_frac)
        y0 = gbot_frac  + (n_grid_rows - 1 - row) * (panel_h_frac + vgap_frac)
        ax.set_position([x0, y0, panel_w_frac, panel_h_frac])

        N = n_idx + 1
        y_approx = fourier_sawtooth(t, N, OMEGA, AMPLITUDE)
        if GRID_SHOW_EXACT:
            ax.plot(t * OMEGA / np.pi, sawtooth_exact,
                    color=DARKGRAY2, linewidth=0.8, linestyle='--')
        curve_color = _grid_cmap(n_idx / 9)
        ax.plot(t * OMEGA / np.pi, y_approx,
                color=curve_color, linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.4, zorder=0)
        ax.set_xlim(t[0] * OMEGA / np.pi, t[-1] * OMEGA / np.pi)
        ax.set_ylim(-1.5 * AMPLITUDE, 1.5 * AMPLITUDE)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.set_title(rf'$N={N}$', fontsize=tick_label_fontsize, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax.tick_params(axis='both', direction=tick_direction,
                       width=tick_width, labelsize=tick_label_fontsize - 1)

        if row == n_grid_rows - 1:
            ax.set_xlabel(r'$\Omega t \,/\, \pi$', fontsize=axis_label_fontsize - 1)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(r'$f(t)$', fontsize=axis_label_fontsize - 1)
        else:
            ax.tick_params(labelleft=False)

    out_path = Path(__file__).parent / 'sawtooth_fourier_grid.png'
    fig_grid.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

plt.show()
