"""
dark_window_theory_investigator.py

Pure-theory investigation of the "dark window" formed by jointly suppressing
harmonic orders in DARK_ORDERS (default: the carrier n=0 and the n=-1
sideband) of a dual-tone phase-modulated drive, as a function of the two
modulation depths beta1 (drive at f) and beta2 (drive at 2f).

At each (beta1, beta2) grid point, the relative phase phi2 is swept (ch1
phase phi1 is held fixed at PHI1_DEG -- only the relative phase phi2-phi1
matters physically) and the metric

    depth(beta1, beta2) = min_phi2  max_{p in DARK_ORDERS} |A_p(beta1, beta2, phi2)|^2

is recorded, in dB: the best achievable joint suppression of every order in
DARK_ORDERS at that (beta1, beta2). A heatmap of this metric is drawn over
the beta1-beta2 grid (centered on BETA1_CENTER/BETA2_CENTER), with
iso-contours at CONTOUR_LEVELS_DB marking where each suppression depth is
achieved.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from graphics import RED2, BLUE2

# ── configuration ─────────────────────────────────────────────────────────────

# beta1-beta2 grid, centered on the nominal design point.
BETA1_CENTER = 2.455
BETA2_CENTER = 1.103
BETA1_HALF_SPAN = 0.035   # grid spans [center - half_span, center + half_span]
BETA2_HALF_SPAN = 0.035
N_BETA1 = 101            # grid resolution (columns)
N_BETA2 = 101            # grid resolution (rows)

PHI1_DEG = 0.0          # ch1 phase (deg); fixed -- only phi2-phi1 matters physically
N_PHI_SEARCH = 1801      # phase-optimization search resolution (grid search over phi2)

# Harmonic orders to jointly suppress (the "dark window"); the metric at each
# (beta1, beta2) is the best achievable (phase-optimized) MAX power across
# these orders.
DARK_ORDERS = (0, -1)

# Bessel series truncation. None = auto, from the grid's max beta + margin.
K_TRUNC = None

# ── contour levels ────────────────────────────────────────────────────────────
CONTOUR_LEVELS_DB = [-60, -50, -40, -30, -20]
# Single color applied to every level, or a list the same length as
# CONTOUR_LEVELS_DB for a distinct color per level.
CONTOUR_COLOR     = '#ffffff'
CONTOUR_LINEWIDTH = 1.5
CONTOUR_LINESTYLE = '-'
CONTOUR_ALPHA      = 1.0
CONTOUR_ZORDER     = 3

CONTOUR_LABEL_SHOW     = True
CONTOUR_LABEL_FONTSIZE = 7
CONTOUR_LABEL_FMT      = '%d dB'
CONTOUR_LABEL_COLOR    = '#ffffff'   # None = match each line's own color

# ── heatmap ───────────────────────────────────────────────────────────────────
HEATMAP_CMAP  = 'viridis_r'   # reversed: darker = more suppression = "better"
HEATMAP_VMIN  = -70.0          # None = auto
HEATMAP_VMAX  = -10.0          # None = auto
SHOW_COLORBAR = True
COLORBAR_LABEL = r'Dark-window depth [dB]  ($\max$ of $|A_0|^2$, $|A_{-1}|^2$, optimized over $\phi_2$)'

# ── design-point marker ────────────────────────────────────────────────────────
SHOW_CENTER_MARKER       = False
CENTER_MARKER            = 'x'
CENTER_MARKERSIZE        = 10
CENTER_MARKER_FACECOLOR  = 'none'
CENTER_MARKER_FACE_ALPHA = 1.0
CENTER_MARKER_EDGECOLOR  = RED2
CENTER_MARKER_EDGE_ALPHA = 1.0
CENTER_MARKER_EDGEWIDTH  = 2.0
CENTER_MARKER_ZORDER     = 5
CENTER_MARKER_LABEL      = 'Design point'

SHOW_GRID   = False
SHOW_LEGEND = True

# Axis limits; None = the beta1/beta2 grid extents above.
XLIM = None
YLIM = None

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm   = 100.0
axes_height_mm  =  80.0
left_mm         =  20.0
right_mm        =  24.0   # room for the colorbar
bottom_mm       =  16.0
top_mm          =   8.0
spine_linewidth =   2.0
tick_width      =   2.0
tick_direction  = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── phase-jitter suppression sampling ─────────────────────────────────────────
# Sample N_SAMPLES (beta1, beta2) points from within the region of the grid
# achieving at least SAMPLE_REGION_DB suppression, perturb each point's own
# optimal phi2 (from the heatmap computation) by Gaussian noise with std
# SAMPLE_PHASE_STD_DEG, and histogram the resulting (generally worse than
# optimal) suppression ratio actually realized at that jittered phase.
SHOW_PHASE_JITTER_HISTOGRAM = True
SAMPLE_REGION_DB     = -50.0   # region: metric_grid_db <= this value
SAMPLE_PHASE_STD_DEG = 0.3     # std of Gaussian phase jitter around the optimal phi2
N_SAMPLES            = 100
SAMPLE_SEED          = None    # None = nondeterministic; set an int for reproducible sampling

# ── histogram style ────────────────────────────────────────────────────────────
HIST_BINS      = 20
HIST_LINESTYLE = '-'
HIST_LINEWIDTH = 1.0
HIST_EDGECOLOR = '#000000'
HIST_EDGE_ALPHA = 1.0
HIST_FACECOLOR = BLUE2
HIST_FACE_ALPHA = 0.85
HIST_ZORDER     = 2

HIST_SHOW_GRID   = False
HIST_SHOW_LEGEND = False

HIST_XLIM = None
HIST_YLIM = None

# Histogram figure geometry (independent of the heatmap figure above).
hist_axes_width_mm   = 100.0
hist_axes_height_mm  =  40.0
hist_left_mm         =  20.0
hist_right_mm        =  10.0
hist_bottom_mm       =  16.0
hist_top_mm          =   8.0

HIST_PUBLICATION_SVG_NAME = 'dark_window_phase_jitter_histogram.svg'
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels, the title, and the legend (from the
# same figure that's shown and PNG-saved below), and additionally saves an
# SVG to SAVE_FOLDER.
FOR_PUBLICATION = False
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUBLICATION_SVG_NAME = 'dark_window_theory_investigator.svg'
# ─────────────────────────────────────────────────────────────────────────────


def _compute_dark_window_grid(beta1_vals: np.ndarray, beta2_vals: np.ndarray, phi1: float,
                               dark_orders, phi_grid: np.ndarray, k_trunc: int):
    """
    For every (beta1, beta2) grid point, the best achievable (phase-optimized)
    joint suppression of every order in dark_orders -- fully vectorized (one
    matrix product per order, rather than looping per grid point).

    Returns (metric_grid_db, best_phi_grid), each shape
    (len(beta2_vals), len(beta1_vals)).
    """
    k = np.arange(-k_trunc, k_trunc + 1)         # (K,)
    E = np.exp(1j * np.outer(k, phi_grid))        # (K, P) -- exp(i*k*phi2)

    n1, n2 = len(beta1_vals), len(beta2_vals)
    worst = np.zeros((n2, n1, len(phi_grid)))

    for order in dark_orders:
        orders_minus_2k = order - 2 * k                            # (K,)
        const = np.exp(1j * orders_minus_2k * phi1)                 # (K,) -- fixed phi1 part
        J1 = jv(orders_minus_2k[None, :], beta1_vals[:, None])      # (n1, K)
        J2 = jv(k[None, :], beta2_vals[:, None])                    # (n2, K)

        coeff = J2[:, None, :] * J1[None, :, :] * const[None, None, :]   # (n2, n1, K)
        A = coeff.reshape(n2 * n1, len(k)) @ E                             # (n2*n1, P)
        power = (np.abs(A) ** 2).reshape(n2, n1, len(phi_grid))
        worst = np.maximum(worst, power)

    best_idx = np.argmin(worst, axis=2)                                          # (n2, n1)
    metric_lin = np.take_along_axis(worst, best_idx[..., None], axis=2)[..., 0]  # (n2, n1)
    metric_db = 10.0 * np.log10(np.maximum(metric_lin, 1e-30))
    best_phi = phi_grid[best_idx]
    return metric_db, best_phi


def _worst_power_at(beta1_arr: np.ndarray, beta2_arr: np.ndarray, phi1: float,
                     phi2_arr: np.ndarray, dark_orders, k_trunc: int) -> np.ndarray:
    """
    max_{p in dark_orders} |A_p|^2, evaluated elementwise at each
    (beta1_arr[i], beta2_arr[i], phi2_arr[i]) triple (not a grid) -- used to
    evaluate the *actual* (generally non-optimal) suppression at a jittered
    phase, as opposed to _compute_dark_window_grid's phase-optimized metric.
    """
    k = np.arange(-k_trunc, k_trunc + 1)              # (K,)
    phase2 = np.exp(1j * np.outer(phi2_arr, k))       # (N, K)
    worst = np.zeros(len(beta1_arr))
    for order in dark_orders:
        orders_minus_2k = order - 2 * k                                    # (K,)
        const = np.exp(1j * orders_minus_2k * phi1)                        # (K,)
        J1 = jv(orders_minus_2k[None, :], beta1_arr[:, None])               # (N, K)
        J2 = jv(k[None, :], beta2_arr[:, None])                             # (N, K)
        A = np.sum(J1 * J2 * const[None, :] * phase2, axis=1)              # (N,)
        worst = np.maximum(worst, np.abs(A) ** 2)
    return worst


def _sample_phase_jitter_suppression(beta1_vals, beta2_vals, metric_grid_db, best_phi_grid,
                                      phi1: float, dark_orders, k_trunc: int):
    """
    Sample N_SAMPLES (beta1, beta2) grid points from within the
    metric_grid_db <= SAMPLE_REGION_DB region (with replacement), perturb
    each sampled point's own optimal phi2 by Gaussian noise (std
    SAMPLE_PHASE_STD_DEG), and evaluate the actual suppression realized at
    that jittered phase.

    Returns (suppression_db, region_count): suppression_db is (N_SAMPLES,);
    region_count is how many distinct grid points satisfy the region cutoff.
    """
    region_i, region_j = np.where(metric_grid_db <= SAMPLE_REGION_DB)
    if len(region_i) == 0:
        raise RuntimeError(
            f"No grid points achieve SAMPLE_REGION_DB={SAMPLE_REGION_DB} dB; "
            "widen the beta1/beta2 span or relax the threshold."
        )

    rng = np.random.default_rng(SAMPLE_SEED)
    pick = rng.integers(0, len(region_i), size=N_SAMPLES)
    i_pick, j_pick = region_i[pick], region_j[pick]

    beta1_samples = beta1_vals[j_pick]
    beta2_samples = beta2_vals[i_pick]
    phi_optimal_samples = best_phi_grid[i_pick, j_pick]

    phase_std_rad = np.deg2rad(SAMPLE_PHASE_STD_DEG)
    phi_jittered = phi_optimal_samples + rng.normal(0.0, phase_std_rad, size=N_SAMPLES)

    worst_lin = _worst_power_at(beta1_samples, beta2_samples, phi1, phi_jittered,
                                 dark_orders, k_trunc)
    suppression_db = 10.0 * np.log10(np.maximum(worst_lin, 1e-30))
    return suppression_db, len(region_i)


def _make_figure():
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )
    return fig, ax


def _build_figure(beta1_vals, beta2_vals, metric_grid_db, title, show_labels: bool):
    """
    Build one heatmap figure. show_labels=True gives the normal, fully-labeled
    figure; False gives the stripped publication variant -- built as its own
    separate figure rather than a mutation of the labeled one.
    """
    fig, ax = _make_figure()

    extent = [beta1_vals[0], beta1_vals[-1], beta2_vals[0], beta2_vals[-1]]
    im = ax.imshow(metric_grid_db, origin='lower', aspect='auto', extent=extent,
                   cmap=HEATMAP_CMAP, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, zorder=1)

    if SHOW_COLORBAR:
        cb = plt.colorbar(im, ax=ax)
        if show_labels:
            cb.set_label(COLORBAR_LABEL, fontsize=axis_label_fontsize)
        cb.ax.tick_params(labelsize=tick_label_fontsize)

    B1, B2 = np.meshgrid(beta1_vals, beta2_vals)
    n_levels = len(CONTOUR_LEVELS_DB)
    colors = CONTOUR_COLOR if isinstance(CONTOUR_COLOR, (list, tuple)) else [CONTOUR_COLOR] * n_levels
    levels_and_colors = sorted(zip(CONTOUR_LEVELS_DB, colors))
    sorted_levels = [lvl for lvl, _ in levels_and_colors]
    sorted_colors = [c for _, c in levels_and_colors]

    cs = ax.contour(B1, B2, metric_grid_db, levels=sorted_levels, colors=sorted_colors,
                     linewidths=CONTOUR_LINEWIDTH, linestyles=CONTOUR_LINESTYLE,
                     alpha=CONTOUR_ALPHA, zorder=CONTOUR_ZORDER)
    # matplotlib >=3.10 draws contours as one artist (no .collections);
    # older versions expose a list of LineCollections instead.
    if hasattr(cs, 'collections'):
        for coll in cs.collections:
            coll.set_capstyle('round')
    else:
        cs.set_capstyle('round')
    if show_labels and CONTOUR_LABEL_SHOW:
        ax.clabel(cs, inline=True, fontsize=CONTOUR_LABEL_FONTSIZE, fmt=CONTOUR_LABEL_FMT,
                  colors=(CONTOUR_LABEL_COLOR if CONTOUR_LABEL_COLOR is not None else None))

    if SHOW_CENTER_MARKER:
        ax.plot([BETA1_CENTER], [BETA2_CENTER], linestyle='none',
                marker=CENTER_MARKER, markersize=CENTER_MARKERSIZE,
                markerfacecolor=mcolors.to_rgba(CENTER_MARKER_FACECOLOR, CENTER_MARKER_FACE_ALPHA)
                                 if CENTER_MARKER_FACECOLOR != 'none' else 'none',
                markeredgecolor=mcolors.to_rgba(CENTER_MARKER_EDGECOLOR, CENTER_MARKER_EDGE_ALPHA),
                markeredgewidth=CENTER_MARKER_EDGEWIDTH, zorder=CENTER_MARKER_ZORDER,
                label=CENTER_MARKER_LABEL)

    ax.set_xlim(XLIM if XLIM is not None else (beta1_vals[0], beta1_vals[-1]))
    ax.set_ylim(YLIM if YLIM is not None else (beta2_vals[0], beta2_vals[-1]))

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction, width=tick_width,
                    labelsize=tick_label_fontsize)

    if show_labels:
        ax.set_xlabel(r'$\beta_1$ (drive at $f$) [rad]', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$\beta_2$ (drive at $2f$) [rad]', fontsize=axis_label_fontsize)
        ax.set_title(title, fontsize=axis_label_fontsize)
        if SHOW_GRID:
            ax.grid()
        if SHOW_LEGEND and SHOW_CENTER_MARKER:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def _build_histogram_figure(suppression_db, title, show_labels: bool):
    """
    Build one histogram figure of the phase-jitter-degraded suppression
    samples. show_labels=True gives the normal, fully-labeled figure; False
    gives the stripped publication variant -- its own separate figure, as in
    _build_figure above.
    """
    fig_w = hist_left_mm + hist_axes_width_mm + hist_right_mm
    fig_h = hist_bottom_mm + hist_axes_height_mm + hist_top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = hist_left_mm   / fig_w,
        right  = 1 - hist_right_mm  / fig_w,
        bottom = hist_bottom_mm / fig_h,
        top    = 1 - hist_top_mm    / fig_h,
    )

    ax.hist(
        suppression_db, bins=HIST_BINS,
        facecolor=mcolors.to_rgba(HIST_FACECOLOR, HIST_FACE_ALPHA),
        edgecolor=mcolors.to_rgba(HIST_EDGECOLOR, HIST_EDGE_ALPHA),
        linewidth=HIST_LINEWIDTH, linestyle=HIST_LINESTYLE, zorder=HIST_ZORDER,
        label=f'n={N_SAMPLES}',
    )

    if HIST_XLIM is not None:
        ax.set_xlim(HIST_XLIM)
    if HIST_YLIM is not None:
        ax.set_ylim(HIST_YLIM)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction, width=tick_width,
                    labelsize=tick_label_fontsize)

    if show_labels:
        ax.set_xlabel('Suppression [dB]  (max of $|A_0|^2$, $|A_{-1}|^2$)',
                       fontsize=axis_label_fontsize)
        ax.set_ylabel('Count', fontsize=axis_label_fontsize)
        ax.set_title(title, fontsize=axis_label_fontsize)
        if HIST_SHOW_GRID:
            ax.grid()
        if HIST_SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


def main():
    beta1_vals = np.linspace(BETA1_CENTER - BETA1_HALF_SPAN, BETA1_CENTER + BETA1_HALF_SPAN, N_BETA1)
    beta2_vals = np.linspace(BETA2_CENTER - BETA2_HALF_SPAN, BETA2_CENTER + BETA2_HALF_SPAN, N_BETA2)
    phi1 = np.deg2rad(PHI1_DEG)
    phi_grid = np.linspace(0.0, 2 * np.pi, N_PHI_SEARCH, endpoint=False)
    k_trunc = K_TRUNC if K_TRUNC is not None else int(2 * max(beta1_vals.max(), beta2_vals.max())) + 20

    print(f"Dark window orders : {DARK_ORDERS}")
    print(f"beta1 grid         : {beta1_vals[0]:.4f} - {beta1_vals[-1]:.4f} rad  ({N_BETA1} points)")
    print(f"beta2 grid         : {beta2_vals[0]:.4f} - {beta2_vals[-1]:.4f} rad  ({N_BETA2} points)")
    print(f"Phase search       : {N_PHI_SEARCH} points,  k_trunc={k_trunc}")
    print("Computing grid...")

    metric_grid_db, best_phi_grid = _compute_dark_window_grid(
        beta1_vals, beta2_vals, phi1, DARK_ORDERS, phi_grid, k_trunc)

    # Global best point found anywhere in the grid.
    i_best, j_best = np.unravel_index(np.argmin(metric_grid_db), metric_grid_db.shape)
    print(f"\nDeepest point in grid:")
    print(f"  beta1={beta1_vals[j_best]:.4f} rad,  beta2={beta2_vals[i_best]:.4f} rad")
    print(f"  depth={metric_grid_db[i_best, j_best]:.2f} dB  "
          f"at phi2={np.degrees(best_phi_grid[i_best, j_best]):.1f} deg")

    # Value at the exact design-point center (nearest grid index).
    j_center = int(np.argmin(np.abs(beta1_vals - BETA1_CENTER)))
    i_center = int(np.argmin(np.abs(beta2_vals - BETA2_CENTER)))
    print(f"\nAt design point (beta1={BETA1_CENTER}, beta2={BETA2_CENTER}):")
    print(f"  depth={metric_grid_db[i_center, j_center]:.2f} dB  "
          f"at phi2={np.degrees(best_phi_grid[i_center, j_center]):.1f} deg  "
          f"(nearest grid point)")

    title = (rf'Dark window (orders {DARK_ORDERS}), $\phi_1={PHI1_DEG:.0f}°$, '
             rf'$\phi_2$ optimized per point')

    fig, ax = _build_figure(beta1_vals, beta2_vals, metric_grid_db, title, show_labels=True)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_figure(beta1_vals, beta2_vals, metric_grid_db, title,
                                          show_labels=False)
        pub_path = Path(SAVE_FOLDER) / PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f"\nSaved: {pub_path}")

    out_path = Path(__file__).parent / 'dark_window_theory_investigator.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")

    if SHOW_PHASE_JITTER_HISTOGRAM:
        suppression_db, region_count = _sample_phase_jitter_suppression(
            beta1_vals, beta2_vals, metric_grid_db, best_phi_grid,
            phi1, DARK_ORDERS, k_trunc)

        print(f"\nPhase-jitter sampling ({N_SAMPLES} samples, "
              f"std={SAMPLE_PHASE_STD_DEG} deg, region <= {SAMPLE_REGION_DB} dB):")
        print(f"  Grid points in region : {region_count} / {metric_grid_db.size}")
        print(f"  Suppression realized  : mean={suppression_db.mean():.2f} dB,  "
              f"std={suppression_db.std():.2f} dB,  "
              f"min={suppression_db.min():.2f} dB,  max={suppression_db.max():.2f} dB")

        hist_title = (rf'Suppression under $\phi_2$ jitter ($\sigma$={SAMPLE_PHASE_STD_DEG}°), '
                      rf'sampled from $\leq${SAMPLE_REGION_DB:.0f} dB region, n={N_SAMPLES}')

        fig_hist, ax_hist = _build_histogram_figure(suppression_db, hist_title, show_labels=True)

        if FOR_PUBLICATION:
            fig_hist_pub, _ax_hist_pub = _build_histogram_figure(
                suppression_db, hist_title, show_labels=False)
            hist_pub_path = Path(SAVE_FOLDER) / HIST_PUBLICATION_SVG_NAME
            fig_hist_pub.savefig(hist_pub_path, format='svg', bbox_inches='tight')
            print(f"Saved: {hist_pub_path}")

        hist_out_path = Path(__file__).parent / 'dark_window_phase_jitter_histogram.png'
        fig_hist.savefig(hist_out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {hist_out_path}")

    plt.show()


if __name__ == '__main__':
    main()
