"""
ce_suppression_tradeoff.py

For dual-tone phase modulation (ch1 at f with depth beta1/phase phi1, ch2 at
2f with depth beta2/phase phi2 -- see dual_tone_amplitudes in
comb_displayer.py), sweeps a target conversion efficiency (CE: the fraction
of total input power delivered to WANTED_ORDER, e.g. +1) across
[CE_MIN_PERCENT, CE_MAX_PERCENT] and, for each target, finds the
(beta1, beta2, phi2) that hits it while minimizing the worse of two
suppression metrics (phi1 fixed at PHI1_DEG as an overall phase reference --
it doesn't affect any |A_p|^2, so fixing it loses no generality):

  carrier suppression  = |A_0|^2                        (carrier power / total input power)
  sideband suppression = |A_UNWANTED_ORDER|^2 / |A_WANTED_ORDER|^2   (unwanted sideband / wanted sideband power)

Both are reported in dB (more negative = better suppressed); the objective is
the minimax of the two (in dB), so neither metric is sacrificed more than
necessary to hit the target CE -- there's a genuine tradeoff here, since
demanding very high conversion efficiency into the wanted order tends to
force either the carrier or the unwanted sideband to leak more.

Two-stage solve per target CE:
  1. differential_evolution (global) minimizes a CE-penalized version of the
     minimax objective -- global search matters here since the Jacobi-Anger
     comb can have multiple local optima in phi2 (same reasoning as
     comb_displayer.py's/optimized_search_viewer.py's best-fit search).
  2. scipy.optimize.minimize (SLSQP, local) polishes that result under an
     exact CE == target equality constraint, minimizing the minimax
     objective directly (no penalty term needed once the constraint itself
     is exact).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import differential_evolution, minimize

from comb_displayer import dual_tone_amplitudes
from graphics import (
    RED2, LIGHTBLUE2, VIOLET2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

# ── target conversion efficiency sweep ────────────────────────────────────────
CE_MIN_PERCENT = 38    # % of total input power delivered to WANTED_ORDER
CE_MAX_PERCENT = 50.0
N_POINTS       = 200
# ─────────────────────────────────────────────────────────────────────────────

# ── orders / phase convention ─────────────────────────────────────────────────
WANTED_ORDER   =  1    # CE target = |A_{WANTED_ORDER}|^2 (fraction of total input power)
UNWANTED_ORDER = -1    # sideband suppression = |A_{UNWANTED_ORDER}|^2 / |A_{WANTED_ORDER}|^2
CARRIER_ORDER  =  0    # carrier suppression  = |A_{CARRIER_ORDER}|^2
PHI1_DEG = 0.0         # ch1 phase, fixed reference (see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

# ── search ─────────────────────────────────────────────────────────────────────
BETA1_BOUNDS    = (0.0, 2.5)     # rad
BETA2_BOUNDS    = (0.0, 2.5)     # rad
PHI2_BOUNDS_DEG = (0.0, 360.0)

# Stage 1 (differential_evolution, global): minimizes
#   max(carrier_supp_dB, sideband_supp_dB) + CE_PENALTY_WEIGHT*(CE_achieved - CE_target)^2   [%^2]
# to land in the right basin (and on the right side of any phi2 ambiguity)
# before stage 2 enforces the CE constraint exactly.
CE_PENALTY_WEIGHT = 5.0
DE_SEED = None   # differential_evolution seed; None = nondeterministic

# Stage 2 (SLSQP, local): polishes stage 1's result under an exact CE ==
# target equality constraint.
SLSQP_MAXITER = 300

# Achieved CE deviating from target by more than this (percentage points)
# prints a warning -- e.g. a target CE beyond what dual-tone modulation can
# actually deliver, or a solver that failed to converge.
CE_WARN_TOL_PERCENT = 0.05
# ─────────────────────────────────────────────────────────────────────────────

# ── plot: carrier suppression series ──────────────────────────────────────────
CARRIER_LINESTYLE  = 'none'
CARRIER_LINEWIDTH  = 2.0
CARRIER_LINE_COLOR = RED2
CARRIER_LINE_ALPHA = 1.0
CARRIER_MARKER            = 'none'
CARRIER_MARKERSIZE        = 5.0
CARRIER_MARKER_FACECOLOR  = None   # None = match line color
CARRIER_MARKER_FACE_ALPHA = 1.0
CARRIER_MARKER_EDGECOLOR  = None   # None = match line color
CARRIER_MARKER_EDGE_ALPHA = 1.0
CARRIER_ZORDER = 2
# ─────────────────────────────────────────────────────────────────────────────

# ── plot: sideband suppression series ─────────────────────────────────────────
SIDEBAND_LINESTYLE  = 'none'
SIDEBAND_LINEWIDTH  = 2.0
SIDEBAND_LINE_COLOR = LIGHTBLUE2
SIDEBAND_LINE_ALPHA = 1.0
SIDEBAND_MARKER            = 'none'
SIDEBAND_MARKERSIZE        = 5.0
SIDEBAND_MARKER_FACECOLOR  = None
SIDEBAND_MARKER_FACE_ALPHA = 1.0
SIDEBAND_MARKER_EDGECOLOR  = None
SIDEBAND_MARKER_EDGE_ALPHA = 1.0
SIDEBAND_ZORDER = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── plot: minimax (worse-of-the-two) series ───────────────────────────────────
MINIMAX_LINESTYLE  = '-'
MINIMAX_LINEWIDTH  = 2.0
MINIMAX_LINE_COLOR = VIOLET2
MINIMAX_LINE_ALPHA = 1.0
MINIMAX_MARKER            = 'o'
MINIMAX_MARKERSIZE        = 5.0
MINIMAX_MARKER_FACECOLOR  = None
MINIMAX_MARKER_FACE_ALPHA = 1.0
MINIMAX_MARKER_EDGECOLOR  = None
MINIMAX_MARKER_EDGE_ALPHA = 1.0
MINIMAX_ZORDER = 4
# ─────────────────────────────────────────────────────────────────────────────

# ── plot layout ────────────────────────────────────────────────────────────────
SHOW_GRID   = False
SHOW_LEGEND = True
XLIM = None   # None = auto (CE_MIN_PERCENT .. CE_MAX_PERCENT with default margin)
YLIM = (-60, 0)

axes_width_mm  = 100.0
axes_height_mm =  40.0
left_mm    = 20.0
right_mm   = 10.0
bottom_mm  = 16.0
top_mm     =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels and the legend, and saves an SVG to
# SAVE_FOLDER (in addition to the normal figure that's still shown/PNG-saved).
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUBLICATION_SVG_NAME = 'ce_suppression_tradeoff.svg'
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Theory / metrics
# ─────────────────────────────────────────────────────────────────────────────

def _k_trunc(beta1, beta2):
    return int(2 * max(beta1, beta2)) + 20


def _order_powers(beta1, beta2, phi2_deg):
    """{order: |A_p|^2} for the three orders this script needs."""
    orders = [CARRIER_ORDER, UNWANTED_ORDER, WANTED_ORDER]
    amps = dual_tone_amplitudes(beta1, beta2, np.deg2rad(PHI1_DEG), np.deg2rad(phi2_deg),
                                 orders, _k_trunc(beta1, beta2))
    return {p: abs(A) ** 2 for p, A in amps.items()}


def _metrics(beta1, beta2, phi2_deg):
    """Returns (ce_percent, carrier_supp_db, sideband_supp_db, minimax_db)."""
    p = _order_powers(beta1, beta2, phi2_deg)
    p_carrier, p_unwanted, p_wanted = p[CARRIER_ORDER], p[UNWANTED_ORDER], p[WANTED_ORDER]

    ce_percent = p_wanted * 100.0
    carrier_supp_db = 10.0 * np.log10(max(p_carrier, 1e-30))
    sideband_supp_db = (10.0 * np.log10(max(p_unwanted, 1e-30))
                         - 10.0 * np.log10(max(p_wanted, 1e-30)))
    minimax_db = max(carrier_supp_db, sideband_supp_db)
    return ce_percent, carrier_supp_db, sideband_supp_db, minimax_db


def _penalized_objective(params, target_ce_percent):
    beta1, beta2, phi2_deg = params
    ce_percent, _, _, minimax_db = _metrics(beta1, beta2, phi2_deg)
    return minimax_db + CE_PENALTY_WEIGHT * (ce_percent - target_ce_percent) ** 2


def _minimax_objective(params):
    beta1, beta2, phi2_deg = params
    return _metrics(beta1, beta2, phi2_deg)[3]


def _ce_percent_of(params):
    beta1, beta2, phi2_deg = params
    return _metrics(beta1, beta2, phi2_deg)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-target solve
# ─────────────────────────────────────────────────────────────────────────────

def _solve_for_target(target_ce_percent):
    """
    Two-stage solve for one target CE (see module docstring). Returns a dict
    with beta1/beta2/phi2_deg, achieved ce_percent, both suppression metrics
    (dB), the minimax (dB), and each stage's convergence flag.
    """
    bounds = [BETA1_BOUNDS, BETA2_BOUNDS, PHI2_BOUNDS_DEG]

    de_result = differential_evolution(_penalized_objective, bounds, args=(target_ce_percent,),
                                        seed=DE_SEED, tol=1e-12, polish=True)

    constraints = [{'type': 'eq', 'fun': lambda p: _ce_percent_of(p) - target_ce_percent}]
    slsqp_result = minimize(_minimax_objective, de_result.x, method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': SLSQP_MAXITER, 'ftol': 1e-10})

    beta1, beta2, phi2_deg = slsqp_result.x
    phi2_deg = float(phi2_deg % 360.0)
    ce_percent, carrier_supp_db, sideband_supp_db, minimax_db = _metrics(beta1, beta2, phi2_deg)

    return {
        'target_ce_percent': float(target_ce_percent),
        'beta1': float(beta1),
        'beta2': float(beta2),
        'phi2_deg': phi2_deg,
        'ce_percent': float(ce_percent),
        'carrier_supp_db': float(carrier_supp_db),
        'sideband_supp_db': float(sideband_supp_db),
        'minimax_db': float(minimax_db),
        'de_success': bool(de_result.success),
        'slsqp_success': bool(slsqp_result.success),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _draw_series(ax, x, y, linestyle, linewidth, color, alpha, marker, markersize,
                  marker_facecolor, marker_face_alpha, marker_edgecolor, marker_edge_alpha,
                  zorder, label):
    face = marker_facecolor if marker_facecolor is not None else color
    edge = marker_edgecolor if marker_edgecolor is not None else color
    ax.plot(x, y, linestyle=linestyle, linewidth=linewidth,
            color=mcolors.to_rgba(color, alpha),
            marker=marker, markersize=markersize,
            markerfacecolor=mcolors.to_rgba(face, marker_face_alpha),
            markeredgecolor=mcolors.to_rgba(edge, marker_edge_alpha),
            solid_capstyle='round', zorder=zorder, label=label)


def _build_tradeoff_figure(ce_targets, carrier_db, sideband_db, minimax_db, show_labels: bool):
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    _draw_series(ax, ce_targets, carrier_db, CARRIER_LINESTYLE, CARRIER_LINEWIDTH,
                 CARRIER_LINE_COLOR, CARRIER_LINE_ALPHA, CARRIER_MARKER, CARRIER_MARKERSIZE,
                 CARRIER_MARKER_FACECOLOR, CARRIER_MARKER_FACE_ALPHA,
                 CARRIER_MARKER_EDGECOLOR, CARRIER_MARKER_EDGE_ALPHA,
                 CARRIER_ZORDER, label='Carrier suppression')
    _draw_series(ax, ce_targets, sideband_db, SIDEBAND_LINESTYLE, SIDEBAND_LINEWIDTH,
                 SIDEBAND_LINE_COLOR, SIDEBAND_LINE_ALPHA, SIDEBAND_MARKER, SIDEBAND_MARKERSIZE,
                 SIDEBAND_MARKER_FACECOLOR, SIDEBAND_MARKER_FACE_ALPHA,
                 SIDEBAND_MARKER_EDGECOLOR, SIDEBAND_MARKER_EDGE_ALPHA,
                 SIDEBAND_ZORDER, label='Sideband suppression')
    _draw_series(ax, ce_targets, minimax_db, MINIMAX_LINESTYLE, MINIMAX_LINEWIDTH,
                 MINIMAX_LINE_COLOR, MINIMAX_LINE_ALPHA, MINIMAX_MARKER, MINIMAX_MARKERSIZE,
                 MINIMAX_MARKER_FACECOLOR, MINIMAX_MARKER_FACE_ALPHA,
                 MINIMAX_MARKER_EDGECOLOR, MINIMAX_MARKER_EDGE_ALPHA,
                 MINIMAX_ZORDER, label='Minimax (worse of the two)')

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    if XLIM is not None:
        ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)

    if show_labels:
        ax.set_xlabel(f'Target CE (order {WANTED_ORDER:+d}) [%]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Suppression [dB]', fontsize=axis_label_fontsize)
        if SHOW_GRID:
            ax.grid()
        if SHOW_LEGEND:
            ax.legend(fontsize=tick_label_fontsize, frameon=False)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ce_targets_percent = np.linspace(CE_MIN_PERCENT, CE_MAX_PERCENT, N_POINTS)
    print(f"Sweeping target CE (order {WANTED_ORDER:+d}) from {CE_MIN_PERCENT:.2f}% to "
          f"{CE_MAX_PERCENT:.2f}% over {N_POINTS} point(s), phi1={PHI1_DEG:.1f} deg fixed...\n")

    header = (f"{'target CE':>10}  {'achieved CE':>12}  {'beta1':>8}  {'beta2':>8}  "
              f"{'phi2':>9}  {'carrier supp':>13}  {'sideband supp':>14}  {'minimax':>9}")
    print(header)
    print('-' * len(header))

    results = []
    for target in ce_targets_percent:
        r = _solve_for_target(float(target))
        results.append(r)

        flag = ''
        if not r['slsqp_success']:
            flag += '  <-- WARNING: SLSQP did not report convergence'
        if abs(r['ce_percent'] - target) > CE_WARN_TOL_PERCENT:
            flag += (f"  <-- WARNING: achieved CE off by {r['ce_percent'] - target:+.3f} pp "
                     f"(target may be unreachable, or the solver stalled)")

        print(f"{target:>9.2f}%  {r['ce_percent']:>11.3f}%  {r['beta1']:>8.4f}  {r['beta2']:>8.4f}  "
              f"{r['phi2_deg']:>7.2f}d  {r['carrier_supp_db']:>11.3f}dB  "
              f"{r['sideband_supp_db']:>12.3f}dB  {r['minimax_db']:>7.3f}dB{flag}")

    ce_targets = np.array([r['target_ce_percent'] for r in results])
    carrier_db = np.array([r['carrier_supp_db'] for r in results])
    sideband_db = np.array([r['sideband_supp_db'] for r in results])
    minimax_db = np.array([r['minimax_db'] for r in results])

    fig, ax = _build_tradeoff_figure(ce_targets, carrier_db, sideband_db, minimax_db,
                                      show_labels=True)

    if FOR_PUBLICATION:
        fig_pub, _ax_pub = _build_tradeoff_figure(ce_targets, carrier_db, sideband_db, minimax_db,
                                                    show_labels=False)
        pub_path = Path(SAVE_FOLDER) / PUBLICATION_SVG_NAME
        fig_pub.savefig(pub_path, format='svg', bbox_inches='tight')
        print(f'\nSaved: {pub_path}')

    out_path = Path(__file__).parent / 'ce_suppression_tradeoff.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

    plt.show()


if __name__ == '__main__':
    main()
