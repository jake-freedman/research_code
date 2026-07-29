"""
Batch nonlinear-phase fit across a folder of dual-tone ch2-phase sweeps.

Each .npz in SWEEP_FOLDER is expected to be a single ch2-phase sweep produced
by bnc_dual_tone_esa_script.py (DualToneSweepData-compatible), typically one
sweep per drive voltage in a series. For each file this script runs the same
physically-motivated joint fit as PUB_CURVE_MODE='nonlinear_phase' in
dual_tone_sweep_analysis.py -- pooling every requested harmonic's data into
one differential_evolution optimization of

    theta(t) = beta1*sin(Wt) + beta2*sin(2Wt+phi+phi0) + beta2_nl*sin(2Wt+phi_NL)

(phi1 = 0 fixed; phi is the swept ch2 phase; phi0 a static calibration offset
on the intentional beta2 term; beta2_nl/phi_NL a fixed-phase nonlinear/
parasitic 2f contribution) -- and nothing else: no raw/sinusoid/phase-
harmonics fit modes, no per-file plots, no interactive sliders. This is meant
to run unattended over many files.

Fitted parameters for every sweep are collected and written to a single
OUTPUT_FILENAME .npz inside SWEEP_FOLDER. That output file is excluded when
this script scans the folder for input sweeps, so re-running it (e.g. after
adding more sweep files) does not try to treat its own prior output as data.

After fitting every sweep, a summary plot of beta2_NL vs. drive voltage is
produced -- one point per successfully-fit sweep.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiment_control'))

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import differential_evolution
from scipy.special import jv as bessel_jv

from dual_tone_sweep_data import DualToneSweepData
from graphics import (
    GREEN2, DARKGRAY2, BLUE2, RED2, VIOLET2, ORANGE2,
    DARKGREEN2, DARKBLUE2, TAN2, PINK2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    left_mm, right_mm, bottom_mm, top_mm,
)

_HARMONIC_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_EXTRA_COLORS = [BLUE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2, DARKGRAY2]


def _harmonic_color(n: int) -> str:
    return _HARMONIC_COLORS.get(n, _EXTRA_COLORS[abs(n) % len(_EXTRA_COLORS)])

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

SWEEP_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\nonlinearity_characterization3"

# Name of the file this script writes into SWEEP_FOLDER. Any .npz with this
# exact basename is skipped when scanning the folder for input sweeps (so
# re-running the script never tries to fit its own prior output).
OUTPUT_FILENAME = 'nonlinear_fit_results.npz'

# Harmonic orders to include in the joint fit for every sweep. A file
# missing some of these is fit with whatever subset it has (a warning is
# printed); a file with none of them is skipped entirely.
FIT_HARMONICS = [-3, -2, -1, 0, 1, 2, 3]

# If OUTPUT_FILENAME already exists, skip re-running the (slow)
# differential_evolution fit for any sweep file whose name is already
# present in it -- its stored parameters are reused as-is, and only new
# sweep files (not in that prior output) get fit. Set to False to always
# refit every file from scratch. Ignored (falls back to refitting
# everything) if the stored file's fit_harmonics don't match FIT_HARMONICS.
REUSE_EXISTING_FITS = True

# Which channel's drive voltage to use as the x-axis of the summary plot.
# 'ch1_voltage' or 'ch2_voltage'. Default is ch1 since beta2_nl is modeled
# as a parasitic term most plausibly driven by the (typically stronger)
# fundamental tone -- switch this if your series instead steps ch2's power.
DRIVE_VOLTAGE_AXIS = 'ch1_voltage'

# X-axis of the beta2_NL summary plot:
#   'drive_voltage' -- drive voltage on DRIVE_VOLTAGE_AXIS (default)
#   'beta1'         -- the fitted beta1 for each sweep instead of voltage
SUMMARY_PLOT_X_AXIS = 'drive_voltage'

# When True: removes title/axis labels/tick labels and saves each of the
# summary plots below (beta2_NL, beta1, beta2, phi_NL, scale-factor bar
# chart) as an SVG, all inside PUBLICATION_SVG_FOLDER (each under its own
# default filename).
FOR_PUBLICATION = True
PUBLICATION_SVG_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"

# ── nonlinear-phase joint fit (see dual_tone_sweep_analysis.py for details) ──
NL_FIT_BETA1_BOUNDS      = (0.0, 2.5)     # rad
NL_FIT_BETA2_BOUNDS      = (0.0, 2.0)     # rad
NL_FIT_BETA2_NL_BOUNDS   = (0.0, 0.5)     # rad
NL_FIT_PHI0_BOUNDS_DEG   = (0.0, 360.0)
NL_FIT_PHI_NL_BOUNDS_DEG = (0.0, 360.0)

# Optional initial guess seeding differential_evolution's population for
# every sweep (still a global search around it, not a strict local
# refinement). Any left None falls back to the midpoint of its bounds.
NL_FIT_GUESS_BETA1      = None
NL_FIT_GUESS_BETA2      = None
NL_FIT_GUESS_BETA2_NL   = None
NL_FIT_GUESS_PHI0_DEG   = None
NL_FIT_GUESS_PHI_NL_DEG = None
NL_FIT_SEED = None   # differential_evolution seed; None = nondeterministic

# When True, the joint fit gets one extra free multiplicative scale factor
# per fitted harmonic (bounded by NL_FIT_SCALE_BOUNDS) to absorb small
# systematic per-sideband calibration errors without biasing the physical
# parameters. When False, every sideband's scale factor is fixed at 1.0.
NL_FIT_PER_SIDEBAND_SCALE = True
NL_FIT_SCALE_BOUNDS       = (0.6, 1.2)

# ── summary plot ──────────────────────────────────────────────────────────
axes_width_mm  = 55.0
axes_height_mm = 45.0

SUMMARY_LINESTYLE          = 'none'
SUMMARY_LINEWIDTH          = 1.5
SUMMARY_LINE_COLOR         = GREEN2
SUMMARY_LINE_ALPHA         = 1.0
SUMMARY_MARKER             = 'o'
SUMMARY_MARKER_SIZE        = 9       # points
SUMMARY_MARKER_FACE_COLOR  = 'same'  # 'same' = match SUMMARY_LINE_COLOR
SUMMARY_MARKER_FACE_ALPHA  = 1.0
SUMMARY_MARKER_EDGE_COLOR  = '#000000'
SUMMARY_MARKER_EDGE_ALPHA  = 1.0
SUMMARY_MARKER_EDGE_WIDTH  = 1.00     # points
SUMMARY_ZORDER             = 2
SUMMARY_SHOW_GRID          = False
SUMMARY_SHOW_LEGEND        = False
SUMMARY_XLIM               = (0,3)    # (xmin, xmax) or None = auto
SUMMARY_YLIM               = None    # (ymin, ymax) or None = auto

# When set to a number, the y-axis plots beta2_nl raised to this power (e.g.
# 0.5 for sqrt(beta2_nl), 2 for beta2_nl**2) instead of raw beta2_nl. Useful
# for checking a suspected power-law scaling by linearizing it against the
# x-axis. None = plot raw beta2_nl (rad) unchanged.
SUMMARY_PLOT_Y_POWER = 1
SUMMARY_POWER_YLABEL = None   # None = generic auto label; set your own to be specific

# Fit a polynomial (default degree 2 -- a parabola) to whatever is actually
# plotted (x_vals, y_vals -- so it matches SUMMARY_PLOT_X_AXIS/Y_POWER above)
# and overlay it as a line. Indices in SUMMARY_FIT_EXCLUDE_INDICES are 0-based
# positions in order of increasing applied drive voltage (not file order, and
# not affected by SUMMARY_PLOT_X_AXIS) -- e.g. [0, 3] excludes the lowest- and
# 4th-lowest-voltage sweeps; negative indices count from the end as in normal
# Python indexing (e.g. -1 = highest voltage). Excluded points are still shown
# as data, just left out of the fit -- use this to drop known outliers.
SUMMARY_FIT_SHOW             = True
SUMMARY_FIT_DEGREE           = 2
SUMMARY_FIT_EXCLUDE_INDICES  = []
SUMMARY_FIT_COLOR            = 'black'
SUMMARY_FIT_LINESTYLE        = '-'
SUMMARY_FIT_LINEWIDTH        = 2.0
SUMMARY_FIT_ALPHA            = 1.0
SUMMARY_FIT_ZORDER           = 1
SUMMARY_FIT_LABEL            = None   # None = auto (e.g. 'degree-2 fit')


# ── beta1 vs. drive voltage plot (separate figure) ────────────────────────
SHOW_BETA1_VS_VOLTAGE_PLOT = True
BETA1_PLOT_AXES_WIDTH_MM  = 55.0
BETA1_PLOT_AXES_HEIGHT_MM = 45.0

BETA1_LINESTYLE          = 'none'
BETA1_LINEWIDTH          = 1.5
BETA1_LINE_COLOR         = LIGHTBLUE2
BETA1_LINE_ALPHA         = 1.0
BETA1_MARKER             = 'o'
BETA1_MARKER_SIZE        = 9       # points
BETA1_MARKER_FACE_COLOR  = 'same'  # 'same' = match BETA1_LINE_COLOR
BETA1_MARKER_FACE_ALPHA  = 1.0
BETA1_MARKER_EDGE_COLOR  = '#000000'
BETA1_MARKER_EDGE_ALPHA  = 1.0
BETA1_MARKER_EDGE_WIDTH  = 1.0     # points
BETA1_ZORDER             = 2
BETA1_SHOW_GRID          = False
BETA1_SHOW_LEGEND        = False
BETA1_XLIM = (0,3)   # (xmin, xmax) or None = auto
BETA1_YLIM = (0,2)

# Fit a line (default degree 1) to beta1 vs. drive voltage and overlay it.
# Indices are 0-based positions in order of increasing voltage (negative
# indices count from the end, as in normal Python indexing).
BETA1_FIT_SHOW            = True
BETA1_FIT_DEGREE          = 1
BETA1_FIT_EXCLUDE_INDICES = []
BETA1_FIT_COLOR           = 'black'
BETA1_FIT_LINESTYLE       = '-'
BETA1_FIT_LINEWIDTH       = 2.0
BETA1_FIT_ALPHA           = 1.0
BETA1_FIT_ZORDER          = 1
BETA1_FIT_LABEL           = None   # None = auto (e.g. 'degree-1 fit')


# ── beta2 vs. drive voltage plot (separate figure) ────────────────────────
SHOW_BETA2_VS_VOLTAGE_PLOT = True
BETA2_PLOT_AXES_WIDTH_MM  = 55.0
BETA2_PLOT_AXES_HEIGHT_MM = 45.0

BETA2_LINESTYLE          = 'none'
BETA2_LINEWIDTH          = 1.5
BETA2_LINE_COLOR         = VIOLET2
BETA2_LINE_ALPHA         = 1.0
BETA2_MARKER             = 'o'
BETA2_MARKER_SIZE        = 9       # points
BETA2_MARKER_FACE_COLOR  = 'same'  # 'same' = match BETA2_LINE_COLOR
BETA2_MARKER_FACE_ALPHA  = 1.0
BETA2_MARKER_EDGE_COLOR  = '#000000'
BETA2_MARKER_EDGE_ALPHA  = 1.0
BETA2_MARKER_EDGE_WIDTH  = 1.0     # points
BETA2_ZORDER             = 2
BETA2_SHOW_GRID          = False
BETA2_SHOW_LEGEND        = False
BETA2_XLIM = (0,3)   # (xmin, xmax) or None = auto
BETA2_YLIM = (0,2)

# Fit a line (default degree 1) to beta2 vs. drive voltage and overlay it.
# Indices are 0-based positions in order of increasing voltage (negative
# indices count from the end, as in normal Python indexing).
BETA2_FIT_SHOW            = True
BETA2_FIT_DEGREE          = 1
BETA2_FIT_EXCLUDE_INDICES = []
BETA2_FIT_COLOR           = 'black'
BETA2_FIT_LINESTYLE       = '-'
BETA2_FIT_LINEWIDTH       = 2.0
BETA2_FIT_ALPHA           = 1.0
BETA2_FIT_ZORDER          = 1
BETA2_FIT_LABEL           = None   # None = auto (e.g. 'degree-1 fit')


# ── phi_NL vs. drive voltage plot (separate figure) ────────────────────────
SHOW_PHI_NL_VS_VOLTAGE_PLOT = True
PHI_NL_PLOT_AXES_WIDTH_MM  = 55.0
PHI_NL_PLOT_AXES_HEIGHT_MM = 45.0

PHI_NL_LINESTYLE          = 'none'
PHI_NL_LINEWIDTH          = 1.5
PHI_NL_LINE_COLOR         = ORANGE2
PHI_NL_LINE_ALPHA         = 1.0
PHI_NL_MARKER             = 'o'
PHI_NL_MARKER_SIZE        = 9       # points
PHI_NL_MARKER_FACE_COLOR  = 'same'  # 'same' = match PHI_NL_LINE_COLOR
PHI_NL_MARKER_FACE_ALPHA  = 1.0
PHI_NL_MARKER_EDGE_COLOR  = '#000000'
PHI_NL_MARKER_EDGE_ALPHA  = 1.0
PHI_NL_MARKER_EDGE_WIDTH  = 1.0     # points
PHI_NL_ZORDER             = 2
PHI_NL_SHOW_GRID          = False
PHI_NL_SHOW_LEGEND        = False
PHI_NL_XLIM = (0,3)  # (xmin, xmax) or None = auto
PHI_NL_YLIM = None   # (ymin, ymax) or None = auto

# Fit a line (default degree 1) to phi_NL vs. drive voltage and overlay it.
# Indices are 0-based positions in order of increasing voltage (negative
# indices count from the end, as in normal Python indexing).
PHI_NL_FIT_SHOW            = False
PHI_NL_FIT_DEGREE          = 1
PHI_NL_FIT_EXCLUDE_INDICES = []
PHI_NL_FIT_COLOR           = 'black'
PHI_NL_FIT_LINESTYLE       = '-'
PHI_NL_FIT_LINEWIDTH       = 2.0
PHI_NL_FIT_ALPHA           = 1.0
PHI_NL_FIT_ZORDER          = 1
PHI_NL_FIT_LABEL           = None   # None = auto (e.g. 'degree-1 fit')


# ── scale-factor summary bar chart (NL_FIT_PER_SIDEBAND_SCALE only) ────────
# One bar per fitted harmonic order: mean correction scale factor pooled
# across all sweeps, with the std shown as an error bar.
SHOW_SCALE_BAR_CHART = True
SCALE_BAR_AXES_WIDTH_MM  = 90.0
SCALE_BAR_AXES_HEIGHT_MM = 60.0

SCALE_BAR_WIDTH          = 0.7
SCALE_BAR_FACE_COLOR     = TAN2
SCALE_BAR_FACE_ALPHA     = 1.0
SCALE_BAR_EDGE_COLOR     = '#000000'
SCALE_BAR_EDGE_ALPHA     = 1.0
SCALE_BAR_EDGE_WIDTH     = 1.0     # points
SCALE_BAR_ERR_COLOR      = '#000000'
SCALE_BAR_ERR_ALPHA      = 1.0
SCALE_BAR_ERR_LINEWIDTH  = 1.5
SCALE_BAR_ERR_CAPSIZE    = 4.0     # points
SCALE_BAR_ZORDER         = 2
SCALE_BAR_SHOW_GRID      = False
SCALE_BAR_SHOW_LEGEND    = False
SCALE_BAR_YLIM = None   # (ymin, ymax) or None = auto


# ── per-sweep diagnostic plots ───────────────────────────────────────────
# When True, every sweep gets its own plot with all recorded sidebands' CE
# curves on one axes (no SPLIT_GROUPS-style splitting), optionally with the
# fitted nonlinear-phase curve overlaid on top of whichever sidebands were
# part of the fit. ALL of these are saved to PER_SWEEP_PLOT_FOLDER; only one
# representative plot (PER_SWEEP_PLOT_SHOW_INDEX) is actually displayed on
# screen, alongside the beta2_NL summary plot -- the rest are closed right
# after saving so dozens of sweeps don't open dozens of windows.
SAVE_PER_SWEEP_PLOTS = True
# Overlay the fitted nonlinear-phase curve on top of the raw data. If a
# sweep's fit failed, its raw data is still plotted/saved either way.
PER_SWEEP_PLOT_SHOW_FIT = True
# 0-based index (in the order sweeps are processed) of the one per-sweep
# plot to actually display; all others are only saved to disk. Clamped to
# the number of sweep files found.
PER_SWEEP_PLOT_SHOW_INDEX = 0
# Folder per-sweep plots are saved into. None = a 'per_sweep_fit_plots'
# subfolder created inside SWEEP_FOLDER.
PER_SWEEP_PLOT_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media\grid4"
PER_SWEEP_PLOT_DPI = 150

# Units for the per-sweep sideband-power plots' y-axis:
#   'percent' -- conversion efficiency, % of carrier (linear)
#   'dBc'     -- relative to carrier, dB (log)
PER_SWEEP_PLOT_UNITS = 'percent'

PER_SWEEP_AXES_WIDTH_MM  = 90.0
PER_SWEEP_AXES_HEIGHT_MM = 60.0

# Raw data points (one series per recorded harmonic). 'same' = follow
# _harmonic_color for that harmonic; set to a fixed color to override.
PER_SWEEP_RAW_LINESTYLE         = 'none'  # 'none' = markers only, no connecting line
PER_SWEEP_RAW_LINEWIDTH         = 0.8
PER_SWEEP_RAW_LINE_ALPHA        = 1.0
PER_SWEEP_RAW_MARKER            = 'o'
PER_SWEEP_RAW_MARKER_SIZE       = 6       # points
PER_SWEEP_RAW_MARKER_FACE_COLOR = 'same'  # 'same' = match that harmonic's color
PER_SWEEP_RAW_MARKER_FACE_ALPHA = 1.0
PER_SWEEP_RAW_MARKER_EDGE_COLOR = 'same'  # 'same' = match that harmonic's color
PER_SWEEP_RAW_MARKER_EDGE_ALPHA = 1.0
PER_SWEEP_RAW_MARKER_EDGE_WIDTH = 0.0     # points
PER_SWEEP_RAW_ZORDER            = 4

# Fitted nonlinear-phase overlay curve (only drawn when PER_SWEEP_PLOT_SHOW_FIT).
# 'same' = match that harmonic's raw-data color; set to a fixed color to override.
PER_SWEEP_FIT_COLOR      = '#000000'
PER_SWEEP_FIT_LINESTYLE  = '-'
PER_SWEEP_FIT_LINEWIDTH  = 1.5
PER_SWEEP_FIT_LINE_ALPHA = 1.0
PER_SWEEP_FIT_ZORDER     = 5

PER_SWEEP_SHOW_GRID   = False
PER_SWEEP_SHOW_LEGEND = True
PER_SWEEP_XLIM = None   # (xmin, xmax) or None = auto
PER_SWEEP_YLIM = None   # (ymin, ymax) or None = auto

# ------------------------------------------------------------------


def _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonics, k_trunc):
    """
    Vectorized Jacobi-Anger amplitudes A_n(phi_i) for
        theta(t) = beta1*sin(Wt) + beta2*sin(2Wt+phi+phi0) + beta2_nl*sin(2Wt+phi_NL),
    phi1 = 0 fixed. phi0 is a static offset added only to the swept beta2
    term. The two 2f terms combine via phasor sum into a single effective
    two-tone drive at 2f:
        Z(phi) = beta2*exp(i*(phi+phi0)) + beta2_nl*exp(i*phi_nl)
        beta_eff = |Z|, phi_eff = angle(Z)
    Returns {harmonic: (M,) complex array}, one amplitude per phi_rad entry.
    """
    phi_rad = np.asarray(phi_rad, dtype=float)
    Z = beta2 * np.exp(1j * (phi_rad + phi0_rad)) + beta2_nl * np.exp(1j * phi_nl_rad)
    beta_eff = np.abs(Z)
    phi_eff = np.angle(Z)

    k = np.arange(-k_trunc, k_trunc + 1)
    Jk_beta_eff = bessel_jv(k[:, None], beta_eff[None, :])       # (K, M)
    phase = np.exp(1j * k[:, None] * phi_eff[None, :])           # (K, M)

    return {
        n: np.sum(bessel_jv(n - 2 * k, beta1)[:, None] * Jk_beta_eff * phase, axis=0)
        for n in harmonics
    }


def _nl_phase_cost(params, phi_rad, harmonics, measured_dbc, k_trunc):
    beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad = params[:5]
    scales = params[5:]   # one per harmonic, in `harmonics` order; empty if disabled
    amps = _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonics, k_trunc)
    err = 0.0
    for idx, n in enumerate(harmonics):
        scale = scales[idx] if len(scales) else 1.0
        pred_dbc = 10.0 * np.log10(np.maximum(scale * np.abs(amps[n]) ** 2, 1e-30))
        err += np.sum((measured_dbc[n] - pred_dbc) ** 2)
    return float(err)


def _fit_one_sweep(data: DualToneSweepData, fit_harmonics, k_trunc):
    """
    Run the joint nonlinear-phase fit for a single sweep.

    Returns a result dict, or None if the fit couldn't be attempted/failed.
    """
    available = [n for n in fit_harmonics if n in data.harmonics]
    missing = sorted(set(fit_harmonics) - set(available))
    if missing:
        print(f"    Warning: harmonics {missing} not present in this file; fitting only {available}.")
    if not available:
        print("    Warning: none of the requested harmonics are present; skipping.")
        return None

    phi_rad_all = np.deg2rad(data.ch2_phases_deg)
    dbc_all = data.normalized_peak_powers_dbm()   # (M, N) dBc
    measured_dbc = {}
    for n in available:
        j = int(np.where(data.harmonics == n)[0][0])
        measured_dbc[n] = dbc_all[:, j]

    bounds = [NL_FIT_BETA1_BOUNDS, NL_FIT_BETA2_BOUNDS, NL_FIT_BETA2_NL_BOUNDS,
              tuple(np.deg2rad(NL_FIT_PHI0_BOUNDS_DEG)),
              tuple(np.deg2rad(NL_FIT_PHI_NL_BOUNDS_DEG))]
    guess = [NL_FIT_GUESS_BETA1, NL_FIT_GUESS_BETA2, NL_FIT_GUESS_BETA2_NL,
             None if NL_FIT_GUESS_PHI0_DEG is None else np.deg2rad(NL_FIT_GUESS_PHI0_DEG),
             None if NL_FIT_GUESS_PHI_NL_DEG is None else np.deg2rad(NL_FIT_GUESS_PHI_NL_DEG)]
    if NL_FIT_PER_SIDEBAND_SCALE:
        bounds += [NL_FIT_SCALE_BOUNDS] * len(available)
        guess += [None] * len(available)
    x0 = [float(np.clip(g if g is not None else 0.5 * (lo + hi), lo, hi))
          for g, (lo, hi) in zip(guess, bounds)]

    try:
        result = differential_evolution(
            _nl_phase_cost, bounds, args=(phi_rad_all, available, measured_dbc, k_trunc),
            seed=NL_FIT_SEED, tol=1e-12, polish=True,
            x0=(x0 if any(g is not None for g in guess) else None))
        if not np.isfinite(result.fun):
            raise RuntimeError("fit produced a non-finite cost")
    except Exception as exc:
        print(f"    Warning: nonlinear-phase fit failed ({exc}); skipping.")
        return None

    beta1_fit, beta2_fit, beta2_nl_fit, phi0_fit_rad, phi_nl_fit_rad = result.x[:5]
    scales_fit = ({n: float(result.x[5 + i]) for i, n in enumerate(available)}
                  if NL_FIT_PER_SIDEBAND_SCALE else {n: 1.0 for n in available})

    print(f"    beta1={beta1_fit:.4f} rad, beta2={beta2_fit:.4f} rad, "
          f"beta2_nl={beta2_nl_fit:.4f} rad, "
          f"phi0={np.degrees(phi0_fit_rad) % 360.0:+7.2f} deg, "
          f"phi_NL={np.degrees(phi_nl_fit_rad) % 360.0:+7.2f} deg, "
          f"cost={result.fun:.4f}")

    amps_fit = _nl_phase_amplitudes(beta1_fit, beta2_fit, beta2_nl_fit, phi0_fit_rad,
                                     phi_nl_fit_rad, phi_rad_all, available, k_trunc)
    for n in available:
        pred_dbc = 10.0 * np.log10(np.maximum(scales_fit[n] * np.abs(amps_fit[n]) ** 2, 1e-30))
        resid_rms = float(np.sqrt(np.mean((measured_dbc[n] - pred_dbc) ** 2)))
        print(f"      p={n:+d}: measured mean {measured_dbc[n].mean():+7.3f} dB, "
              f"RMS residual {resid_rms:6.3f} dB"
              + (f", scale={scales_fit[n]:.4f}" if NL_FIT_PER_SIDEBAND_SCALE else ""))

    return {
        'beta1': float(beta1_fit),
        'beta2': float(beta2_fit),
        'beta2_nl': float(beta2_nl_fit),
        'phi0_deg': float(np.degrees(phi0_fit_rad) % 360.0),
        'phi_nl_deg': float(np.degrees(phi_nl_fit_rad) % 360.0),
        'cost': float(result.fun),
        'fit_harmonics': available,
        'scales': scales_fit,
    }


def _nl_predicted_ce(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonic, k_trunc,
                      scale=1.0, units='dBc'):
    """Predicted conversion efficiency for one harmonic at the given ch2
    phase(s), used for the per-sweep fit-overlay curve. units: 'percent' or
    'dBc'."""
    amps = _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, [harmonic], k_trunc)
    frac = scale * np.abs(amps[harmonic]) ** 2
    if units == 'percent':
        return frac * 100.0
    return 10.0 * np.log10(np.maximum(frac, 1e-30))


def _plot_sweep_with_fit(fname: str, data: DualToneSweepData, result, show_fit: bool, k_trunc: int):
    """
    One axes with every recorded sideband's raw conversion efficiency (% or
    dBc, per PER_SWEEP_PLOT_UNITS) vs. ch2 phase, plus (if show_fit and the
    fit succeeded) the fitted nonlinear-phase curve overlaid on whichever
    sidebands were part of that fit. No SPLIT_GROUPS-style splitting --
    everything on one plot, per the per-sweep diagnostic.
    """
    fig, ax = _make_fig(PER_SWEEP_AXES_WIDTH_MM, PER_SWEEP_AXES_HEIGHT_MM)
    dbc_all = data.normalized_peak_powers_dbm()   # (M, N) dBc
    if PER_SWEEP_PLOT_UNITS == 'percent':
        y_all = 10.0 ** (dbc_all / 10.0) * 100.0
        ylabel = 'Conversion efficiency [%]'
    else:
        y_all = dbc_all
        ylabel = 'Sideband power [dBc]'
    x = data.ch2_phases_deg
    order_idx = np.argsort(x)

    for j, n in enumerate(data.harmonics):
        n = int(n)
        color = _harmonic_color(n)
        face_color_base = color if PER_SWEEP_RAW_MARKER_FACE_COLOR == 'same' else PER_SWEEP_RAW_MARKER_FACE_COLOR
        edge_color_base = color if PER_SWEEP_RAW_MARKER_EDGE_COLOR == 'same' else PER_SWEEP_RAW_MARKER_EDGE_COLOR
        line_rgba = mcolors.to_rgba(color, alpha=PER_SWEEP_RAW_LINE_ALPHA)
        face_rgba = mcolors.to_rgba(face_color_base, alpha=PER_SWEEP_RAW_MARKER_FACE_ALPHA)
        edge_rgba = mcolors.to_rgba(edge_color_base, alpha=PER_SWEEP_RAW_MARKER_EDGE_ALPHA)
        ax.plot(x[order_idx], y_all[order_idx, j],
                linestyle=PER_SWEEP_RAW_LINESTYLE, linewidth=PER_SWEEP_RAW_LINEWIDTH, color=line_rgba,
                marker=PER_SWEEP_RAW_MARKER, markersize=PER_SWEEP_RAW_MARKER_SIZE,
                markerfacecolor=face_rgba, markeredgecolor=edge_rgba,
                markeredgewidth=PER_SWEEP_RAW_MARKER_EDGE_WIDTH,
                solid_capstyle='round', zorder=PER_SWEEP_RAW_ZORDER, label=f'n={n:+d}')

    if show_fit and result is not None:
        x_fit = np.linspace(x.min(), x.max(), 300)
        phi_fit_rad = np.deg2rad(x_fit)
        phi0_rad = np.deg2rad(result['phi0_deg'])
        phi_nl_rad = np.deg2rad(result['phi_nl_deg'])
        for n in result['fit_harmonics']:
            y_fit = _nl_predicted_ce(result['beta1'], result['beta2'], result['beta2_nl'],
                                      phi0_rad, phi_nl_rad, phi_fit_rad, n, k_trunc,
                                      scale=result['scales'].get(n, 1.0),
                                      units=PER_SWEEP_PLOT_UNITS)
            fit_color_base = _harmonic_color(n) if PER_SWEEP_FIT_COLOR == 'same' else PER_SWEEP_FIT_COLOR
            fit_rgba = mcolors.to_rgba(fit_color_base, alpha=PER_SWEEP_FIT_LINE_ALPHA)
            ax.plot(x_fit, y_fit, linestyle=PER_SWEEP_FIT_LINESTYLE, linewidth=PER_SWEEP_FIT_LINEWIDTH,
                    color=fit_rgba, solid_capstyle='round', zorder=PER_SWEEP_FIT_ZORDER)

    if PER_SWEEP_XLIM is not None:
        ax.set_xlim(PER_SWEEP_XLIM)
    if PER_SWEEP_YLIM is not None:
        ax.set_ylim(PER_SWEEP_YLIM)
    ax.grid(PER_SWEEP_SHOW_GRID)

    ax.set_xlabel('Ch2 phase [deg]', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_title(fname, fontsize=tick_label_fontsize)
    if PER_SWEEP_SHOW_LEGEND:
        ax.legend(fontsize=tick_label_fontsize, ncol=2)
    ax.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)
    return fig, ax


def _discover_sweep_files(folder: str, output_filename: str):
    """All .npz in folder, sorted, excluding one named output_filename."""
    all_npz = sorted(glob.glob(os.path.join(folder, '*.npz')))
    return [p for p in all_npz if os.path.basename(p) != output_filename]


def _load_cached_fits(output_path: str, fit_harmonics):
    """
    Load a previous run's saved results, keyed by filename, for reuse.

    Returns {} if output_path doesn't exist, can't be read, or was written
    with a different FIT_HARMONICS setting than the current one (reusing
    fits across a settings change would silently mix incompatible results).
    """
    if not os.path.exists(output_path):
        return {}
    try:
        d = np.load(output_path, allow_pickle=True)
        stored_harmonics = [int(n) for n in d['fit_harmonics']]
        fnames_stored = [str(n) for n in d['filenames']]
        scales_matrix = d['scales'] if 'scales' in d else None
    except Exception as exc:
        print(f"Warning: could not read existing {output_path} ({exc}); refitting everything.")
        return {}

    if stored_harmonics != list(fit_harmonics):
        print(f"Existing {os.path.basename(output_path)} was fit with harmonics "
              f"{stored_harmonics}, but FIT_HARMONICS is now {list(fit_harmonics)}; "
              f"ignoring cache and refitting everything.")
        return {}

    cached = {}
    for i, fname in enumerate(fnames_stored):
        row_scales = {}
        if scales_matrix is not None:
            for j, n in enumerate(stored_harmonics):
                v = scales_matrix[i, j]
                if np.isfinite(v):
                    row_scales[n] = float(v)
        cached[fname] = {
            'beta1': float(d['beta1'][i]),
            'beta2': float(d['beta2'][i]),
            'beta2_nl': float(d['beta2_nl'][i]),
            'phi0_deg': float(d['phi0_deg'][i]),
            'phi_nl_deg': float(d['phi_nl_deg'][i]),
            'cost': float(d['cost'][i]),
            'fit_harmonics': sorted(row_scales) if row_scales else list(stored_harmonics),
            'scales': row_scales if row_scales else {n: 1.0 for n in stored_harmonics},
        }
    return cached


def _drive_voltage(data: DualToneSweepData, axis: str) -> float:
    log20 = 10.0 * np.log10(20.0)
    powers_dbm = data.ch1_powers_dbm if axis == 'ch1_voltage' else data.ch2_powers_dbm
    return float(10.0 ** ((powers_dbm[0] - log20) / 20.0))


def _make_fig(w_mm, h_mm):
    mm = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=(
        (left_mm + w_mm + right_mm) * mm,
        (bottom_mm + h_mm + top_mm) * mm,
    ))
    fig.subplots_adjust(
        left=left_mm / (left_mm + w_mm + right_mm),
        right=(left_mm + w_mm) / (left_mm + w_mm + right_mm),
        bottom=bottom_mm / (bottom_mm + h_mm + top_mm),
        top=(bottom_mm + h_mm) / (bottom_mm + h_mm + top_mm),
    )
    return fig, ax


def _fit_and_overlay(ax, x_vals, y_vals, degree, exclude_indices, drive_voltages, fnames,
                      color, alpha, linestyle, linewidth, zorder, label, what, xlim=None):
    """
    Fit a degree-N polynomial to (x_vals, y_vals) and overlay it on ax.
    exclude_indices are 0-based positions in order of increasing
    drive_voltages (negative indices count from the end, as in normal
    Python indexing) -- excluded points are still shown as data elsewhere,
    just left out of this fit. The overlaid line spans xlim (xmin, xmax) if
    given, else the data's own min/max. Returns the fit coefficients
    (highest power first, as from np.polyfit), or None if there weren't
    enough points.
    """
    voltage_order = np.argsort(drive_voltages)
    n_sweeps = len(voltage_order)
    normalized_exclude = [i if i >= 0 else i + n_sweeps for i in exclude_indices]
    valid_exclude = sorted(set(i for i in normalized_exclude if 0 <= i < n_sweeps))
    fit_mask = np.ones(n_sweeps, dtype=bool)
    fit_mask[voltage_order[valid_exclude]] = False
    n_fit_pts = int(fit_mask.sum())
    if n_fit_pts < degree + 1:
        print(f"\nWarning: only {n_fit_pts} point(s) available for a degree-{degree} "
              f"fit of {what} (need {degree + 1}); skipping.")
        return None
    if valid_exclude:
        excluded_names = [fnames[voltage_order[i]] for i in valid_exclude]
        print(f"\nExcluding {len(valid_exclude)} sweep(s) from the {what} fit "
              f"(voltage-order indices {valid_exclude}): {excluded_names}")
    fit_coeffs = np.polyfit(x_vals[fit_mask], y_vals[fit_mask], degree)
    print(f"Degree-{degree} fit to {n_fit_pts} point(s) for {what}, "
          f"coefficients (highest power first): {fit_coeffs}")
    x_fit_min, x_fit_max = xlim if xlim is not None else (x_vals.min(), x_vals.max())
    x_fit = np.linspace(x_fit_min, x_fit_max, 200)
    y_fit = np.polyval(fit_coeffs, x_fit)
    fit_rgba = mcolors.to_rgba(color, alpha=alpha)
    ax.plot(x_fit, y_fit, linestyle=linestyle, linewidth=linewidth, color=fit_rgba,
            solid_capstyle='round', zorder=zorder, label=label or f'degree-{degree} fit')
    return fit_coeffs


def main():
    if not os.path.isdir(SWEEP_FOLDER):
        raise NotADirectoryError(f"Not a directory: {SWEEP_FOLDER}")

    output_path = os.path.join(SWEEP_FOLDER, OUTPUT_FILENAME)
    sweep_paths = _discover_sweep_files(SWEEP_FOLDER, OUTPUT_FILENAME)
    if not sweep_paths:
        raise RuntimeError(
            f"No sweep .npz files found in {SWEEP_FOLDER} "
            f"(excluding {OUTPUT_FILENAME}, if present)."
        )

    print(f"Found {len(sweep_paths)} sweep file(s) in {SWEEP_FOLDER}")
    print(f"Output will be written to {output_path} "
          f"(excluded from future scans of this folder)\n")

    k_trunc_nl = int(2 * max(NL_FIT_BETA1_BOUNDS[1],
                              NL_FIT_BETA2_BOUNDS[1] + NL_FIT_BETA2_NL_BOUNDS[1])) + 20

    plot_folder = PER_SWEEP_PLOT_FOLDER or os.path.join(SWEEP_FOLDER, 'per_sweep_fit_plots')
    show_index = max(0, min(PER_SWEEP_PLOT_SHOW_INDEX, len(sweep_paths) - 1))
    n_plotted = 0
    shown_fig = None

    cached = _load_cached_fits(output_path, FIT_HARMONICS) if REUSE_EXISTING_FITS else {}
    if cached:
        print(f"Reusing {len(cached)} previously-fit sweep(s) from "
              f"{os.path.basename(output_path)}; only new/unfit sweeps will be (re)fit.\n")

    fnames, drive_voltages = [], []
    beta1_list, beta2_list, beta2_nl_list = [], [], []
    phi0_list, phi_nl_list, cost_list = [], [], []
    scales_rows = []                 # one dict per successfully-fit sweep
    scales_by_harmonic = {}          # n -> list of scale values, pooled across sweeps

    for path in sweep_paths:
        fname = os.path.basename(path)
        try:
            data = DualToneSweepData.from_file(path)
        except Exception as exc:
            print(f"Warning: could not load {fname} ({exc}); skipping.")
            continue

        if fname in cached:
            print(f"Using cached fit for {fname} (skipping refit).")
            result = cached[fname]
        else:
            print(f"Fitting {fname} ...")
            result = _fit_one_sweep(data, FIT_HARMONICS, k_trunc_nl)

        if SAVE_PER_SWEEP_PLOTS:
            os.makedirs(plot_folder, exist_ok=True)
            fig_s, _ = _plot_sweep_with_fit(fname, data, result, PER_SWEEP_PLOT_SHOW_FIT, k_trunc_nl)
            out_name = os.path.splitext(fname)[0] + '_fit.png'
            fig_s.savefig(os.path.join(plot_folder, out_name), dpi=PER_SWEEP_PLOT_DPI, bbox_inches='tight')
            print(f"    Saved plot -> {os.path.join(plot_folder, out_name)}")
            if n_plotted == show_index:
                shown_fig = fig_s
            else:
                plt.close(fig_s)
            n_plotted += 1

        if result is None:
            continue

        fnames.append(fname)
        drive_voltages.append(_drive_voltage(data, DRIVE_VOLTAGE_AXIS))
        beta1_list.append(result['beta1'])
        beta2_list.append(result['beta2'])
        beta2_nl_list.append(result['beta2_nl'])
        phi0_list.append(result['phi0_deg'])
        phi_nl_list.append(result['phi_nl_deg'])
        cost_list.append(result['cost'])
        scales_rows.append(result['scales'])
        for n, v in result['scales'].items():
            scales_by_harmonic.setdefault(n, []).append(v)

    if not fnames:
        raise RuntimeError("No sweep could be fit successfully; nothing to save or plot.")

    drive_voltages = np.array(drive_voltages)
    beta1_arr = np.array(beta1_list)
    beta2_arr = np.array(beta2_list)
    beta2_nl_arr = np.array(beta2_nl_list)
    phi0_arr = np.array(phi0_list)
    phi_nl_arr = np.array(phi_nl_list)
    cost_arr = np.array(cost_list)

    scales_matrix = np.full((len(fnames), len(FIT_HARMONICS)), np.nan)
    for i, row in enumerate(scales_rows):
        for j, n in enumerate(FIT_HARMONICS):
            if n in row:
                scales_matrix[i, j] = row[n]

    np.savez_compressed(
        output_path,
        filenames=np.array(fnames),
        drive_voltage_axis=np.array(DRIVE_VOLTAGE_AXIS),
        drive_voltages=drive_voltages,
        fit_harmonics=np.array(FIT_HARMONICS),
        beta1=beta1_arr,
        beta2=beta2_arr,
        beta2_nl=beta2_nl_arr,
        phi0_deg=phi0_arr,
        phi_nl_deg=phi_nl_arr,
        cost=cost_arr,
        scales=scales_matrix,   # (n_files, len(FIT_HARMONICS)); NaN where not fit
    )
    print(f"\nDone. Fit {len(fnames)}/{len(sweep_paths)} sweep(s). Saved results to {output_path}")

    # ── scale factor statistics table (always printed, independent of the
    # per-sweep-plot option above) ──────────────────────────────────────
    if not NL_FIT_PER_SIDEBAND_SCALE:
        print("\nNL_FIT_PER_SIDEBAND_SCALE is False; every scale factor is fixed "
              "at 1.0 (no statistics to report).")
    elif scales_by_harmonic:
        print("\nScale factor statistics (per sideband, pooled across all fit sweeps):")
        print(f"  {'order':>5}  {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
        for n in sorted(scales_by_harmonic):
            vals = np.array(scales_by_harmonic[n])
            print(f"  {n:>+5d}  {len(vals):>4d}  {vals.mean():>8.4f}  {vals.std():>8.4f}  "
                  f"{vals.min():>8.4f}  {vals.max():>8.4f}")

    # ── summary plot: beta2_NL vs. drive voltage (or beta1) ───────────────
    if SUMMARY_PLOT_X_AXIS == 'beta1':
        x_vals = beta1_arr
        axis_label = r'$\beta_1$ [rad]'
    else:
        x_vals = drive_voltages
        axis_label = (r'Ch1 drive voltage [V$_\mathrm{rms}$]' if DRIVE_VOLTAGE_AXIS == 'ch1_voltage'
                      else r'Ch2 drive voltage [V$_\mathrm{rms}$]')
    order = np.argsort(x_vals)

    if SUMMARY_PLOT_Y_POWER is not None and SUMMARY_PLOT_Y_POWER != 1:
        y_vals = np.power(beta2_nl_arr, SUMMARY_PLOT_Y_POWER)
        y_label = SUMMARY_POWER_YLABEL or rf'$\beta_{{2,\mathrm{{NL}}}}^{{{SUMMARY_PLOT_Y_POWER:g}}}$ [a.u.]'
        y_legend_label = rf'$\beta_{{2,\mathrm{{NL}}}}^{{{SUMMARY_PLOT_Y_POWER:g}}}$'
    else:
        y_vals = beta2_nl_arr
        y_label = r'$\beta_{2,\mathrm{NL}}$ [rad]'
        y_legend_label = r'$\beta_{2,\mathrm{NL}}$'

    line_color = mcolors.to_rgba(SUMMARY_LINE_COLOR, alpha=SUMMARY_LINE_ALPHA)
    face_color_base = SUMMARY_LINE_COLOR if SUMMARY_MARKER_FACE_COLOR == 'same' else SUMMARY_MARKER_FACE_COLOR
    face_color = mcolors.to_rgba(face_color_base, alpha=SUMMARY_MARKER_FACE_ALPHA)
    edge_color = mcolors.to_rgba(SUMMARY_MARKER_EDGE_COLOR, alpha=SUMMARY_MARKER_EDGE_ALPHA)

    fig, ax = _make_fig(axes_width_mm, axes_height_mm)
    ax.plot(x_vals[order], y_vals[order],
            linestyle=SUMMARY_LINESTYLE, linewidth=SUMMARY_LINEWIDTH, color=line_color,
            marker=SUMMARY_MARKER, markersize=SUMMARY_MARKER_SIZE,
            markerfacecolor=face_color, markeredgecolor=edge_color,
            markeredgewidth=SUMMARY_MARKER_EDGE_WIDTH,
            solid_capstyle='round', zorder=SUMMARY_ZORDER,
            label=y_legend_label)

    if SUMMARY_FIT_SHOW:
        _fit_and_overlay(ax, x_vals, y_vals, SUMMARY_FIT_DEGREE, SUMMARY_FIT_EXCLUDE_INDICES,
                          drive_voltages, fnames, SUMMARY_FIT_COLOR, SUMMARY_FIT_ALPHA,
                          SUMMARY_FIT_LINESTYLE, SUMMARY_FIT_LINEWIDTH, SUMMARY_FIT_ZORDER,
                          SUMMARY_FIT_LABEL, 'beta2_NL summary', xlim=SUMMARY_XLIM)

    if SUMMARY_XLIM is not None:
        ax.set_xlim(SUMMARY_XLIM)
    if SUMMARY_YLIM is not None:
        ax.set_ylim(SUMMARY_YLIM)
    ax.grid(SUMMARY_SHOW_GRID)
    if SUMMARY_SHOW_LEGEND:
        ax.legend(fontsize=tick_label_fontsize, frameon=False)

    ax.set_xlabel(axis_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)

    if FOR_PUBLICATION:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        svg_path = os.path.join(PUBLICATION_SVG_FOLDER, 'beta2_nl_summary.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path}")

    # ── separate figure: beta1 vs. drive voltage ──────────────────────────
    if SHOW_BETA1_VS_VOLTAGE_PLOT:
        order_b1 = np.argsort(drive_voltages)
        b1_line_color = mcolors.to_rgba(BETA1_LINE_COLOR, alpha=BETA1_LINE_ALPHA)
        b1_face_base = BETA1_LINE_COLOR if BETA1_MARKER_FACE_COLOR == 'same' else BETA1_MARKER_FACE_COLOR
        b1_face_color = mcolors.to_rgba(b1_face_base, alpha=BETA1_MARKER_FACE_ALPHA)
        b1_edge_color = mcolors.to_rgba(BETA1_MARKER_EDGE_COLOR, alpha=BETA1_MARKER_EDGE_ALPHA)

        fig_b1, ax_b1 = _make_fig(BETA1_PLOT_AXES_WIDTH_MM, BETA1_PLOT_AXES_HEIGHT_MM)
        ax_b1.plot(drive_voltages[order_b1], beta1_arr[order_b1],
                   linestyle=BETA1_LINESTYLE, linewidth=BETA1_LINEWIDTH, color=b1_line_color,
                   marker=BETA1_MARKER, markersize=BETA1_MARKER_SIZE,
                   markerfacecolor=b1_face_color, markeredgecolor=b1_edge_color,
                   markeredgewidth=BETA1_MARKER_EDGE_WIDTH,
                   solid_capstyle='round', zorder=BETA1_ZORDER, label=r'$\beta_1$')

        if BETA1_FIT_SHOW:
            beta1_fit_coeffs = _fit_and_overlay(
                ax_b1, drive_voltages, beta1_arr, BETA1_FIT_DEGREE, BETA1_FIT_EXCLUDE_INDICES,
                drive_voltages, fnames, BETA1_FIT_COLOR, BETA1_FIT_ALPHA,
                BETA1_FIT_LINESTYLE, BETA1_FIT_LINEWIDTH, BETA1_FIT_ZORDER,
                BETA1_FIT_LABEL, 'beta1 vs voltage', xlim=BETA1_XLIM)
            if beta1_fit_coeffs is not None:
                if BETA1_FIT_DEGREE == 1:
                    slope, intercept = beta1_fit_coeffs
                    if slope > 0:
                        vpi_fit = np.pi / slope
                        print(f"Vpi (from beta1 = slope*V + intercept, Vpi = pi/slope): "
                              f"{vpi_fit:.4f} V  (intercept = {intercept:.4f} rad, "
                              f"ideally ~0 for a pure phase modulator)")
                    else:
                        print(f"Warning: beta1-vs-voltage fit slope is non-positive "
                              f"({slope:.4g}); cannot compute a physical Vpi.")
                else:
                    print(f"Note: BETA1_FIT_DEGREE={BETA1_FIT_DEGREE} != 1; "
                          f"skipping Vpi (only defined for a linear beta1-vs-voltage fit).")

        if BETA1_XLIM is not None:
            ax_b1.set_xlim(BETA1_XLIM)
        if BETA1_YLIM is not None:
            ax_b1.set_ylim(BETA1_YLIM)
        ax_b1.grid(BETA1_SHOW_GRID)
        if BETA1_SHOW_LEGEND:
            ax_b1.legend(fontsize=tick_label_fontsize, frameon=False)

        b1_axis_label = (r'Ch1 drive voltage [V$_\mathrm{rms}$]' if DRIVE_VOLTAGE_AXIS == 'ch1_voltage'
                          else r'Ch2 drive voltage [V$_\mathrm{rms}$]')
        ax_b1.set_xlabel(b1_axis_label, fontsize=axis_label_fontsize)
        ax_b1.set_ylabel(r'$\beta_1$ [rad]', fontsize=axis_label_fontsize)
        ax_b1.tick_params(axis='both', direction=tick_direction,
                           width=tick_width, labelsize=tick_label_fontsize)
        for side in ['top', 'bottom', 'left', 'right']:
            ax_b1.spines[side].set_linewidth(spine_linewidth)

        if FOR_PUBLICATION:
            ax_b1.set_xlabel('')
            ax_b1.set_ylabel('')
            ax_b1.tick_params(labelbottom=False, labelleft=False)
            legend_b1 = ax_b1.get_legend()
            if legend_b1 is not None:
                legend_b1.remove()
            svg_path_b1 = os.path.join(PUBLICATION_SVG_FOLDER, 'beta1_summary.svg')
            fig_b1.savefig(svg_path_b1, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path_b1}")

    # ── separate figure: beta2 vs. drive voltage ──────────────────────────
    if SHOW_BETA2_VS_VOLTAGE_PLOT:
        order_b2 = np.argsort(drive_voltages)
        b2_line_color = mcolors.to_rgba(BETA2_LINE_COLOR, alpha=BETA2_LINE_ALPHA)
        b2_face_base = BETA2_LINE_COLOR if BETA2_MARKER_FACE_COLOR == 'same' else BETA2_MARKER_FACE_COLOR
        b2_face_color = mcolors.to_rgba(b2_face_base, alpha=BETA2_MARKER_FACE_ALPHA)
        b2_edge_color = mcolors.to_rgba(BETA2_MARKER_EDGE_COLOR, alpha=BETA2_MARKER_EDGE_ALPHA)

        fig_b2, ax_b2 = _make_fig(BETA2_PLOT_AXES_WIDTH_MM, BETA2_PLOT_AXES_HEIGHT_MM)
        ax_b2.plot(drive_voltages[order_b2], beta2_arr[order_b2],
                   linestyle=BETA2_LINESTYLE, linewidth=BETA2_LINEWIDTH, color=b2_line_color,
                   marker=BETA2_MARKER, markersize=BETA2_MARKER_SIZE,
                   markerfacecolor=b2_face_color, markeredgecolor=b2_edge_color,
                   markeredgewidth=BETA2_MARKER_EDGE_WIDTH,
                   solid_capstyle='round', zorder=BETA2_ZORDER, label=r'$\beta_2$')

        if BETA2_FIT_SHOW:
            _fit_and_overlay(ax_b2, drive_voltages, beta2_arr, BETA2_FIT_DEGREE, BETA2_FIT_EXCLUDE_INDICES,
                              drive_voltages, fnames, BETA2_FIT_COLOR, BETA2_FIT_ALPHA,
                              BETA2_FIT_LINESTYLE, BETA2_FIT_LINEWIDTH, BETA2_FIT_ZORDER,
                              BETA2_FIT_LABEL, 'beta2 vs voltage', xlim=BETA2_XLIM)

        if BETA2_XLIM is not None:
            ax_b2.set_xlim(BETA2_XLIM)
        if BETA2_YLIM is not None:
            ax_b2.set_ylim(BETA2_YLIM)
        ax_b2.grid(BETA2_SHOW_GRID)
        if BETA2_SHOW_LEGEND:
            ax_b2.legend(fontsize=tick_label_fontsize, frameon=False)

        b2_axis_label = (r'Ch1 drive voltage [V$_\mathrm{rms}$]' if DRIVE_VOLTAGE_AXIS == 'ch1_voltage'
                          else r'Ch2 drive voltage [V$_\mathrm{rms}$]')
        ax_b2.set_xlabel(b2_axis_label, fontsize=axis_label_fontsize)
        ax_b2.set_ylabel(r'$\beta_2$ [rad]', fontsize=axis_label_fontsize)
        ax_b2.tick_params(axis='both', direction=tick_direction,
                           width=tick_width, labelsize=tick_label_fontsize)
        for side in ['top', 'bottom', 'left', 'right']:
            ax_b2.spines[side].set_linewidth(spine_linewidth)

        if FOR_PUBLICATION:
            ax_b2.set_xlabel('')
            ax_b2.set_ylabel('')
            ax_b2.tick_params(labelbottom=False, labelleft=False)
            legend_b2 = ax_b2.get_legend()
            if legend_b2 is not None:
                legend_b2.remove()
            svg_path_b2 = os.path.join(PUBLICATION_SVG_FOLDER, 'beta2_summary.svg')
            fig_b2.savefig(svg_path_b2, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path_b2}")

    # ── separate figure: phi_NL vs. drive voltage ──────────────────────────
    if SHOW_PHI_NL_VS_VOLTAGE_PLOT:
        order_pn = np.argsort(drive_voltages)
        pn_line_color = mcolors.to_rgba(PHI_NL_LINE_COLOR, alpha=PHI_NL_LINE_ALPHA)
        pn_face_base = PHI_NL_LINE_COLOR if PHI_NL_MARKER_FACE_COLOR == 'same' else PHI_NL_MARKER_FACE_COLOR
        pn_face_color = mcolors.to_rgba(pn_face_base, alpha=PHI_NL_MARKER_FACE_ALPHA)
        pn_edge_color = mcolors.to_rgba(PHI_NL_MARKER_EDGE_COLOR, alpha=PHI_NL_MARKER_EDGE_ALPHA)

        fig_pn, ax_pn = _make_fig(PHI_NL_PLOT_AXES_WIDTH_MM, PHI_NL_PLOT_AXES_HEIGHT_MM)
        ax_pn.plot(drive_voltages[order_pn], phi_nl_arr[order_pn],
                   linestyle=PHI_NL_LINESTYLE, linewidth=PHI_NL_LINEWIDTH, color=pn_line_color,
                   marker=PHI_NL_MARKER, markersize=PHI_NL_MARKER_SIZE,
                   markerfacecolor=pn_face_color, markeredgecolor=pn_edge_color,
                   markeredgewidth=PHI_NL_MARKER_EDGE_WIDTH,
                   solid_capstyle='round', zorder=PHI_NL_ZORDER, label=r'$\phi_{\mathrm{NL}}$')

        if PHI_NL_FIT_SHOW:
            _fit_and_overlay(ax_pn, drive_voltages, phi_nl_arr, PHI_NL_FIT_DEGREE, PHI_NL_FIT_EXCLUDE_INDICES,
                              drive_voltages, fnames, PHI_NL_FIT_COLOR, PHI_NL_FIT_ALPHA,
                              PHI_NL_FIT_LINESTYLE, PHI_NL_FIT_LINEWIDTH, PHI_NL_FIT_ZORDER,
                              PHI_NL_FIT_LABEL, 'phi_NL vs voltage', xlim=PHI_NL_XLIM)

        if PHI_NL_XLIM is not None:
            ax_pn.set_xlim(PHI_NL_XLIM)
        if PHI_NL_YLIM is not None:
            ax_pn.set_ylim(PHI_NL_YLIM)
        ax_pn.grid(PHI_NL_SHOW_GRID)
        if PHI_NL_SHOW_LEGEND:
            ax_pn.legend(fontsize=tick_label_fontsize, frameon=False)

        pn_axis_label = (r'Ch1 drive voltage [V$_\mathrm{rms}$]' if DRIVE_VOLTAGE_AXIS == 'ch1_voltage'
                          else r'Ch2 drive voltage [V$_\mathrm{rms}$]')
        ax_pn.set_xlabel(pn_axis_label, fontsize=axis_label_fontsize)
        ax_pn.set_ylabel(r'$\phi_{\mathrm{NL}}$ [deg]', fontsize=axis_label_fontsize)
        ax_pn.tick_params(axis='both', direction=tick_direction,
                           width=tick_width, labelsize=tick_label_fontsize)
        for side in ['top', 'bottom', 'left', 'right']:
            ax_pn.spines[side].set_linewidth(spine_linewidth)

        if FOR_PUBLICATION:
            ax_pn.set_xlabel('')
            ax_pn.set_ylabel('')
            ax_pn.tick_params(labelbottom=False, labelleft=False)
            legend_pn = ax_pn.get_legend()
            if legend_pn is not None:
                legend_pn.remove()
            svg_path_pn = os.path.join(PUBLICATION_SVG_FOLDER, 'phi_nl_summary.svg')
            fig_pn.savefig(svg_path_pn, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path_pn}")

    # ── separate figure: scale-factor summary bar chart ────────────────────
    if SHOW_SCALE_BAR_CHART:
        if not NL_FIT_PER_SIDEBAND_SCALE:
            print("\nNL_FIT_PER_SIDEBAND_SCALE is False; skipping the scale-factor bar chart "
                  "(every scale factor is fixed at 1.0).")
        elif not scales_by_harmonic:
            print("\nNo per-sideband scale factors available; skipping the scale-factor bar chart.")
        else:
            bar_orders = sorted(scales_by_harmonic)
            bar_means = np.array([np.mean(scales_by_harmonic[n]) for n in bar_orders])
            bar_stds = np.array([np.std(scales_by_harmonic[n]) for n in bar_orders])

            bar_face_color = mcolors.to_rgba(SCALE_BAR_FACE_COLOR, alpha=SCALE_BAR_FACE_ALPHA)
            bar_edge_color = mcolors.to_rgba(SCALE_BAR_EDGE_COLOR, alpha=SCALE_BAR_EDGE_ALPHA)
            bar_err_color = mcolors.to_rgba(SCALE_BAR_ERR_COLOR, alpha=SCALE_BAR_ERR_ALPHA)

            fig_sb, ax_sb = _make_fig(SCALE_BAR_AXES_WIDTH_MM, SCALE_BAR_AXES_HEIGHT_MM)
            x_bar = np.arange(len(bar_orders))
            ax_sb.bar(x_bar, bar_means, width=SCALE_BAR_WIDTH,
                      color=bar_face_color, edgecolor=bar_edge_color, linewidth=SCALE_BAR_EDGE_WIDTH,
                      zorder=SCALE_BAR_ZORDER, label='scale factor',
                      yerr=bar_stds, capsize=SCALE_BAR_ERR_CAPSIZE,
                      error_kw=dict(ecolor=bar_err_color, elinewidth=SCALE_BAR_ERR_LINEWIDTH,
                                    capthick=SCALE_BAR_ERR_LINEWIDTH, zorder=SCALE_BAR_ZORDER + 1))
            ax_sb.set_xticks(x_bar)
            ax_sb.set_xticklabels([f'{n:+d}' for n in bar_orders])

            if SCALE_BAR_YLIM is not None:
                ax_sb.set_ylim(SCALE_BAR_YLIM)
            ax_sb.grid(SCALE_BAR_SHOW_GRID)
            if SCALE_BAR_SHOW_LEGEND:
                ax_sb.legend(fontsize=tick_label_fontsize, frameon=False)

            ax_sb.set_xlabel('Harmonic order', fontsize=axis_label_fontsize)
            ax_sb.set_ylabel('Correction scale factor', fontsize=axis_label_fontsize)
            ax_sb.tick_params(axis='both', direction=tick_direction,
                               width=tick_width, labelsize=tick_label_fontsize)
            for side in ['top', 'bottom', 'left', 'right']:
                ax_sb.spines[side].set_linewidth(spine_linewidth)

            if FOR_PUBLICATION:
                ax_sb.set_xlabel('')
                ax_sb.set_ylabel('')
                ax_sb.tick_params(labelbottom=False, labelleft=False)
                legend_sb = ax_sb.get_legend()
                if legend_sb is not None:
                    legend_sb.remove()
                svg_path_sb = os.path.join(PUBLICATION_SVG_FOLDER, 'scale_factor_bar.svg')
                fig_sb.savefig(svg_path_sb, format='svg', bbox_inches='tight')
                print(f"Saved: {svg_path_sb}")

    plt.show()


if __name__ == '__main__':
    main()
