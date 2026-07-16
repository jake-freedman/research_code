"""
Analyse a power-sweep heterodyne harmonic recording produced by
vna_power_heterodyne_sweep() in vna_power_harmonic_esa_script.py
or bnc_power_heterodyne_sweep() in bnc_power_harmonic_esa_script.py.

Same as power_harmonic_sweep_analysis.py, but additionally estimates V_pi
from user-supplied "anchor points": known modulation-depth values at a
critical feature of a sideband curve (e.g. a Bessel-function zero, where a
given harmonic's power dips to a minimum). For each anchor, the RF voltage
at which that harmonic's power is minimized within a search window is
located, and V_pi is estimated from the linear relation
beta(V) = (pi/V_pi)*V:

    V_pi = pi * V_dip / beta_anchor

A separate V_pi estimate is produced for every anchor point provided. Each
anchor's V_pi can also be used to overlay theoretical Bessel sideband curves
J_n(beta(V))^2 (pure single-tone phase modulation) on the sideband plot.

A per-harmonic efficiency correction is also available: at one reference
voltage, each harmonic's measured conversion efficiency is compared to its
theoretical value, and the resulting per-harmonic ratio can be applied to
every voltage in that harmonic's curve (e.g. to remove a harmonic-dependent
systematic loss in the detection chain).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
from scipy.special import jv as bessel_jv
from power_harmonic_sweep_data import PowerHeterodyneSweepData
from path_utils import local_path
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# Font sizes for axis and tick labels (set explicitly here rather than
# inherited from graphics.py, so this script's labels stay fixed regardless
# of that module's defaults).
axis_label_fontsize = 10
tick_label_fontsize = 8

# Same harmonic color scheme as dual_tone_sweep_analysis.py / dual_tone_sweep_data.py
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

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\really_good_bnc_power_sweep_mode2.npz"

# X-axis for the sideband/calibration plots: 'voltage' (V_rms) or 'dbm'.
# (Anchor V_pi values are always computed from the true RF voltage,
# regardless of this setting -- it only affects how the plots and anchor
# search windows are displayed.)
X_AXIS = 'voltage'

# Normalize sideband powers by the per-step calibration carrier level?
#   False      → y-axis in dBm  (raw ESA power)
#   True       → y-axis in dBc  (relative to optical carrier, log scale)
#   'percent'  → y-axis in %    (fraction of carrier power, linear scale)
NORMALIZE = 'percent'

# Show the calibration (carrier-beat) power vs drive level?
SHOW_CALIBRATION = False

# Show the legend on the sideband power plot?
SHOW_LEGEND = False

# None = show all recorded harmonics; list = show only those orders.
HARMONICS_TO_SHOW = None

# Y-axis limits for sideband power plot. None = auto.
if NORMALIZE == 'percent':
    POWER_YMIN = -5
    POWER_YMAX = 105
else:
    POWER_YMIN = None
    POWER_YMAX = None

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40

# ── sideband curve style ──────────────────────────────────────────────────────
SIDEBAND_LINESTYLE        = 'none'
SIDEBAND_LINEWIDTH        = 1.5
SIDEBAND_MARKER           = 'o'
SIDEBAND_MARKERSIZE       = 6
SIDEBAND_MARKER_FACECOLOR = None   # None = match the harmonic's line color
SIDEBAND_MARKER_EDGECOLOR = None   # None = match the harmonic's line color
SIDEBAND_MARKER_EDGEWIDTH = 0.00
SIDEBAND_ALPHA            = 1
SIDEBAND_ZORDER           = 2

# ── V_pi anchor points ────────────────────────────────────────────────────────
# Each anchor identifies a known Bessel-function critical point (a null or a
# local maximum) of one recorded harmonic:
#   'harmonic'  -> which recorded harmonic's power curve to search
#   'beta_zero' -> the known theoretical modulation depth (rad) at that
#                  critical point -- see 'kind' below for which kind of
#                  point this must be
#   'window'    -> (xmin, xmax) search window, in the units of X_AXIS, that
#                  brackets just that one critical point (needed since a
#                  harmonic can have several across the sweep)
#   'kind'      -> 'null' (default): look for the harmonic's power MINIMUM
#                     in the window (a Bessel zero, e.g. 2.405, 5.520,
#                     8.654, ... for J0; 3.832, 7.016, ... for J1; 5.136,
#                     8.417, ... for J2)
#                  'max': look for the harmonic's power MAXIMUM in the
#                     window (a Bessel local max, e.g. 1.8412 for J1's
#                     first max, 3.0542 for J2's first max, 4.2012 for
#                     J3's first max)
#                  'inflection': look for the point of steepest slope
#                     |dP_n/dV| in the window (a curvature zero of
#                     J_n(beta)^2 -- these sit between a null and the
#                     adjacent max and aren't standard tabulated constants,
#                     so beta_zero here should come from a numerical root
#                     search for your harmonic; they're the most
#                     sensitive points but the noisiest landmarks, since
#                     locating them means numerically differentiating the
#                     power curve twice)
#   'label'     -> optional description used in printouts/plot annotations
# ANCHOR_POINTS = [
#     {'harmonic': 0, 'beta_zero': 2.405, 'window': (1.0, 1.5), 'label': 'J0 1st null'},
#     {'harmonic': 1, 'beta_zero': 3.832, 'window': (1.5, 2.0), 'label': 'J1 1st null'},
#     {'harmonic': 1, 'beta_zero': 1.8412, 'window': (0.5, 1.0), 'kind': 'max', 'label': 'J1 1st max'},
#     {'harmonic': 2, 'beta_zero': 3.0542, 'window': (1.2, 1.8), 'kind': 'max', 'label': 'J2 1st max'},
# ]

ANCHOR_POINTS = [
    {'harmonic': 1, 'beta_zero': 1.8412, 'window': (0.5, 2.0), 'kind': 'max', 'label': 'J1 1st max'},
    {'harmonic': 0, 'beta_zero': 2.405, 'window': (0.5, 2.5), 'label': 'J0 1st null'},

    # Inflection points (curvature zeros of J_n(beta)^2, numerically rooted --
    # not standard tabulated constants). Voltages below use V_pi = 1.5605 V,
    # just for picking a sensible search window; the fit doesn't depend on
    # that guess, only on beta_zero and where the window actually brackets
    # the real inflection in your data.
    # {'harmonic': 0, 'beta_zero': 1.0820, 'window': (0.3, 0.8), 'kind': 'inflection', 'label': 'J0 1st inflection'},
    # {'harmonic': 1, 'beta_zero': 0.9116, 'window': (0.2, 0.7), 'kind': 'inflection', 'label': 'J1 1st inflection'},
    # {'harmonic': 2, 'beta_zero': 2.0646, 'window': (0.7, 1.8), 'kind': 'inflection', 'label': 'J2 1st inflection'},
]

# ── anchor marker style ───────────────────────────────────────────────────────
SHOW_ANCHOR_MARKERS = False
ANCHOR_MARKER_COLOR = RED2
ANCHOR_MARKER_STYLE = 'x'
ANCHOR_MARKER_SIZE  = 60     # scatter marker size (points^2)
ANCHOR_MARKER_WIDTH = 1.5    # marker edge/stroke width (points)
ANCHOR_ZORDER        = 5

SHOW_ANCHOR_LABELS   = False
ANCHOR_LABEL_FONTSIZE = 7
ANCHOR_LABEL_COLOR    = ANCHOR_MARKER_COLOR
ANCHOR_LABEL_OFFSET_PT = (0, 8)   # (x, y) offset in points from the marker

# ── derivative plot ────────────────────────────────────────────────────────────
# Plot d(sideband power)/dx vs drive level, in the same NORMALIZE/X_AXIS units
# as the main sideband plot (so a %-mode plot gives %/V, etc). When
# SHOW_ANCHOR_MARKERS is also on, anchors are marked here too, at the same
# located voltage -- particularly useful for 'inflection' anchors, since
# they're defined as the extremum of this exact derivative.
SHOW_DERIVATIVE_PLOT = False
DERIVATIVE_LINESTYLE  = '-'
DERIVATIVE_LINEWIDTH  = 1.5
DERIVATIVE_MARKER     = 'o'
DERIVATIVE_MARKERSIZE = 4
DERIVATIVE_ALPHA      = 1.0
DERIVATIVE_ZORDER     = 2
DERIVATIVE_SVG_NAME   = 'vpi_anchor_derivative.svg'

# ── theoretical sideband overlay (from anchor V_pi) ───────────────────────────
# Overlay theoretical pure-phase-modulation Bessel sideband curves
# J_n(beta(V))^2, with beta(V) = pi*V/V_pi. A single V_pi is used: the mean
# of the anchors selected by THEORY_ANCHORS (a list of ANCHOR_POINTS indices
# and/or anchor 'label' strings), or the mean of ALL successfully found
# anchors if THEORY_ANCHORS is None. In raw-dBm mode (NORMALIZE = False) this
# requires per-step calibration (cal_spectra) to convert the Bessel fraction
# to dBm.
SHOW_THEORY = True
THEORY_ANCHORS = None
# Which harmonics to draw theory curves for. None = same as HARMONICS_TO_SHOW
# (or all recorded harmonics if that is also None).
THEORY_HARMONICS = [0,1,2]

THEORY_LINESTYLE  = '--'
THEORY_LINEWIDTH  = 1.5
THEORY_MARKER     = None
THEORY_MARKERSIZE = 3
THEORY_COLOR      = '#000000'   # None = match the harmonic's sideband color
THEORY_ALPHA      = 1.0
THEORY_ZORDER     = 10

# ── per-harmonic efficiency correction ────────────────────────────────────────
# At the RF voltage nearest CORRECTION_VOLTAGE (in true V_rms, regardless of
# X_AXIS), each harmonic's measured conversion efficiency is compared to its
# theoretical J_n(beta)^2 value (beta from CORRECTION_VPI_SOURCE's V_pi).
# The resulting per-harmonic ratio (theory / measured) can then be applied
# to that harmonic's data at every voltage, so its curve is shifted by a
# fixed factor to match theory at the reference point. Requires per-step
# calibration (cal_spectra).
CORRECTION_VOLTAGE = 1.1   # reference RF voltage in V_rms; None = disabled
# Which anchor's V_pi supplies the theory reference at CORRECTION_VOLTAGE:
# None = mean V_pi across all successfully found anchors; otherwise an
# ANCHOR_POINTS index or anchor 'label' string.
CORRECTION_VPI_SOURCE = None
# Apply the per-harmonic correction factor to the plotted sideband data?
APPLY_CORRECTION = True

# When APPLY_CORRECTION is on, show a slider (+ an on/off checkbox) under the
# sideband plot: the slider drags the correction reference voltage live, and
# the checkbox toggles whether the correction is applied at all -- both
# recompute each harmonic's plotted curve in place (theory/measured ratio at
# the current voltage, or the raw uncorrected data when unchecked), without
# re-running the script. Purely a plot interaction; the CORRECTION_VOLTAGE
# setting above (and its printed summary) is unchanged.
INTERACTIVE_CORRECTION_SLIDER = True
# Slider range in true V_rms. None = the full measured sweep range.
CORRECTION_SLIDER_RANGE = None
CORRECTION_SLIDER_LABEL = 'Correction V'
CORRECTION_SLIDER_HEIGHT_MM = 8
CORRECTION_SLIDER_PAD_MM    = 6   # gap between the plot and the controls

CORRECTION_CHECKBOX_LABEL      = 'Apply'
CORRECTION_CHECKBOX_WIDTH_MM   = 22
CORRECTION_CONTROL_GAP_MM      = 4   # horizontal gap between slider and checkbox

# ── multi-voltage correction sampling ─────────────────────────────────────────
# Alternative to a single CORRECTION_VOLTAGE: linearly sample
# CORRECTION_N_SAMPLES voltages over CORRECTION_VOLTAGE_RANGE and compute a
# correction factor per harmonic at each sampled voltage (same theory/measured
# ratio as compute_correction_factors, using CORRECTION_VPI_SOURCE's V_pi). A
# sample is excluded for a given harmonic if its measured conversion
# efficiency there is below CORRECTION_MIN_CE (a linear fraction, e.g. 0.05 =
# 5%) -- too close to zero to give a reliable ratio. The mean and std of the
# surviving per-harmonic factors are then overlaid as a corrected curve
# (mean) with a shaded uncertainty cloud (+-std), independent of
# APPLY_CORRECTION above.
SHOW_CORRECTED_CLOUD = False
CORRECTION_VOLTAGE_RANGE = (0.5, 2.5)   # (vmin, vmax), true V_rms
CORRECTION_N_SAMPLES = 20
CORRECTION_MIN_CE = 0.05   # 5%, linear fraction of carrier power
# Which harmonics get a corrected-cloud overlay. None = same as
# HARMONICS_TO_SHOW (or all recorded harmonics if that is also None).
CORRECTED_HARMONICS = None

CORRECTED_LINE_COLOR   = None   # None = match the harmonic's sideband color
CORRECTED_LINESTYLE    = '-'
CORRECTED_LINEWIDTH    = 1.5
CORRECTED_ALPHA        = 1.0
CORRECTED_ZORDER       = 4

CORRECTED_CLOUD_COLOR  = None   # None = match the harmonic's sideband color
CORRECTED_CLOUD_ALPHA         = 0.25   # 1-sigma band (darker/more opaque)
CORRECTED_CLOUD_ALPHA_2SIGMA  = 0.12   # 2-sigma band (lighter)
CORRECTED_CLOUD_ZORDER        = 3
CORRECTED_CLOUD_ZORDER_2SIGMA = 2      # drawn behind the 1-sigma band

# ── figure size (mm) ───────────────────────────────────────────────────────────
axes_width_mm  = 100
axes_height_mm = 40

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels and the legend, and saves the sideband
# (and calibration, if shown) plot as SVGs. Markers/lines are left exactly as
# configured above (SIDEBAND_*/THEORY_*/CORRECTED_*) -- no separate styling.
FOR_PUBLICATION = False
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"
SIDEBAND_SVG_NAME    = 'vpi_anchor_sidebands.svg'
CALIBRATION_SVG_NAME = 'vpi_anchor_calibration.svg'
# ─────────────────────────────────────────────────────────────────────────────


def load_averaged(filepath: str) -> PowerHeterodyneSweepData:
    """
    Load a power-sweep heterodyne .npz, averaging out a leading repeat axis
    if present (spectra shaped (R, M, N, K) instead of (M, N, K); cal_spectra
    (R, M, K) instead of (M, K)), as recorded when n_repeats > 1. Averaging
    is done in linear power, then converted back to dBm.
    """
    d = np.load(filepath)
    spectra = d['spectra']
    n_repeats = int(d['n_repeats']) if 'n_repeats' in d else 1

    if spectra.ndim == 4:
        spectra = 10.0 * np.log10((10.0 ** (spectra / 10.0)).mean(axis=0))
    elif spectra.ndim != 3:
        raise ValueError(f"Unexpected spectra shape {spectra.shape} in {filepath}")

    cal = d['cal_spectra'] if 'cal_spectra' in d else None
    if cal is not None and cal.ndim == 3:
        cal = 10.0 * np.log10((10.0 ** (cal / 10.0)).mean(axis=0))

    data = PowerHeterodyneSweepData.__new__(PowerHeterodyneSweepData)
    data.cw_freq          = float(d['cw_freq'])
    data.cw_powers        = d['cw_powers']
    data.harmonics        = d['harmonics'].astype(int)
    data.heterodyne_shift = float(d['heterodyne_shift'])
    data.offsets_hz       = d['offsets_hz']
    data.spectra          = spectra
    data.window_hz        = float(d['window_hz'])
    data.cal_spectra      = cal
    data.filepath         = filepath
    data.n_repeats        = n_repeats
    return data


def _x_values(data: PowerHeterodyneSweepData, x_axis: str):
    if x_axis == 'dbm':
        return data.cw_powers, 'Drive power [dBm]', 'dBm'
    return data.rf_voltage_rms(), r'RF voltage [V$_\mathrm{rms}$]', 'V_rms'


def _peak_values(data: PowerHeterodyneSweepData, normalize, correction: dict | None = None):
    """
    Same peak-power transform plot_peak_powers() uses, so anchor markers
    land exactly on the plotted curve regardless of NORMALIZE.

    If `correction` (a {harmonic: factor} dict) is given, each harmonic's
    measured fraction of carrier power is scaled by that factor first
    (applied additively in dBc, i.e. multiplicatively in linear power);
    requires cal_spectra.
    """
    if correction is not None:
        if data.cal_spectra is None:
            raise RuntimeError(
                "APPLY_CORRECTION requires per-step calibration (cal_spectra)."
            )
        dbc = data.normalized_peak_powers_dbm().copy()
        for j, n in enumerate(data.harmonics):
            factor = correction.get(int(n), 1.0)
            dbc[:, j] += 10.0 * np.log10(max(factor, 1e-30))

        if normalize == 'percent':
            return 10.0 ** (dbc / 10.0) * 100.0
        if normalize:
            return dbc
        return dbc + data.cal_peak_power_dbm()[:, np.newaxis]

    if normalize == 'percent':
        return 10.0 ** (data.normalized_peak_powers_dbm() / 10.0) * 100.0
    if normalize:
        return data.normalized_peak_powers_dbm()
    return data.peak_powers_dbm()


def _harmonic_idx(data: PowerHeterodyneSweepData, n: int) -> int:
    idx = np.where(data.harmonics == n)[0]
    if len(idx) == 0:
        raise ValueError(f"Harmonic {n} not in dataset. Available: {list(data.harmonics)}")
    return int(idx[0])


def _build_harmonic_colors(harmonics) -> dict:
    """Map each harmonic number to a color, using _HARMONIC_COLORS with
    _EXTRA_COLORS as a fallback for orders not in that table."""
    extra_iter = iter(_EXTRA_COLORS)
    return {int(n): _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000')) for n in harmonics}


def _make_figure(w_mm: float, h_mm: float):
    mm = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=((_left_mm + w_mm + _right_mm) * mm,
                                     (_bottom_mm + h_mm + _top_mm) * mm))
    fig.subplots_adjust(
        left=_left_mm / (_left_mm + w_mm + _right_mm),
        right=(_left_mm + w_mm) / (_left_mm + w_mm + _right_mm),
        bottom=_bottom_mm / (_bottom_mm + h_mm + _top_mm),
        top=(_bottom_mm + h_mm) / (_bottom_mm + h_mm + _top_mm),
    )
    return fig, ax


def _style_axes(ax):
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    for side in ax.spines.values():
        side.set_linewidth(spine_linewidth)


def find_anchor_vpi(data: PowerHeterodyneSweepData, anchor: dict, x: np.ndarray):
    """
    Within anchor['window'] (in the units of x), find the step at which
    anchor['harmonic']'s peak power is extremal -- a minimum (a Bessel null),
    a maximum (a Bessel local max), or an inflection (steepest |dP/dV|, a
    curvature zero), per anchor.get('kind', 'null') -- and estimate V_pi
    from the known beta_zero there via
    beta(V) = (pi/V_pi)*V  =>  V_pi = pi * V_ext / beta_zero.

    Returns (ext_idx, v_ext, vpi). v_ext is the true RF voltage at the
    extremum (independent of X_AXIS), so vpi is always in volts.
    """
    idx_h = _harmonic_idx(data, anchor['harmonic'])
    peaks_dbm = data.peak_powers_dbm()[:, idx_h]

    lo, hi = anchor['window']
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise ValueError(f"No data points in window {anchor['window']}")

    window_idx = np.flatnonzero(mask)
    kind = anchor.get('kind', 'null')
    if kind == 'null':
        ext_idx = window_idx[np.argmin(peaks_dbm[window_idx])]
    elif kind == 'max':
        ext_idx = window_idx[np.argmax(peaks_dbm[window_idx])]
    elif kind == 'inflection':
        # Inflection = extremum of dP/dV (a curvature zero). Must be
        # computed in linear power vs. true RF voltage -- dB compresses the
        # curve's shape, and differentiating against a nonlinear X_AXIS
        # (e.g. 'dbm') would shift the located point -- so both are used
        # explicitly here regardless of X_AXIS/NORMALIZE.
        v_rms_all = data.rf_voltage_rms()
        lin = 10.0 ** (peaks_dbm / 10.0)
        dlin_dv = np.gradient(lin, v_rms_all)
        ext_idx = window_idx[np.argmax(np.abs(dlin_dv[window_idx]))]
    else:
        raise ValueError(f"Unknown anchor kind {kind!r}; use 'null', 'max', or 'inflection'.")

    v_ext = data.rf_voltage_rms()[ext_idx]
    vpi = np.pi * v_ext / anchor['beta_zero']
    return int(ext_idx), float(v_ext), float(vpi)


def _resolve_vpi_source(anchor_results: list, source):
    """
    Resolve a V_pi source selector to a single V_pi value.
    None -> mean V_pi across all successfully found anchors.
    int/str -> that specific ANCHOR_POINTS index or anchor 'label'.
    """
    found = [r for r in anchor_results if r is not None]
    if not found:
        raise RuntimeError("No anchor V_pi available.")

    if source is None:
        return float(np.mean([vpi for _, vpi in found]))

    for i, r in enumerate(anchor_results):
        if r is None:
            continue
        label, vpi = r
        if i == source or label == source:
            return vpi
    raise ValueError(f"Source {source!r} did not match any successfully found anchor.")


def _resolve_vpi_average(anchor_results: list, selectors) -> float:
    """
    Resolve a single V_pi to use for a theory curve: the mean of the anchors
    selected by `selectors` (a list of ANCHOR_POINTS indices and/or anchor
    'label' strings), or the mean of ALL successfully found anchors if
    selectors is None.
    """
    found = [(i, label, vpi) for i, r in enumerate(anchor_results) if r is not None
             for label, vpi in [r]]
    if not found:
        raise RuntimeError("No anchor V_pi available.")

    if selectors is None:
        vpis = [vpi for _, _, vpi in found]
    else:
        vpis = [vpi for i, label, vpi in found if i in selectors or label in selectors]
        if not vpis:
            raise ValueError(f"THEORY_ANCHORS {selectors!r} matched no successfully found anchor.")

    return float(np.mean(vpis))


def compute_correction_factors(data: PowerHeterodyneSweepData, vpi_ref: float, v_ref: float):
    """
    At the RF voltage nearest v_ref, compare each harmonic's measured
    conversion efficiency (fraction of carrier power) to its theoretical
    J_n(beta)^2 value, with beta = pi*v_ref/vpi_ref.

    Returns (factors, idx_ref, v_actual), where
    factors = {harmonic: theory_fraction / measured_fraction} -- multiplying
    a harmonic's measured fraction by its factor matches theory exactly at
    that reference point. v_actual is the actual swept voltage nearest v_ref.
    """
    if data.cal_spectra is None:
        raise RuntimeError(
            "CORRECTION_VOLTAGE requires per-step calibration (cal_spectra)."
        )
    v_rms = data.rf_voltage_rms()
    idx_ref = int(np.argmin(np.abs(v_rms - v_ref)))
    v_actual = float(v_rms[idx_ref])
    beta_ref = np.pi * v_actual / vpi_ref

    measured_dbc = data.normalized_peak_powers_dbm()[idx_ref]   # (N,)
    factors = {}
    for j, n in enumerate(data.harmonics):
        n = int(n)
        theory_frac = bessel_jv(n, beta_ref) ** 2
        measured_frac = 10.0 ** (measured_dbc[j] / 10.0)
        factors[n] = theory_frac / max(measured_frac, 1e-30)
    return factors, idx_ref, v_actual


def compute_correction_factors_multi(data: PowerHeterodyneSweepData, vpi_ref: float,
                                      v_min: float, v_max: float, n_samples: int,
                                      min_ce: float):
    """
    Sample n_samples voltages linearly over [v_min, v_max] (true V_rms), and
    at each one compare every harmonic's measured conversion efficiency to
    its theoretical J_n(beta)^2 value (beta = pi*V/vpi_ref). A harmonic's
    sample is discarded if its measured fraction there is below min_ce.

    Returns (factors_mean, factors_std, n_used): dicts keyed by harmonic,
    missing an entry wherever no sample survived for that harmonic.
    """
    if data.cal_spectra is None:
        raise RuntimeError(
            "Multi-voltage correction requires per-step calibration (cal_spectra)."
        )
    v_rms = data.rf_voltage_rms()
    dbc_all = data.normalized_peak_powers_dbm()   # (M, N)
    v_targets = np.linspace(v_min, v_max, n_samples)

    samples = {int(n): [] for n in data.harmonics}
    for v_target in v_targets:
        idx = int(np.argmin(np.abs(v_rms - v_target)))
        beta = np.pi * v_rms[idx] / vpi_ref
        for j, n in enumerate(data.harmonics):
            n = int(n)
            measured_frac = 10.0 ** (dbc_all[idx, j] / 10.0)
            if measured_frac < min_ce:
                continue
            theory_frac = bessel_jv(n, beta) ** 2
            samples[n].append(theory_frac / max(measured_frac, 1e-30))

    factors_mean, factors_std, n_used = {}, {}, {}
    for n, vals in samples.items():
        if vals:
            factors_mean[n] = float(np.mean(vals))
            factors_std[n] = float(np.std(vals))
            n_used[n] = len(vals)
    return factors_mean, factors_std, n_used


def plot_corrected_cloud(ax, data: PowerHeterodyneSweepData, x: np.ndarray, normalize,
                          harmonics, factors_mean: dict, factors_std: dict,
                          harmonic_colors: dict):
    """
    For each harmonic with a valid factors_mean/factors_std entry, overlay
    the corrected curve (measured * mean factor) as a solid line, with
    shaded +-1-sigma (darker) and +-2-sigma (lighter) clouds around it.
    """
    for n in harmonics:
        n = int(n)
        if n not in factors_mean:
            continue
        j = _harmonic_idx(data, n)
        mean_curve = _peak_values(data, normalize, {n: factors_mean[n]})[:, j]
        low1 = _peak_values(
            data, normalize, {n: max(factors_mean[n] - factors_std[n], 1e-30)})[:, j]
        high1 = _peak_values(
            data, normalize, {n: factors_mean[n] + factors_std[n]})[:, j]
        low2 = _peak_values(
            data, normalize, {n: max(factors_mean[n] - 2 * factors_std[n], 1e-30)})[:, j]
        high2 = _peak_values(
            data, normalize, {n: factors_mean[n] + 2 * factors_std[n]})[:, j]

        color = harmonic_colors.get(n, '#000000')
        line_color = CORRECTED_LINE_COLOR if CORRECTED_LINE_COLOR is not None else color
        cloud_color = CORRECTED_CLOUD_COLOR if CORRECTED_CLOUD_COLOR is not None else color

        ax.fill_between(x, low2, high2, color=cloud_color,
                         alpha=CORRECTED_CLOUD_ALPHA_2SIGMA, zorder=CORRECTED_CLOUD_ZORDER_2SIGMA,
                         linewidth=0)
        ax.fill_between(x, low1, high1, color=cloud_color,
                         alpha=CORRECTED_CLOUD_ALPHA, zorder=CORRECTED_CLOUD_ZORDER,
                         linewidth=0)
        ax.plot(x, mean_curve, color=line_color, linestyle=CORRECTED_LINESTYLE,
                linewidth=CORRECTED_LINEWIDTH, alpha=CORRECTED_ALPHA,
                zorder=CORRECTED_ZORDER, label=f'Harmonic {n} corrected (mean±std)')


def plot_sideband_powers(data: PowerHeterodyneSweepData, x: np.ndarray, xlabel: str,
                          normalize, harmonic_colors: dict, correction: dict | None = None):
    """Sideband power curves, fully styled from the module-level SIDEBAND_*
    settings, colored per-harmonic via harmonic_colors. Returns (fig, ax,
    lines), where lines is {harmonic: Line2D} so callers (e.g. the
    interactive correction slider) can update the curves in place."""
    peaks = _peak_values(data, normalize, correction)
    if normalize == 'percent':
        ylabel = 'Sideband power [% of carrier]'
    elif normalize:
        ylabel = 'Sideband power [dBc]'
    else:
        ylabel = 'Peak power [dBm]'

    show_set = set(HARMONICS_TO_SHOW) if HARMONICS_TO_SHOW is not None else None

    fig, ax = _make_figure(axes_width_mm, axes_height_mm)
    lines = {}
    for j, n in enumerate(data.harmonics):
        n = int(n)
        if show_set is not None and n not in show_set:
            continue
        color = harmonic_colors[n]
        face = SIDEBAND_MARKER_FACECOLOR if SIDEBAND_MARKER_FACECOLOR is not None else color
        edge = SIDEBAND_MARKER_EDGECOLOR if SIDEBAND_MARKER_EDGECOLOR is not None else color
        label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n}'
        line, = ax.plot(
            x, peaks[:, j],
            color=color, linestyle=SIDEBAND_LINESTYLE, linewidth=SIDEBAND_LINEWIDTH,
            marker=SIDEBAND_MARKER, markersize=SIDEBAND_MARKERSIZE,
            markerfacecolor=face, markeredgecolor=edge,
            markeredgewidth=SIDEBAND_MARKER_EDGEWIDTH,
            alpha=SIDEBAND_ALPHA, zorder=SIDEBAND_ZORDER, label=label,
        )
        lines[n] = line

    if POWER_YMIN is not None or POWER_YMAX is not None:
        ax.set_ylim([POWER_YMIN, POWER_YMAX])
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    _style_axes(ax)
    return fig, ax, lines


def add_correction_controls(fig, ax, lines: dict, data: PowerHeterodyneSweepData,
                             normalize, vpi_ref: float, v_range: tuple[float, float],
                             init_voltage: float) -> tuple[Slider, CheckButtons]:
    """
    Add a Slider + an on/off CheckButtons under `ax` for the correction
    reference voltage. `lines` is {harmonic: Line2D} from
    plot_sideband_powers(); on every slider move or checkbox click, each
    line's y-data is recomputed in place -- correction factors at the current
    slider voltage when the checkbox is checked, or the raw uncorrected data
    when it's not. Shrinks `ax` slightly to make room, within the figure's
    existing size.

    Returns (slider, checkbox) -- the caller must keep a reference to both
    (e.g. local variables held until plt.show()), or they will stop
    responding once garbage-collected.
    """
    pos = ax.get_position()
    fig_h_in = fig.get_figheight()
    fig_w_in = fig.get_figwidth()
    slider_h_frac = (CORRECTION_SLIDER_HEIGHT_MM / 25.4) / fig_h_in
    pad_frac = (CORRECTION_SLIDER_PAD_MM / 25.4) / fig_h_in

    new_bottom = pos.y0 + slider_h_frac + pad_frac
    ax.set_position([pos.x0, new_bottom, pos.width, pos.y0 + pos.height - new_bottom])

    checkbox_w_frac = (CORRECTION_CHECKBOX_WIDTH_MM / 25.4) / fig_w_in
    gap_frac = (CORRECTION_CONTROL_GAP_MM / 25.4) / fig_w_in
    slider_w_frac = pos.width - checkbox_w_frac - gap_frac

    slider_ax = fig.add_axes([pos.x0, pos.y0, slider_w_frac, slider_h_frac])
    slider = Slider(slider_ax, CORRECTION_SLIDER_LABEL, v_range[0], v_range[1],
                     valinit=init_voltage)

    checkbox_ax = fig.add_axes([pos.x0 + slider_w_frac + gap_frac, pos.y0,
                                 checkbox_w_frac, slider_h_frac])
    checkbox = CheckButtons(checkbox_ax, [CORRECTION_CHECKBOX_LABEL], [True])

    state = {'enabled': True, 'voltage': init_voltage}

    def _redraw():
        correction = None
        if state['enabled']:
            correction, _, _ = compute_correction_factors(data, vpi_ref, state['voltage'])
        peaks = _peak_values(data, normalize, correction)
        for n, line in lines.items():
            j = _harmonic_idx(data, n)
            line.set_ydata(peaks[:, j])
        fig.canvas.draw_idle()

    def _on_slider_change(v_target):
        state['voltage'] = v_target
        _redraw()

    def _on_checkbox_click(_label):
        state['enabled'] = not state['enabled']
        _redraw()

    slider.on_changed(_on_slider_change)
    checkbox.on_clicked(_on_checkbox_click)
    return slider, checkbox


def plot_ce_derivative(data: PowerHeterodyneSweepData, x: np.ndarray, xlabel: str,
                        normalize, harmonic_colors: dict, correction: dict | None = None):
    """
    Plot d(sideband power)/dx vs drive level, per harmonic, in the same
    NORMALIZE/X_AXIS units as plot_sideband_powers(). Returns (fig, ax,
    derivatives), where derivatives is {harmonic: dP/dx array} so anchor
    markers can be placed on this curve without recomputing it.
    """
    peaks = _peak_values(data, normalize, correction)
    if normalize == 'percent':
        ylabel = r'$d(\mathrm{CE})/dx$  [%/unit]'
    elif normalize:
        ylabel = r'$d(\mathrm{dBc})/dx$  [dB/unit]'
    else:
        ylabel = r'$d(\mathrm{dBm})/dx$  [dB/unit]'

    show_set = set(HARMONICS_TO_SHOW) if HARMONICS_TO_SHOW is not None else None

    fig, ax = _make_figure(axes_width_mm, axes_height_mm)
    derivatives = {}
    for j, n in enumerate(data.harmonics):
        n = int(n)
        if show_set is not None and n not in show_set:
            continue
        dydx = np.gradient(peaks[:, j], x)
        derivatives[n] = dydx
        color = harmonic_colors[n]
        label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n}'
        ax.plot(
            x, dydx,
            color=color, linestyle=DERIVATIVE_LINESTYLE, linewidth=DERIVATIVE_LINEWIDTH,
            marker=DERIVATIVE_MARKER, markersize=DERIVATIVE_MARKERSIZE,
            alpha=DERIVATIVE_ALPHA, zorder=DERIVATIVE_ZORDER, label=label,
        )

    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':', zorder=0)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    _style_axes(ax)
    return fig, ax, derivatives


def plot_theory_curves(ax, data: PowerHeterodyneSweepData, x: np.ndarray, normalize,
                        harmonics, vpi_entries: list, harmonic_colors: dict):
    """
    Overlay theoretical pure-PM Bessel sideband curves J_n(beta(V))^2 at the
    measured voltages, for each (label, vpi) in vpi_entries.
    """
    v_rms = data.rf_voltage_rms()
    cal_dbm = data.cal_peak_power_dbm() if data.cal_spectra is not None else None

    for label, vpi in vpi_entries:
        beta = np.pi * v_rms / vpi
        for n in harmonics:
            n = int(n)
            p_frac = bessel_jv(n, beta) ** 2

            if normalize == 'percent':
                y_theory = 100.0 * p_frac
            elif normalize:
                y_theory = 10.0 * np.log10(p_frac + 1e-30)
            else:
                if cal_dbm is None:
                    raise RuntimeError(
                        "Raw-dBm theory curves require per-step calibration "
                        "(cal_spectra); use NORMALIZE='percent'/True, or "
                        "record with per_step_calibration=True."
                    )
                y_theory = cal_dbm + 10.0 * np.log10(p_frac + 1e-30)

            color = THEORY_COLOR if THEORY_COLOR is not None else harmonic_colors.get(n, '#000000')
            ax.plot(
                x, y_theory,
                color=color, linestyle=THEORY_LINESTYLE, linewidth=THEORY_LINEWIDTH,
                marker=THEORY_MARKER, markersize=THEORY_MARKERSIZE,
                alpha=THEORY_ALPHA, zorder=THEORY_ZORDER,
                label=f'{label}: theory n={n}',
            )


def main():
    data = load_averaged(local_path(DATA_FILE))

    normalize = NORMALIZE
    if normalize and data.cal_spectra is None:
        print("Warning: NORMALIZE requested but file has no per-step calibration; "
              "falling back to raw dBm.")
        normalize = False

    x, xlabel, x_unit = _x_values(data, X_AXIS)
    harmonic_colors = _build_harmonic_colors(data.harmonics)

    print(f"Loaded: {DATA_FILE}")
    print(f"  CW frequency     : {data.cw_freq / 1e9:.4f} GHz")
    print(f"  Drive powers     : {data.cw_powers[0]:+.1f} to "
          f"{data.cw_powers[-1]:+.1f} dBm ({len(data.cw_powers)} steps)")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    print(f"  Cal spectra      : {'yes' if data.cal_spectra is not None else 'no'}")
    print(f"  Repeats averaged : {data.n_repeats}")

    # ── anchor V_pi extraction (done first: correction factors need it) ──────
    print("\nV_pi anchor points:")
    anchor_results = []   # aligned with ANCHOR_POINTS; None where extraction failed
    for anchor in ANCHOR_POINTS:
        label = anchor.get('label', f"harmonic {anchor['harmonic']}")
        try:
            dip_idx, v_dip, vpi = find_anchor_vpi(data, anchor, x)
        except ValueError as e:
            print(f"  {label}: {e}")
            anchor_results.append(None)
            continue

        anchor_results.append((label, vpi))
        kind_word = anchor.get('kind', 'null')
        print(f"  {label}: {kind_word} at {x[dip_idx]:.4f} {x_unit} "
              f"({v_dip:.4f} V_rms), beta={anchor['beta_zero']:.4f} rad "
              f"->  V_pi = {vpi:.4f} V")

    vpis = [r[1] for r in anchor_results if r is not None]
    if vpis:
        print(f"\n  Mean V_pi across {len(vpis)} anchor(s): "
              f"{np.mean(vpis):.4f} V  (std {np.std(vpis):.4f} V)")

    # ── per-harmonic efficiency correction ────────────────────────────────────
    correction = None
    vpi_ref = None
    if APPLY_CORRECTION:
        if CORRECTION_VOLTAGE is None:
            print("\nWarning: APPLY_CORRECTION is on but CORRECTION_VOLTAGE is "
                  "None; skipping correction.")
        else:
            vpi_ref = _resolve_vpi_source(anchor_results, CORRECTION_VPI_SOURCE)
            correction, idx_ref, v_actual = compute_correction_factors(
                data, vpi_ref, CORRECTION_VOLTAGE)
            print(f"\nEfficiency correction at V={v_actual:.4f} V_rms "
                  f"(nearest to requested {CORRECTION_VOLTAGE:.4f} V_rms, "
                  f"V_pi={vpi_ref:.4f} V):")
            for n, factor in correction.items():
                print(f"  Harmonic {n}: factor = {factor:.4f}  "
                      f"({10.0 * np.log10(max(factor, 1e-30)):+.2f} dB)")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig_pow, ax_pow, lines_pow = plot_sideband_powers(data, x, xlabel, normalize, harmonic_colors,
                                                       correction=correction)

    correction_controls = None
    if APPLY_CORRECTION and INTERACTIVE_CORRECTION_SLIDER and vpi_ref is not None:
        v_range = CORRECTION_SLIDER_RANGE
        if v_range is None:
            v_rms_all = data.rf_voltage_rms()
            v_range = (float(v_rms_all.min()), float(v_rms_all.max()))
        correction_controls = add_correction_controls(
            fig_pow, ax_pow, lines_pow, data, normalize, vpi_ref, v_range, v_actual)

    fig_cal, ax_cal = None, None
    if SHOW_CALIBRATION and data.cal_spectra is not None:
        fig_cal, ax_cal = data.plot_calibration(
            x_axis=X_AXIS,
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            ymin=CAL_YMIN,
            ymax=CAL_YMAX,
        )

    peak_vals = _peak_values(data, normalize, correction)

    if SHOW_ANCHOR_MARKERS:
        for anchor, result in zip(ANCHOR_POINTS, anchor_results):
            if result is None:
                continue
            label, vpi = result
            dip_idx, _, _ = find_anchor_vpi(data, anchor, x)
            idx_h = _harmonic_idx(data, anchor['harmonic'])
            y_dip = peak_vals[dip_idx, idx_h]
            ax_pow.scatter([x[dip_idx]], [y_dip],
                           color=ANCHOR_MARKER_COLOR, marker=ANCHOR_MARKER_STYLE,
                           s=ANCHOR_MARKER_SIZE, linewidths=ANCHOR_MARKER_WIDTH,
                           zorder=ANCHOR_ZORDER)
            if SHOW_ANCHOR_LABELS:
                ax_pow.annotate(
                    f'{label}\n$V_\\pi$={vpi:.3f} V',
                    xy=(x[dip_idx], y_dip), xytext=ANCHOR_LABEL_OFFSET_PT,
                    textcoords='offset points', fontsize=ANCHOR_LABEL_FONTSIZE,
                    color=ANCHOR_LABEL_COLOR, ha='center', va='bottom',
                )

    fig_deriv, ax_deriv = None, None
    if SHOW_DERIVATIVE_PLOT:
        fig_deriv, ax_deriv, derivatives = plot_ce_derivative(
            data, x, xlabel, normalize, harmonic_colors, correction=correction)

        if SHOW_ANCHOR_MARKERS:
            for anchor, result in zip(ANCHOR_POINTS, anchor_results):
                if result is None:
                    continue
                label, vpi = result
                dip_idx, _, _ = find_anchor_vpi(data, anchor, x)
                harmonic = anchor['harmonic']
                if harmonic not in derivatives:
                    continue
                y_dip = derivatives[harmonic][dip_idx]
                ax_deriv.scatter([x[dip_idx]], [y_dip],
                                 color=ANCHOR_MARKER_COLOR, marker=ANCHOR_MARKER_STYLE,
                                 s=ANCHOR_MARKER_SIZE, linewidths=ANCHOR_MARKER_WIDTH,
                                 zorder=ANCHOR_ZORDER)
                if SHOW_ANCHOR_LABELS:
                    ax_deriv.annotate(
                        f'{label}\n$V_\\pi$={vpi:.3f} V',
                        xy=(x[dip_idx], y_dip), xytext=ANCHOR_LABEL_OFFSET_PT,
                        textcoords='offset points', fontsize=ANCHOR_LABEL_FONTSIZE,
                        color=ANCHOR_LABEL_COLOR, ha='center', va='bottom',
                    )

        if SHOW_LEGEND:
            ax_deriv.legend(fontsize=tick_label_fontsize, frameon=False)

    if SHOW_CORRECTED_CLOUD:
        vpi_ref_cloud = _resolve_vpi_source(anchor_results, CORRECTION_VPI_SOURCE)
        v_min, v_max = CORRECTION_VOLTAGE_RANGE
        factors_mean, factors_std, n_used = compute_correction_factors_multi(
            data, vpi_ref_cloud, v_min, v_max, CORRECTION_N_SAMPLES, CORRECTION_MIN_CE)
        print(f"\nMulti-voltage efficiency correction ({CORRECTION_N_SAMPLES} samples "
              f"over {v_min:.4f}-{v_max:.4f} V_rms, V_pi={vpi_ref_cloud:.4f} V):")
        for n in data.harmonics:
            n = int(n)
            if n in factors_mean:
                print(f"  Harmonic {n}: factor = {factors_mean[n]:.4f} ± "
                      f"{factors_std[n]:.4f}  (n={n_used[n]} samples used)")
            else:
                print(f"  Harmonic {n}: no samples passed "
                      f"CORRECTION_MIN_CE={CORRECTION_MIN_CE:.3f}")

        corrected_harmonics = (
            CORRECTED_HARMONICS if CORRECTED_HARMONICS is not None
            else (HARMONICS_TO_SHOW if HARMONICS_TO_SHOW is not None else list(data.harmonics))
        )
        plot_corrected_cloud(ax_pow, data, x, normalize, corrected_harmonics,
                              factors_mean, factors_std, harmonic_colors)

    if SHOW_THEORY:
        try:
            vpi_theory = _resolve_vpi_average(anchor_results, THEORY_ANCHORS)
        except (RuntimeError, ValueError) as e:
            print(f"\nSHOW_THEORY is on but {e}")
        else:
            n_used_theory = sum(1 for r in anchor_results if r is not None)
            theory_label = (f'Theory (mean of {n_used_theory} anchor(s))' if THEORY_ANCHORS is None
                             else f'Theory (mean of {THEORY_ANCHORS})')
            print(f"\nTheory curve V_pi = {vpi_theory:.4f} V ({theory_label})")

            theory_harmonics = (
                THEORY_HARMONICS if THEORY_HARMONICS is not None
                else (HARMONICS_TO_SHOW if HARMONICS_TO_SHOW is not None else list(data.harmonics))
            )
            plot_theory_curves(ax_pow, data, x, normalize, theory_harmonics,
                                [(theory_label, vpi_theory)], harmonic_colors)

    if SHOW_LEGEND:
        ax_pow.legend(fontsize=tick_label_fontsize, frameon=False)

    if FOR_PUBLICATION:
        import os as _os

        def _apply_pub_style(fig, ax, svg_name):
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelbottom=False, labelleft=False)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

            path = _os.path.join(SAVE_FOLDER, svg_name)
            fig.savefig(path, format='svg', bbox_inches='tight')
            print(f"Saved: {path}")

        _apply_pub_style(fig_pow, ax_pow, SIDEBAND_SVG_NAME)
        if fig_cal is not None:
            _apply_pub_style(fig_cal, ax_cal, CALIBRATION_SVG_NAME)
        if fig_deriv is not None:
            _apply_pub_style(fig_deriv, ax_deriv, DERIVATIVE_SVG_NAME)

    plt.show()


if __name__ == '__main__':
    main()
