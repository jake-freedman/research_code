"""
Analyse a power-sweep heterodyne harmonic recording produced by
vna_power_heterodyne_sweep() in vna_power_harmonic_esa_script.py
or bnc_power_heterodyne_sweep() in bnc_power_harmonic_esa_script.py.

Same as power_harmonic_sweep_analysis.py, but the modulation-depth plot adds:
  - ROOT_N: plot beta**(1/ROOT_N) instead of beta itself (e.g. 3 for a cube
    root, useful for checking a cubic/THG-like beta ~ V^3 dependence).
  - FIT_DEGREE: fit an arbitrary-degree polynomial (not just a line) to the
    (possibly root-transformed) modulation depth, and report where that fit
    crosses beta = pi.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'experiment_control'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import jn as bessel_jn
from power_harmonic_sweep_data import PowerHeterodyneSweepData
from path_utils import local_path
from graphics import (
    BLUE2, RED2, GREEN2, ORANGE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d21_wg16a_p5\very_nice_thg.npz"

# X-axis for all plots: 'voltage' (V_rms) or 'dbm' (drive power in dBm).
X_AXIS = 'voltage'

# Normalize sideband powers by the per-step calibration carrier level?
#   False      → y-axis in dBm  (raw ESA power)
#   True       → y-axis in dBc  (relative to optical carrier, log scale)
#   'percent'  → y-axis in %    (fraction of carrier power, linear scale)
# Requires the file to have been recorded with per_step_calibration=True;
# falls back to False automatically if cal_spectra is absent.
NORMALIZE = False

# Show the calibration (carrier-beat) power vs drive level?
SHOW_CALIBRATION = True

# Harmonic pair for β extraction: J_num(β) / J_den(β).
HARMONIC_NUMERATOR = 1
HARMONIC_DENOMINATOR = 0

# Initial guess for β (rad).
BETA_GUESS = 2.0

# Seed each step's fsolve() with the previous step's computed beta instead of
# always starting from BETA_GUESS? Only the first step (lowest drive level)
# uses BETA_GUESS; robust when beta increases monotonically with drive level.
CARRY_FORWARD_GUESS = True

# Manually override which harmonic pair (num, den) is used for beta
# extraction over specific regions of the sweep, keyed by an (xmin, xmax)
# tuple in the units of X_AXIS. Steps whose x falls in none of these regions
# fall back to (HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR). Empty/{} = use
# that pair everywhere.
# HARMONIC_PAIR_BY_REGION = {
#     (0.0, 1.0): (1, 0),
#     (1.0, 5): (3, 2),
# }

HARMONIC_PAIR_BY_REGION = {}
# Plot beta**(1/ROOT_N) instead of beta itself.
#   1 -> beta          2 -> sqrt(beta)          3 -> cube root of beta, ...
ROOT_N = 1

# Plot the modulation-depth axes log-log (log(beta) vs log(drive level))
# instead of linear? A power-law beta ~ x^n shows up as a straight line with
# slope n. BETA_YMIN/BETA_YMAX's auto-fallback (0, 1.1*max) is skipped here
# since log scales can't include 0 -- set them explicitly if needed.
LOG_LOG_PLOT = True

# Overlay a polynomial fit and report where it crosses beta = pi?
FIT = True
# int  -> full polynomial of that degree (orders 0, 1, ..., FIT_DEGREE), as
#         in the original script (1 = line).
# list -> sparse fit using only these powers of x, e.g. [1, 3] fits
#         y = a1*x + a3*x^3 with no constant, quadratic, or higher terms.
FIT_DEGREE = [3]

# Restrict the fit to this (xmin, xmax) range, in the units of X_AXIS
# (e.g. avoid a noisy or saturated tail). The fit curve is still drawn over
# the full sweep range. None = fit using all data points.
FIT_RANGE = [2, 3]
# FIT_RANGE = [0.2, 1.8]

# Show each individual repeat's beta points (semi-transparent) behind the
# averaged points? Only has an effect when the file has n_repeats > 1.
SHOW_REPEAT_POINTS = True

# ── modulation-depth plot style ────────────────────────────────────────────────
DATA_COLOR       = BLUE2
DATA_LINEWIDTH   = 1.5
DATA_LINESTYLE   = 'none'
DATA_MARKER      = 'o'
DATA_MARKERSIZE  = 6
DATA_ZORDER      = 2

REPEAT_POINT_COLOR  = None   # None = match DATA_COLOR
REPEAT_POINT_ALPHA  = 0.25
REPEAT_POINT_SIZE   = 10      # scatter marker size (points^2)
REPEAT_POINT_ZORDER = 1

FIT_COLOR     = '#bbbbbb'
FIT_LINEWIDTH = 2.5
FIT_LINESTYLE = '--'
FIT_ZORDER    = 3

TARGET_LINE_COLOR  = 'gray'
TARGET_LINE_WIDTH  = 0.8
TARGET_LINE_STYLE  = ':'
TARGET_LINE_ZORDER = 0

# Figure size (mm), applied to all three plots (peak powers, calibration,
# and modulation depth).
AXES_WIDTH_MM  = 100
AXES_HEIGHT_MM = 50
# ─────────────────────────────────────────────────────────────────────────────

# Y-axis limits for sideband power plot. None = auto.
if NORMALIZE == 'percent':
    POWER_YMIN = 0
    POWER_YMAX = 100
else:
    POWER_YMIN = None
    POWER_YMAX = None

# Y-axis limits for the modulation-depth plot (in the plotted, possibly
# root-transformed, units). None = auto.
BETA_YMIN = -0.01
BETA_YMAX = None

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40


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

    spectra_all = None
    if spectra.ndim == 4:
        spectra_all = spectra   # (R, M, N, K), raw per-repeat spectra in dBm
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
    data.spectra_all      = spectra_all
    return data


def _repeat_view(data: PowerHeterodyneSweepData, r: int) -> PowerHeterodyneSweepData:
    """A PowerHeterodyneSweepData sharing data's metadata but with spectra
    from repeat `r` only (no averaging), for per-repeat beta extraction."""
    rep = PowerHeterodyneSweepData.__new__(PowerHeterodyneSweepData)
    rep.cw_freq          = data.cw_freq
    rep.cw_powers        = data.cw_powers
    rep.harmonics         = data.harmonics
    rep.heterodyne_shift = data.heterodyne_shift
    rep.offsets_hz        = data.offsets_hz
    rep.spectra           = data.spectra_all[r]
    rep.window_hz         = data.window_hz
    rep.cal_spectra       = None
    rep.filepath          = data.filepath
    rep.n_repeats         = 1
    return rep


def _x_values(data: PowerHeterodyneSweepData, x_axis: str):
    if x_axis == 'dbm':
        return data.cw_powers, 'Drive power [dBm]', 'dBm'
    return data.rf_voltage_rms(), r'RF voltage [V$_\mathrm{rms}$]', 'V_rms'


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


def _fit_orders(fit_degree) -> list:
    """Powers of x used in the fit basis, ascending. An int gives the full
    0..fit_degree set; a list/tuple gives just those (sparse) powers."""
    if isinstance(fit_degree, (list, tuple, np.ndarray)):
        return sorted(set(int(p) for p in fit_degree))
    return list(range(int(fit_degree) + 1))


def _fit_poly(x: np.ndarray, y: np.ndarray, orders: list) -> np.ndarray:
    """Least-squares fit y = sum_i coeffs[i] * x**orders[i]."""
    basis = np.vstack([x ** p for p in orders]).T
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return coeffs


def _eval_poly(x: np.ndarray, coeffs: np.ndarray, orders: list) -> np.ndarray:
    return sum(c * x ** p for c, p in zip(coeffs, orders))


def _poly_desc_coeffs(coeffs: np.ndarray, orders: list) -> np.ndarray:
    """Dense coefficient array (index 0 = highest power), for np.roots."""
    dense = np.zeros(max(orders) + 1)
    for c, p in zip(coeffs, orders):
        dense[p] = c
    return dense[::-1]


def _style_axes(ax):
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    for side in ax.spines.values():
        side.set_linewidth(spine_linewidth)


def _beta_guess_for(data: PowerHeterodyneSweepData):
    """BETA_GUESS as a scalar (fsolve carries it forward step-to-step) or,
    with CARRY_FORWARD_GUESS off, as a fixed per-step array so every step
    restarts from BETA_GUESS independently."""
    if CARRY_FORWARD_GUESS:
        return BETA_GUESS
    return np.full(len(data.cw_powers), BETA_GUESS)


def _harmonic_idx(data: PowerHeterodyneSweepData, n: int) -> int:
    idx = np.where(data.harmonics == n)[0]
    if len(idx) == 0:
        raise ValueError(f"Harmonic {n} not in dataset. Available: {list(data.harmonics)}")
    return int(idx[0])


def _pair_for_x(x_val: float):
    """Look up the (num, den) harmonic pair to use at drive level x_val from
    HARMONIC_PAIR_BY_REGION; falls back to (HARMONIC_NUMERATOR,
    HARMONIC_DENOMINATOR) if x_val isn't inside any listed region."""
    for (lo, hi), pair in HARMONIC_PAIR_BY_REGION.items():
        if lo <= x_val <= hi:
            return pair
    return HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR


def modulation_depth_by_region(data: PowerHeterodyneSweepData, x: np.ndarray,
                                beta_guess) -> np.ndarray:
    """
    Like PowerHeterodyneSweepData.modulation_depth(), but the harmonic pair
    used at each step is looked up from HARMONIC_PAIR_BY_REGION based on
    that step's drive level x, instead of one fixed pair for the whole sweep.
    """
    peaks = data.peak_powers_dbm()   # (M, N) dBm
    n_steps = len(data.cw_powers)

    carry_forward = np.ndim(beta_guess) == 0
    guesses = (np.full(n_steps, float(beta_guess)) if carry_forward
               else np.asarray(beta_guess, dtype=float))

    betas = np.empty(n_steps)
    for i in range(n_steps):
        num, den = _pair_for_x(x[i])
        row = peaks[i]
        p_num = 10.0 ** (row[_harmonic_idx(data, num)] / 10.0)
        p_den = 10.0 ** (row[_harmonic_idx(data, den)] / 10.0)
        target = np.sqrt(p_num / p_den)

        def residual(beta, t=target, n=num, d=den):
            return bessel_jn(n, beta) / bessel_jn(d, beta) - t

        betas[i] = float(fsolve(residual, guesses[i])[0])
        if carry_forward and i + 1 < n_steps:
            guesses[i + 1] = betas[i]

    return betas


def _compute_betas(data: PowerHeterodyneSweepData, x: np.ndarray) -> np.ndarray:
    guess = _beta_guess_for(data)
    if HARMONIC_PAIR_BY_REGION:
        return modulation_depth_by_region(data, x, guess)
    return data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR, guess)


def plot_modulation_depth(data: PowerHeterodyneSweepData):
    """Modulation depth (optionally root-transformed) vs drive level, with
    an optional arbitrary-degree polynomial fit."""
    x, xlabel, x_unit = _x_values(data, X_AXIS)
    betas = _compute_betas(data, x)

    y = betas ** (1.0 / ROOT_N) if ROOT_N != 1 else betas
    ylabel = (r'Modulation depth $\beta$ [rad]' if ROOT_N == 1
              else rf'$\beta^{{1/{ROOT_N}}}$ [rad$^{{1/{ROOT_N}}}$]')

    fig, ax = _make_figure(AXES_WIDTH_MM, AXES_HEIGHT_MM)

    if SHOW_REPEAT_POINTS and data.spectra_all is not None:
        repeat_color = REPEAT_POINT_COLOR if REPEAT_POINT_COLOR is not None else DATA_COLOR
        for r in range(data.spectra_all.shape[0]):
            betas_r = _compute_betas(_repeat_view(data, r), x)
            y_r = betas_r ** (1.0 / ROOT_N) if ROOT_N != 1 else betas_r
            ax.scatter(x, y_r, color=repeat_color, alpha=REPEAT_POINT_ALPHA,
                       s=REPEAT_POINT_SIZE, linewidths=0, zorder=REPEAT_POINT_ZORDER)

    ax.plot(x, y, color=DATA_COLOR, linewidth=DATA_LINEWIDTH, linestyle=DATA_LINESTYLE,
            marker=DATA_MARKER, markersize=DATA_MARKERSIZE, zorder=DATA_ZORDER)

    if FIT:
        orders = _fit_orders(FIT_DEGREE)
        fit_desc = (f'Degree-{FIT_DEGREE}' if isinstance(FIT_DEGREE, int)
                    else f'Orders {orders}')

        if FIT_RANGE is not None:
            fit_mask = (x >= FIT_RANGE[0]) & (x <= FIT_RANGE[1])
            if fit_mask.sum() <= len(orders):
                raise ValueError(
                    f"Only {fit_mask.sum()} points in FIT_RANGE for a {fit_desc} fit."
                )
        else:
            fit_mask = np.ones_like(x, dtype=bool)

        coeffs = _fit_poly(x[fit_mask], y[fit_mask], orders)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = _eval_poly(x_fit, coeffs, orders)
        ax.plot(x_fit, y_fit, color=FIT_COLOR, linewidth=FIT_LINEWIDTH, linestyle=FIT_LINESTYLE,
                 label=f'{fit_desc} fit', zorder=FIT_ZORDER)

        terms = []
        for c, p in sorted(zip(coeffs, orders), key=lambda cp: -cp[1]):
            terms.append(f"{c:+.6g}" if p == 0 else
                         f"{c:+.6g}*x" if p == 1 else
                         f"{c:+.6g}*x^{p}")
        print(f"{fit_desc} fit polynomial: y = {' '.join(terms)}  (x in {x_unit})")

        target = np.pi ** (1.0 / ROOT_N)
        dense = _poly_desc_coeffs(coeffs, orders)
        dense[-1] -= target
        roots = np.roots(dense)
        real_roots = roots[np.abs(roots.imag) < 1e-6].real
        in_range = real_roots[(real_roots >= x.min()) & (real_roots <= x.max())]
        x_pi = in_range[0] if len(in_range) else (real_roots[0] if len(real_roots) else None)
        if x_pi is not None:
            print(f"{fit_desc} fit (root {ROOT_N}): crosses β=π at "
                  f"{x_pi:.4f} {x_unit}")
        else:
            print(f"{fit_desc} fit (root {ROOT_N}): no real crossing at β=π.")

        ax.axhline(target, color=TARGET_LINE_COLOR, linewidth=TARGET_LINE_WIDTH,
                   linestyle=TARGET_LINE_STYLE, zorder=TARGET_LINE_ZORDER)
        ax.legend(fontsize=tick_label_fontsize, frameon=False)

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if LOG_LOG_PLOT:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if BETA_YMIN is not None or BETA_YMAX is not None:
            ax.set_ylim([BETA_YMIN, BETA_YMAX])
    else:
        ymin = BETA_YMIN if BETA_YMIN is not None else 0.0
        ymax = BETA_YMAX if BETA_YMAX is not None else 1.1 * y.max()
        ax.set_ylim([ymin, ymax])
    _style_axes(ax)
    return fig, ax


def main():
    data = load_averaged(local_path(DATA_FILE))

    normalize = NORMALIZE
    if normalize and data.cal_spectra is None:
        print("Warning: NORMALIZE requested but file has no per-step calibration; "
              "falling back to raw dBm.")
        normalize = False

    print(f"Loaded: {DATA_FILE}")
    print(f"  CW frequency     : {data.cw_freq / 1e9:.4f} GHz")
    print(f"  Drive powers     : {data.cw_powers[0]:+.1f} to "
          f"{data.cw_powers[-1]:+.1f} dBm ({len(data.cw_powers)} steps)")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    print(f"  Cal spectra      : {'yes' if data.cal_spectra is not None else 'no'}")
    print(f"  Repeats averaged : {data.n_repeats}")
    print(f"  Extracting β from J{HARMONIC_NUMERATOR}(β) / "
          f"J{HARMONIC_DENOMINATOR}(β)")

    data.plot_peak_powers(
        normalize=normalize,
        x_axis=X_AXIS,
        axes_width_mm=AXES_WIDTH_MM,
        axes_height_mm=AXES_HEIGHT_MM,
        ymin=POWER_YMIN,
        ymax=POWER_YMAX,
    )

    if SHOW_CALIBRATION and data.cal_spectra is not None:
        data.plot_calibration(
            x_axis=X_AXIS,
            axes_width_mm=AXES_WIDTH_MM,
            axes_height_mm=AXES_HEIGHT_MM,
            ymin=CAL_YMIN,
            ymax=CAL_YMAX,
        )

    try:
        plot_modulation_depth(data)
    except ValueError as e:
        print(f"Could not extract β: {e}")

    plt.show()


if __name__ == '__main__':
    main()
