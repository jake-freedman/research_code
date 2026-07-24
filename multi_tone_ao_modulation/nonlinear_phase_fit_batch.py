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

SWEEP_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5"

# Name of the file this script writes into SWEEP_FOLDER. Any .npz with this
# exact basename is skipped when scanning the folder for input sweeps (so
# re-running the script never tries to fit its own prior output).
OUTPUT_FILENAME = 'nonlinear_fit_results.npz'

# Harmonic orders to include in the joint fit for every sweep. A file
# missing some of these is fit with whatever subset it has (a warning is
# printed); a file with none of them is skipped entirely.
FIT_HARMONICS = [-2, -1, 0, 1, 2]

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

# ── nonlinear-phase joint fit (see dual_tone_sweep_analysis.py for details) ──
NL_FIT_BETA1_BOUNDS      = (0.0, 3.0)     # rad
NL_FIT_BETA2_BOUNDS      = (0.0, 3.0)     # rad
NL_FIT_BETA2_NL_BOUNDS   = (0.0, 3.0)     # rad
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
axes_width_mm  = 90.0
axes_height_mm = 60.0

# ── per-sweep diagnostic plots ───────────────────────────────────────────
# When True, every sweep gets its own plot with all recorded sidebands' CE
# curves on one axes (no SPLIT_GROUPS-style splitting), optionally with the
# fitted nonlinear-phase curve overlaid on top of whichever sidebands were
# part of the fit. ALL of these are saved to PER_SWEEP_PLOT_FOLDER; only one
# representative plot (PER_SWEEP_PLOT_SHOW_INDEX) is actually displayed on
# screen, alongside the beta2_NL summary plot -- the rest are closed right
# after saving so dozens of sweeps don't open dozens of windows.
SAVE_PER_SWEEP_PLOTS = False
# Overlay the fitted nonlinear-phase curve on top of the raw data. If a
# sweep's fit failed, its raw data is still plotted/saved either way.
PER_SWEEP_PLOT_SHOW_FIT = True
# 0-based index (in the order sweeps are processed) of the one per-sweep
# plot to actually display; all others are only saved to disk. Clamped to
# the number of sweep files found.
PER_SWEEP_PLOT_SHOW_INDEX = 0
# Folder per-sweep plots are saved into. None = a 'per_sweep_fit_plots'
# subfolder created inside SWEEP_FOLDER.
PER_SWEEP_PLOT_FOLDER = None
PER_SWEEP_PLOT_DPI = 150

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


def _nl_predicted_dbc(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonic, k_trunc, scale=1.0):
    """Predicted CE [dBc] for one harmonic at the given ch2 phase(s), used
    for the per-sweep fit-overlay curve."""
    amps = _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, [harmonic], k_trunc)
    return 10.0 * np.log10(np.maximum(scale * np.abs(amps[harmonic]) ** 2, 1e-30))


def _plot_sweep_with_fit(fname: str, data: DualToneSweepData, result, show_fit: bool, k_trunc: int):
    """
    One axes with every recorded sideband's raw CE [dBc] vs. ch2 phase, plus
    (if show_fit and the fit succeeded) the fitted nonlinear-phase curve
    overlaid on whichever sidebands were part of that fit. No SPLIT_GROUPS-
    style splitting -- everything on one plot, per the per-sweep diagnostic.
    """
    fig, ax = _make_fig(axes_width_mm, axes_height_mm)
    dbc_all = data.normalized_peak_powers_dbm()   # (M, N) dBc
    x = data.ch2_phases_deg
    order_idx = np.argsort(x)

    for j, n in enumerate(data.harmonics):
        n = int(n)
        ax.plot(x[order_idx], dbc_all[order_idx, j], marker='o', markersize=3,
                linewidth=0.8, color=_harmonic_color(n), label=f'n={n:+d}')

    if show_fit and result is not None:
        x_fit = np.linspace(x.min(), x.max(), 300)
        phi_fit_rad = np.deg2rad(x_fit)
        phi0_rad = np.deg2rad(result['phi0_deg'])
        phi_nl_rad = np.deg2rad(result['phi_nl_deg'])
        for n in result['fit_harmonics']:
            y_fit = _nl_predicted_dbc(result['beta1'], result['beta2'], result['beta2_nl'],
                                       phi0_rad, phi_nl_rad, phi_fit_rad, n, k_trunc,
                                       scale=result['scales'].get(n, 1.0))
            ax.plot(x_fit, y_fit, color=_harmonic_color(n), linewidth=1.5, zorder=5)

    ax.set_xlabel('Ch2 phase [deg]', fontsize=axis_label_fontsize)
    ax.set_ylabel('Sideband power [dBc]', fontsize=axis_label_fontsize)
    ax.set_title(fname, fontsize=tick_label_fontsize)
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

    # ── summary plot: beta2_NL vs. drive voltage ──────────────────────────
    order = np.argsort(drive_voltages)
    fig, ax = _make_fig(axes_width_mm, axes_height_mm)
    ax.plot(drive_voltages[order], beta2_nl_arr[order],
            marker='o', markersize=5, linewidth=1.5,
            color=GREEN2, markeredgecolor=DARKGRAY2, markeredgewidth=0.5)

    axis_label = (r'Ch1 drive voltage [V$_\mathrm{rms}$]' if DRIVE_VOLTAGE_AXIS == 'ch1_voltage'
                  else r'Ch2 drive voltage [V$_\mathrm{rms}$]')
    ax.set_xlabel(axis_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$\beta_{2,\mathrm{NL}}$ [rad]', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', direction=tick_direction,
                    width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)

    plt.show()


if __name__ == '__main__':
    main()
