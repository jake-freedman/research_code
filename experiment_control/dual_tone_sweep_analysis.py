"""
Analyse a dual-tone sweep recording produced by bnc_dual_tone_esa_script.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from dual_tone_sweep_data import DualToneSweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\paper_data\phase_sweep_dual_tone_sweep_2026-07-28-17-54-14.npz"
# DATA_FILE = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\phase_sweep_dual_tone_sweep_2026-07-17-11-09-05.npz"

# If set to a filepath, skip the nonlinear_phase differential_evolution fit
# entirely (PUB_CURVE_MODE = 'nonlinear_phase' only) and load these saved
# parameters instead -- still drawn/slider-capable exactly as a fresh fit
# would be. Use this to reuse a previously-computed fit without re-running
# the (slow) global search, e.g. after fine-tuning via the interactive
# sliders and wanting to lock those values in. None = fit normally; on a
# successful fresh fit, the result is also auto-saved next to DATA_FILE (see
# console output for the exact path) so it can be pointed to here later.
PRESAVED_FIT_PATH = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\paper_data\phase_sweep_dual_tone_sweep_2026-07-28-17-54-14_nl_fit.json"
# X-axis for all plots. One of:
#   'drive_freq'   — fundamental drive frequency f
#   'ch1_power'    — channel 1 output power (dBm)
#   'ch2_power'    — channel 2 output power (dBm)
#   'ch1_voltage'  — channel 1 RMS voltage at 50 Ω (V)
#   'ch2_voltage'  — channel 2 RMS voltage at 50 Ω (V)
#   'ch1_phase'    — channel 1 phase offset (deg)
#   'ch2_phase'    — channel 2 phase offset (deg)
#   'stability'    — step index (use when all parameters are held constant to
#                    check measurement repeatability over time)
X_AXIS = 'ch2_phase'

# Normalize sideband powers by the per-step calibration carrier level?
#   False      → y-axis in dBm  (raw ESA power)
#   True       → y-axis in dBc  (relative to optical carrier, log scale)
#   'percent'  → y-axis in %    (fraction of carrier power, linear scale)
NORMALIZE = 'percent'

# Show the calibration (carrier-beat) power vs sweep parameter?
SHOW_CALIBRATION = True

# Calibration reference used for normalization:
#   'auto'        — use whatever cal_spectra was recorded (per-step RF-off,
#                   sideband-sum proxy, or tiled preamble — whatever the script saved)
#   'J0_preamble' — override with the initial RF-off preamble measurement
#                   (ref_cal_spectrum), using it as a flat reference across all
#                   steps; useful for sideband-sum data where you want to express
#                   sideband powers as a fraction of the known initial comb power
CALIBRATION_REF = 'auto'

# When n_sweep_repeats > 1: show individual repeat points as semi-transparent
# scatter behind the mean curve?
SHOW_REPEAT_POINTS = False

# When n_sweep_repeats > 1: shade ± 1 std band around the mean curve?
SHOW_ERROR_BAND = False

# Plot a single repeat instead of the mean across all repeats?
# None  → show the mean (default)
# 0, 1, … → show only that repeat index (0-based); scatter/error band are suppressed
REPEAT_INDEX = None

# Y-axis limits for sideband power plot (dBm or dBc). None = auto.
if NORMALIZE == 'percent':
    POWER_YMIN = 0
    POWER_YMAX = 101
else:
      POWER_YMIN = -30
      POWER_YMAX = 5

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40

# Initial β guess for single-tone preamble extraction.
BETA_GUESS_REF = 2.0

# ── sideband filter ──────────────────────────────────────────────────────────
# None = show all recorded harmonics; list = show only those orders.
# (ignored when SPLIT_FIGURES = True)
HARMONICS_TO_SHOW = [-3, -2, -1, 0, 1, 2, 3]   # e.g. [-1, 1] to show only ±1

# ── figure size ───────────────────────────────────────────────────────────────
axes_width_mm  = 40
axes_height_mm = 40

# ── split figures ─────────────────────────────────────────────────────────────
# When True, produce three separate power figures instead of one.
# Each dict: harmonics to include, y-limits, axes size, and SVG filename.
# Optional 'polar_diam_mm' key: this group's polar axes diameter (mm) when
# SHOW_POLAR_CE_PLOT -- omit/None = auto: max(that group's w_mm, h_mm).
SPLIT_FIGURES = True
SPLIT_GROUPS = [
    {'harmonics': [0],      'ymin': 0,   'ymax': 15,  'w_mm': 50,  'h_mm': 15,  'marker_pt': 5,  'svg': 'dual_tone_sweep_powers_0.svg'},
    {'harmonics': [-1, 1],  'ymin': -2,  'ymax': 58,  'w_mm': 50,  'h_mm': 40,  'marker_pt': 5,  'svg': 'dual_tone_sweep_powers_pm1.svg'},
    {'harmonics': [-2, 2],  'ymin': 0,   'ymax': 15,  'w_mm': 50,  'h_mm': 15,  'marker_pt': 5,  'svg': 'dual_tone_sweep_powers_pm2.svg'},
    {'harmonics': [-3, 3],  'ymin': 0,   'ymax': 8,  'w_mm': 50,  'h_mm': 15,  'marker_pt': 5,  'svg': 'dual_tone_sweep_powers_pm3.svg'},
]

# ── polar plotting ─────────────────────────────────────────────────────────────
# When True, every produced power plot (each SPLIT_GROUPS figure, or the
# single combined figure when SPLIT_FIGURES=False) is drawn in polar form
# instead of cartesian: angle = the swept phase (X_AXIS, must be 'ch1_phase'
# or 'ch2_phase'), radius = the sideband power/CE value (per NORMALIZE).
# Axes are forced square. Any active curve fit (PUB_CURVE_MODE) is drawn in
# polar form too. Only the radial (constant-value) gridlines are shown -- the
# angular (constant-phase) rays are suppressed. Falls back to cartesian with
# a warning if X_AXIS isn't a phase axis.
SHOW_POLAR_CE_PLOT = True

# Polar axes diameter (mm) for the single combined figure (SPLIT_FIGURES =
# False). None = auto: max(axes_width_mm, axes_height_mm). Each SPLIT_GROUPS
# dict can set its own 'polar_diam_mm' key the same way (missing/None there
# = auto: max(that group's w_mm, h_mm)) -- see SPLIT_GROUPS above.
POLAR_DIAMETER_MM = 37

# Number of radial gridline circles (constant-value rings) shown inside each
# polar plot, evenly spaced strictly between its center and outer edge
# (matplotlib silently drops a ring placed exactly at the center, so the
# center/edge themselves are never counted as one of these rings).
# None = matplotlib's automatic tick count.
POLAR_N_RINGS = 3

# ── calibration scaling ───────────────────────────────────────────────────────
# Per-harmonic multiplicative correction applied to the conversion efficiency
# after all normalization. Keys are harmonic orders (int); missing orders are
# left unchanged. Set to None to disable.
# Example: {1: 2.0, -1: 2.0} doubles the measured ±1 sideband power.
CALIBRATION_DICT = None # {-2: 1/0.8051, -1: 1/0.8421, 0: 1/0.979, 1: 1/0.9011, 2: 1/0.8627}

# ── publication export ────────────────────────────────────────────────────────
# When True: removes axis/tick labels, legend, and title; saves SVGs.
FOR_PUBLICATION = True
SAVE_FOLDER = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\media"

# Marker style for publication plot. Scatter is drawn first; curves on top.
# PUB_MARKER_EDGE_COLOR / PUB_CURVE_COLOR: 'same' = match combline color.
PUB_MARKER_PT           = 8      # marker diameter in points
PUB_MARKER_ALPHA        = 1      # fill alpha (edge is always fully opaque)
PUB_MARKER_EDGE_COLOR   = 'same'
PUB_MARKER_EDGE_WIDTH   = 0.5      # edge stroke width in points
PUB_MARKER_STRIDE       = 1      # plot every Nth point (1 = all, 2 = every other, etc.)

PUB_CURVE_SHOW          = True      # draw a curve relative to scatter markers
PUB_CURVE_BEHIND        = False      # True = curve behind points; False = curve on top
PUB_CURVE_COLOR         = 'black'   # 'same' = keep combline color
PUB_CURVE_WIDTH         = 2.0       # linewidth in points
# 'raw'             → plot the data line directly
# 'sinusoid'        → fit A·cos(ω·x + φ) + C to each combline (independent
#                      amplitude/phase per frequency component) and plot the fit
# 'phase_harmonics' → fit C + 2*sum_{m=1}^{N} A_m·cos(m·phi + m·phi0) to each
#                      combline: a single shared phase offset phi0, with a free
#                      amplitude A_m per harmonic m=1..N of the swept phase.
#                      Assumes X_AXIS is a phase in degrees (ch1_phase or
#                      ch2_phase); phi = deg2rad(x). N (PUB_PHASE_HARMONIC_N)
#                      is the same for every sideband.
# 'nonlinear_phase' → a physically-motivated JOINT fit (one simultaneous
#                      differential_evolution optimization pooling every
#                      displayed combline's data at once, unlike the two modes
#                      above which fit each combline independently) of
#                        theta(t) = beta1*sin(Wt) + beta2*sin(2Wt+phi+phi0) + beta2_nl*sin(2Wt+phi_NL)
#                      phi1 = 0 fixed reference; phi is the swept ch2 phase
#                      (X_AXIS must be 'ch2_phase'); phi0 is a static phase
#                      offset applied only to the swept beta2 term (e.g. a
#                      calibration offset between commanded and true ch2
#                      phase). beta2_nl/phi_NL model a fixed-phase
#                      nonlinear/parasitic 2f contribution that doesn't track
#                      the swept ch2 phase, interfering with the intentional
#                      beta2 term as phi sweeps. See NL_FIT_* below for search
#                      bounds/initial guess.
PUB_CURVE_MODE          = 'nonlinear_phase'
# Per-combline period count for the sinusoid frequency initial guess (PUB_CURVE_MODE = 'sinusoid').
# Keys are harmonic orders (int); missing orders default to 1.
PUB_SINUSOID_N_PERIODS  = {0:(1,2,3), -2: (1,2,3), 2: (1,2,3), -1: (1,2,3), 1: (1,2,3)}
# Number of phase harmonics N to fit (PUB_CURVE_MODE = 'phase_harmonics'), same for every sideband.
PUB_PHASE_HARMONIC_N    = 3

# ── nonlinear-phase joint fit (PUB_CURVE_MODE = 'nonlinear_phase') ────────────
NL_FIT_BETA1_BOUNDS      = (0.0, 3.0)     # rad
NL_FIT_BETA2_BOUNDS      = (0.0, 3.0)     # rad
NL_FIT_BETA2_NL_BOUNDS   = (0.0, 3.0)     # rad
NL_FIT_PHI0_BOUNDS_DEG   = (0.0, 360.0)   # static offset added to the swept phi (beta2 term only)
NL_FIT_PHI_NL_BOUNDS_DEG = (0.0, 360.0)

# Optional initial guess seeding differential_evolution's population (still a
# global search around it, not a strict local refinement -- useful to steer
# it toward the intended solution if the comb has near-degenerate fits). Any
# left None falls back to the midpoint of its search bounds.
NL_FIT_GUESS_BETA1      = None
NL_FIT_GUESS_BETA2      = None
NL_FIT_GUESS_BETA2_NL   = None
NL_FIT_GUESS_PHI0_DEG   = None
NL_FIT_GUESS_PHI_NL_DEG = None
NL_FIT_SEED = None   # differential_evolution seed; None = nondeterministic

# When True, the joint fit gets one extra free multiplicative scale factor per
# displayed harmonic (applied to that harmonic's predicted linear power,
# bounded by NL_FIT_SCALE_BOUNDS) to absorb small systematic per-sideband
# calibration/measurement errors without biasing beta1/beta2/beta2_nl/phi0/phi_NL.
# When False (default), every sideband's scale factor is fixed at 1.0.
NL_FIT_PER_SIDEBAND_SCALE = True
NL_FIT_SCALE_BOUNDS       = (0.6, 1.2)

# When True, opens a small control window with 5 sliders (beta1, beta2,
# beta2_nl, phi0, phi_NL) that let you manually override the fitted values
# and see every displayed nonlinear_phase overlay curve -- across all
# SPLIT_GROUPS figures at once, if SPLIT_FIGURES -- update live as you drag
# them. Useful to visually check whether the automatic fit landed in the
# right basin, or to explore/fine-tune it by eye. If the automatic joint fit
# failed, the sliders still appear (so you can search by hand), starting from
# NL_FIT_GUESS_*/bounds-midpoint instead of a fitted value.
NL_FIT_INTERACTIVE_SLIDERS = True
NL_SLIDER_WIDTH_MM  = 70.0
NL_SLIDER_HEIGHT_MM = 8.0
NL_SLIDER_GAP_MM    = 4.0     # vertical gap between sliders
NL_SLIDER_MARGIN_LEFT_MM   = 30.0
NL_SLIDER_MARGIN_RIGHT_MM  = 15.0
NL_SLIDER_MARGIN_TOP_MM    = 8.0
NL_SLIDER_MARGIN_BOTTOM_MM = 8.0


def main():
    data = DualToneSweepData.from_file(local_path(DATA_FILE))

    if CALIBRATION_REF == 'J0_preamble':
        if data.ref_cal_spectrum is None:
            print("Warning: no preamble J0 found in file; using recorded cal_spectra.")
        else:
            data.cal_spectra = np.tile(
                data.ref_cal_spectrum[:len(data.offsets_hz)],
                (len(data.drive_freqs), 1),
            )

    if REPEAT_INDEX is not None:
        if data.spectra_all is None:
            print("Warning: REPEAT_INDEX set but file has only one repeat; showing that sweep.")
        else:
            n_reps = data.spectra_all.shape[0]
            if not (0 <= REPEAT_INDEX < n_reps):
                raise ValueError(f"REPEAT_INDEX {REPEAT_INDEX} out of range (file has {n_reps} repeats).")
            data.spectra = data.spectra_all[REPEAT_INDEX]
            if data.cal_spectra_all is not None:
                data.cal_spectra = data.cal_spectra_all[REPEAT_INDEX]
            data.spectra_all = None      # suppress scatter / error band
            data.cal_spectra_all = None
            data.n_repeats = 1

    if CALIBRATION_DICT:
        for j, n in enumerate(data.harmonics):
            factor = CALIBRATION_DICT.get(int(n), 1.0)
            if factor != 1.0:
                db_shift = 10.0 * np.log10(max(factor, 1e-30))
                data.spectra[:, j, :] += db_shift
                if data.spectra_all is not None:
                    data.spectra_all[:, :, j, :] += db_shift

    print(f"Loaded: {DATA_FILE}")
    try:
        beta1, beta2 = data.single_tone_modulation_depths(beta_guess=BETA_GUESS_REF, )
        if beta1 is not None:
            print(f"  β1 (ch1-only, f) : {beta1:.4f} rad  ({np.degrees(beta1):.2f}°)")
        if beta2 is not None:
            print(f"  β2 (ch2-only, 2f): {beta2:.4f} rad  ({np.degrees(beta2):.2f}°)")
    except RuntimeError:
        pass
    print(f"  Steps            : {len(data.drive_freqs)}")
    print(f"  Drive freq range : {data.drive_freqs.min() / 1e9:.4f} – "
          f"{data.drive_freqs.max() / 1e9:.4f} GHz")
    print(f"  Ch1 power range  : {data.ch1_powers_dbm.min():+.1f} – "
          f"{data.ch1_powers_dbm.max():+.1f} dBm")
    print(f"  Ch2 power range  : {data.ch2_powers_dbm.min():+.1f} – "
          f"{data.ch2_powers_dbm.max():+.1f} dBm")
    print(f"  Ch1 phase range  : {data.ch1_phases_deg.min():.1f} – "
          f"{data.ch1_phases_deg.max():.1f} deg")
    print(f"  Ch2 phase range  : {data.ch2_phases_deg.min():.1f} – "
          f"{data.ch2_phases_deg.max():.1f} deg")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    if data.ref_cal_spectrum is not None:
        print(f"  Preamble J0      : {float(data.ref_cal_spectrum.max()):.2f} dBm")

    import os as _os
    import json as _json
    import matplotlib.colors as _mc
    from scipy.optimize import curve_fit as _curve_fit
    from scipy.optimize import differential_evolution as _differential_evolution
    from scipy.special import jv as _bessel_jv
    from matplotlib.widgets import Slider as _Slider

    def _parse_order(lbl):
        parts = lbl.split()
        try:
            return int(parts[-1]) if parts[0] == 'Harmonic' else None
        except ValueError:
            return None

    def _make_sin_model(omegas):
        """Return a sum-of-sinusoids model with fixed frequencies given by omegas."""
        def _model(x, *params):
            # params: A0, phi0, A1, phi1, ..., C  (2*N + 1 values)
            total = np.full_like(x, params[-1], dtype=float)
            for k, om in enumerate(omegas):
                total += params[2 * k] * np.cos(om * x + params[2 * k + 1])
            return total
        return _model

    def _make_phase_harmonic_model(n_harm):
        """
        f(phi_deg) = C + 2 * sum_{m=1}^{n_harm} A_m * cos(m*phi_rad + m*phi0),
        phi_rad = deg2rad(phi_deg). A single phi0 is shared across all m.
        params: A_1, ..., A_n_harm, phi0, C  (n_harm + 2 values).
        """
        def _model(x_deg, *params):
            phi_rad = np.deg2rad(x_deg)
            phi0 = params[n_harm]
            C = params[n_harm + 1]
            total = np.full_like(phi_rad, C, dtype=float)
            for m in range(1, n_harm + 1):
                total = total + 2.0 * params[m - 1] * np.cos(m * phi_rad + m * phi0)
            return total
        return _model

    def _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonics, k_trunc):
        """
        Vectorized Jacobi-Anger amplitudes A_n(phi_i) for
            theta(t) = beta1*sin(Wt) + beta2*sin(2Wt+phi+phi0) + beta2_nl*sin(2Wt+phi_NL),
        phi1 = 0 fixed. phi0 is a static offset added only to the swept beta2
        term. The two 2f terms (beta2 at the swept phi+phi0, beta2_nl at the
        fixed phi_nl) combine via phasor sum into a single effective two-tone
        drive at 2f:
            Z(phi) = beta2*exp(i*(phi+phi0)) + beta2_nl*exp(i*phi_nl)
            beta_eff = |Z|, phi_eff = angle(Z)
        so this reduces to the same dual_tone_amplitudes() (comb_displayer.py)
        formula, just with (beta2, phi2) replaced by (beta_eff(phi), phi_eff(phi))
        -- computed here directly (not via that function) so every phi_rad
        point is evaluated in one vectorized pass instead of a Python loop.
        Returns {harmonic: (M,) complex array}, one amplitude per phi_rad entry.
        """
        phi_rad = np.asarray(phi_rad, dtype=float)
        Z = beta2 * np.exp(1j * (phi_rad + phi0_rad)) + beta2_nl * np.exp(1j * phi_nl_rad)
        beta_eff = np.abs(Z)
        phi_eff = np.angle(Z)

        k = np.arange(-k_trunc, k_trunc + 1)
        Jk_beta_eff = _bessel_jv(k[:, None], beta_eff[None, :])       # (K, M)
        phase = np.exp(1j * k[:, None] * phi_eff[None, :])            # (K, M)

        return {
            n: np.sum(_bessel_jv(n - 2 * k, beta1)[:, None] * Jk_beta_eff * phase, axis=0)
            for n in harmonics
        }

    def _nl_phase_cost(params, phi_rad, harmonics, measured_dbc, k_trunc, free_scale_harmonics):
        beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad = params[:5]
        scales = params[5:]   # one per entry of free_scale_harmonics, in that order
        scale_map = {n: scales[i] for i, n in enumerate(free_scale_harmonics)}
        amps = _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, harmonics, k_trunc)
        err = 0.0
        for n in harmonics:
            # harmonics with a CALIBRATION_DICT entry are already corrected in
            # the raw data (see the CALIBRATION_DICT pre-shift above), so
            # their scale is fixed at 1.0 here rather than re-fit.
            scale = scale_map.get(n, 1.0)
            pred_dbc = 10.0 * np.log10(np.maximum(scale * np.abs(amps[n]) ** 2, 1e-30))
            err += np.sum((measured_dbc[n] - pred_dbc) ** 2)
        return float(err)

    def _nl_curve_y(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, x_deg, order, k_trunc, scale=1.0):
        """One harmonic's predicted y (in whatever units NORMALIZE gives) at
        the given x_deg (ch2 phase, degrees) -- shared by both the initial
        overlay draw and every slider-driven update."""
        phi_rad = np.deg2rad(x_deg)
        amps = _nl_phase_amplitudes(beta1, beta2, beta2_nl, phi0_rad, phi_nl_rad, phi_rad, [order], k_trunc)
        frac = np.abs(amps[order]) ** 2 * scale
        if NORMALIZE == 'percent':
            return frac * 100.0
        if NORMALIZE:
            return 10.0 * np.log10(np.maximum(frac, 1e-30))
        cal_dbm_fit = np.interp(x_deg, data.ch2_phases_deg, data.cal_peak_power_dbm())
        return cal_dbm_fit + 10.0 * np.log10(np.maximum(frac, 1e-30))

    def _add_nl_fit_sliders(nl_fit_lines, initial):
        """
        Opens a small control figure with sliders for beta1, beta2, beta2_nl,
        phi0, phi_NL, plus (if NL_FIT_PER_SIDEBAND_SCALE) one more per
        displayed harmonic's power scale factor, that update every
        (line, order, x_fit_deg, k_trunc) tuple in nl_fit_lines live as
        they're dragged -- across every figure any of those lines belongs to
        (SPLIT_FIGURES may spread them over several). Returns
        (fig_ctrl, sliders) -- the caller must keep a reference to both (e.g.
        local variables held until plt.show()), or they will stop responding
        once garbage-collected.
        """
        specs = [
            ('beta1',    NL_FIT_BETA1_BOUNDS,      initial['beta1']),
            ('beta2',    NL_FIT_BETA2_BOUNDS,      initial['beta2']),
            ('beta2_nl', NL_FIT_BETA2_NL_BOUNDS,   initial['beta2_nl']),
            ('phi0',     NL_FIT_PHI0_BOUNDS_DEG,   np.degrees(initial['phi0_rad']) % 360.0),
            ('phi_NL',   NL_FIT_PHI_NL_BOUNDS_DEG, np.degrees(initial['phi_nl_rad']) % 360.0),
        ]
        scale_harmonics = sorted(initial['scales']) if NL_FIT_PER_SIDEBAND_SCALE else []
        for n in scale_harmonics:
            specs.append((f'scale[{n:+d}]', NL_FIT_SCALE_BOUNDS, initial['scales'][n]))
        n_sliders = len(specs)

        fig_w_mm = NL_SLIDER_MARGIN_LEFT_MM + NL_SLIDER_WIDTH_MM + NL_SLIDER_MARGIN_RIGHT_MM
        fig_h_mm = (NL_SLIDER_MARGIN_BOTTOM_MM + n_sliders * NL_SLIDER_HEIGHT_MM
                    + (n_sliders - 1) * NL_SLIDER_GAP_MM + NL_SLIDER_MARGIN_TOP_MM)
        mm = 1.0 / 25.4
        fig_ctrl = plt.figure(figsize=(fig_w_mm * mm, fig_h_mm * mm))
        fig_ctrl.canvas.manager.set_window_title('Nonlinear-phase fit controls')

        sliders = []
        for i, (label, (lo, hi), val0) in enumerate(specs):
            top_mm = fig_h_mm - NL_SLIDER_MARGIN_TOP_MM - i * (NL_SLIDER_HEIGHT_MM + NL_SLIDER_GAP_MM)
            bottom_mm = top_mm - NL_SLIDER_HEIGHT_MM
            ax_s = fig_ctrl.add_axes([
                NL_SLIDER_MARGIN_LEFT_MM / fig_w_mm, bottom_mm / fig_h_mm,
                NL_SLIDER_WIDTH_MM / fig_w_mm, NL_SLIDER_HEIGHT_MM / fig_h_mm,
            ])
            sliders.append(_Slider(ax_s, label, lo, hi, valinit=float(np.clip(val0, lo, hi))))

        def _on_change(_val):
            beta1_s, beta2_s, beta2_nl_s = sliders[0].val, sliders[1].val, sliders[2].val
            phi0_rad_s = np.deg2rad(sliders[3].val)
            phi_nl_rad_s = np.deg2rad(sliders[4].val)
            scale_s = {n: sliders[5 + i].val for i, n in enumerate(scale_harmonics)}
            figs_to_redraw = set()
            for line, order, x_fit_deg, k_trunc in nl_fit_lines:
                line.set_ydata(_nl_curve_y(beta1_s, beta2_s, beta2_nl_s, phi0_rad_s, phi_nl_rad_s,
                                            x_fit_deg, order, k_trunc, scale=scale_s.get(order, 1.0)))
                figs_to_redraw.add(line.figure)
            for f in figs_to_redraw:
                f.canvas.draw_idle()

        for s in sliders:
            s.on_changed(_on_change)

        return fig_ctrl, sliders

    def _apply_pub_style(fig, ax, svg_name, marker_pt=PUB_MARKER_PT):
        is_polar = ax.name == 'polar'
        if is_polar:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
            ax.set_title('')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelbottom=False, labelleft=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        _lines = [(ln, ln.get_xdata().copy(), ln.get_ydata().copy(), ln.get_color(),
                   _parse_order(ln.get_label()))
                  for ln in ax.lines]
        for _, xd_all, yd_all, fc, _ in _lines:
            rgba = list(_mc.to_rgba(fc))
            rgba[3] = PUB_MARKER_ALPHA
            ec = fc if PUB_MARKER_EDGE_COLOR == 'same' else PUB_MARKER_EDGE_COLOR
            xd = xd_all[::PUB_MARKER_STRIDE]
            yd = yd_all[::PUB_MARKER_STRIDE]
            ax.scatter(xd, yd,
                       s=marker_pt ** 2,
                       facecolors=[rgba] * len(xd),
                       edgecolors=ec,
                       linewidths=PUB_MARKER_EDGE_WIDTH,
                       zorder=2)
        if PUB_CURVE_SHOW:
            cc_fn = lambda fc: fc if PUB_CURVE_COLOR == 'same' else PUB_CURVE_COLOR
            curve_z = 1 if PUB_CURVE_BEHIND else 5
            if PUB_CURVE_MODE == 'raw':
                for ln, _, _, fc, _ in _lines:
                    ln.set_color(cc_fn(fc))
                    ln.set_linewidth(PUB_CURVE_WIDTH)
                    ln.set_zorder(curve_z)
            elif PUB_CURVE_MODE == 'sinusoid':
                for ln, _, _, _, _ in _lines:
                    ln.set_visible(False)
                _fit_unit = '%' if NORMALIZE == 'percent' else ('dBc' if NORMALIZE else 'dBm')
                print("\n  Sinusoid fit parameters (A*cos(omega*x + phi) + C):")
                for _, xd_all, yd_all, fc, order in _lines:
                    x_span = xd_all[-1] - xd_all[0]
                    if x_span == 0 or len(xd_all) < 4:
                        continue
                    periods_raw = (PUB_SINUSOID_N_PERIODS.get(order, 1)
                                   if isinstance(PUB_SINUSOID_N_PERIODS, dict)
                                   else PUB_SINUSOID_N_PERIODS)
                    periods = (periods_raw,) if not isinstance(periods_raw, (list, tuple)) else tuple(periods_raw)
                    omegas = tuple(2 * np.pi * n / x_span for n in periods)
                    model = _make_sin_model(omegas)
                    A0 = (yd_all.max() - yd_all.min()) / 2
                    C0 = (yd_all.max() + yd_all.min()) / 2
                    p0 = [v for _ in omegas for v in (A0 / len(omegas), 0.0)] + [C0]
                    try:
                        popt, _ = _curve_fit(model, xd_all, yd_all, p0=p0, maxfev=10000)
                        x_fit = np.linspace(xd_all.min(), xd_all.max(), 500)
                        ax.plot(x_fit, model(x_fit, *popt),
                                color=cc_fn(fc), linewidth=PUB_CURVE_WIDTH, zorder=curve_z)

                        label = f"n={order}" if order is not None else "unknown harmonic"
                        print(f"    {label}:")
                        for k, n_period in enumerate(periods):
                            A_k, phi_k = popt[2 * k], popt[2 * k + 1]
                            if A_k < 0:
                                A_k, phi_k = -A_k, phi_k + np.pi
                            phi_k = (phi_k + np.pi) % (2 * np.pi) - np.pi
                            print(f"      component {k} (n={n_period} period(s)/sweep): "
                                  f"A={A_k:.4f} {_fit_unit},  phi={np.degrees(phi_k):+7.2f} deg")
                        print(f"      offset C = {popt[-1]:.4f} {_fit_unit}")
                    except RuntimeError:
                        print(f"  Warning: sinusoid fit failed for harmonic {order}; skipping.")
            elif PUB_CURVE_MODE == 'phase_harmonics':
                for ln, _, _, _, _ in _lines:
                    ln.set_visible(False)
                if X_AXIS not in ('ch1_phase', 'ch2_phase'):
                    print(f"  Warning: PUB_CURVE_MODE='phase_harmonics' assumes a phase "
                          f"X_AXIS in degrees; current X_AXIS={X_AXIS!r}.")
                _fit_unit = '%' if NORMALIZE == 'percent' else ('dBc' if NORMALIZE else 'dBm')
                n_harm = PUB_PHASE_HARMONIC_N
                model = _make_phase_harmonic_model(n_harm)
                print(f"\n  Phase-harmonic fit parameters "
                      f"(C + 2*sum_m=1^{n_harm} A_m*cos(m*phi + m*phi0)):")
                for _, xd_all, yd_all, fc, order in _lines:
                    if len(xd_all) < n_harm + 2:
                        print(f"  Warning: not enough points to fit N={n_harm} harmonics "
                              f"for harmonic {order}; skipping.")
                        continue
                    # _make_phase_harmonic_model's x_deg always expects degrees;
                    # xd_all is already in radians here if the axes is polar.
                    xd_all_deg = np.degrees(xd_all) if is_polar else xd_all
                    C0 = float(np.mean(yd_all))
                    A0 = (yd_all.max() - yd_all.min()) / max(4 * n_harm, 1)
                    p0 = [A0] * n_harm + [0.0, C0]
                    try:
                        popt, _ = _curve_fit(model, xd_all_deg, yd_all, p0=p0, maxfev=20000)
                        x_fit_deg = np.linspace(xd_all_deg.min(), xd_all_deg.max(), 500)
                        x_fit_plot = np.deg2rad(x_fit_deg) if is_polar else x_fit_deg
                        ax.plot(x_fit_plot, model(x_fit_deg, *popt),
                                color=cc_fn(fc), linewidth=PUB_CURVE_WIDTH, zorder=curve_z)

                        label = f"n={order}" if order is not None else "unknown harmonic"
                        phi0_deg = (np.degrees(popt[n_harm]) + 180.0) % 360.0 - 180.0
                        print(f"    {label}:")
                        print(f"      phi0 = {phi0_deg:+7.2f} deg")
                        for m in range(1, n_harm + 1):
                            print(f"      A_{m} = {popt[m - 1]:.4f} {_fit_unit}")
                        print(f"      C = {popt[-1]:.4f} {_fit_unit}")
                    except RuntimeError:
                        print(f"  Warning: phase-harmonic fit failed for harmonic {order}; skipping.")
            elif PUB_CURVE_MODE == 'nonlinear_phase':
                # nl_draw_params/fit_harmonics come from the ONE joint fit
                # done once below (shared across every SPLIT_GROUPS axes) --
                # unlike 'sinusoid'/'phase_harmonics' above, nothing is re-fit
                # here. nl_draw_params is the fitted result if the automatic
                # fit succeeded, or (only when NL_FIT_INTERACTIVE_SLIDERS is
                # on) a guess/bounds-midpoint fallback so the sliders have a
                # starting curve to drag from. If neither is available, or a
                # harmonic on this axes wasn't part of the fit, its raw data
                # is simply left visible rather than hidden with nothing to
                # replace it.
                if nl_draw_params is None:
                    print("  Warning: nonlinear_phase fit unavailable; leaving raw data visible.")
                else:
                    for ln, _, _, _, order in _lines:
                        if order in fit_harmonics:
                            ln.set_visible(False)
                    for _, xd_all, yd_all, fc, order in _lines:
                        if order is None or order not in fit_harmonics:
                            continue
                        # _nl_curve_y always expects x_deg in degrees; xd_all is
                        # already in radians here if the axes is polar.
                        xd_all_deg = np.degrees(xd_all) if is_polar else xd_all
                        x_fit_deg = np.linspace(xd_all_deg.min(), xd_all_deg.max(), 500)
                        x_fit_plot = np.deg2rad(x_fit_deg) if is_polar else x_fit_deg
                        y_fit = _nl_curve_y(nl_draw_params['beta1'], nl_draw_params['beta2'],
                                             nl_draw_params['beta2_nl'], nl_draw_params['phi0_rad'],
                                             nl_draw_params['phi_nl_rad'],
                                             x_fit_deg, order, nl_draw_params['k_trunc'],
                                             scale=nl_draw_params['scales'].get(order, 1.0))
                        [line] = ax.plot(x_fit_plot, y_fit, color=cc_fn(fc),
                                          linewidth=PUB_CURVE_WIDTH, zorder=curve_z)
                        nl_fit_lines.append((line, order, x_fit_deg, nl_draw_params['k_trunc']))
        fig.savefig(_os.path.join(SAVE_FOLDER, svg_name), format='svg', bbox_inches='tight')
        print(f"Saved: {_os.path.join(SAVE_FOLDER, svg_name)}")

    # ── nonlinear-phase joint fit: computed ONCE here (pooling every
    # displayed harmonic's data across all SPLIT_GROUPS at once), then just
    # looked up by _apply_pub_style above for each axes' overlay curve.
    # nl_fit_result is the automatically-fitted parameters (None if that
    # fit failed); nl_draw_params is what actually gets drawn/slider-driven --
    # the fit if it succeeded, else (only with NL_FIT_INTERACTIVE_SLIDERS) a
    # NL_FIT_GUESS_*/bounds-midpoint fallback so the sliders have a starting
    # curve. If nl_draw_params ends up None, _apply_pub_style leaves each
    # harmonic's raw data visible instead of hiding it with nothing to replace it.
    fit_harmonics = []
    nl_fit_result = None
    nl_draw_params = None
    nl_fit_lines = []
    if FOR_PUBLICATION and PUB_CURVE_MODE == 'nonlinear_phase':
        if SPLIT_FIGURES:
            display_harmonics = sorted(set(h for grp in SPLIT_GROUPS for h in grp['harmonics']))
        else:
            display_harmonics = (sorted(set(HARMONICS_TO_SHOW)) if HARMONICS_TO_SHOW is not None
                                  else sorted(int(n) for n in data.harmonics))
        fit_harmonics = [n for n in display_harmonics if n in data.harmonics]

        if X_AXIS != 'ch2_phase':
            print(f"\n  Warning: PUB_CURVE_MODE='nonlinear_phase' assumes X_AXIS='ch2_phase' "
                  f"(the swept phase in the model); current X_AXIS={X_AXIS!r}.")
        if not fit_harmonics:
            print("\n  Warning: nonlinear_phase fit has no displayed harmonics to fit; skipping.")
        else:
            k_trunc_nl = int(2 * max(NL_FIT_BETA1_BOUNDS[1],
                                      NL_FIT_BETA2_BOUNDS[1] + NL_FIT_BETA2_NL_BOUNDS[1])) + 20

            if PRESAVED_FIT_PATH:
                try:
                    with open(PRESAVED_FIT_PATH, 'r') as f:
                        saved = _json.load(f)
                    nl_fit_result = {
                        'beta1': float(saved['beta1']), 'beta2': float(saved['beta2']),
                        'beta2_nl': float(saved['beta2_nl']),
                        'phi0_rad': float(saved['phi0_rad']), 'phi_nl_rad': float(saved['phi_nl_rad']),
                        'scales': {int(k): float(v) for k, v in saved.get('scales', {}).items()},
                        'k_trunc': int(saved.get('k_trunc', k_trunc_nl)),
                    }
                    print(f"\n  Loaded presaved nonlinear_phase fit from {PRESAVED_FIT_PATH}:")
                    print(f"    beta1     = {nl_fit_result['beta1']:.4f} rad")
                    print(f"    beta2     = {nl_fit_result['beta2']:.4f} rad")
                    print(f"    beta2_nl  = {nl_fit_result['beta2_nl']:.4f} rad")
                    print(f"    phi0      = {np.degrees(nl_fit_result['phi0_rad']) % 360.0:+7.2f} deg")
                    print(f"    phi_NL    = {np.degrees(nl_fit_result['phi_nl_rad']) % 360.0:+7.2f} deg")
                    for n in fit_harmonics:
                        print(f"    scale[{n:+d}] = {nl_fit_result['scales'].get(n, 1.0):.4f}")

                    phi_rad_all = np.deg2rad(data.ch2_phases_deg)
                    dbc_all = data.normalized_peak_powers_dbm()   # (M, N) dBc
                    measured_dbc = {}
                    for n in fit_harmonics:
                        j = int(np.where(data.harmonics == n)[0][0])
                        measured_dbc[n] = dbc_all[:, j]
                    amps_fit = _nl_phase_amplitudes(
                        nl_fit_result['beta1'], nl_fit_result['beta2'], nl_fit_result['beta2_nl'],
                        nl_fit_result['phi0_rad'], nl_fit_result['phi_nl_rad'],
                        phi_rad_all, fit_harmonics, nl_fit_result['k_trunc'])
                    print(f"    {'order':>5}   {'measured (mean)':>16}   {'RMS residual':>12}")
                    for n in fit_harmonics:
                        scale = nl_fit_result['scales'].get(n, 1.0)
                        pred_dbc = 10.0 * np.log10(np.maximum(scale * np.abs(amps_fit[n]) ** 2, 1e-30))
                        resid = measured_dbc[n] - pred_dbc
                        print(f"    p={n:+d}   {measured_dbc[n].mean():>14.3f} dB   "
                              f"{np.sqrt(np.mean(resid ** 2)):>10.3f} dB")
                except Exception as e:
                    print(f"\n  Warning: failed to load PRESAVED_FIT_PATH ({e}); "
                          f"running the fit normally instead.")
                    nl_fit_result = None

            if nl_fit_result is None:
                # Harmonics with a CALIBRATION_DICT entry are already corrected
                # in the raw data (see the CALIBRATION_DICT pre-shift above) --
                # their per-sideband scale is fixed at 1.0 rather than
                # free-fit, to avoid applying that correction twice.
                free_scale_harmonics = ([n for n in fit_harmonics
                                          if not (CALIBRATION_DICT and int(n) in CALIBRATION_DICT)]
                                         if NL_FIT_PER_SIDEBAND_SCALE else [])
                calibrated_harmonics = [n for n in fit_harmonics if n not in free_scale_harmonics]

                bounds = [NL_FIT_BETA1_BOUNDS, NL_FIT_BETA2_BOUNDS, NL_FIT_BETA2_NL_BOUNDS,
                          tuple(np.deg2rad(NL_FIT_PHI0_BOUNDS_DEG)),
                          tuple(np.deg2rad(NL_FIT_PHI_NL_BOUNDS_DEG))]
                guess = [NL_FIT_GUESS_BETA1, NL_FIT_GUESS_BETA2, NL_FIT_GUESS_BETA2_NL,
                         None if NL_FIT_GUESS_PHI0_DEG is None else np.deg2rad(NL_FIT_GUESS_PHI0_DEG),
                         None if NL_FIT_GUESS_PHI_NL_DEG is None else np.deg2rad(NL_FIT_GUESS_PHI_NL_DEG)]
                if free_scale_harmonics:
                    bounds += [NL_FIT_SCALE_BOUNDS] * len(free_scale_harmonics)
                    guess += [None] * len(free_scale_harmonics)
                x0 = [float(np.clip(g if g is not None else 0.5 * (lo + hi), lo, hi))
                      for g, (lo, hi) in zip(guess, bounds)]
                fallback_params = {
                    'beta1': x0[0], 'beta2': x0[1], 'beta2_nl': x0[2],
                    'phi0_rad': x0[3], 'phi_nl_rad': x0[4],
                    'scales': {n: x0[5 + i] for i, n in enumerate(free_scale_harmonics)},
                    'k_trunc': k_trunc_nl,
                }

                try:
                    phi_rad_all = np.deg2rad(data.ch2_phases_deg)
                    dbc_all = data.normalized_peak_powers_dbm()   # (M, N) dBc
                    measured_dbc = {}
                    for n in fit_harmonics:
                        j = int(np.where(data.harmonics == n)[0][0])
                        measured_dbc[n] = dbc_all[:, j]

                    result = _differential_evolution(
                        _nl_phase_cost, bounds,
                        args=(phi_rad_all, fit_harmonics, measured_dbc, k_trunc_nl, free_scale_harmonics),
                        seed=NL_FIT_SEED, tol=1e-12, polish=True,
                        x0=(x0 if any(g is not None for g in guess) else None))
                    if not np.isfinite(result.fun):
                        raise RuntimeError(
                            "fit produced a non-finite cost (check for NaN/invalid values in the data)")
                    beta1_fit, beta2_fit, beta2_nl_fit, phi0_fit_rad, phi_nl_fit_rad = result.x[:5]
                    phi0_fit_deg = float(np.degrees(phi0_fit_rad) % 360.0)
                    phi_nl_fit_deg = float(np.degrees(phi_nl_fit_rad) % 360.0)
                    scales_fit = {n: float(result.x[5 + i]) for i, n in enumerate(free_scale_harmonics)}

                    print("\n  Nonlinear-phase joint fit "
                          "(theta = beta1*sin(Wt) + beta2*sin(2Wt+phi+phi0) + beta2_nl*sin(2Wt+phi_NL)):")
                    print(f"    beta1     = {beta1_fit:.4f} rad")
                    print(f"    beta2     = {beta2_fit:.4f} rad")
                    print(f"    beta2_nl  = {beta2_nl_fit:.4f} rad")
                    print(f"    phi0      = {phi0_fit_deg:+7.2f} deg")
                    print(f"    phi_NL    = {phi_nl_fit_deg:+7.2f} deg")
                    if NL_FIT_PER_SIDEBAND_SCALE:
                        for n in fit_harmonics:
                            tag = '  (fixed via CALIBRATION_DICT)' if n in calibrated_harmonics else ''
                            print(f"    scale[{n:+d}] = {scales_fit.get(n, 1.0):.4f}{tag}")
                    print(f"    Sum of squared dBc residuals: {result.fun:.4f}")

                    amps_fit = _nl_phase_amplitudes(beta1_fit, beta2_fit, beta2_nl_fit, phi0_fit_rad,
                                                     phi_nl_fit_rad, phi_rad_all, fit_harmonics, k_trunc_nl)
                    print(f"    {'order':>5}   {'measured (mean)':>16}   {'RMS residual':>12}")
                    for n in fit_harmonics:
                        pred_dbc = 10.0 * np.log10(np.maximum(scales_fit.get(n, 1.0) * np.abs(amps_fit[n]) ** 2, 1e-30))
                        resid = measured_dbc[n] - pred_dbc
                        print(f"    p={n:+d}   {measured_dbc[n].mean():>14.3f} dB   "
                              f"{np.sqrt(np.mean(resid ** 2)):>10.3f} dB")

                    nl_fit_result = {
                        'beta1': float(beta1_fit), 'beta2': float(beta2_fit),
                        'beta2_nl': float(beta2_nl_fit),
                        'phi0_rad': float(phi0_fit_rad), 'phi_nl_rad': float(phi_nl_fit_rad),
                        'scales': scales_fit,
                        'k_trunc': k_trunc_nl,
                    }

                    save_path = _os.path.join(
                        _os.path.dirname(local_path(DATA_FILE)),
                        _os.path.splitext(_os.path.basename(DATA_FILE))[0] + '_nl_fit.json')
                    try:
                        with open(save_path, 'w') as f:
                            _json.dump({
                                'beta1': nl_fit_result['beta1'], 'beta2': nl_fit_result['beta2'],
                                'beta2_nl': nl_fit_result['beta2_nl'],
                                'phi0_rad': nl_fit_result['phi0_rad'], 'phi_nl_rad': nl_fit_result['phi_nl_rad'],
                                'scales': {str(k): v for k, v in nl_fit_result['scales'].items()},
                                'k_trunc': nl_fit_result['k_trunc'],
                                'fit_harmonics': fit_harmonics,
                                'source_data_file': DATA_FILE,
                            }, f, indent=2)
                        print(f"\n  Saved nonlinear_phase fit parameters to {save_path}")
                    except Exception as e:
                        print(f"\n  Warning: could not save fit parameters ({e}).")
                except Exception as e:
                    print(f"\n  Warning: nonlinear_phase joint fit failed ({e}); "
                          f"{'sliders will start from a guess/bounds-midpoint fallback' if NL_FIT_INTERACTIVE_SLIDERS else 'leaving raw data visible'}.")
                    nl_fit_result = None

            if nl_fit_result is not None:
                nl_draw_params = nl_fit_result
            elif NL_FIT_INTERACTIVE_SLIDERS:
                print(f"\n  Sliders starting from: beta1={fallback_params['beta1']:.4f} rad, "
                      f"beta2={fallback_params['beta2']:.4f} rad, "
                      f"beta2_nl={fallback_params['beta2_nl']:.4f} rad, "
                      f"phi0={np.degrees(fallback_params['phi0_rad']) % 360.0:+7.2f} deg, "
                      f"phi_NL={np.degrees(fallback_params['phi_nl_rad']) % 360.0:+7.2f} deg"
                      + (f", scales={fallback_params['scales']}" if NL_FIT_PER_SIDEBAND_SCALE else ""))
                nl_draw_params = fallback_params

    _polar_active = SHOW_POLAR_CE_PLOT
    if _polar_active and X_AXIS not in ('ch1_phase', 'ch2_phase'):
        print(f"\n  Warning: SHOW_POLAR_CE_PLOT assumes a phase X_AXIS "
              f"(angle = swept phase); current X_AXIS={X_AXIS!r}. Falling back to cartesian.")
        _polar_active = False

    def _set_polar_rings(ax):
        """POLAR_N_RINGS whole-number CE rings, evenly spaced by the
        LARGEST whole-number increment for which the leftover center circle
        (from the plot's center out to the innermost ring) is still at
        least as large as that increment -- computed from this axes' own
        range, since each SPLIT_GROUPS entry spans different CE values. The
        outermost ring sits exactly one increment inside the outer edge,
        and the rest step inward from there. If the range can't fit
        POLAR_N_RINGS rings under that center-circle constraint even at the
        minimum increment of 1, the ring count is reduced (with a warning)
        rather than shrinking the increment below 1 or violating the
        constraint. Ticks exactly at either axis limit are excluded, since
        matplotlib silently drops a tick placed exactly at the center (r=0
        degenerates to a point, not a ring)."""
        y_lo, y_hi = ax.get_ylim()
        span = y_hi - y_lo
        if span <= 0:
            return
        n = POLAR_N_RINGS
        step = int(np.floor(span / (n + 1)))
        while n > 0 and step < 1:
            n -= 1
            step = int(np.floor(span / (n + 1))) if n > 0 else step
        if n <= 0:
            print(f"\n  Warning: range ({y_lo:g}, {y_hi:g}) is too small to fit "
                  f"any whole-number CE ring while keeping the center circle "
                  f"at least as large as the increment.")
            return
        if n < POLAR_N_RINGS:
            print(f"\n  Warning: only {n} of {POLAR_N_RINGS} requested whole-number "
                  f"CE rings fit in range ({y_lo:g}, {y_hi:g}) while keeping the "
                  f"center circle at least as large as the increment (step={step}).")
        outer = y_hi - step
        rings = outer - step * np.arange(n)
        ax.set_yticks(rings)

    if SPLIT_FIGURES:
        for grp in SPLIT_GROUPS:
            diam_mm = grp.get('polar_diam_mm') or max(grp['w_mm'], grp['h_mm'])
            fig_g, ax_g = data.plot_peak_powers(
                normalize=NORMALIZE,
                x_axis=X_AXIS,
                axes_width_mm=diam_mm if _polar_active else grp['w_mm'],
                axes_height_mm=diam_mm if _polar_active else grp['h_mm'],
                ymin=grp['ymin'],
                ymax=grp['ymax'],
                show_points=SHOW_REPEAT_POINTS,
                show_error=SHOW_ERROR_BAND,
                harmonics=grp['harmonics'],
                show_line_markers=not FOR_PUBLICATION,
                polar=_polar_active,
            )
            if _polar_active and POLAR_N_RINGS is not None:
                _set_polar_rings(ax_g)
            if FOR_PUBLICATION:
                _apply_pub_style(fig_g, ax_g, grp['svg'],
                                 marker_pt=grp.get('marker_pt', PUB_MARKER_PT))
    else:
        diam_mm = POLAR_DIAMETER_MM or max(axes_width_mm, axes_height_mm)
        fig_pow, ax_pow = data.plot_peak_powers(
            normalize=NORMALIZE,
            x_axis=X_AXIS,
            axes_width_mm=diam_mm if _polar_active else axes_width_mm,
            axes_height_mm=diam_mm if _polar_active else axes_height_mm,
            ymin=POWER_YMIN,
            ymax=POWER_YMAX,
            show_points=SHOW_REPEAT_POINTS,
            show_error=SHOW_ERROR_BAND,
            harmonics=HARMONICS_TO_SHOW,
            show_line_markers=not FOR_PUBLICATION,
            polar=_polar_active,
        )
        if _polar_active and POLAR_N_RINGS is not None:
            _set_polar_rings(ax_pow)
        if FOR_PUBLICATION:
            _apply_pub_style(fig_pow, ax_pow, 'dual_tone_sweep_powers.svg')

    if SHOW_CALIBRATION:
        fig_cal, ax_cal = data.plot_calibration(
            x_axis=X_AXIS,
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            ymin=CAL_YMIN,
            ymax=CAL_YMAX,
        )

        if FOR_PUBLICATION:
            import os as _os
            ax_cal.set_xlabel('')
            ax_cal.set_ylabel('')
            ax_cal.tick_params(labelbottom=False, labelleft=False)
            fig_cal.savefig(_os.path.join(SAVE_FOLDER, 'dual_tone_sweep_calibration.svg'),
                            format='svg', bbox_inches='tight')
            print(f"Saved: {_os.path.join(SAVE_FOLDER, 'dual_tone_sweep_calibration.svg')}")

    _nl_slider_controls = None
    if NL_FIT_INTERACTIVE_SLIDERS and nl_draw_params is not None and nl_fit_lines:
        _nl_slider_controls = _add_nl_fit_sliders(nl_fit_lines, nl_draw_params)

    plt.show()


if __name__ == '__main__':
    main()
