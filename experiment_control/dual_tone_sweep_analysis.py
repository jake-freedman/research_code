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

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\phase_sweep_dual_tone_sweep_2026-06-24-10-17-19.npz"
# DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\phase_sweep_dual_tone_sweep_2026-07-17-11-09-05.npz"
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
      POWER_YMIN = -90
      POWER_YMAX = 5

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40

# Initial β guess for single-tone preamble extraction.
BETA_GUESS_REF = 2.0

# ── sideband filter ──────────────────────────────────────────────────────────
# None = show all recorded harmonics; list = show only those orders.
# (ignored when SPLIT_FIGURES = True)
HARMONICS_TO_SHOW = [-2, -1, 0, 1, 2]   # e.g. [-1, 1] to show only ±1

# ── figure size ───────────────────────────────────────────────────────────────
axes_width_mm  = 100
axes_height_mm = 55

# ── split figures ─────────────────────────────────────────────────────────────
# When True, produce three separate power figures instead of one.
# Each dict: harmonics to include, y-limits, axes size, and SVG filename.
SPLIT_FIGURES = False
SPLIT_GROUPS = [
    {'harmonics': [0],      'ymin': 0,   'ymax': 18,  'w_mm': 85,  'h_mm': 20,  'marker_pt': 4,  'svg': 'dual_tone_sweep_powers_0.svg'},
    {'harmonics': [-1, 1],  'ymin': -2,  'ymax': 53,  'w_mm': 85,  'h_mm': 55,  'marker_pt': 8,  'svg': 'dual_tone_sweep_powers_pm1.svg'},
    {'harmonics': [-2, 2],  'ymin': 0,   'ymax': 18,  'w_mm': 85,  'h_mm': 20,  'marker_pt': 4,  'svg': 'dual_tone_sweep_powers_pm2.svg'},
]

# ── calibration scaling ───────────────────────────────────────────────────────
# Per-harmonic multiplicative correction applied to the conversion efficiency
# after all normalization. Keys are harmonic orders (int); missing orders are
# left unchanged. Set to None to disable.
# Example: {1: 2.0, -1: 2.0} doubles the measured ±1 sideband power.
CALIBRATION_DICT = None # {-2: 1.12, -1: 1.33, 0: 1.08, 1: 1.168, 2: 1.136}

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
PUB_CURVE_MODE          = 'phase_harmonics'
# Per-combline period count for the sinusoid frequency initial guess (PUB_CURVE_MODE = 'sinusoid').
# Keys are harmonic orders (int); missing orders default to 1.
PUB_SINUSOID_N_PERIODS  = {0:(1,2,3), -2: (1,2,3), 2: (1,2,3), -1: (1,2,3), 1: (1,2,3)}
# Number of phase harmonics N to fit (PUB_CURVE_MODE = 'phase_harmonics'), same for every sideband.
PUB_PHASE_HARMONIC_N    = 3


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
    import matplotlib.colors as _mc
    from scipy.optimize import curve_fit as _curve_fit

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

    def _apply_pub_style(fig, ax, svg_name, marker_pt=PUB_MARKER_PT):
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
                    C0 = float(np.mean(yd_all))
                    A0 = (yd_all.max() - yd_all.min()) / max(4 * n_harm, 1)
                    p0 = [A0] * n_harm + [0.0, C0]
                    try:
                        popt, _ = _curve_fit(model, xd_all, yd_all, p0=p0, maxfev=20000)
                        x_fit = np.linspace(xd_all.min(), xd_all.max(), 500)
                        ax.plot(x_fit, model(x_fit, *popt),
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
        fig.savefig(_os.path.join(SAVE_FOLDER, svg_name), format='svg', bbox_inches='tight')
        print(f"Saved: {_os.path.join(SAVE_FOLDER, svg_name)}")

    if SPLIT_FIGURES:
        for grp in SPLIT_GROUPS:
            fig_g, ax_g = data.plot_peak_powers(
                normalize=NORMALIZE,
                x_axis=X_AXIS,
                axes_width_mm=grp['w_mm'],
                axes_height_mm=grp['h_mm'],
                ymin=grp['ymin'],
                ymax=grp['ymax'],
                show_points=SHOW_REPEAT_POINTS,
                show_error=SHOW_ERROR_BAND,
                harmonics=grp['harmonics'],
                show_line_markers=not FOR_PUBLICATION,
            )
            if FOR_PUBLICATION:
                _apply_pub_style(fig_g, ax_g, grp['svg'],
                                 marker_pt=grp.get('marker_pt', PUB_MARKER_PT))
    else:
        fig_pow, ax_pow = data.plot_peak_powers(
            normalize=NORMALIZE,
            x_axis=X_AXIS,
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            ymin=POWER_YMIN,
            ymax=POWER_YMAX,
            show_points=SHOW_REPEAT_POINTS,
            show_error=SHOW_ERROR_BAND,
            harmonics=HARMONICS_TO_SHOW,
            show_line_markers=not FOR_PUBLICATION,
        )
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

    plt.show()


if __name__ == '__main__':
    main()
