"""
BNC 855B dark-window / combline optimizer.

Three mutually exclusive objectives, selected by `mode`:

  'minimize' (default) — searches for the (P1, P2, phi2) state that
    minimizes a group of sideband orders simultaneously (minimax on
    conversion efficiency, matching the 'dark_window' criterion in
    comb_finder.py): the objective is min over phase of max over
    dark_orders of CE [dBc].

  'maximize' — searches for the (P1, P2, phi2) state that maximizes a
    single provided combline order: the objective is max over phase of
    CE [dBc] at maximize_order.

  'flatness' — searches for the (P1, P2, phi2) state that makes the powers
    of a group of comblines as equal as possible: the objective is min over
    phase of (max - min) CE [dB] across flatness_orders. Only the spread is
    optimized; the common level is unconstrained.

Either way the run then deeply characterizes the state it finds.

Stage 1 — power search:
  Nelder-Mead simplex over (P1, P2), hard-clipped to user-supplied bounds.
  Each objective evaluation sets the powers, records an RF-off J0 carrier
  calibration, then scans the full 0-360 deg of ch2 phase (N points) while
  measuring the target-order sideband peak(s). Changing power scrambles the
  global phase, so the full circle must be re-scanned at every power point
  and the J0 calibration is never refreshed inside a phase scan. The coarse
  grid optimum is then sharpened by a short adaptive-step phase descent so
  the objective measures the true peak/null depth rather than how close a
  grid point happened to land to it. With search_in_voltage=True the
  simplex runs in RMS-voltage coordinates instead of dBm (hardware is still
  commanded in dBm) — the sideband physics is linear in drive voltage, so
  this conditions the landscape better when the bounds span many dB.

Stage 2 — phase refinement:
  Powers are fixed at the best point found. After a fresh J0 calibration and
  one more full-circle scan (the phase-to-physical mapping was scrambled by
  the final power change), phi2 is refined by adaptive-step downhill descent:
  probe phi +/- step, move to the lower side if it improves, otherwise halve
  the step, until the step falls below phase_tol_deg.

Stage 3 — deep measurement:
  With (P1*, P2*, phi2*) left untouched (any power or output toggle would
  scramble phi2*), a broader range of harmonics is measured n_deep_repeats
  times back-to-back. The stage-2 J0 calibration is used for all repeats.

Early stop: if stop_at_metric_dbc is set, the run ends the moment any single
phase measurement in stage 1 or 2 reaches that value in the direction implied
by mode (<= for 'minimize'/'flatness', >= for 'maximize') — the optimization
halts and the deep measurement starts immediately with the signal generator
left exactly as-is, using the J0 calibration from the current phase scan.

Everything is saved in one .npz: the deep measurement is DualToneSweepData-
compatible (repeats mapped to the steps axis, constant drive arrays) and the
full optimization trajectory is included as extra opt_* / refine_* keys.
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'


def _sweep_avg(esa, n_avg: int):
    if n_avg == 1:
        _, pwr = esa.sweep()
        return np.asarray(pwr, dtype=float)
    accum = None
    for _ in range(n_avg):
        _, pwr = esa.sweep()
        lin = 10.0 ** (np.asarray(pwr, dtype=float) / 10.0)
        accum = lin if accum is None else accum + lin
    return 10.0 * np.log10(accum / n_avg)


_LOG20 = 10.0 * np.log10(20.0)


def _dbm_to_vrms(p_dbm):
    """RMS voltage into 50 ohm for a power in dBm (as in voltage_linspace)."""
    return 10.0 ** ((p_dbm - _LOG20) / 20.0)


def _vrms_to_dbm(v):
    return 20.0 * np.log10(v) + _LOG20


class _TargetReached(Exception):
    """Raised mid-search when a phase measurement hits stop_at_metric_dbc."""


def bnc_dark_window_optimizer(
    cw_freq: float,
    ch1_power_min_dbm: float,
    ch1_power_max_dbm: float,
    ch2_power_min_dbm: float,
    ch2_power_max_dbm: float,
    ch1_power_guess_dbm: float,
    ch2_power_guess_dbm: float,
    n_phases: int = 36,
    mode: str = 'minimize',
    dark_orders=(0, -1),
    maximize_order: int = None,
    flatness_orders=None,
    deep_harmonics=(-3, -2, -1, 0, 1, 2, 3),
    n_deep_repeats: int = 10,
    power_initial_step_dbm: float = 0.5,
    power_xatol_dbm: float = 0.05,
    power_fatol_db: float = 0.25,
    power_max_evals: int = 30,
    search_in_voltage: bool = False,
    stop_at_metric_dbc: float = None,
    eval_phase_refine_iters: int = 8,
    phase_tol_deg: float = 0.5,
    phase_max_iters: int = 25,
    heterodyne_shift: float = 125e6,
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    averages_per_point: int = 1,
    use_cxa: bool = False,
    data_folder: str = '.',
    optional_name: str = '',
    plot: bool = True,
) -> str:
    """
    Dark-window search + deep measurement.

    Parameters
    ----------
    cw_freq : float
        Ch1 drive frequency f in Hz. Ch2 is driven at 2*f.
    ch1_power_min_dbm, ch1_power_max_dbm : float
        Hard bounds on ch1 power in dBm. The optimizer never commands a
        power outside this range.
    ch2_power_min_dbm, ch2_power_max_dbm : float
        Hard bounds on ch2 power in dBm.
    ch1_power_guess_dbm, ch2_power_guess_dbm : float
        Starting point for the Nelder-Mead power search.
    n_phases : int
        Phase points per full-circle scan at each power evaluation.
    mode : str
        'minimize' (default) — minimize the worst CE among dark_orders
        (dark-window search). 'maximize' — maximize the CE of the single
        maximize_order combline. 'flatness' — minimize the peak-to-peak
        spread in CE across flatness_orders, i.e. make those comblines as
        equal in power as possible.
    dark_orders : sequence of int
        Used when mode='minimize'. Sideband orders whose worst CE is
        minimized. Default (0, -1).
    maximize_order : int, optional
        Used when mode='maximize'. The single combline order whose CE is
        maximized. Required when mode='maximize'.
    flatness_orders : sequence of int, optional
        Used when mode='flatness'. The combline orders whose powers should
        be made as equal as possible (at least 2 required). Required when
        mode='flatness'.
    deep_harmonics : sequence of int
        Harmonic orders recorded in the final deep measurement.
    n_deep_repeats : int
        Back-to-back repeats of the deep measurement.
    power_initial_step_dbm : float
        Size of the initial Nelder-Mead simplex edges in dBm.
    power_xatol_dbm : float
        Nelder-Mead termination tolerance on the powers in dBm.
    power_fatol_db : float
        Nelder-Mead termination tolerance on the metric in dB.
    power_max_evals : int
        Maximum objective evaluations (phase scans) in the power stage.
    search_in_voltage : bool
        If True, the Nelder-Mead simplex works in RMS-voltage coordinates
        (50 ohm) instead of dBm; the hardware is still commanded in dBm and
        the bounds still apply exactly. Voltage coordinates condition the
        landscape better when the power bounds span many dB; within a
        couple of dB the two are nearly equivalent. power_initial_step_dbm
        and power_xatol_dbm are converted to their voltage equivalents at
        the guess point. Default False.
    stop_at_metric_dbc : float, optional
        Early-stop target, in the direction implied by mode: for
        mode='minimize' or 'flatness' the run stops the moment a measurement
        is <= this value (e.g. -40 stops once the worst dark order is at or
        below -40 dBc; for flatness, e.g. 1.0 stops once the spread is at or
        below 1 dB); for mode='maximize' it stops the moment a measurement
        is >= this value (e.g. -3 stops once maximize_order reaches -3 dBc).
        Either way the optimization halts immediately and the deep
        measurement begins with the signal generator left exactly as-is;
        the J0 calibration from the current phase scan is reused. None
        (default) disables early stopping.
    eval_phase_refine_iters : int
        Maximum phase-descent iterations run after the coarse scan inside
        each power evaluation (2 measurements each). Sharpens the coarse
        grid minimum to the true null depth so the objective is
        reproducible. 0 disables per-evaluation refinement.
    phase_tol_deg : float
        Phase descent stops once the probe step falls below this (applies
        both per-evaluation and in stage 2).
    phase_max_iters : int
        Maximum stage-2 refinement iterations (2 measurements each).
    heterodyne_shift : float
        LO offset in Hz. ESA windows sit at abs(n*f + shift).
    window_hz : float
        Half-width of each ESA window in Hz.
    esa_freq_step : float
        Frequency step within each ESA window in Hz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz.
    esa_ref_level : float
        ESA reference level in dBm.
    settle_time_s : float
        Wait after each phase/power change and after re-enabling outputs.
    averages_per_point : int
        ESA sweeps to linear-average per measurement.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA.
    data_folder : str
        Directory in which to save the .npz file.
    optional_name : str
        Label prepended to the saved filename.
    plot : bool
        If True, plot the optimization trace and deep measurement.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    if mode not in ('minimize', 'maximize', 'flatness'):
        raise ValueError(f"mode must be 'minimize', 'maximize', or 'flatness', got {mode!r}")
    if mode == 'maximize' and maximize_order is None:
        raise ValueError("maximize_order must be given when mode='maximize'.")
    if mode == 'flatness' and (flatness_orders is None or len(list(flatness_orders)) < 2):
        raise ValueError("flatness_orders (>= 2 orders) must be given when mode='flatness'.")

    dark_orders = list(dark_orders)
    flatness_orders = list(flatness_orders) if flatness_orders is not None else []
    if mode == 'maximize':
        target_orders = [int(maximize_order)]
    elif mode == 'flatness':
        target_orders = flatness_orders
    else:
        target_orders = dark_orders

    # sign flips the metric so that "better" is always "smaller" downstream
    # (Nelder-Mead minimizes, and the phase-search comparisons use argmin).
    # minimize/flatness: smaller raw metric is better. maximize: larger is.
    sign = -1.0 if mode == 'maximize' else 1.0
    # unit_label documents what the raw metric numerically represents.
    unit_label = 'dB spread' if mode == 'flatness' else 'dBc'

    def better(a, b):
        """True if raw metric a is better than raw metric b, given mode."""
        return (sign * a) < (sign * b)

    def reduce_peaks(vals):
        """Combine per-order CE values [dBc] into the scalar raw metric."""
        return (max(vals) - min(vals)) if mode == 'flatness' else max(vals)

    deep_harmonics = list(deep_harmonics)
    phase_grid = np.linspace(0.0, 360.0, n_phases, endpoint=False)
    p1_lo, p1_hi = float(ch1_power_min_dbm), float(ch1_power_max_dbm)
    p2_lo, p2_hi = float(ch2_power_min_dbm), float(ch2_power_max_dbm)

    def clip_powers(p1, p2):
        c1 = float(np.clip(p1, p1_lo, p1_hi))
        c2 = float(np.clip(p2, p2_lo, p2_hi))
        if c1 != p1 or c2 != p2:
            print(f"  (clipped {p1:+.2f},{p2:+.2f} -> {c1:+.2f},{c2:+.2f} dBm)")
        return c1, c2

    g1, g2 = clip_powers(ch1_power_guess_dbm, ch2_power_guess_dbm)

    os.makedirs(data_folder, exist_ok=True)
    tag = {
        'minimize': 'dark_window_opt_',
        'maximize': 'maximize_combline_opt_',
        'flatness': 'flatten_combs_opt_',
    }[mode]
    fname = (
        f'{optional_name}{tag}'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(data_folder, fname)

    if mode == 'minimize':
        mode_line = f"  dark orders : {dark_orders} (minimize worst)"
    elif mode == 'maximize':
        mode_line = f"  target order: {target_orders[0]} (maximize)"
    else:
        mode_line = f"  flatness combs: {flatness_orders} (equalize)"
    title = {'minimize': 'Dark-window', 'maximize': 'Combline-maximize',
             'flatness': 'Combline-flatness'}[mode]
    print(
        f"{title} optimizer\n"
        f"  cw_freq     : {cw_freq / 1e9:.4f} GHz  (ch2 at {2 * cw_freq / 1e9:.4f} GHz)\n"
        f"  ch1 bounds  : {p1_lo:+.2f} to {p1_hi:+.2f} dBm (guess {g1:+.2f})\n"
        f"  ch2 bounds  : {p2_lo:+.2f} to {p2_hi:+.2f} dBm (guess {g2:+.2f})\n"
        f"{mode_line}\n"
        f"  phase grid  : {n_phases} points / 360 deg\n"
        f"  deep stage  : harmonics {deep_harmonics} x {n_deep_repeats} repeats\n"
    )

    # ── state shared across stages ───────────────────────────────────────
    offsets_hz = None
    K = None

    # power-stage trajectory
    traj_p1, traj_p2 = [], []
    traj_metric, traj_best_phase = [], []
    traj_phase_metrics = []          # (E, n_phases) CE dBc curves

    # refinement trajectory
    refine_phases, refine_metrics = [], []

    # deep measurement
    deep_spectra = []                # grows to (R, N_harm, K)
    deep_cal = None                  # (K,) J0 used for all deep repeats
    best = {}                        # filled as stages complete
    early_stop = False               # True if stop_at_metric_dbc was hit

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:

            def esa_window(center):
                esa.configure(
                    start_freq=center - window_hz,
                    stop_freq=center + window_hz,
                    freq_step=esa_freq_step, res_bw=esa_res_bw,
                    ref_level=esa_ref_level, attenuation=0.0,
                )

            def measure_j0_cal():
                """RF-off carrier spectrum. Scrambles the global phase."""
                nonlocal offsets_hz, K
                sig.disable_all_outputs()
                esa_window(heterodyne_shift)
                pwr = _sweep_avg(esa, averages_per_point)
                if offsets_hz is None:
                    offsets_hz = np.linspace(-window_hz, window_hz, len(pwr))
                    K = len(offsets_hz)
                sig.enable_all_outputs()
                time.sleep(settle_time_s)
                return pwr[:K]

            def order_peaks_dbm(orders):
                """Peak dBm in each order's window at the current state."""
                peaks = []
                for n in orders:
                    esa_window(abs(n * cw_freq + heterodyne_shift))
                    peaks.append(float(_sweep_avg(esa, averages_per_point)[:K].max()))
                return peaks

            # Live sig-gen state, kept so an early stop knows exactly what
            # is currently applied without touching the instrument again.
            current = dict(p1=g1, p2=g2, cal=None, phi2=0.0, metric=np.inf)

            def phase_metric(phi, cal_peak):
                """Raw (unsigned) mode-specific metric at commanded ch2
                phase phi: max CE over target_orders ('minimize'/'maximize',
                collapsing to a single order's CE for 'maximize'), or the
                peak-to-peak CE spread across target_orders ('flatness')."""
                sig.set_phase(2, float(phi) % 360.0)
                time.sleep(settle_time_s)
                peaks = order_peaks_dbm(target_orders)
                m = reduce_peaks([p - cal_peak for p in peaks])
                if stop_at_metric_dbc is not None and sign * m <= sign * stop_at_metric_dbc:
                    current['phi2'] = float(phi) % 360.0
                    current['metric'] = m
                    raise _TargetReached
                return m

            def full_phase_scan(cal_peak):
                """Scan the full circle; return (metrics (n_phases,), best_phi, best_m)."""
                metrics = np.array([phase_metric(ph, cal_peak) for ph in phase_grid])
                k = int(np.argmin(sign * metrics))
                return metrics, float(phase_grid[k]), float(metrics[k])

            def descend_phase(phi, f_phi, cal_peak, max_iters,
                              record_ph=None, record_m=None, verbose=False):
                """Adaptive-step downhill (mode-aware) search on phi2 from
                (phi, f_phi).

                Probes phi +/- step; moves to whichever side is better,
                otherwise halves the step, until step < phase_tol_deg or
                max_iters is reached. Returns (phi % 360, f_phi).
                """
                step = 360.0 / n_phases / 2.0
                for _ in range(max_iters):
                    if step < phase_tol_deg:
                        break
                    f_plus = phase_metric(phi + step, cal_peak)
                    f_minus = phase_metric(phi - step, cal_peak)
                    if record_ph is not None:
                        record_ph += [phi + step, phi - step]
                        record_m += [f_plus, f_minus]
                    cand_phi, cand_f = ((phi + step, f_plus) if better(f_plus, f_minus)
                                        else (phi - step, f_minus))
                    if better(cand_f, f_phi):
                        phi, f_phi = cand_phi, cand_f
                        if verbose:
                            print(f"  moved to phi2={phi % 360.0:.2f} deg: {f_phi:+.2f} {unit_label}")
                    else:
                        step /= 2.0
                return phi % 360.0, f_phi

            sig.configure_channel(1, cw_freq, g1, 0.0)
            sig.configure_channel(2, 2.0 * cw_freq, g2, 0.0)

            try:
                # ── Stage 1: Nelder-Mead over (P1, P2) ───────────────────
                # Simplex coordinates are either dBm (default) or RMS
                # voltage; everything downstream of from_coord() is dBm.
                if search_in_voltage:
                    to_coord, from_coord = _dbm_to_vrms, _vrms_to_dbm
                    # map the dBm tolerance through the local slope dV/dP at
                    # the smaller-voltage channel (conservative for both dims)
                    xatol = (np.log(10.0) / 20.0) * power_xatol_dbm * min(
                        _dbm_to_vrms(g1), _dbm_to_vrms(g2))
                    print("=== Stage 1: power search (Nelder-Mead, voltage coords) ===")
                else:
                    to_coord = from_coord = lambda p: p
                    xatol = power_xatol_dbm
                    print("=== Stage 1: power search (Nelder-Mead) ===")

                def objective(x):
                    p1, p2 = clip_powers(from_coord(x[0]), from_coord(x[1]))
                    sig.set_power(1, p1)
                    sig.set_power(2, p2)
                    time.sleep(settle_time_s)
                    cal = measure_j0_cal()
                    cal_peak = float(cal.max())
                    current.update(p1=p1, p2=p2, cal=cal)
                    metrics, best_phi, m = full_phase_scan(cal_peak)
                    # Sharpen the coarse grid optimum to the true peak/null
                    # depth so the objective is reproducible run to run.
                    best_phi, m = descend_phase(
                        best_phi, m, cal_peak, eval_phase_refine_iters,
                    )
                    traj_p1.append(p1)
                    traj_p2.append(p2)
                    traj_metric.append(m)
                    traj_best_phase.append(best_phi)
                    traj_phase_metrics.append(metrics)
                    print(
                        f"  eval {len(traj_metric):2d}: "
                        f"P1={p1:+.2f} P2={p2:+.2f} dBm -> "
                        f"{m:+.2f} {unit_label} @ phi2={best_phi:.1f} deg"
                    )
                    return sign * m

                s1 = power_initial_step_dbm if g1 + power_initial_step_dbm <= p1_hi else -power_initial_step_dbm
                s2 = power_initial_step_dbm if g2 + power_initial_step_dbm <= p2_hi else -power_initial_step_dbm
                initial_simplex = np.array([
                    [to_coord(g1), to_coord(g2)],
                    [to_coord(np.clip(g1 + s1, p1_lo, p1_hi)), to_coord(g2)],
                    [to_coord(g1), to_coord(np.clip(g2 + s2, p2_lo, p2_hi))],
                ])
                minimize(
                    objective, x0=np.array([to_coord(g1), to_coord(g2)]),
                    method='Nelder-Mead',
                    bounds=[
                        (to_coord(p1_lo), to_coord(p1_hi)),
                        (to_coord(p2_lo), to_coord(p2_hi)),
                    ],
                    options=dict(
                        initial_simplex=initial_simplex,
                        xatol=xatol,
                        fatol=power_fatol_db,
                        maxfev=power_max_evals,
                    ),
                )

                # Trust the measured trajectory over res.x (noisy metric)
                k_best = int(np.argmin(sign * np.array(traj_metric)))
                p1_star, p2_star = traj_p1[k_best], traj_p2[k_best]
                print(
                    f"Power stage done ({len(traj_metric)} evals): "
                    f"P1*={p1_star:+.2f} P2*={p2_star:+.2f} dBm, "
                    f"{traj_metric[k_best]:+.2f} {unit_label}\n"
                )

                # ── Stage 2: phase refinement at fixed powers ─────────────
                # Setting the powers scrambles the phase mapping, so
                # calibrate and re-scan the full circle before descending.
                print("=== Stage 2: phase refinement ===")
                sig.set_power(1, p1_star)
                sig.set_power(2, p2_star)
                time.sleep(settle_time_s)
                deep_cal = measure_j0_cal()
                cal_peak = float(deep_cal.max())
                current.update(p1=p1_star, p2=p2_star, cal=deep_cal)

                coarse_metrics, phi, f_phi = full_phase_scan(cal_peak)
                refine_phases.append(phi)
                refine_metrics.append(f_phi)
                print(f"  coarse: {f_phi:+.2f} {unit_label} @ phi2={phi:.1f} deg")

                phi_star, f_phi = descend_phase(
                    phi, f_phi, cal_peak, phase_max_iters,
                    record_ph=refine_phases, record_m=refine_metrics,
                    verbose=True,
                )
                print(f"Phase stage done: phi2*={phi_star:.2f} deg, {f_phi:+.2f} {unit_label}\n")

                best = dict(p1=p1_star, p2=p2_star, phi2=phi_star, metric=f_phi)
                sig.set_phase(2, phi_star)
                time.sleep(settle_time_s)

            except _TargetReached:
                early_stop = True
                best = dict(p1=current['p1'], p2=current['p2'],
                            phi2=current['phi2'], metric=current['metric'])
                deep_cal = current['cal']
                cmp_str = '>=' if mode == 'maximize' else '<='
                print(
                    f"\nTarget reached: {best['metric']:+.2f} {unit_label} {cmp_str} "
                    f"{stop_at_metric_dbc:+.2f} {unit_label} at "
                    f"P1={best['p1']:+.2f} dBm, P2={best['p2']:+.2f} dBm, "
                    f"phi2={best['phi2']:.2f} deg.\n"
                    f"Skipping remaining optimization; signal generator "
                    f"left untouched.\n"
                )

            # ── Stage 3: deep measurement ─────────────────────────────────
            # Powers, phase, and outputs must not be touched here or the
            # state is lost; all repeats share the last J0 calibration.
            print("=== Stage 3: deep measurement ===")
            for r in range(n_deep_repeats):
                rows = []
                for n in deep_harmonics:
                    esa_window(abs(n * cw_freq + heterodyne_shift))
                    rows.append(_sweep_avg(esa, averages_per_point)[:K])
                deep_spectra.append(rows)
                print(f"  repeat {r + 1}/{n_deep_repeats}")

            sig.disable_all_outputs()
            esa_window(heterodyne_shift)
            esa.set_continuous(True)
            print("ESA: continuous sweep at carrier beat window.")

    except Exception as exc:
        print(f"ERROR: {exc}")
        if deep_spectra:
            print(f"Saving partial deep measurement ({len(deep_spectra)}/{n_deep_repeats} repeats)...")
        elif traj_metric:
            partial = full_path.replace('.npz', '_trajectory_partial.npz')
            np.savez_compressed(
                partial,
                opt_ch1_dbm=np.array(traj_p1),
                opt_ch2_dbm=np.array(traj_p2),
                opt_metric_dbc=np.array(traj_metric),
                opt_best_phase_deg=np.array(traj_best_phase),
                opt_phase_metrics_dbc=np.array(traj_phase_metrics),
                opt_phase_grid_deg=phase_grid,
                refine_phases_deg=np.array(refine_phases),
                refine_metrics_dbc=np.array(refine_metrics),
                dark_orders=np.array(dark_orders),
                flatness_orders=np.array(flatness_orders),
                mode=np.array(mode),
                target_orders=np.array(target_orders),
            )
            print(f"Saved trajectory only to {partial}")
            raise
        else:
            raise

    R = len(deep_spectra)
    spectra = np.array(deep_spectra)                    # (R, N_harm, K)
    cal_spectra = np.tile(deep_cal, (R, 1))             # (R, K)

    save_kwargs = dict(
        # DualToneSweepData-compatible: deep repeats mapped to the steps axis
        drive_freqs=np.ones(R) * cw_freq,
        ch1_powers_dbm=np.ones(R) * best['p1'],
        ch2_powers_dbm=np.ones(R) * best['p2'],
        ch1_phases_deg=np.zeros(R),
        ch2_phases_deg=np.ones(R) * best['phi2'],
        harmonics=np.array(deep_harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra,
        cal_spectra=cal_spectra,
        n_sweep_repeats=np.array(1),
        ref_freq=np.array(cw_freq),
        ref_cal_spectrum=deep_cal,
        ch1_enabled=np.array(True),
        ch2_enabled=np.array(True),
        # optimization result + trajectory
        best_ch1_dbm=np.array(best['p1']),
        best_ch2_dbm=np.array(best['p2']),
        best_ch2_phase_deg=np.array(best['phi2']),
        best_metric_dbc=np.array(best['metric']),
        mode=np.array(mode),
        dark_orders=np.array(dark_orders),
        flatness_orders=np.array(flatness_orders),
        target_orders=np.array(target_orders),
        maximize_order=np.array(np.nan if maximize_order is None else maximize_order),
        ch1_power_bounds_dbm=np.array([p1_lo, p1_hi]),
        ch2_power_bounds_dbm=np.array([p2_lo, p2_hi]),
        power_guess_dbm=np.array([g1, g2]),
        search_in_voltage=np.array(search_in_voltage),
        stop_at_metric_dbc=np.array(
            np.nan if stop_at_metric_dbc is None else stop_at_metric_dbc),
        early_stopped=np.array(early_stop),
        opt_ch1_dbm=np.array(traj_p1),
        opt_ch2_dbm=np.array(traj_p2),
        opt_metric_dbc=np.array(traj_metric),
        opt_best_phase_deg=np.array(traj_best_phase),
        opt_phase_metrics_dbc=np.array(traj_phase_metrics),
        opt_phase_grid_deg=phase_grid,
        refine_phases_deg=np.array(refine_phases),
        refine_metrics_dbc=np.array(refine_metrics),
    )
    np.savez_compressed(full_path, **save_kwargs)
    print(
        f"Done. P1*={best['p1']:+.2f} dBm, P2*={best['p2']:+.2f} dBm, "
        f"phi2*={best['phi2']:.2f} deg, metric {best['metric']:+.2f} {unit_label}\n"
        f"Saved {R} deep repeat(s) to {full_path}"
    )

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        evals = np.arange(1, len(traj_metric) + 1)
        ax1.plot(evals, traj_metric, marker='o', markersize=3, label='power stage')
        if refine_metrics:
            r_evals = len(traj_metric) + np.arange(1, len(refine_metrics) + 1)
            ax1.plot(r_evals, refine_metrics, marker='.', linestyle='none',
                     label='phase refinement')
        ax1.axhline(best['metric'], color='k', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('Evaluation')
        if mode == 'minimize':
            ylabel = 'max CE over dark orders [dBc]'
        elif mode == 'maximize':
            ylabel = f'CE at order {target_orders[0]} [dBc]'
        else:
            ylabel = f'CE spread over {flatness_orders} [dB]'
        ax1.set_ylabel(ylabel)
        ax1.legend()

        peaks_dbm = spectra.max(axis=-1)                 # (R, N_harm)
        cal_peak = float(deep_cal.max())
        ce_dbc = peaks_dbm - cal_peak
        reps = np.arange(1, R + 1)
        for j, n in enumerate(deep_harmonics):
            ax2.plot(reps, ce_dbc[:, j], marker='o', markersize=3, label=f'n={n}')
        ax2.set_xlabel('Deep repeat')
        ax2.set_ylabel('Sideband power [dBc]')
        ax2.legend(fontsize=8, ncol=2)

        plt.tight_layout()
        plt.show()

    return full_path


def main():
    DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\comb_finding"

    bnc_dark_window_optimizer(
        cw_freq=1.130e9,
        ch1_power_min_dbm=19,
        ch1_power_max_dbm=26,
        ch2_power_min_dbm=0,
        ch2_power_max_dbm=19,
        ch1_power_guess_dbm=21,
        ch2_power_guess_dbm=12,
        n_phases=36,
        mode='maximize',   # or 'maximize' with maximize_order=<int>,
                            # or 'flatness' with flatness_orders=<list of int>
        dark_orders=(0, -1),
        maximize_order=1,
        flatness_orders=[-3, -2, -1, 0, 1, 2, 3],
        deep_harmonics=(-3, -2, -1, 0, 1, 2, 3),
        n_deep_repeats=100,
        power_initial_step_dbm=0.5,
        power_xatol_dbm=0.05,
        power_fatol_db=0.25,
        power_max_evals=30,
        search_in_voltage=True,
        stop_at_metric_dbc=1,
        eval_phase_refine_iters=8,
        phase_tol_deg=0.2,
        phase_max_iters=25,
        heterodyne_shift=125e6,
        window_hz=1e3,
        esa_freq_step=1e3 / 1001,
        esa_res_bw=3e3,
        esa_ref_level=-40,
        settle_time_s=0.05,
        averages_per_point=1,
        use_cxa=True,
        data_folder=DATA_FOLDER,
        optional_name='',
    )


if __name__ == '__main__':
    main()
