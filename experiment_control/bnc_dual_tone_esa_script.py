"""
BNC 855B dual-tone sweep with heterodyne harmonic-tracking ESA.

Channel 1 drives at fundamental frequency f; channel 2 drives at 2*f.
All five sweep parameters (f, ch1/ch2 power, ch1/ch2 phase) are provided
as equal-length arrays of length M. For parameters that are not being swept,
pass a constant array: e.g. np.ones(M) * 10.0 for a fixed 10 dBm power.

Before the sweep, two single-tone preamble measurements are made:
  • ch1 on only (at drive_freqs[0]) — saved as ch1_ref_spectra (N, K)
  • ch2 on only (at 2*drive_freqs[0]) — saved as ch2_ref_spectra (N, K)
These let DualToneSweepData extract single-tone modulation depths β1 and β2.

At each step the script:
  1. Turns off both channels and records a calibration spectrum at the
     carrier beat (heterodyne_shift Hz) to track total optical comb power.
  2. Turns on both channels and records spectra at abs(n*f + heterodyne_shift)
     for each requested harmonic n. The absolute-value mapping correctly places
     negative harmonics (n < 0) at their physical ESA frequency |n|*f − shift.

Data is saved as a .npz file compatible with DualToneSweepData.
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
from dual_tone_sweep_data import DualToneSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5"


def _sweep_avg(esa, n_avg: int, return_all: bool = False):
    """
    Take n_avg ESA sweeps and return their linear-averaged power in dBm.

    When return_all=True, returns (mean_dBm, stack) where stack has shape
    (n_avg, K) with each individual sweep in dBm.
    """
    if n_avg == 1:
        _, pwr = esa.sweep()
        arr = np.asarray(pwr, dtype=float)
        return (arr, arr[np.newaxis, :]) if return_all else arr
    sweeps = []
    accum = None
    for _ in range(n_avg):
        _, pwr = esa.sweep()
        arr = np.asarray(pwr, dtype=float)
        sweeps.append(arr)
        lin = 10.0 ** (arr / 10.0)
        accum = lin if accum is None else accum + lin
    mean_db = 10.0 * np.log10(accum / n_avg)
    return (mean_db, np.stack(sweeps)) if return_all else mean_db


def bnc_dual_tone_sweep(
    drive_freqs,
    ch1_powers_dbm,
    ch2_powers_dbm,
    ch1_phases_deg,
    ch2_phases_deg,
    harmonics=(1, 2, 3),
    heterodyne_shift: float = 125e6,
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
    use_cxa: bool = False,
    ch1_enabled: bool = True,
    ch2_enabled: bool = True,
    per_step_calibration: bool = True,
    sideband_sum_calibration: bool = False,
    averages_per_point: int = 1,
    n_sweep_repeats: int = 1,
    save_raw_averages: bool = False,
    plot: bool = True,
) -> str:
    """
    Dual-tone sweep: step through arrays of drive parameters and record
    harmonic sideband spectra plus a calibration spectrum at each step.

    Parameters
    ----------
    drive_freqs : array-like, shape (M,)
        Channel 1 drive frequencies f in Hz. Channel 2 is set to 2*f.
    ch1_powers_dbm : array-like, shape (M,)
        Channel 1 output powers in dBm.
    ch2_powers_dbm : array-like, shape (M,)
        Channel 2 output powers in dBm.
    ch1_phases_deg : array-like, shape (M,)
        Channel 1 phase offsets in degrees.
    ch2_phases_deg : array-like, shape (M,)
        Channel 2 phase offsets in degrees.
    harmonics : sequence of int
        Harmonic numbers n to record. ESA windows centred at n*f + shift.
    heterodyne_shift : float
        LO offset in Hz. Default 125 MHz.
    window_hz : float
        Half-width of each ESA window in Hz. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window. Default 250 kHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 10 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait time after switching outputs on or off before sweeping. Default 0.1 s.
    optional_name : str
        Label prepended to the saved filename.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA.
    ch1_enabled : bool
        If True (default), channel 1 is active during the sweep. Set to False
        to keep ch1 completely off; the ch1-only preamble is also skipped.
    ch2_enabled : bool
        If True (default), channel 2 is active during the sweep. Set to False
        to keep ch2 completely off; the ch2-only preamble is also skipped.
    per_step_calibration : bool
        If True (default), RF is briefly turned off at each step to record the
        carrier beat as a calibration. If False, the single preamble calibration
        is broadcast across all steps and RF stays on for the entire sweep.
    sideband_sum_calibration : bool
        If True, no RF toggle is done during the sweep. After the sweep the
        total sideband power (sum of all harmonic peak powers in linear) at
        each step is compared to step 0 to infer fractional optical-power drift.
        Combined with the RF-off preamble measurement P_cal_0 this gives:
            P_cal[i] = P_cal_0 × (Σ_n P_n[i]) / (Σ_n P_n[0])
        Overrides per_step_calibration when both are True. Default False.
    averages_per_point : int
        Number of ESA sweeps to take and average at each measurement point.
        Averaging is done in linear (mW) before converting back to dBm.
        Default 1 (no averaging).
    n_sweep_repeats : int
        Number of times to repeat the full M-step sweep. When > 1, per-repeat
        spectra are saved as 'spectra_all' (R, M, N, K) and the linear mean
        across repeats is saved as 'spectra' (M, N, K). If an error aborts a
        repeat midway, that partial repeat is discarded; only n_sweep_repeats=1
        preserves partial data (matching the single-sweep behaviour).
        Default 1.
    save_raw_averages : bool
        When True and averages_per_point > 1 and n_sweep_repeats == 1, save
        every individual ESA sweep (before averaging) as 'spectra_avgs'
        (M, N, A, K). Enables per-average scatter points in the analysis
        script. Has no effect when n_sweep_repeats > 1 (use spectra_all
        instead) or when averages_per_point == 1. Default False.
    plot : bool
        If True, plot sideband peak powers after saving.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    drive_freqs    = np.asarray(drive_freqs,    dtype=float)
    ch1_powers_dbm = np.asarray(ch1_powers_dbm, dtype=float)
    ch2_powers_dbm = np.asarray(ch2_powers_dbm, dtype=float)
    ch1_phases_deg = np.asarray(ch1_phases_deg, dtype=float)
    ch2_phases_deg = np.asarray(ch2_phases_deg, dtype=float)
    harmonics = list(harmonics)
    M = len(drive_freqs)

    print(
        f"Starting dual-tone sweep: {M} steps × {n_sweep_repeats} repeat(s), "
        f"harmonics {harmonics}, shift {heterodyne_shift / 1e6:.1f} MHz\n"
        f"  f range    : {drive_freqs.min() / 1e9:.4f} – {drive_freqs.max() / 1e9:.4f} GHz\n"
        f"  ch1 power  : {ch1_powers_dbm.min():+.1f} – {ch1_powers_dbm.max():+.1f} dBm\n"
        f"  ch2 power  : {ch2_powers_dbm.min():+.1f} – {ch2_powers_dbm.max():+.1f} dBm\n"
        f"  ch1 phase  : {ch1_phases_deg.min():.1f} – {ch1_phases_deg.max():.1f} deg\n"
        f"  ch2 phase  : {ch2_phases_deg.min():.1f} – {ch2_phases_deg.max():.1f} deg\n"
        f"  averages   : {averages_per_point} per point"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}dual_tone_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    offsets_hz = None
    repeats_spectra = []   # completed repeats: list of R lists, each (M, N) of arrays
    repeats_cal = []       # completed cal repeats: list of R lists, each (M,) of arrays
    current_spectra = []   # steps collected so far in the current (possibly partial) repeat
    current_cal = []

    # Preamble reference arrays (set inside the try block; needed in finally path)
    ref_cal_spectrum = None
    ch1_ref_spectra = ch2_ref_spectra = None
    ch1_ref_carrier_spectrum = ch2_ref_carrier_spectrum = None
    ref_freq = float(drive_freqs[0])

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:

            # ----------------------------------------------------------------
            # Single-tone preamble (at first step's parameters)
            # ----------------------------------------------------------------
            sig.disable_all_outputs()
            if ch1_enabled:
                sig.configure_channel(1, ref_freq, ch1_powers_dbm[0], ch1_phases_deg[0])
            if ch2_enabled:
                sig.configure_channel(2, 2.0 * ref_freq, ch2_powers_dbm[0], ch2_phases_deg[0])

            # Collect raw (un-truncated) preamble rows; K is not known until the
            # carrier calibration below, which is done last so it is as close in
            # time as possible to the first sweep step.
            ch1_ref_rows_raw = []
            ch1_ref_carrier_raw = None
            ch2_ref_rows_raw = []
            ch2_ref_carrier_raw = None

            if ch1_enabled:
                print(f"Preamble ch1-only: f={ref_freq / 1e9:.4f} GHz, "
                      f"{ch1_powers_dbm[0]:+.1f} dBm ...")
                sig.enable_output(1)
                time.sleep(settle_time_s)
                for n in harmonics:
                    center = abs(n * ref_freq + heterodyne_shift)
                    esa.configure(
                        start_freq=center - window_hz, stop_freq=center + window_hz,
                        freq_step=esa_freq_step, res_bw=esa_res_bw,
                        ref_level=esa_ref_level, attenuation=0.0,
                    )
                    ch1_ref_rows_raw.append(_sweep_avg(esa, averages_per_point))
                esa.configure(
                    start_freq=heterodyne_shift - window_hz,
                    stop_freq=heterodyne_shift + window_hz,
                    freq_step=esa_freq_step, res_bw=esa_res_bw,
                    ref_level=esa_ref_level, attenuation=0.0,
                )
                ch1_ref_carrier_raw = _sweep_avg(esa, averages_per_point)

            if ch2_enabled:
                print(f"Preamble ch2-only: 2f={2 * ref_freq / 1e9:.4f} GHz, "
                      f"{ch2_powers_dbm[0]:+.1f} dBm ...")
                sig.disable_output(1)   # harmless if ch1 was never enabled
                sig.enable_output(2)
                time.sleep(settle_time_s)
                for n in harmonics:
                    center = abs(n * ref_freq + heterodyne_shift)
                    esa.configure(
                        start_freq=center - window_hz, stop_freq=center + window_hz,
                        freq_step=esa_freq_step, res_bw=esa_res_bw,
                        ref_level=esa_ref_level, attenuation=0.0,
                    )
                    ch2_ref_rows_raw.append(_sweep_avg(esa, averages_per_point))
                esa.configure(
                    start_freq=heterodyne_shift - window_hz,
                    stop_freq=heterodyne_shift + window_hz,
                    freq_step=esa_freq_step, res_bw=esa_res_bw,
                    ref_level=esa_ref_level, attenuation=0.0,
                )
                ch2_ref_carrier_raw = _sweep_avg(esa, averages_per_point)

            # Carrier calibration done last — immediately before the sweep starts —
            # to minimise the temporal gap between the J0 reference and step 0.
            sig.disable_all_outputs()
            print("Carrier calibration (RF off) ...")
            esa.configure(
                start_freq=heterodyne_shift - window_hz,
                stop_freq=heterodyne_shift + window_hz,
                freq_step=esa_freq_step, res_bw=esa_res_bw,
                ref_level=esa_ref_level, attenuation=0.0,
            )
            pwr = _sweep_avg(esa, averages_per_point)
            offsets_hz = np.linspace(-window_hz, window_hz, len(pwr))
            K = len(offsets_hz)
            ref_cal_spectrum = pwr[:K]

            # Truncate preamble rows to K now that K is known.
            if ch1_ref_rows_raw:
                ch1_ref_spectra = np.array([r[:K] for r in ch1_ref_rows_raw])
                ch1_ref_carrier_spectrum = ch1_ref_carrier_raw[:K]
            if ch2_ref_rows_raw:
                ch2_ref_spectra = np.array([r[:K] for r in ch2_ref_rows_raw])
                ch2_ref_carrier_spectrum = ch2_ref_carrier_raw[:K]

            print("Preamble complete.\n")

            # ----------------------------------------------------------------
            # Main sweep — RF stays on across all steps and all repeats.
            # Only phase is written between steps; freq/power only when changed.
            # ----------------------------------------------------------------
            if ch1_enabled:
                sig.configure_channel(1, drive_freqs[0], ch1_powers_dbm[0], ch1_phases_deg[0])
            if ch2_enabled:
                sig.configure_channel(2, 2.0 * drive_freqs[0], ch2_powers_dbm[0], ch2_phases_deg[0])
            if ch1_enabled and ch2_enabled:
                sig.enable_all_outputs()
            elif ch1_enabled:
                sig.enable_output(1)
            elif ch2_enabled:
                sig.enable_output(2)
            time.sleep(settle_time_s)

            # Track actual instrument state to skip redundant SCPI writes.
            prev_f       = drive_freqs[0]
            prev_ch1_pow = ch1_powers_dbm[0]
            prev_ch2_pow = ch2_powers_dbm[0]

            # Raw per-average sweeps (only collected when save_raw_averages applies)
            do_save_raw = save_raw_averages and averages_per_point > 1 and n_sweep_repeats == 1
            all_raw_avgs = []   # (M, N) list of (A, K) arrays

            for r in range(n_sweep_repeats):
                if n_sweep_repeats > 1:
                    print(f"--- Repeat {r + 1}/{n_sweep_repeats} ---")
                current_spectra = []
                current_cal = []

                for i in range(M):
                    f = drive_freqs[i]

                    if ch1_enabled:
                        if f != prev_f or ch1_powers_dbm[i] != prev_ch1_pow:
                            sig.set_frequency(1, f)
                            sig.set_power(1, ch1_powers_dbm[i])
                        sig.set_phase(1, ch1_phases_deg[i])
                    if ch2_enabled:
                        if f != prev_f or ch2_powers_dbm[i] != prev_ch2_pow:
                            sig.set_frequency(2, 2.0 * f)
                            sig.set_power(2, ch2_powers_dbm[i])
                        sig.set_phase(2, ch2_phases_deg[i])
                    prev_f       = f
                    prev_ch1_pow = ch1_powers_dbm[i]
                    prev_ch2_pow = ch2_powers_dbm[i]
                    time.sleep(settle_time_s)

                    if per_step_calibration and not sideband_sum_calibration:
                        sig.disable_all_outputs()
                        esa.configure(
                            start_freq=heterodyne_shift - window_hz,
                            stop_freq=heterodyne_shift + window_hz,
                            freq_step=esa_freq_step,
                            res_bw=esa_res_bw,
                            ref_level=esa_ref_level,
                            attenuation=0.0,
                        )
                        current_cal.append(_sweep_avg(esa, averages_per_point))
                        if ch1_enabled and ch2_enabled:
                            sig.enable_all_outputs()
                        elif ch1_enabled:
                            sig.enable_output(1)
                        elif ch2_enabled:
                            sig.enable_output(2)
                        time.sleep(settle_time_s)

                    harmonic_spectra = []
                    raw_avgs_this_step = [] if do_save_raw else None
                    for n in harmonics:
                        center = abs(n * f + heterodyne_shift)
                        esa.configure(
                            start_freq=center - window_hz,
                            stop_freq=center + window_hz,
                            freq_step=esa_freq_step,
                            res_bw=esa_res_bw,
                            ref_level=esa_ref_level,
                            attenuation=0.0,
                        )
                        if do_save_raw:
                            mean_pwr, raw_stack = _sweep_avg(esa, averages_per_point, return_all=True)
                            harmonic_spectra.append(mean_pwr)
                            raw_avgs_this_step.append(raw_stack)  # (A, K_raw)
                        else:
                            harmonic_spectra.append(_sweep_avg(esa, averages_per_point))
                    current_spectra.append(harmonic_spectra)
                    if do_save_raw:
                        all_raw_avgs.append(raw_avgs_this_step)  # list grows to M

                    print(
                        f"  Step {i + 1}/{M}"
                        + (f" (repeat {r + 1})" if n_sweep_repeats > 1 else "")
                        + f": f={f / 1e9:.4f} GHz, "
                        f"ch1={ch1_powers_dbm[i]:+.1f} dBm/{ch1_phases_deg[i]:.1f}°, "
                        f"ch2={ch2_powers_dbm[i]:+.1f} dBm/{ch2_phases_deg[i]:.1f}°"
                    )

                # Repeat complete — move to permanent storage
                repeats_spectra.append(current_spectra)
                if per_step_calibration and not sideband_sum_calibration:
                    repeats_cal.append(current_cal)
                current_spectra = []
                current_cal = []

            sig.disable_all_outputs()

            # Leave the ESA in a continuous-sweep monitoring view centred on
            # the heterodyne carrier beat so it is easy to inspect after the run.
            esa.configure(
                start_freq=heterodyne_shift - window_hz,
                stop_freq=heterodyne_shift + window_hz,
                freq_step=esa_freq_step,
                res_bw=esa_res_bw,
                ref_level=esa_ref_level,
                attenuation=0.0,
            )
            esa.set_continuous(True)
            print("ESA: continuous sweep at carrier beat window.")

    except Exception as exc:
        print(f"ERROR: {exc}")
        if n_sweep_repeats == 1 and current_spectra:
            # Single-repeat mode: preserve partial data (existing behaviour)
            repeats_spectra.append(current_spectra)
            if per_step_calibration and not sideband_sum_calibration and current_cal:
                repeats_cal.append(current_cal)
            print(f"Saving partial sweep ({len(current_spectra)}/{M} steps)...")
        elif current_spectra:
            print(
                f"Partial repeat {len(repeats_spectra) + 1} "
                f"({len(current_spectra)}/{M} steps) discarded; "
                f"saving {len(repeats_spectra)} complete repeat(s)."
            )
        if not repeats_spectra:
            raise

    if offsets_hz is None:
        raise RuntimeError("No data collected; offsets_hz not set.")

    K = len(offsets_hz)
    R = len(repeats_spectra)
    M_saved = len(repeats_spectra[0])  # steps per repeat (may be < M if partial)

    # Build (R, M_saved, N, K) array in dBm
    spectra_all = np.array([
        [[s[:K] for s in row] for row in rpt]
        for rpt in repeats_spectra
    ])
    # Linear-domain mean across repeats → (M_saved, N, K)
    spectra_mean = (
        10.0 * np.log10(np.mean(10.0 ** (spectra_all / 10.0), axis=0))
        if R > 1 else spectra_all[0]
    )

    if per_step_calibration and not sideband_sum_calibration:
        cal_all = np.array([[s[:K] for s in rpt] for rpt in repeats_cal])  # (R, M_saved, K)
        cal_mean = (
            10.0 * np.log10(np.mean(10.0 ** (cal_all / 10.0), axis=0))
            if R > 1 else cal_all[0]
        )
    elif sideband_sum_calibration:
        # No RF toggle: estimate per-step optical power from the sideband sum.
        # P_cal[i] = P_cal_0 × P_proxy[i] / P_proxy_ref
        # where P_proxy_ref is the sideband sum at step 0 of the very first sweep
        # (repeat 0), used as a fixed reference across all repeats.
        P_cal_0_lin = 10.0 ** (float(ref_cal_spectrum[:K].max()) / 10.0)  # mW
        peaks_lin_all = 10.0 ** (spectra_all.max(axis=3) / 10.0)  # (R, M_saved, N)
        P_proxy_all = peaks_lin_all.sum(axis=2)                    # (R, M_saved)
        P_proxy_ref = float(P_proxy_all[0, 0])                     # step 0 of repeat 0
        P_proxy_mean = P_proxy_all.mean(axis=0)                    # (M_saved,) mean across repeats
        cal_est_dbm = 10.0 * np.log10(P_cal_0_lin * P_proxy_mean / P_proxy_ref)
        cal_mean = np.tile(cal_est_dbm[:, np.newaxis], (1, K))     # (M_saved, K)
    else:
        cal_mean = np.tile(ref_cal_spectrum[:K], (M_saved, 1))

    save_kwargs = dict(
        drive_freqs=drive_freqs[:M_saved],
        ch1_powers_dbm=ch1_powers_dbm[:M_saved],
        ch2_powers_dbm=ch2_powers_dbm[:M_saved],
        ch1_phases_deg=ch1_phases_deg[:M_saved],
        ch2_phases_deg=ch2_phases_deg[:M_saved],
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_mean,
        cal_spectra=cal_mean,
        n_sweep_repeats=np.array(R),
        ref_freq=np.array(ref_freq),
        ref_cal_spectrum=ref_cal_spectrum[:K],
        ch1_enabled=np.array(ch1_enabled),
        ch2_enabled=np.array(ch2_enabled),
    )
    if ch1_ref_spectra is not None:
        save_kwargs['ch1_ref_spectra'] = ch1_ref_spectra
        save_kwargs['ch1_ref_carrier_spectrum'] = ch1_ref_carrier_spectrum
    if ch2_ref_spectra is not None:
        save_kwargs['ch2_ref_spectra'] = ch2_ref_spectra
        save_kwargs['ch2_ref_carrier_spectrum'] = ch2_ref_carrier_spectrum
    if R > 1:
        save_kwargs['spectra_all'] = spectra_all  # (R, M, N, K)
        if per_step_calibration and not sideband_sum_calibration:
            save_kwargs['cal_spectra_all'] = cal_all  # (R, M, K)
    if do_save_raw and all_raw_avgs:
        # all_raw_avgs: list of M_saved items, each a list of N arrays (A, K_raw)
        spectra_avgs = np.array([
            [step[j][:, :K] for j in range(len(harmonics))]
            for step in all_raw_avgs
        ])  # (M_saved, N, A, K)
        save_kwargs['spectra_avgs'] = spectra_avgs
        save_kwargs['n_averages_per_point'] = np.array(averages_per_point)

    np.savez_compressed(full_path, **save_kwargs)
    print(f"Done. Saved {R} repeat(s) × {M_saved}/{M} steps to {full_path}")
    if plot:
        data = DualToneSweepData.from_file(full_path)
        data.plot_peak_powers()
        plt.show()
    return full_path


def main():

    M = 360
    f0 = 1.130e9

    # Example: sweep channel 2 phase 0–360°, all other params fixed
    drive_freqs    = np.ones(M) * f0
    ch1_powers_dbm = np.ones(M) * (23.8) # max 25
    # ch1_powers_dbm = np.linspace(0, 23.5, M)
    ch2_powers_dbm = np.ones(M) * 11 # max 20
    # ch2_powers_dbm = np.linspace(-10, 15, M)
    ch1_phases_deg = np.ones(M) * 0.0
    # ch2_phases_deg = np.ones(M) * 100
    ch2_phases_deg = np.linspace(0, 360, M, endpoint=True)

    bnc_dual_tone_sweep(
        drive_freqs=drive_freqs,
        ch1_powers_dbm=ch1_powers_dbm,
        ch2_powers_dbm=ch2_powers_dbm,
        ch1_phases_deg=ch1_phases_deg,
        ch2_phases_deg=ch2_phases_deg,
        harmonics=(-2, -1,0,1, 2),
        heterodyne_shift=125e6,
        window_hz=1e6,
        esa_freq_step=2e6/1001,
        esa_res_bw=10e3,
        esa_ref_level=-20,
        settle_time_s=0.05,
        optional_name='phase_sweep_',
        use_cxa=True,
        per_step_calibration=False,
        averages_per_point=1,
        n_sweep_repeats=1,
        sideband_sum_calibration=True,
        ch2_enabled=True,
        ch1_enabled=True,
        save_raw_averages=True,
    )


if __name__ == '__main__':
    main()
