"""
BNC 855B phase reset reproducibility test.

At each of N iterations the script performs a deliberate reset sequence and
then measures the dual-tone harmonic sideband powers:

  1. Set both channel amplitudes to 0 dBm (outputs remain enabled).
  2. Restore amplitudes to CH1_POWER_DBM and CH2_POWER_DBM.
  3. Set ch1 phase to 0° and ch2 phase to 0°.
  4. Set ch2 phase to CH2_PHASE_DEG.
  5. Turn outputs off briefly to record a carrier calibration spectrum.
  6. Re-enable outputs and record sideband spectra at each requested harmonic.

Each step in the settle_time_s wait applies after each amplitude or phase
change and after re-enabling outputs following calibration.

The saved .npz is fully compatible with DualToneSweepData. The N iterations
map to the M-steps axis; drive_freqs, ch1/ch2 powers, and ch1/ch2 phases are
constant arrays of length N.
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


def bnc_phase_reset_test(
    cw_freq: float,
    ch1_power_dbm: float,
    ch2_power_dbm: float,
    ch2_phase_deg: float,
    n_iterations: int,
    harmonics=(1, 2, 3),
    heterodyne_shift: float = 125e6,
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    data_folder: str = '.',
    optional_name: str = '',
    use_cxa: bool = False,
    averages_per_point: int = 1,
    n_sweep_repeats: int = 1,
    plot: bool = True,
) -> str:
    """
    Phase reset reproducibility test.

    Parameters
    ----------
    cw_freq : float
        Ch1 drive frequency in Hz. Ch2 is driven at 2*cw_freq.
    ch1_power_dbm : float
        Ch1 output power in dBm (restored to this after each reset).
    ch2_power_dbm : float
        Ch2 output power in dBm (restored to this after each reset).
    ch2_phase_deg : float
        Ch2 phase in degrees applied at the end of each reset sequence.
        Ch1 phase is always held at 0°.
    n_iterations : int
        Number of reset-and-measure trials per repeat.
    harmonics : sequence of int
        Harmonic orders n to record. ESA windows at abs(n*cw_freq + shift).
    heterodyne_shift : float
        LO offset in Hz. Default 125 MHz.
    window_hz : float
        Half-width of each ESA window in Hz. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window in Hz. Default 250 kHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 10 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait after each amplitude or phase change, and after re-enabling
        outputs following calibration. Default 0.1 s.
    data_folder : str
        Directory in which to save the .npz file. Default current directory.
    optional_name : str
        Label prepended to the saved filename.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA. Default False.
    averages_per_point : int
        ESA sweeps to average at each measurement point (linear). Default 1.
    n_sweep_repeats : int
        Number of times to repeat the full n_iterations sequence. When > 1,
        per-repeat spectra are saved as spectra_all (R, N, n_harm, K) and the
        linear mean across repeats is saved as spectra (N, n_harm, K).
        Default 1.
    plot : bool
        If True, plot sideband CE [dBc] vs iteration after saving. Default True.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    harmonics = list(harmonics)
    N = n_iterations

    print(
        f"Phase reset test: {N} iterations × {n_sweep_repeats} repeat(s)\n"
        f"  cw_freq    : {cw_freq / 1e9:.4f} GHz  (ch2 at {2 * cw_freq / 1e9:.4f} GHz)\n"
        f"  ch1 power  : {ch1_power_dbm:+.2f} dBm\n"
        f"  ch2 power  : {ch2_power_dbm:+.2f} dBm\n"
        f"  ch2 phase  : {ch2_phase_deg:.2f} deg\n"
        f"  harmonics  : {harmonics}\n"
        f"  shift      : {heterodyne_shift / 1e6:.1f} MHz\n"
        f"  averages   : {averages_per_point} per point"
    )

    os.makedirs(data_folder, exist_ok=True)
    fname = (
        f'{optional_name}phase_reset_test_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(data_folder, fname)

    offsets_hz = None
    K = None
    ref_cal_spectrum = None
    repeats_spectra = []
    repeats_cal = []
    current_spectra = []
    current_cal = []

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:

            # ----------------------------------------------------------------
            # Preamble: RF-off carrier calibration before the first iteration.
            # ----------------------------------------------------------------
            sig.configure_channel(1, cw_freq, ch1_power_dbm, 0.0)
            sig.configure_channel(2, 2.0 * cw_freq, ch2_power_dbm, ch2_phase_deg)
            sig.disable_all_outputs()

            print("Preamble: RF-off carrier calibration ...")
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
            print(f"  Carrier: {float(ref_cal_spectrum.max()):.2f} dBm\n")

            sig.enable_all_outputs()
            time.sleep(settle_time_s)

            # ----------------------------------------------------------------
            # Main loop
            # ----------------------------------------------------------------
            for r in range(n_sweep_repeats):
                if n_sweep_repeats > 1:
                    print(f"--- Repeat {r + 1}/{n_sweep_repeats} ---")
                current_spectra = []
                current_cal = []

                for i in range(N):
                    # ── Reset sequence ────────────────────────────────────────
                    sig.set_power(1, 0.0)
                    sig.set_power(2, 0.0)
                    time.sleep(settle_time_s)

                    sig.set_power(1, ch1_power_dbm)
                    sig.set_power(2, ch2_power_dbm)
                    time.sleep(settle_time_s)

                    sig.set_phase(1, 0.0)
                    sig.set_phase(2, 0.0)
                    time.sleep(settle_time_s)

                    sig.set_phase(2, ch2_phase_deg)
                    time.sleep(settle_time_s)

                    # ── Per-iteration carrier calibration ─────────────────────
                    sig.disable_all_outputs()
                    esa.configure(
                        start_freq=heterodyne_shift - window_hz,
                        stop_freq=heterodyne_shift + window_hz,
                        freq_step=esa_freq_step, res_bw=esa_res_bw,
                        ref_level=esa_ref_level, attenuation=0.0,
                    )
                    current_cal.append(_sweep_avg(esa, averages_per_point)[:K])
                    sig.enable_all_outputs()
                    time.sleep(settle_time_s)

                    # ── Harmonic spectra ──────────────────────────────────────
                    harmonic_spectra = []
                    for n in harmonics:
                        center = abs(n * cw_freq + heterodyne_shift)
                        esa.configure(
                            start_freq=center - window_hz,
                            stop_freq=center + window_hz,
                            freq_step=esa_freq_step, res_bw=esa_res_bw,
                            ref_level=esa_ref_level, attenuation=0.0,
                        )
                        harmonic_spectra.append(_sweep_avg(esa, averages_per_point)[:K])
                    current_spectra.append(harmonic_spectra)

                    print(
                        f"  Iter {i + 1}/{N}"
                        + (f"  (repeat {r + 1})" if n_sweep_repeats > 1 else "")
                    )

                repeats_spectra.append(current_spectra)
                repeats_cal.append(current_cal)
                current_spectra = []
                current_cal = []

            sig.disable_all_outputs()
            esa.configure(
                start_freq=heterodyne_shift - window_hz,
                stop_freq=heterodyne_shift + window_hz,
                freq_step=esa_freq_step, res_bw=esa_res_bw,
                ref_level=esa_ref_level, attenuation=0.0,
            )
            esa.set_continuous(True)
            print("ESA: continuous sweep at carrier beat window.")

    except Exception as exc:
        print(f"ERROR: {exc}")
        if n_sweep_repeats == 1 and current_spectra:
            repeats_spectra.append(current_spectra)
            if current_cal:
                repeats_cal.append(current_cal)
            print(f"Saving partial run ({len(current_spectra)}/{N} iterations)...")
        elif current_spectra:
            print(
                f"Partial repeat {len(repeats_spectra) + 1} "
                f"({len(current_spectra)}/{N} iters) discarded; "
                f"saving {len(repeats_spectra)} complete repeat(s)."
            )
        if not repeats_spectra:
            raise

    if offsets_hz is None:
        raise RuntimeError("No data collected; offsets_hz not set.")

    K = len(offsets_hz)
    R = len(repeats_spectra)
    N_saved = len(repeats_spectra[0])

    # (R, N_saved, n_harmonics, K)
    spectra_all = np.array([
        [[s for s in row] for row in rpt]
        for rpt in repeats_spectra
    ])
    spectra_mean = (
        10.0 * np.log10(np.mean(10.0 ** (spectra_all / 10.0), axis=0))
        if R > 1 else spectra_all[0]
    )

    # (R, N_saved, K)
    cal_all = np.array([[s for s in rpt] for rpt in repeats_cal])
    cal_mean = (
        10.0 * np.log10(np.mean(10.0 ** (cal_all / 10.0), axis=0))
        if R > 1 else cal_all[0]
    )

    save_kwargs = dict(
        drive_freqs=np.ones(N_saved) * cw_freq,
        ch1_powers_dbm=np.ones(N_saved) * ch1_power_dbm,
        ch2_powers_dbm=np.ones(N_saved) * ch2_power_dbm,
        ch1_phases_deg=np.zeros(N_saved),
        ch2_phases_deg=np.ones(N_saved) * ch2_phase_deg,
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_mean,
        cal_spectra=cal_mean,
        n_sweep_repeats=np.array(R),
        ref_freq=np.array(cw_freq),
        ref_cal_spectrum=ref_cal_spectrum[:K],
        ch1_enabled=np.array(True),
        ch2_enabled=np.array(True),
    )
    if R > 1:
        save_kwargs['spectra_all'] = spectra_all
        save_kwargs['cal_spectra_all'] = cal_all

    np.savez_compressed(full_path, **save_kwargs)
    print(f"Done. Saved {R} repeat(s) × {N_saved}/{N} iterations to {full_path}")

    if plot:
        peaks_dbm = spectra_mean.max(axis=-1)           # (N_saved, n_harmonics)
        cal_peaks_dbm = cal_mean.max(axis=-1)            # (N_saved,)
        ce_dbc = peaks_dbm - cal_peaks_dbm[:, np.newaxis]

        fig, ax = plt.subplots()
        iters = np.arange(1, N_saved + 1)
        for j, n in enumerate(harmonics):
            ax.plot(iters, ce_dbc[:, j], marker='o', markersize=3, label=f'n={n}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Sideband power [dBc]')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return full_path


def main():
    DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5"

    bnc_phase_reset_test(
        cw_freq=1.130e9,
        ch1_power_dbm=19.93,
        ch2_power_dbm=15.23,
        ch2_phase_deg=100.0,
        n_iterations=20,
        harmonics=(-2, -1, 0, 1, 2),
        heterodyne_shift=125e6,
        window_hz=1e6,
        esa_freq_step=2e6 / 1001,
        esa_res_bw=10e3,
        esa_ref_level=-20,
        settle_time_s=0.05,
        data_folder=DATA_FOLDER,
        optional_name='phase_reset_',
        use_cxa=True,
        averages_per_point=1,
        n_sweep_repeats=1,
    )


if __name__ == '__main__':
    main()
