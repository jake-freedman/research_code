"""
BNC 855B fixed-frequency power sweep with heterodyne harmonic-tracking ESA.

Mirrors vna_power_harmonic_esa_script.py but uses the BNC 855B-12 signal
generator (channel 1) as the CW source instead of the VNA.

The BNC is held at a fixed CW frequency and stepped through a list of output
powers. At each power level the ESA records a narrow spectrum centred on each
requested harmonic (n*f_cw + heterodyne_shift). All data is saved as a single
.npz file that is fully compatible with PowerHeterodyneSweepData in
power_harmonic_sweep_data.py.
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
from power_harmonic_sweep_data import PowerHeterodyneSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\calibration_sweep"


def voltage_linspace(p_start_dbm: float, p_stop_dbm: float, n: int) -> np.ndarray:
    """
    Return n powers in dBm that are evenly spaced in RMS voltage (50 Ω sinusoid).

    Converts the dBm bounds to voltages, spaces those linearly, then converts
    back to dBm. Pass the result directly as cw_powers to bnc_power_heterodyne_sweep.

    Parameters
    ----------
    p_start_dbm, p_stop_dbm : float
        Power bounds in dBm defining the voltage range.
    n : int
        Number of steps.
    """
    _log20 = 10.0 * np.log10(20.0)
    v_start = 10.0 ** ((p_start_dbm - _log20) / 20.0)
    v_stop  = 10.0 ** ((p_stop_dbm  - _log20) / 20.0)
    v_rms = np.linspace(v_start, v_stop, n)
    return 20.0 * np.log10(v_rms) + _log20


def bnc_power_heterodyne_sweep(
    cw_freq: float,
    cw_powers,
    heterodyne_shift: float = 125e6,
    harmonics=(0, 1),
    comb_spacing: float = None,
    n_repeats: int = 1,
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    output_enable_settle_s: float = 0.0,
    optional_name: str = '',
    use_cxa: bool = False,
    per_step_calibration: bool = False,
    homodyne: bool = False,
    plot: bool = True,
) -> str:
    """
    Step the BNC 855B through output powers (channel 1) at a fixed CW frequency
    and record a narrow ESA spectrum centred on n*comb_spacing + heterodyne_shift
    for each harmonic.

    Data is saved in the same .npz format as vna_power_heterodyne_sweep() and
    can be loaded by PowerHeterodyneSweepData.

    Parameters
    ----------
    cw_freq : float
        Fixed BNC channel 1 drive frequency in Hz.
    cw_powers : array-like
        BNC channel 1 output powers in dBm.
    heterodyne_shift : float
        Offset of the LO from the signal in Hz. Default 125 MHz.
    harmonics : sequence of int
        Harmonic numbers to record. 0 = carrier beat. Default (0, 1).
    comb_spacing : float, optional
        Frequency spacing between comb teeth in Hz. Sideband n is looked up at
        n*comb_spacing + heterodyne_shift. Defaults to cw_freq, which recovers
        the original behaviour when drive and comb spacing are the same.
    n_repeats : int
        Number of times to repeat the full power sweep. Default 1 (single pass).
        Saved spectra have shape (n_repeats, M, N, K); analysis plots show
        individual repeats as faint traces with the mean as the main line.
    window_hz : float
        Half-width of each harmonic window in Hz. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window in Hz. Default 250 kHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 10 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait time after setting each power level, and after re-enabling the
        RF output following per_step_calibration. Default 0.1 s.
    output_enable_settle_s : float
        Extra delay inserted right before the harmonics loop begins at each
        step, on top of settle_time_s. harmonics is a fixed list, so whichever
        order is first always gets measured with the least elapsed time since
        the most recent disturbance (the power step, or the RF output being
        toggled off/on for per_step_calibration) -- any switching/settling
        transient from that disturbance disproportionately corrupts that one
        order, while later orders benefit from the time taken by the
        intervening sweeps. This parameter gives the transient extra time to
        decay before the loop starts, independent of settle_time_s so it can
        be tuned without slowing down every other wait. Default 0.0 (no
        extra delay -- original behaviour).
    optional_name : str
        Label prepended to the saved filename.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA. Default False.
    per_step_calibration : bool
        If True, the RF output is briefly turned off at each step to record a
        carrier-beat spectrum at heterodyne_shift. These are saved as
        'cal_spectra' (M, K) and enable dBc / percent normalisation in
        the analysis script. Ignored when homodyne=True. Default False.
    homodyne : bool
        If True, look for beatnotes at abs(n * comb_spacing) rather than
        abs(n * comb_spacing + heterodyne_shift). Use when there is no AOM
        frequency offset (direct detection). Note: n=0 falls at DC and cannot
        be measured; per_step_calibration is also unavailable. Default False.
    plot : bool
        If True, plot peak powers and modulation depth after saving. Default True.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    if comb_spacing is None:
        comb_spacing = cw_freq
    cw_powers = np.asarray(cw_powers)
    harmonics = list(harmonics)
    if homodyne and per_step_calibration:
        print("Warning: per_step_calibration is unavailable in homodyne mode (carrier is at DC); ignoring.")
        per_step_calibration = False
    comb_note = (
        f", comb spacing {comb_spacing / 1e9:.4f} GHz"
        if comb_spacing != cw_freq else ""
    )
    repeat_note = f", {n_repeats} repeats" if n_repeats > 1 else ""
    detection_note = "homodyne" if homodyne else f"shift {heterodyne_shift / 1e6:.1f} MHz"
    print(
        f"Starting BNC power sweep: {len(cw_powers)} steps "
        f"({cw_powers[0]:+.1f} to {cw_powers[-1]:+.1f} dBm) "
        f"at {cw_freq / 1e9:.4f} GHz{comb_note}{repeat_note}, "
        f"harmonics {harmonics}, {detection_note}"
        + (", per-step calibration ON" if per_step_calibration else "")
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}power_harmonic_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    # all_repeats_spectra: list of completed (M, N, K) arrays, one per repeat
    # all_repeats_cal:     list of completed (M, K) arrays, one per repeat
    all_repeats_spectra = []
    all_repeats_cal = []
    offsets_hz = None
    K = None

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:
            sig.disable_all_outputs()
            sig.configure_channel(1, cw_freq, cw_powers[0])
            sig.enable_output(1)

            for r in range(n_repeats):
                repeat_spectra = []   # M lists, each N arrays
                repeat_cal = []       # M arrays

                for i, power in enumerate(cw_powers):
                    sig.set_power(1, power)
                    time.sleep(settle_time_s)

                    if per_step_calibration:
                        sig.disable_output(1)
                        esa.configure(
                            start_freq=heterodyne_shift - window_hz,
                            stop_freq=heterodyne_shift + window_hz,
                            freq_step=esa_freq_step,
                            res_bw=esa_res_bw,
                            ref_level=esa_ref_level,
                            attenuation=0.0,
                        )
                        _, cal_pwr = esa.sweep()
                        repeat_cal.append(cal_pwr)
                        if offsets_hz is None:
                            K = len(cal_pwr)
                            offsets_hz = np.linspace(-window_hz, window_hz, K)
                        sig.enable_output(1)
                        time.sleep(settle_time_s)

                    # Extra settle right before the harmonics loop so the
                    # order measured first (fixed by `harmonics`) doesn't
                    # disproportionately catch the tail of the power-step /
                    # output-toggle transient. See output_enable_settle_s.
                    time.sleep(output_enable_settle_s)

                    harmonic_spectra = []
                    for n in harmonics:
                        center = abs(n * comb_spacing) if homodyne else abs(n * comb_spacing + heterodyne_shift)
                        esa.configure(
                            start_freq=center - window_hz,
                            stop_freq=center + window_hz,
                            freq_step=esa_freq_step,
                            res_bw=esa_res_bw,
                            ref_level=esa_ref_level,
                            attenuation=0.0,
                        )
                        _, power_db = esa.sweep()
                        harmonic_spectra.append(power_db)
                        if offsets_hz is None:
                            K = len(power_db)
                            offsets_hz = np.linspace(-window_hz, window_hz, K)

                    repeat_spectra.append(harmonic_spectra)
                    rep_label = f"Rep {r + 1}/{n_repeats}, " if n_repeats > 1 else ""
                    print(f"{rep_label}step {i + 1}/{len(cw_powers)}: {power:+.1f} dBm done.")

                # Repeat fully completed — store it.
                all_repeats_spectra.append(
                    np.array([[s[:K] for s in row] for row in repeat_spectra])
                )
                if per_step_calibration and repeat_cal:
                    all_repeats_cal.append(np.array([s[:K] for s in repeat_cal]))

            sig.disable_output(1)

    except Exception as exc:
        r_done = len(all_repeats_spectra)
        print(f"ERROR during repeat {r_done + 1}/{n_repeats}: {exc}")
        if r_done == 0:
            raise
        print(f"Saving {r_done}/{n_repeats} completed repeat(s)...")

    R_done = len(all_repeats_spectra)
    spectra_arr = np.array(all_repeats_spectra)  # (R, M, N, K)

    save_kwargs = dict(
        cw_freq=np.array(cw_freq),
        comb_spacing=np.array(comb_spacing),
        n_repeats=np.array(n_repeats),
        cw_powers=cw_powers,
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(0.0 if homodyne else heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_arr,
    )
    if per_step_calibration and all_repeats_cal:
        save_kwargs['cal_spectra'] = np.array(all_repeats_cal)  # (R, M, K)

    np.savez_compressed(full_path, **save_kwargs)

    print(f"Done. Saved {R_done}/{n_repeats} repeat(s) to {full_path}")
    if plot:
        data = PowerHeterodyneSweepData.from_file(full_path)
        data.plot_peak_powers()
        data.plot_modulation_depth()
        plt.show()
    return full_path


def main():
    cw_powers = voltage_linspace(-10, 25, 100)
    # 25 max for f1, 20 max for f2

    bnc_power_heterodyne_sweep(
        cw_freq=1*1.120e9,
        homodyne=False,
        cw_powers=cw_powers,
        heterodyne_shift=125e6,
        harmonics=(-3, -2, -1, 0, 1, 2, 3),
        window_hz=1e3,
        esa_freq_step=2e3/1001,
        esa_res_bw=10e3,
        esa_ref_level=-40,
        settle_time_s=0.01,
        output_enable_settle_s=0.1,
        optional_name='wg21_power',
        use_cxa=True,
        per_step_calibration=True,
        n_repeats=5
    )


if __name__ == '__main__':
    main()
