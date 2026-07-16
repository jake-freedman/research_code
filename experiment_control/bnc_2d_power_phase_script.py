"""
BNC 855B 2D power × phase grid sweep.

Sweeps a 2D grid over ch1 (f) and ch2 (2f) output powers, voltage-spaced
between user-supplied dBm bounds. At each grid point a phase sweep is run:
ch2 phase is stepped through a provided array while ch1 phase is held at 0.

Each grid point is saved as a separate .npz file (DualToneSweepData-compatible)
inside a timestamped sub-folder. A grid_meta.npz in that folder records the
power grid and is read by 2d_power_phase_analysis.py.
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
import os
import time
from datetime import datetime
import numpy as np

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5\comb_finding"


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


def voltage_linspace(p_start_dbm: float, p_stop_dbm: float, n: int) -> np.ndarray:
    """n powers evenly spaced in RMS voltage (50 Ω) between the two dBm bounds."""
    _log20 = 10.0 * np.log10(20.0)
    v_start = 10.0 ** ((p_start_dbm - _log20) / 20.0)
    v_stop  = 10.0 ** ((p_stop_dbm  - _log20) / 20.0)
    return 20.0 * np.log10(np.linspace(v_start, v_stop, n)) + _log20


def bnc_2d_power_phase_sweep(
    cw_freq: float,
    ch1_power_min_dbm: float,
    ch1_power_max_dbm: float,
    ch1_n_points: int,
    ch2_power_min_dbm: float,
    ch2_power_max_dbm: float,
    ch2_n_points: int,
    ch2_phases_deg,
    harmonics=(1, 2, 3),
    heterodyne_shift: float = 125e6,
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    averages_per_point: int = 1,
    n_grid_repeats: int = 1,
    use_cxa: bool = False,
    optional_name: str = '',
) -> str:
    """
    2D power × phase sweep.

    Parameters
    ----------
    cw_freq : float
        Ch1 drive frequency f in Hz. Ch2 is driven at 2*f.
    ch1_power_min_dbm, ch1_power_max_dbm : float
        Ch1 output power range in dBm. The grid is evenly spaced in RMS voltage.
    ch1_n_points : int
        Number of ch1 power grid points.
    ch2_power_min_dbm, ch2_power_max_dbm : float
        Ch2 output power range in dBm. Voltage-spaced.
    ch2_n_points : int
        Number of ch2 power grid points.
    ch2_phases_deg : array-like
        Phase values in degrees to sweep for ch2 at each grid point.
        Ch1 phase is held at 0 throughout.
    harmonics : sequence of int
        Harmonic numbers n to record. ESA windows at abs(n*f + shift).
    heterodyne_shift : float
        LO offset in Hz.
    window_hz : float
        Half-width of each ESA window in Hz.
    esa_freq_step : float
        Frequency step within each ESA window in Hz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz.
    esa_ref_level : float
        ESA reference level in dBm.
    settle_time_s : float
        Wait time after changing phase or power before measuring.
    averages_per_point : int
        ESA sweeps to average at each measurement point (linear average).
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA.
    optional_name : str
        Label prepended to the output folder name.

    Returns
    -------
    str
        Path to the folder containing grid_meta.npz and per-point .npz files.
    """
    ch1_powers = voltage_linspace(ch1_power_min_dbm, ch1_power_max_dbm, ch1_n_points)
    ch2_powers = voltage_linspace(ch2_power_min_dbm, ch2_power_max_dbm, ch2_n_points)
    ch2_phases_deg = np.asarray(ch2_phases_deg, dtype=float)
    harmonics = list(harmonics)
    P  = len(ch2_phases_deg)
    n1 = len(ch1_powers)
    n2 = len(ch2_powers)

    folder = os.path.join(
        DATA_FOLDER,
        f'{optional_name}2d_power_phase_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
    )
    os.makedirs(folder, exist_ok=True)

    print(
        f"2D power-phase sweep: {n1}×{n2} grid, {P} phase steps, {n_grid_repeats} repeat(s)\n"
        f"  ch1 : {ch1_powers.min():+.1f} – {ch1_powers.max():+.1f} dBm\n"
        f"  ch2 : {ch2_powers.min():+.1f} – {ch2_powers.max():+.1f} dBm\n"
        f"  phase: {ch2_phases_deg.min():.1f} – {ch2_phases_deg.max():.1f} deg\n"
        f"  folder: {folder}\n"
    )

    esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)

    # accumulators: all_spectra[i][j] = list of (P, N, K) arrays, one per repeat
    all_spectra  = [[[] for _ in range(n2)] for _ in range(n1)]
    all_cal      = [[[] for _ in range(n2)] for _ in range(n1)]

    with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:

        # Global preamble: one RF-off calibration before the grid starts
        sig.disable_all_outputs()
        esa.configure(
            start_freq=heterodyne_shift - window_hz,
            stop_freq=heterodyne_shift + window_hz,
            freq_step=esa_freq_step,
            res_bw=esa_res_bw,
            ref_level=esa_ref_level,
            attenuation=0.0,
        )
        ref_cal_raw = _sweep_avg(esa, averages_per_point)
        offsets_hz = np.linspace(-window_hz, window_hz, len(ref_cal_raw))
        K = len(offsets_hz)
        ref_cal_spectrum = ref_cal_raw[:K]
        print(f"Preamble J0: {float(ref_cal_spectrum.max()):.2f} dBm\n")

        sig.configure_channel(1, cw_freq, ch1_powers[0], 0.0)
        sig.configure_channel(2, 2.0 * cw_freq, ch2_powers[0], 0.0)

        for rep in range(n_grid_repeats):
            if n_grid_repeats > 1:
                print(f"\n=== Grid sweep {rep + 1}/{n_grid_repeats} ===")
            for i, ch1_pwr in enumerate(ch1_powers):
                sig.set_power(1, ch1_pwr)
                for j, ch2_pwr in enumerate(ch2_powers):
                    sig.set_power(2, ch2_pwr)
                    print(
                        f"Grid ({i + 1}/{n1}, {j + 1}/{n2}): "
                        f"ch1={ch1_pwr:+.1f} dBm, ch2={ch2_pwr:+.1f} dBm"
                    )
                    try:
                        # Per-grid-point RF-off calibration
                        sig.disable_all_outputs()
                        esa.configure(
                            start_freq=heterodyne_shift - window_hz,
                            stop_freq=heterodyne_shift + window_hz,
                            freq_step=esa_freq_step,
                            res_bw=esa_res_bw,
                            ref_level=esa_ref_level,
                            attenuation=0.0,
                        )
                        cal_spectrum = _sweep_avg(esa, averages_per_point)[:K]

                        # RF on for the phase sweep
                        sig.enable_all_outputs()
                        time.sleep(settle_time_s)

                        spectra_this = []   # grows to (P, N, K)
                        for phase in ch2_phases_deg:
                            sig.set_phase(2, float(phase))
                            time.sleep(settle_time_s)
                            harmonic_rows = []
                            for n in harmonics:
                                center = abs(n * cw_freq + heterodyne_shift)
                                esa.configure(
                                    start_freq=center - window_hz,
                                    stop_freq=center + window_hz,
                                    freq_step=esa_freq_step,
                                    res_bw=esa_res_bw,
                                    ref_level=esa_ref_level,
                                    attenuation=0.0,
                                )
                                harmonic_rows.append(_sweep_avg(esa, averages_per_point)[:K])
                            spectra_this.append(harmonic_rows)

                        sig.disable_all_outputs()

                        all_spectra[i][j].append(np.array(spectra_this))  # (P, N, K)
                        all_cal[i][j].append(cal_spectrum)                # (K,)

                    except Exception as exc:
                        print(f"  ERROR: {exc}; skipping grid point ({i}, {j}) repeat {rep}.")
                        try:
                            sig.disable_all_outputs()
                        except Exception:
                            pass

        sig.disable_all_outputs()

    # Save all grid points now that all repeats are complete
    for i, ch1_pwr in enumerate(ch1_powers):
        for j, ch2_pwr in enumerate(ch2_powers):
            reps_s = all_spectra[i][j]
            reps_c = all_cal[i][j]
            if not reps_s:
                print(f"  No data for grid ({i}, {j}); skipping save.")
                continue

            spectra_stack = np.stack(reps_s)            # (R, P, N, K)
            lin = 10.0 ** (spectra_stack / 10.0)
            spectra_mean = 10.0 * np.log10(lin.mean(axis=0))   # (P, N, K)

            cal_stack = np.stack(reps_c)                # (R, K)
            cal_lin = 10.0 ** (cal_stack / 10.0)
            cal_mean = 10.0 * np.log10(cal_lin.mean(axis=0))   # (K,)

            fname = os.path.join(folder, f'grid_{i:02d}_{j:02d}.npz')
            save_kwargs = dict(
                drive_freqs=np.ones(P) * cw_freq,
                ch1_powers_dbm=np.ones(P) * ch1_pwr,
                ch2_powers_dbm=np.ones(P) * ch2_pwr,
                ch1_phases_deg=np.zeros(P),
                ch2_phases_deg=ch2_phases_deg,
                harmonics=np.array(harmonics),
                heterodyne_shift=np.array(heterodyne_shift),
                window_hz=np.array(window_hz),
                esa_freq_step_hz=np.array(esa_freq_step),
                offsets_hz=offsets_hz,
                spectra=spectra_mean,
                cal_spectra=np.tile(cal_mean, (P, 1)),
                n_sweep_repeats=np.array(len(reps_s)),
                ref_freq=np.array(cw_freq),
                ref_cal_spectrum=ref_cal_spectrum,
                ch1_enabled=np.array(True),
                ch2_enabled=np.array(True),
                grid_i=np.array(i),
                grid_j=np.array(j),
            )
            if len(reps_s) > 1:
                save_kwargs['spectra_all'] = spectra_stack                          # (R, P, N, K)
                save_kwargs['cal_spectra_all'] = np.tile(
                    cal_stack[:, np.newaxis, :], (1, P, 1)                          # (R, P, K)
                )
            np.savez_compressed(fname, **save_kwargs)
            print(f"  saved → {fname}  ({len(reps_s)} repeat(s))")

    np.savez_compressed(
        os.path.join(folder, 'grid_meta.npz'),
        ch1_powers_dbm=ch1_powers,
        ch2_powers_dbm=ch2_powers,
        cw_freq=np.array(cw_freq),
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        ch2_phases_deg=ch2_phases_deg,
        n_grid_repeats=np.array(n_grid_repeats),
    )

    print(f"\nDone. Results in: {folder}")
    return folder


def main():
    P = 18
    bnc_2d_power_phase_sweep(
        cw_freq=1.130e9,
        ch1_power_min_dbm=18,
        ch1_power_max_dbm=21.5,
        ch1_n_points=5,
        ch2_power_min_dbm=13,
        ch2_power_max_dbm=17,
        ch2_n_points=5,
        ch2_phases_deg=np.linspace(0, 360, P, endpoint=True),
        harmonics=(-3,-2,-1,0,1,2,3),
        heterodyne_shift=125e6,
        window_hz=2e6,
        esa_freq_step=2e6/1001,
        esa_res_bw=10e3,
        esa_ref_level=-40,
        settle_time_s=0.05,
        averages_per_point=1,
        n_grid_repeats=1,
        use_cxa=True,
    )


if __name__ == '__main__':
    main()
