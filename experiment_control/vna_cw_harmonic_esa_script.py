"""
VNA CW frequency sweep with harmonic-tracking ESA.

For each VNA CW frequency the ESA records a narrow spectrum centred on each
requested harmonic of that frequency. All data is saved as a single .npz file
that can be loaded by HarmonicSweepData in harmonic_sweep_data.py.
"""

from vna_control import VNA
from esa_control import ESA
from cxa_control import CXA
from harmonic_sweep_data import HarmonicSweepData
from heterodyne_sweep_data import HeterodyneSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w3_d2-3_wg5b_p5"


def _reset_vna(vna) -> None:
    """Return the VNA to a safe low-power frequency sweep on continuous trigger."""
    vna.configure(
        start_freq=100e6,
        stop_freq=5e9,
        freq_step=10e6,
        power_dbm=-20.0,
    )
    vna._inst.write('SENS1:SWE:MODE CONT')


def vna_cw_harmonic_sweep(
    cw_freqs,
    cw_power: float,
    harmonics=(0, 1),
    window_hz: float = 2e6,
    esa_freq_step: float = 1e6,
    esa_res_bw: float = 1000e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
    use_cxa: bool = False,
    plot: bool = True,
) -> str:
    """
    Step the VNA through CW frequencies and record a narrow ESA spectrum around
    each harmonic of the drive frequency at every step.

    Parameters
    ----------
    cw_freqs : array-like
        VNA CW frequencies in Hz.
    cw_power : float
        VNA output power in dBm.
    harmonics : sequence of int
        Harmonic numbers to record. Default (1, 2, 3).
    window_hz : float
        Half-width of each harmonic window in Hz. The ESA sweeps
        [n*f_cw - window_hz, n*f_cw + window_hz]. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each harmonic window in Hz. Default 1 MHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 100 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait time after setting each CW frequency. Default 0.1 s.
    optional_name : str
        Label prepended to the saved filename.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    cw_freqs = np.asarray(cw_freqs)
    harmonics = list(harmonics)
    print(
        f"Starting harmonic sweep: {len(cw_freqs)} CW steps "
        f"({cw_freqs[0] / 1e9:.4f} to {cw_freqs[-1] / 1e9:.4f} GHz), "
        f"harmonics {harmonics}, window ±{window_hz / 1e6:.1f} MHz"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}harmonic_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    # all_spectra[i][j] = 1-D power array for cw_freqs[i], harmonics[j]
    all_spectra = []
    offsets_hz = None  # determined from the first sweep

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with VNA(VNA_RESOURCE_STRING) as vna, esa_cls(esa_addr) as esa:
            vna.set_cw_mode(cw_freqs[0], cw_power)

            for i, f_cw in enumerate(cw_freqs):
                vna.set_cw_freq(f_cw)
                time.sleep(settle_time_s)

                harmonic_spectra = []
                for n in harmonics:
                    center = n * f_cw
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

                all_spectra.append(harmonic_spectra)
                print(
                    f"Step {i + 1}/{len(cw_freqs)}: "
                    f"{f_cw / 1e9:.4f} GHz done."
                )

            _reset_vna(vna)

    except Exception as exc:
        print(f"ERROR at step {len(all_spectra) + 1}/{len(cw_freqs)}: {exc}")
        if not all_spectra:
            raise
        print(f"Saving partial data ({len(all_spectra)} of {len(cw_freqs)} steps)...")

    # Stack to (M, N, K), truncating to the reference K if any sweep returned
    # a different length (instrument snapping).
    K = len(offsets_hz)
    spectra_arr = np.array([[s[:K] for s in row] for row in all_spectra])
    completed_freqs = cw_freqs[:len(all_spectra)]

    np.savez_compressed(
        full_path,
        cw_freqs=completed_freqs,
        harmonics=np.array(harmonics),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_arr,
    )

    print(
        f"Done. Saved {len(all_spectra)}/{len(cw_freqs)} steps to {full_path}"
    )
    if plot:
        data = HarmonicSweepData.from_file(full_path)
        data.plot_modulation_depth()
        data.plot_harmonic_spectra()
        plt.show()
    return full_path


def vna_cw_heterodyne_sweep(
    cw_freqs,
    cw_power: float,
    heterodyne_shift: float = 125e6,
    harmonics=(0, 1, 2, 3),
    window_hz: float = 2e6,
    esa_freq_step: float = 1e6,
    esa_res_bw: float = 100e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
    use_cxa: bool = False,
    plot: bool = True,
) -> str:
    """
    Step the VNA through CW frequencies and record a narrow ESA spectrum centred
    on n*f_cw + heterodyne_shift for each requested harmonic n.

    Harmonic 0 records the carrier beat at heterodyne_shift itself, which is
    needed to extract modulation depth via J1(β)/J0(β).

    Parameters
    ----------
    cw_freqs : array-like
        VNA CW frequencies in Hz.
    cw_power : float
        VNA output power in dBm.
    heterodyne_shift : float
        Offset of the LO from the signal, in Hz. ESA centre for harmonic n is
        n*f_cw + heterodyne_shift. Default 125 MHz.
    harmonics : sequence of int
        Harmonic numbers to record. 0 = carrier beat. Default (0, 1, 2, 3).
    window_hz : float
        Half-width of each harmonic window in Hz. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window in Hz. Default 1 MHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 100 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait time after setting each CW frequency. Default 0.1 s.
    optional_name : str
        Label prepended to the saved filename.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    cw_freqs = np.asarray(cw_freqs)
    harmonics = list(harmonics)
    print(
        f"Starting heterodyne sweep: {len(cw_freqs)} CW steps "
        f"({cw_freqs[0] / 1e9:.4f} to {cw_freqs[-1] / 1e9:.4f} GHz), "
        f"harmonics {harmonics}, shift {heterodyne_shift / 1e6:.1f} MHz, "
        f"window ±{window_hz / 1e6:.1f} MHz"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}heterodyne_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    all_spectra = []
    offsets_hz = None

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with VNA(VNA_RESOURCE_STRING) as vna, esa_cls(esa_addr) as esa:
            vna.set_cw_mode(cw_freqs[0], cw_power)

            for i, f_cw in enumerate(cw_freqs):
                vna.set_cw_freq(f_cw)

                harmonic_spectra = []
                for n in harmonics:
                    center = n * f_cw + heterodyne_shift
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

                    time.sleep(settle_time_s)

                all_spectra.append(harmonic_spectra)
                print(
                    f"Step {i + 1}/{len(cw_freqs)}: "
                    f"{f_cw / 1e9:.4f} GHz done."
                )

            _reset_vna(vna)

    except Exception as exc:
        print(f"ERROR at step {len(all_spectra) + 1}/{len(cw_freqs)}: {exc}")
        if not all_spectra:
            raise
        print(f"Saving partial data ({len(all_spectra)} of {len(cw_freqs)} steps)...")

    K = len(offsets_hz)
    spectra_arr = np.array([[s[:K] for s in row] for row in all_spectra])
    completed_freqs = cw_freqs[:len(all_spectra)]

    np.savez_compressed(
        full_path,
        cw_freqs=completed_freqs,
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_arr,
    )

    print(
        f"Done. Saved {len(all_spectra)}/{len(cw_freqs)} steps to {full_path}"
    )
    if plot:
        data = HeterodyneSweepData.from_file(full_path)
        data.plot_peak_powers()
        data.plot_modulation_depth()
        plt.show()
    return full_path


def main():
    center_freq = 2*1.130e9
    span = 100e6
    cw_freqs = np.linspace(center_freq - span/2, center_freq + span/2, 300)
    # cw_freqs = np.linspace(0.5e9, 5e9, 2500)

    # vna_cw_harmonic_sweep(
    #     cw_freqs=cw_freqs,
    #     cw_power=10,
    #     harmonics=(1, 2, 3),
    #     window_hz=2e6,
    #     esa_freq_step=1e6,
    #     esa_res_bw=10e3,
    #     esa_ref_level=-40,
    #     settle_time_s=0.05,
    #     optional_name='libbu2_w15_die1-2_mzm_c3_',
    # )

    vna_cw_heterodyne_sweep(
        cw_freqs=cw_freqs,
        cw_power=5,
        heterodyne_shift=125e6,
        harmonics=(0, 1,),
        window_hz=2e6,
        esa_freq_step=2e6/1001,
        esa_res_bw=10e3,
        esa_ref_level=-40,
        settle_time_s=0.01,
        optional_name='test',
        use_cxa=True
    )


if __name__ == '__main__':
    main()