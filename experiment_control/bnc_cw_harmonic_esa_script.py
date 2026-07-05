"""
BNC 855B CW frequency sweep with harmonic-tracking ESA.

Mirrors vna_cw_harmonic_esa_script.py but uses the BNC 855B-12 signal
generator (channel 1) as the CW source instead of the VNA.

For each drive frequency the ESA records a narrow spectrum centred on each
requested harmonic of that frequency. All data is saved as a single .npz file
that is fully compatible with HarmonicSweepData and HeterodyneSweepData in
the existing analysis scripts.
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
from harmonic_sweep_data import HarmonicSweepData
from heterodyne_sweep_data import HeterodyneSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5"


def bnc_cw_harmonic_sweep(
    cw_freqs,
    cw_power: float,
    harmonics=(0, 1),
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
    Step the BNC 855B through CW frequencies (channel 1) and record a narrow
    ESA spectrum around each harmonic of the drive frequency at every step.

    Data is saved in the same .npz format as vna_cw_harmonic_esa_script.py
    and can be loaded by HarmonicSweepData.

    Parameters
    ----------
    cw_freqs : array-like
        BNC channel 1 frequencies in Hz.
    cw_power : float
        BNC channel 1 output power in dBm.
    harmonics : sequence of int
        Harmonic numbers to record. Default (1, 2, 3).
    window_hz : float
        Half-width of each harmonic window in Hz. Default 2 MHz.
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
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA. Default False.
    plot : bool
        If True, plot harmonic spectra and modulation depth after saving.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    cw_freqs = np.asarray(cw_freqs)
    harmonics = list(harmonics)
    print(
        f"Starting BNC harmonic sweep: {len(cw_freqs)} CW steps "
        f"({cw_freqs[0] / 1e9:.4f} to {cw_freqs[-1] / 1e9:.4f} GHz), "
        f"harmonics {harmonics}, window ±{window_hz / 1e6:.1f} MHz"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}harmonic_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    all_spectra = []
    offsets_hz = None

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:
            sig.disable_all_outputs()
            sig.configure_channel(1, cw_freqs[0], cw_power)
            sig.enable_output(1)

            for i, f_cw in enumerate(cw_freqs):
                sig.set_frequency(1, f_cw)
                time.sleep(settle_time_s)

                harmonic_spectra = []
                for n in harmonics:
                    center = abs(n * f_cw)
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
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_arr,
    )

    print(f"Done. Saved {len(all_spectra)}/{len(cw_freqs)} steps to {full_path}")
    if plot:
        data = HarmonicSweepData.from_file(full_path)
        data.plot_modulation_depth()
        data.plot_harmonic_spectra()
        plt.show()
    return full_path


def bnc_cw_heterodyne_sweep(
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
    Step the BNC 855B through CW frequencies (channel 1) and record a narrow
    ESA spectrum centred on n*f_cw + heterodyne_shift for each harmonic n.

    Harmonic 0 records the carrier beat at heterodyne_shift itself, which is
    needed to extract modulation depth via J1(β)/J0(β).

    Data is saved in the same .npz format as vna_cw_heterodyne_sweep() and
    can be loaded by HeterodyneSweepData.

    Parameters
    ----------
    cw_freqs : array-like
        BNC channel 1 frequencies in Hz.
    cw_power : float
        BNC channel 1 output power in dBm.
    heterodyne_shift : float
        Offset of the LO from the signal in Hz. Default 125 MHz.
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
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA. Default False.
    plot : bool
        If True, plot peak powers and modulation depth after saving.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    cw_freqs = np.asarray(cw_freqs)
    harmonics = list(harmonics)
    print(
        f"Starting BNC heterodyne sweep: {len(cw_freqs)} CW steps "
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
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:
            sig.disable_all_outputs()
            sig.configure_channel(1, cw_freqs[0], cw_power)
            sig.enable_output(1)

            for i, f_cw in enumerate(cw_freqs):
                sig.set_frequency(1, f_cw)

                harmonic_spectra = []
                for n in harmonics:
                    center = abs(n * f_cw + heterodyne_shift)
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

    print(f"Done. Saved {len(all_spectra)}/{len(cw_freqs)} steps to {full_path}")
    if plot:
        data = HeterodyneSweepData.from_file(full_path)
        data.plot_peak_powers()
        data.plot_modulation_depth()
        plt.show()
    return full_path


def main():
    center_freq = 1.145e9
    span = 100e6
    # cw_freqs = np.linspace(center_freq - span / 2, center_freq + span / 2, 50)
    cw_freqs = np.linspace(100e6, 3.5e9, 3500)

    bnc_cw_heterodyne_sweep(
        cw_freqs=cw_freqs,
        cw_power=5,
        heterodyne_shift=125e6,
        harmonics=(0, 1),
        window_hz=2e6,
        esa_freq_step=2e6/1001,
        esa_res_bw=10e3,
        esa_ref_level=-40,
        settle_time_s=0.01,
        optional_name='test_',
        use_cxa=True,
    )


if __name__ == '__main__':
    main()
