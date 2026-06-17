"""
VNA fixed-frequency power sweep with heterodyne harmonic-tracking ESA.

The VNA is held at a fixed CW frequency and stepped through a list of output
powers. At each power level the ESA records a narrow spectrum centred on each
requested harmonic (n*f_cw + heterodyne_shift). All data is saved as a single
.npz file that can be loaded by PowerHeterodyneSweepData in
power_harmonic_sweep_data.py to plot modulation depth vs RF voltage.
"""

from vna_control import VNA
from esa_control import ESA
from cxa_control import CXA
from power_harmonic_sweep_data import PowerHeterodyneSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"


def voltage_linspace(p_start_dbm: float, p_stop_dbm: float, n: int) -> np.ndarray:
    """
    Return n powers in dBm that are evenly spaced in RMS voltage (50 Ω sinusoid).

    Converts the dBm bounds to voltages, spaces those linearly, then converts
    back to dBm. Pass the result directly as cw_powers to vna_power_heterodyne_sweep.

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


def vna_power_heterodyne_sweep(
    cw_freq: float,
    cw_powers,
    heterodyne_shift: float = 125e6,
    harmonics=(0, 1),
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
    use_cxa: bool = False,
    plot: bool = True,
) -> str:
    """
    Step the VNA through output powers at a fixed CW frequency and record a
    narrow ESA spectrum centred on n*f_cw + heterodyne_shift for each harmonic.

    Parameters
    ----------
    cw_freq : float
        Fixed VNA CW frequency in Hz.
    cw_powers : array-like
        VNA output powers in dBm.
    heterodyne_shift : float
        Offset of the LO from the signal in Hz. ESA centre for harmonic n is
        n*f_cw + heterodyne_shift. Default 125 MHz.
    harmonics : sequence of int
        Harmonic numbers to record. 0 = carrier beat. Default (0, 1).
    window_hz : float
        Half-width of each harmonic window in Hz. Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window in Hz. Default 250 kHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz. Default 10 kHz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Wait time after setting each power level. Default 0.1 s.
    optional_name : str
        Label prepended to the saved filename.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA. Default False.
    plot : bool
        If True, plot peak powers and modulation depth after saving. Default True.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    cw_powers = np.asarray(cw_powers)
    harmonics = list(harmonics)
    print(
        f"Starting power sweep: {len(cw_powers)} steps "
        f"({cw_powers[0]:+.1f} to {cw_powers[-1]:+.1f} dBm) "
        f"at {cw_freq / 1e9:.4f} GHz, "
        f"harmonics {harmonics}, shift {heterodyne_shift / 1e6:.1f} MHz"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}power_harmonic_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    all_spectra = []
    offsets_hz = None

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with VNA(VNA_RESOURCE_STRING) as vna, esa_cls(esa_addr) as esa:
            vna.set_cw_mode(cw_freq, cw_powers[0])

            for i, power in enumerate(cw_powers):
                vna.set_cw_power(power)
                time.sleep(settle_time_s)

                harmonic_spectra = []
                for n in harmonics:
                    center = n * cw_freq + heterodyne_shift
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
                    f"Step {i + 1}/{len(cw_powers)}: "
                    f"{power:+.1f} dBm done."
                )

            vna.cw_off()

    except Exception as exc:
        print(f"ERROR at step {len(all_spectra) + 1}/{len(cw_powers)}: {exc}")
        if not all_spectra:
            raise
        print(f"Saving partial data ({len(all_spectra)} of {len(cw_powers)} steps)...")

    K = len(offsets_hz)
    spectra_arr = np.array([[s[:K] for s in row] for row in all_spectra])
    completed_powers = cw_powers[:len(all_spectra)]

    np.savez_compressed(
        full_path,
        cw_freq=np.array(cw_freq),
        cw_powers=completed_powers,
        harmonics=np.array(harmonics),
        heterodyne_shift=np.array(heterodyne_shift),
        window_hz=np.array(window_hz),
        esa_freq_step_hz=np.array(esa_freq_step),
        offsets_hz=offsets_hz,
        spectra=spectra_arr,
    )

    print(
        f"Done. Saved {len(all_spectra)}/{len(cw_powers)} steps to {full_path}"
    )
    if plot:
        data = PowerHeterodyneSweepData.from_file(full_path)
        data.plot_peak_powers()
        data.plot_modulation_depth()
        plt.show()
    return full_path


def main():
    # Evenly spaced in RMS voltage between -20 and +10 dBm bounds:
    cw_powers = voltage_linspace(-20, 12, 40)
    # Or evenly spaced in dBm:
    # cw_powers = np.linspace(-20, 10, 40)

    vna_power_heterodyne_sweep(
        cw_freq=1*1.164e9,
        cw_powers=cw_powers,
        heterodyne_shift=125e6,
        harmonics=(0, 1),
        window_hz=2e6,
        esa_freq_step=0.25e6,
        esa_res_bw=10e3,
        esa_ref_level=-40,
        settle_time_s=0.05,
        optional_name='test_',
        use_cxa=True,
    )


if __name__ == '__main__':
    main()
