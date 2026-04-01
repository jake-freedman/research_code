"""
Combined VNA CW power sweep + ESA script.

The VNA is held at a fixed CW frequency and stepped through a list of output
powers. At each power level the ESA records a spectrum. All spectra are saved
as individual CSV files in a timestamped folder.
"""

from vna_control import VNA
from esa_control import ESA
from esa_sweep_data import PowerSweepData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"


def vna_power_esa_sweep(
    cw_freq: float,
    cw_powers,
    esa_start_freq: float,
    esa_stop_freq: float,
    esa_freq_step: float,
    esa_res_bw: float,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
):
    """
    Step the VNA through output powers at a fixed CW frequency and record
    an ESA spectrum at each power level.

    Parameters
    ----------
    cw_freq : float
        Fixed VNA CW frequency in Hz.
    cw_powers : array-like
        List or array of VNA output powers in dBm.
    esa_start_freq : float
        ESA start frequency in Hz.
    esa_stop_freq : float
        ESA stop frequency in Hz.
    esa_freq_step : float
        ESA frequency step in Hz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz.
    esa_ref_level : float
        ESA reference level in dBm. Default 0.
    settle_time_s : float
        Time to wait after setting each power level before triggering the
        ESA sweep. Default 0.1 s.
    optional_name : str
        Optional label prepended to each saved filename.

    Returns
    -------
    str
        Path to the folder where all CSV files were saved.
    """
    cw_powers = np.asarray(cw_powers)
    print(f"Starting power sweep: {len(cw_powers)} steps from "
          f"{cw_powers[0]:+.1f} to {cw_powers[-1]:+.1f} dBm")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = f'{optional_name}power_sweep_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
    full_path = os.path.join(DATA_FOLDER, fname)

    spectra = []
    esa_freqs = None

    try:
        with VNA(VNA_RESOURCE_STRING) as vna, ESA(ESA_RESOURCE_STRING) as esa:

            esa.configure(
                start_freq=esa_start_freq,
                stop_freq=esa_stop_freq,
                freq_step=esa_freq_step,
                res_bw=esa_res_bw,
                ref_level=esa_ref_level,
            )

            vna.set_cw_mode(cw_freq, cw_powers[0])

            for i, power in enumerate(cw_powers):
                vna.set_cw_power(power)
                time.sleep(settle_time_s)
                esa_freqs, power_db = esa.sweep()
                spectra.append(power_db)
                print(f"Step {i + 1}/{len(cw_powers)}: power = {power:+.1f} dBm done.")

            vna.cw_off()

    except Exception as exc:
        print(f"ERROR at step {len(spectra) + 1}/{len(cw_powers)}: {exc}")
        if not spectra:
            raise
        print(f"Saving partial data ({len(spectra)} of {len(cw_powers)} steps)...")

    # Build and save single file using only the steps that completed.
    completed_powers = cw_powers[:len(spectra)]
    spectra_arr = np.column_stack(spectra)          # (n_esa_points, n_completed)
    data = np.column_stack([esa_freqs, spectra_arr])
    pow_header = 'cw_power_dbm:' + ','.join(f'{p:.2f}' for p in completed_powers)
    col_header = 'esa_freq_hz,' + ','.join(f'power_dbm_step{i}' for i in range(len(completed_powers)))
    np.savetxt(full_path, data, delimiter=',', header=pow_header + '\n' + col_header, comments='')

    print(f"Done. Saved {len(spectra)}/{len(cw_powers)} steps to {full_path}")
    return full_path


def plot_power_esa_results(filepath: str, ymin: float = -100.0, ymax: float = 0.0) -> tuple:
    """
    Plot all ESA spectra from a vna_power_esa run on the same axes, coloured
    from light to dark purple across the power steps.

    Parameters
    ----------
    filepath : str
        Path to the single CSV file produced by vna_power_esa_sweep().
    ymin, ymax : float
        Y-axis limits in dBm.

    Returns
    -------
    fig, ax
    """
    data = PowerSweepData.from_file(filepath)
    return data.plot(ymin=ymin, ymax=ymax)


def main():

    cw_powers = np.linspace(-20, 10, 30)  # -20 to +10 dBm in 1 dB steps

    filepath = vna_power_esa_sweep(
        cw_freq=2.533e9,
        cw_powers=cw_powers,
        esa_start_freq=2e9,
        esa_stop_freq=8e9,
        esa_freq_step=1e6,
        esa_res_bw=1e3,
        esa_ref_level=-40,
        settle_time_s=0.1,
        optional_name='libbu2_w15_die1-2_mzm_c3_',
    )

    fig, ax = plot_power_esa_results(filepath, ymin=-120, ymax=-80)
    plt.show()


if __name__ == '__main__':

    main()
