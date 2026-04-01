"""
Combined VNA CW + ESA sweep script.

The VNA is stepped through a list of CW frequencies. At each frequency the
ESA records a spectrum. All spectra are saved as individual CSV files in a
timestamped folder, and can be plotted overlaid afterwards.
"""

from vna_control import VNA
from esa_control import ESA, ESAData
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"


def vna_cw_esa_sweep(
    cw_freqs,
    cw_power: float,
    esa_start_freq: float,
    esa_stop_freq: float,
    esa_freq_step: float,
    esa_res_bw: float,
    esa_ref_level: float = 0.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
):
    """
    Step the VNA through CW frequencies and record an ESA spectrum at each.

    Parameters
    ----------
    cw_freqs : array-like
        List or array of VNA CW frequencies in Hz.
    cw_power : float
        VNA output power in dBm.
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
        Time to wait after setting each CW frequency before triggering the
        ESA sweep. Default 0.1 s.
    optional_name : str
        Optional label prepended to each saved filename.

    Returns
    -------
    str
        Path to the folder where all CSV files were saved.
    """
    cw_freqs = np.asarray(cw_freqs)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = f'{optional_name}cw_freq_sweep_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
    full_path = os.path.join(DATA_FOLDER, fname)

    with VNA(VNA_RESOURCE_STRING) as vna, ESA(ESA_RESOURCE_STRING) as esa:

        esa.configure(
            start_freq=esa_start_freq,
            stop_freq=esa_stop_freq,
            freq_step=esa_freq_step,
            res_bw=esa_res_bw,
            ref_level=esa_ref_level,
        )

        vna.set_cw_mode(cw_freqs[0], cw_power)

        spectra = []
        for i, freq in enumerate(cw_freqs):
            vna.set_cw_freq(freq)
            time.sleep(settle_time_s)
            esa_freqs, power = esa.sweep()
            spectra.append(power)
            print(f"Step {i + 1}/{len(cw_freqs)}: CW = {freq / 1e9:.4f} GHz done.")

        vna.cw_off()

    # Build and save single file:
    # header line 1: swept CW frequencies
    # columns: esa_freq_hz, power_at_step_0, power_at_step_1, ...
    spectra = np.column_stack(spectra)  # shape (n_esa_points, n_steps)
    data = np.column_stack([esa_freqs, spectra])
    cw_header = 'cw_freq_hz:' + ','.join(f'{f:.2f}' for f in cw_freqs)
    col_header = 'esa_freq_hz,' + ','.join(f'power_dbm_step{i}' for i in range(len(cw_freqs)))
    np.savetxt(full_path, data, delimiter=',', header=cw_header + '\n' + col_header, comments='')

    print(f"Done. Saved to {full_path}")
    return full_path


def plot_cw_esa_results(folder: str, ymin: float = -100.0, ymax: float = 0.0) -> tuple:
    """
    Plot all ESA spectra from a vna_cw_esa run on the same axes, coloured
    from light to dark green across the CW frequency steps.

    Parameters
    ----------
    folder : str
        Path to the folder produced by vna_cw_esa_sweep().
    ymin, ymax : float
        Y-axis limits in dBm.

    Returns
    -------
    fig, ax
    """
    csvs = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')],
        key=os.path.getmtime,
    )
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder!r}")

    n = len(csvs)
    colors = [plt.cm.Greens(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    left_mm, right_mm = 14.0, 5.0
    bottom_mm, top_mm = 12.0, 5.0
    axes_width_mm, axes_height_mm = 180.0, 100.0
    mm = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=(
        (left_mm + axes_width_mm + right_mm) * mm,
        (bottom_mm + axes_height_mm + top_mm) * mm,
    ))
    fig.subplots_adjust(
        left=left_mm / (left_mm + axes_width_mm + right_mm),
        right=(left_mm + axes_width_mm) / (left_mm + axes_width_mm + right_mm),
        bottom=bottom_mm / (bottom_mm + axes_height_mm + top_mm),
        top=(bottom_mm + axes_height_mm) / (bottom_mm + axes_height_mm + top_mm),
    )

    for csv, color in zip(csvs, colors):
        data = ESAData.from_file(csv)
        ax.plot(data.freqs / 1e9, data.power_db, color=color, linewidth=1.0)

    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Frequency [GHz]', fontsize=10)
    ax.set_ylabel('Power [dBm]', fontsize=10)
    ax.tick_params(axis='both', direction='in', width=2, labelsize=8)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)

    return fig, ax


def main():

    cw_freqs = np.linspace(2.53e9, 2.55e9, 100)  # 11 steps from 2.5 to 2.6 GHz

    folder = vna_cw_esa_sweep(
        cw_freqs=cw_freqs,
        cw_power=10,
        esa_start_freq=2.525e9,
        esa_stop_freq=2.555e9,
        esa_freq_step=0.1e6,
        esa_res_bw=1e3,
        esa_ref_level=-40,
        settle_time_s=0.1,
        optional_name='libbu2_w15_die1-2_mzm_c3_esa',
    )

    fig, ax = plot_cw_esa_results(folder, ymin=-100, ymax=-40)
    plt.show()


if __name__ == '__main__':

    main()
