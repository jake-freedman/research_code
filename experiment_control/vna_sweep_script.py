from vna_control import VNA, S11Data, S11S21Data
import os
import time
from datetime import datetime
import numpy as np
import pyvisa
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'


def main():

    plot_from_file()

    # mzm_stability_test(n = 5, delay_s = 1.0)

    folder_path = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"

    # fig, ax = plot_stability_s21(
    #     r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\mzm_stability_test_2026-04-01-11-24-53",
    #     ymin=-120, ymax=-60,
    # )
    # plt.show()
    # fig.savefig(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\stability_1s.png", dpi = 300)




    # print(pyvisa.ResourceManager().list_resources())

    # with VNA(VNA_RESOURCE_STRING) as vna:



    #     vna.configure(
    #         start_freq = 0.1e9,
    #         stop_freq = 5e9,
    #         freq_step = 1e6,
    #         power_dbm = 10,
    #         ifbw = 1000,
    #         cal_set = 'CalSet_1'
    #     )

    #     vna.apply_calibration()

    #     freqs, s11, s21 = vna.sweep_s11_s21()
    #     path = vna.save_s11_s21(freqs, s11, s21, folder=folder_path, optional_name='libbu2_w15_die1-2_mzm_c3')
    #     vna.plot_s11_s21(freqs, s11, s21, s21_ymin=-130, s21_ymax=-60, s11_ymin=-30, s11_ymax=1, xmin=0, xmax=5.1)
    #     plt.show()

    #     freqs, s11 = vna.sweep_s11()
    #     path = vna.save_s11(freqs, s11, folder = folder_path, optional_name = 'libbu2_w15_die1-2_mzm_c2')

    #     vna.plot_s11(freqs, s11)
    #     plt.show()


def mzm_stability_test(n: int, delay_s: float, optional_name: str = ''):
    base_folder = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"
    run_folder = os.path.join(base_folder, 'mzm_stability_test_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(run_folder)

    with VNA(VNA_RESOURCE_STRING) as vna:
        vna.configure(
            start_freq=0.1e9,
            stop_freq=5e9,
            freq_step=1e6,
            power_dbm=10,
            ifbw=1000,
            cal_set='CalSet_1',
        )
        vna.apply_calibration()

        for i in range(n):
            freqs, s11, s21 = vna.sweep_s11_s21()
            vna.save_s11_s21(freqs, s11, s21, folder=run_folder, optional_name=optional_name)
            print(f"Sweep {i + 1}/{n} complete")
            if i < n - 1:
                time.sleep(delay_s)

    print(f"Saved {n} sweeps to {run_folder}")


def plot_stability_s21(folder: str, ymin: float = -30.0, ymax: float = 5.0) -> tuple:
    csvs = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')],
        key=os.path.getmtime,
    )
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder!r}")

    n = len(csvs)
    colors = [plt.cm.Blues(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    left_mm, right_mm = 20.0, 5.0
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
        data = S11S21Data.from_file(csv)
        s21_db = 20.0 * np.log10(np.abs(data.s21_complex) + 1e-300)
        ax.plot(data.freqs / 1e9, s21_db, color=color, linewidth=1.0)

    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Frequency [GHz]', fontsize=10)
    ax.set_ylabel(r'$S_{21}$ [dB]', fontsize=10)
    ax.tick_params(axis='both', direction='in', width=2, labelsize=8)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)

    return fig, ax


def plot_from_file():
    # data = S11Data.from_file(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\s11_....csv")
    # data.plot_s11(ymin=-13, ymax = 1)
    # plt.show()

    data = S11S21Data.from_file(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\s11_s21_libbu2_w15_die1-2_mzm_c32026-04-01-11-12-47.csv")
    fig, _, _ = data.plot_s11_s21(s21_ymin=-110, s21_ymax=-60, s11_ymin=-20, s11_ymax=-5, xmin=2, xmax=3)
    plt.show()

    # fig.savefig(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\s21_mzm_zoomin.png", dpi = 300)


if __name__ == '__main__':

    main()
