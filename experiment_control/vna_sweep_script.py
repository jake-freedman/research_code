from vna_control import VNA, S11Data
import pyvisa
import matplotlib.pyplot as plt

VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'


def main():

    plot_from_file()

    # folder_path = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"

    # print(pyvisa.ResourceManager().list_resources())

    # with VNA(VNA_RESOURCE_STRING) as vna:

    #     vna.configure(
    #         start_freq = 10e6,
    #         stop_freq = 5e9,
    #         freq_step = 1e6,
    #         power_dbm = -10,
    #         ifbw = 10000,
    #         cal_set = 'CalSet_1'
    #     )

    #     vna.apply_calibration()

    #     freqs, s11 = vna.sweep_s11()
    #     path = vna.save_s11(freqs, s11, folder = folder_path, optional_name = 'libbu2_w15_die1-2_mzm_c2')

    #     vna.plot_s11(freqs, s11)
    #     plt.show()


def plot_from_file():
    data = S11Data.from_folder(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data")
    data.plot_s11(ymin=-13, ymax = 1)
    plt.show()


if __name__ == '__main__':

    main()