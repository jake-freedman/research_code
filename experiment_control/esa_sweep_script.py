from esa_control import ESA, ESAData
import os
import matplotlib.pyplot as plt
from path_utils import local_path

ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'

DATA_FOLDER = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\esa_esa_test2026-04-01-15-49-41.csv"


def list_resources():
    import pyvisa
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print(f"Found {len(resources)} resource(s):")
    for r in resources:
        print(f"  {r}")
    rm.close()


def main():

    # list_resources()

    # with ESA(ESA_RESOURCE_STRING) as esa:

    #     esa.configure(
    #         start_freq=2e9,
    #         stop_freq=8e9,
    #         freq_step=1e6,
    #         res_bw=1e3,
    #     )

    #     freqs, power = esa.sweep()
    #     esa.save(freqs, power, folder=DATA_FOLDER, optional_name='esa_test')
    #     esa.plot(freqs, power, ymin = -120, ymax = -80)
    #     plt.show()

    data_file = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\esa_esa_test2026-04-01-15-49-41.csv"
    data = ESAData.from_file(local_path(data_file))

    beta = data.modulation_depth(mod_freq=2.533e9)
    print(f"Modulation depth: beta = {beta:.4f} rad")

    data.plot()
    plt.show()


if __name__ == '__main__':

    main()
