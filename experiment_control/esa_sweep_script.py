from esa_control import ESA, ESAData
import os
import matplotlib.pyplot as plt

ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"


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

    with ESA(ESA_RESOURCE_STRING) as esa:

        esa.configure(
            start_freq=2e9,
            stop_freq=8e9,
            freq_step=1e6,
            res_bw=1e3,
        )

        freqs, power = esa.sweep()
        esa.save(freqs, power, folder=DATA_FOLDER, optional_name='esa_test')
        esa.plot(freqs, power, ymin = -120, ymax = -80)
        plt.show()


if __name__ == '__main__':

    main()
