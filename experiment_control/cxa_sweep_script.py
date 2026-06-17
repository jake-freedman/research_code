from cxa_control import CXA, CXAData
import matplotlib.pyplot as plt
from path_utils import local_path

CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"


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

    with CXA(CXA_RESOURCE_STRING) as cxa:

        cxa.configure(
            start_freq=100e6,
            stop_freq=150e6,
            freq_step=1e6,
            res_bw=1e6,
        )

        freqs, power = cxa.sweep()
        path = cxa.save(freqs, power, folder=local_path(DATA_FOLDER), optional_name='cxa_sweep')
        print(f"Saved to {path}")
        cxa.plot(freqs, power, ymin=-100, ymax=-20)
        plt.show()


if __name__ == '__main__':

    main()
