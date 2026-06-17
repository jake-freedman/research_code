"""
Analyse a bias-voltage S11 sweep produced by bias_s11_sweep() in
mechanical_resonance_tuning_script.py.
"""

import matplotlib.pyplot as plt
from bias_s11_sweep_data import BiasS11SweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\s11_w2_d21_wg5a_p5_2026-06-15-19-27-36.csv"

# S11 overlay plot limits.
YMIN = -3.0   # dB
YMAX = -1.5   # dB
XMIN = 1.0    # GHz, None = auto
XMAX = 1.2    # GHz, None = auto

# Frequency window for resonance search (GHz). Only the S11 minimum within
# [FREQ_MIN, FREQ_MAX] is tracked. Set to None to search the full range.
FREQ_MIN = 1.0   # GHz
FREQ_MAX = 1.2   # GHz

# Y-axis limits for the resonance frequency plot (GHz). None = auto.
RES_YMIN = None
RES_YMAX = None


def main():
    data = BiasS11SweepData.from_file(local_path(DATA_FILE))

    print(f"Loaded: {DATA_FILE}")
    print(f"  Bias voltages : {data.voltages[0]:.2f} to {data.voltages[-1]:.2f} V "
          f"({len(data.voltages)} steps)")
    print(f"  Frequencies   : {data.freqs[0] / 1e9:.4f} to "
          f"{data.freqs[-1] / 1e9:.4f} GHz ({len(data.freqs)} points)")

    data.plot(ymin=YMIN, ymax=YMAX, xmin=XMIN, xmax=XMAX)
    data.plot_resonance_freq_vs_voltage(
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        ymin=RES_YMIN,
        ymax=RES_YMAX,
    )
    plt.show()


if __name__ == '__main__':
    main()
