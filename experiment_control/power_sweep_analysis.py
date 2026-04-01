"""
Analyse a VNA power sweep + ESA recording.

Reads a single-file power sweep CSV produced by vna_power_esa_script.py and
plots the peak ESA power within a user-defined frequency window as a function
of VNA output power.
"""

import matplotlib.pyplot as plt

from esa_sweep_data import PowerSweepData

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\libbu2_w15_die1-2_mzm_c3_power_sweep_2026-04-01-16-35-43.csv"

# Frequency window for peak search (Hz)
FREQ_MIN = 5e9
FREQ_MAX = 6e9

# Y-axis limits for peak-vs-power plot (dBm). Set to None for auto.
YMIN = None
YMAX = None


def main():
    data = PowerSweepData.from_file(DATA_FILE)

    # Print a summary
    print(f"Loaded: {DATA_FILE}")
    print(f"  VNA powers : {data.cw_powers[0]:+.1f} to {data.cw_powers[-1]:+.1f} dBm "
          f"({len(data.cw_powers)} steps)")
    print(f"  ESA range  : {data.esa_freqs[0]/1e9:.4f} to {data.esa_freqs[-1]/1e9:.4f} GHz "
          f"({len(data.esa_freqs)} points)")
    print(f"  Peak window: {FREQ_MIN/1e9:.4f} to {FREQ_MAX/1e9:.4f} GHz")

    # Plot all spectra (waterfall overview)
    fig_spec, ax_spec = data.plot(ymin=YMIN, ymax=YMAX)
    ax_spec.axvline(FREQ_MIN / 1e9, color='k', linewidth=0.8, linestyle='--')
    ax_spec.axvline(FREQ_MAX / 1e9, color='k', linewidth=0.8, linestyle='--')

    # Plot peak power vs VNA power
    fig_peak, ax_peak = data.plot_peak_vs_power(
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        ymin=YMIN,
        ymax=YMAX,
    )

    plt.show()


if __name__ == '__main__':
    main()
