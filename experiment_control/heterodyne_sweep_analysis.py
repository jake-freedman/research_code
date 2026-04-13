"""
Analyse a heterodyne harmonic sweep produced by vna_cw_heterodyne_sweep()
in vna_cw_harmonic_esa_script.py.
"""

import matplotlib.pyplot as plt
from heterodyne_sweep_data import HeterodyneSweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\libbu2_w16_die1-2_lpm_4_heterodyne_sweep_2026-04-07-15-11-37.npz"
# Harmonic pair for β extraction.
# Default: J1(β)/J0(β) — 1st harmonic over carrier beat.
HARMONIC_NUMERATOR = 1
HARMONIC_DENOMINATOR = 0

# Y-axis limits for β plot (rad). None = auto.
BETA_YMIN = None
BETA_YMAX = None

# Y-axis limits for peak power plot (dBm). None = auto.
POWER_YMIN = None
POWER_YMAX = None


def main():
    data = HeterodyneSweepData.from_file(local_path(DATA_FILE))

    print(f"Loaded: {DATA_FILE}")
    print(f"  CW frequencies   : {data.cw_freqs[0] / 1e9:.4f} to "
          f"{data.cw_freqs[-1] / 1e9:.4f} GHz ({len(data.cw_freqs)} steps)")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    print(f"  Window           : \u00b1{data.window_hz / 1e6:.1f} MHz per harmonic")
    print(f"  Extracting \u03b2 from J{HARMONIC_NUMERATOR}(\u03b2) / "
          f"J{HARMONIC_DENOMINATOR}(\u03b2)")

    # Peak power at each harmonic vs CW frequency
    data.plot_peak_powers(ymin=POWER_YMIN, ymax=POWER_YMAX)

    # Modulation depth vs CW frequency
    data.plot_modulation_depth(
        harmonic_numerator=HARMONIC_NUMERATOR,
        harmonic_denominator=HARMONIC_DENOMINATOR,
        ymin=BETA_YMIN,
        ymax=BETA_YMAX,
    )

    plt.show()


if __name__ == '__main__':
    main()
