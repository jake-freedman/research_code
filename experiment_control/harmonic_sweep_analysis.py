"""
Analyse a harmonic tracking sweep produced by vna_cw_harmonic_esa_script.py.

Loads the .npz file, prints a summary, plots modulation depth β vs CW
frequency, and optionally shows the waterfall of harmonic spectra.
"""

import matplotlib.pyplot as plt
from harmonic_sweep_data import HarmonicSweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\libbu2_w15_die1-2_mzm_c3_harmonic_sweep_2026-04-03-15-40-43.npz"

# Harmonics used for β extraction (must both be present in the dataset)
HARMONIC_A = 1
HARMONIC_B = 3

# Y-axis limits for the β plot (rad). Set to None for auto.
BETA_YMIN = None
BETA_YMAX = None

# Y-axis limits for the spectral waterfall plots (dBm). Set to None for auto.
SPEC_YMIN = None
SPEC_YMAX = None

# Set to True to also show waterfall plots for every harmonic in the dataset.
SHOW_HARMONIC_SPECTRA = True


def main():
    data = HarmonicSweepData.from_file(local_path(DATA_FILE))

    print(f"Loaded: {DATA_FILE}")
    print(f"  CW frequencies : {data.cw_freqs[0] / 1e9:.4f} to "
          f"{data.cw_freqs[-1] / 1e9:.4f} GHz ({len(data.cw_freqs)} steps)")
    print(f"  Harmonics      : {list(data.harmonics)}")
    print(f"  Window         : \u00b1{data.window_hz / 1e6:.1f} MHz per harmonic")
    print(f"  Points/window  : {len(data.offsets_hz)}")
    print(f"  Extracting \u03b2 from harmonics {HARMONIC_A} and {HARMONIC_B}")

    # Modulation depth vs CW frequency
    fig_beta, ax_beta = data.plot_modulation_depth(
        harmonic_a=HARMONIC_A,
        harmonic_b=HARMONIC_B,
        ymin=BETA_YMIN,
        ymax=BETA_YMAX,
    )

    # Peak power at each harmonic vs CW frequency
    if SHOW_HARMONIC_SPECTRA:
        data.plot_harmonic_spectra(
            ymin=SPEC_YMIN,
            ymax=SPEC_YMAX,
        )

    plt.show()


if __name__ == '__main__':
    main()