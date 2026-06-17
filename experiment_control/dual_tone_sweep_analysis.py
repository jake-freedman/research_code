"""
Analyse a dual-tone sweep recording produced by bnc_dual_tone_esa_script.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from dual_tone_sweep_data import DualToneSweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\phase_sweep_dual_tone_sweep_2026-06-17-12-50-07.npz"
# X-axis for all plots. One of:
#   'drive_freq'   — fundamental drive frequency f
#   'ch1_power'    — channel 1 output power (dBm)
#   'ch2_power'    — channel 2 output power (dBm)
#   'ch1_voltage'  — channel 1 RMS voltage at 50 Ω (V)
#   'ch2_voltage'  — channel 2 RMS voltage at 50 Ω (V)
#   'ch1_phase'    — channel 1 phase offset (deg)
#   'ch2_phase'    — channel 2 phase offset (deg)
#   'stability'    — step index (use when all parameters are held constant to
#                    check measurement repeatability over time)
X_AXIS = 'ch2_phase'

# Normalize sideband powers by the per-step calibration carrier level?
#   False      → y-axis in dBm  (raw ESA power)
#   True       → y-axis in dBc  (relative to optical carrier, log scale)
#   'percent'  → y-axis in %    (fraction of carrier power, linear scale)
NORMALIZE = 'percent'

# Show the calibration (carrier-beat) power vs sweep parameter?
SHOW_CALIBRATION = True

# Calibration reference used for normalization:
#   'auto'        — use whatever cal_spectra was recorded (per-step RF-off,
#                   sideband-sum proxy, or tiled preamble — whatever the script saved)
#   'J0_preamble' — override with the initial RF-off preamble measurement
#                   (ref_cal_spectrum), using it as a flat reference across all
#                   steps; useful for sideband-sum data where you want to express
#                   sideband powers as a fraction of the known initial comb power
CALIBRATION_REF = 'auto'

# When n_sweep_repeats > 1: show individual repeat points as semi-transparent
# scatter behind the mean curve?
SHOW_REPEAT_POINTS = True

# When n_sweep_repeats > 1: shade ± 1 std band around the mean curve?
SHOW_ERROR_BAND = True

# Plot a single repeat instead of the mean across all repeats?
# None  → show the mean (default)
# 0, 1, … → show only that repeat index (0-based); scatter/error band are suppressed
REPEAT_INDEX = None

# Y-axis limits for sideband power plot (dBm or dBc). None = auto.
if NORMALIZE == 'percent':
    POWER_YMIN = 0
    POWER_YMAX = 60
else:
      POWER_YMIN = -90
      POWER_YMAX = 5

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40

# Initial β guess for single-tone preamble extraction.
BETA_GUESS_REF = 2.0


def main():
    data = DualToneSweepData.from_file(local_path(DATA_FILE))

    if CALIBRATION_REF == 'J0_preamble':
        if data.ref_cal_spectrum is None:
            print("Warning: no preamble J0 found in file; using recorded cal_spectra.")
        else:
            data.cal_spectra = np.tile(
                data.ref_cal_spectrum[:len(data.offsets_hz)],
                (len(data.drive_freqs), 1),
            )

    if REPEAT_INDEX is not None:
        if data.spectra_all is None:
            print("Warning: REPEAT_INDEX set but file has only one repeat; showing that sweep.")
        else:
            n_reps = data.spectra_all.shape[0]
            if not (0 <= REPEAT_INDEX < n_reps):
                raise ValueError(f"REPEAT_INDEX {REPEAT_INDEX} out of range (file has {n_reps} repeats).")
            data.spectra = data.spectra_all[REPEAT_INDEX]
            if data.cal_spectra_all is not None:
                data.cal_spectra = data.cal_spectra_all[REPEAT_INDEX]
            data.spectra_all = None      # suppress scatter / error band
            data.cal_spectra_all = None
            data.n_repeats = 1

    print(f"Loaded: {DATA_FILE}")
    try:
        beta1, beta2 = data.single_tone_modulation_depths(beta_guess=BETA_GUESS_REF, )
        if beta1 is not None:
            print(f"  β1 (ch1-only, f) : {beta1:.4f} rad  ({np.degrees(beta1):.2f}°)")
        if beta2 is not None:
            print(f"  β2 (ch2-only, 2f): {beta2:.4f} rad  ({np.degrees(beta2):.2f}°)")
    except RuntimeError:
        pass
    print(f"  Steps            : {len(data.drive_freqs)}")
    print(f"  Drive freq range : {data.drive_freqs.min() / 1e9:.4f} – "
          f"{data.drive_freqs.max() / 1e9:.4f} GHz")
    print(f"  Ch1 power range  : {data.ch1_powers_dbm.min():+.1f} – "
          f"{data.ch1_powers_dbm.max():+.1f} dBm")
    print(f"  Ch2 power range  : {data.ch2_powers_dbm.min():+.1f} – "
          f"{data.ch2_powers_dbm.max():+.1f} dBm")
    print(f"  Ch1 phase range  : {data.ch1_phases_deg.min():.1f} – "
          f"{data.ch1_phases_deg.max():.1f} deg")
    print(f"  Ch2 phase range  : {data.ch2_phases_deg.min():.1f} – "
          f"{data.ch2_phases_deg.max():.1f} deg")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    if data.ref_cal_spectrum is not None:
        print(f"  Preamble J0      : {float(data.ref_cal_spectrum.max()):.2f} dBm")

    data.plot_peak_powers(
        normalize=NORMALIZE,
        x_axis=X_AXIS,
        ymin=POWER_YMIN,
        ymax=POWER_YMAX,
        show_points=SHOW_REPEAT_POINTS,
        show_error=SHOW_ERROR_BAND,
    )

    if SHOW_CALIBRATION:
        data.plot_calibration(
            x_axis=X_AXIS,
            ymin=CAL_YMIN,
            ymax=CAL_YMAX,
        )

    plt.show()


if __name__ == '__main__':
    main()
