"""
Analyse a power-sweep heterodyne harmonic recording produced by
vna_power_heterodyne_sweep() in vna_power_harmonic_esa_script.py
or bnc_power_heterodyne_sweep() in bnc_power_harmonic_esa_script.py.
"""

import matplotlib.pyplot as plt
from power_harmonic_sweep_data import PowerHeterodyneSweepData
from path_utils import local_path

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\phase_sweep_dual_tone_sweep_2026-06-24-14-03-42.npz"

# X-axis for all plots: 'voltage' (V_rms) or 'dbm' (drive power in dBm).
X_AXIS = 'dbm'

# Normalize sideband powers by the per-step calibration carrier level?
#   False      → y-axis in dBm  (raw ESA power)
#   True       → y-axis in dBc  (relative to optical carrier, log scale)
#   'percent'  → y-axis in %    (fraction of carrier power, linear scale)
# Requires the file to have been recorded with per_step_calibration=True;
# falls back to False automatically if cal_spectra is absent.
NORMALIZE = False

# Show the calibration (carrier-beat) power vs drive level?
# Only available when per_step_calibration=True was used.
SHOW_CALIBRATION = True

# Harmonic pair for β extraction: J_num(β) / J_den(β).
HARMONIC_NUMERATOR = 1
HARMONIC_DENOMINATOR = 0

# Initial guess for β (rad).
BETA_GUESS = 2.0

# Overlay a linear fit to estimate V_π (voltage mode) or P_π (dBm mode)?
FIT_VPI = True

# Y-axis limits for sideband power plot. None = auto.
if NORMALIZE == 'percent':
    POWER_YMIN = 0
    POWER_YMAX = 100
else:
    POWER_YMIN = None
    POWER_YMAX = None

# Y-axis limits for β plot (rad). None = auto.
BETA_YMIN = 0
BETA_YMAX = 5

# Y-axis limits for calibration plot (dBm). None = auto.
CAL_YMIN = -80
CAL_YMAX = -40


def main():
    data = PowerHeterodyneSweepData.from_file(local_path(DATA_FILE))

    normalize = NORMALIZE
    if normalize and data.cal_spectra is None:
        print("Warning: NORMALIZE requested but file has no per-step calibration; "
              "falling back to raw dBm.")
        normalize = False

    print(f"Loaded: {DATA_FILE}")
    print(f"  CW frequency     : {data.cw_freq / 1e9:.4f} GHz")
    print(f"  Drive powers     : {data.cw_powers[0]:+.1f} to "
          f"{data.cw_powers[-1]:+.1f} dBm ({len(data.cw_powers)} steps)")
    print(f"  Harmonics        : {list(data.harmonics)}")
    print(f"  Heterodyne shift : {data.heterodyne_shift / 1e6:.1f} MHz")
    print(f"  Window           : ±{data.window_hz / 1e6:.1f} MHz per harmonic")
    print(f"  Cal spectra      : {'yes' if data.cal_spectra is not None else 'no'}")
    print(f"  Extracting β from J{HARMONIC_NUMERATOR}(β) / "
          f"J{HARMONIC_DENOMINATOR}(β)")

    data.plot_peak_powers(
        normalize=normalize,
        x_axis=X_AXIS,
        ymin=POWER_YMIN,
        ymax=POWER_YMAX,
    )

    if SHOW_CALIBRATION and data.cal_spectra is not None:
        data.plot_calibration(
            x_axis=X_AXIS,
            ymin=CAL_YMIN,
            ymax=CAL_YMAX,
        )

    try:
        data.plot_modulation_depth(
            harmonic_numerator=HARMONIC_NUMERATOR,
            harmonic_denominator=HARMONIC_DENOMINATOR,
            beta_guess=BETA_GUESS,
            x_axis=X_AXIS,
            ymin=BETA_YMIN,
            ymax=BETA_YMAX,
            fit=FIT_VPI,
        )
    except ValueError as e:
        print(f"Could not extract β: {e}")

    plt.show()


if __name__ == '__main__':
    main()
