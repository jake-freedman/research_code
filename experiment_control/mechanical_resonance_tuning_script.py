"""
Mechanical resonance tuning via DC bias voltage on the Keithley 2450 SMU.

Also contains bias_s11_sweep(), which steps through an array of DC bias
voltages and records a VNA S11 sweep at each one, saving all data to a
single .npz file readable by BiasS11SweepData in bias_s11_sweep_data.py.
"""

import os
import time
from datetime import datetime

import numpy as np
import pyvisa

from keithley_control import Keithley2450
from vna_control import VNA

# ------------------------------------------------------------------
# Instrument addresses
# ------------------------------------------------------------------

SMU_RESOURCE_STRING = 'USB0::0x05E6::0x2450::04615671::INSTR'
VNA_RESOURCE_STRING = 'TCPIP0::Localhost::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data"

# ------------------------------------------------------------------
# apply_bias settings
# ------------------------------------------------------------------

BIAS_VOLTAGE = 50.0        # V — voltage to hold
COMPLIANCE_CURRENT = 10e-3  # A — current limit

# ------------------------------------------------------------------
# bias_s11_sweep settings
# ------------------------------------------------------------------

BIAS_VOLTAGES = np.linspace(-100, 100, 21)  # V — array of voltages to sweep

# VNA
START_FREQ    = 0.5e9   # Hz
STOP_FREQ     = 1.5e9   # Hz
NUM_POINTS    = 5000
VNA_POWER_DBM = 0.0     # dBm
VNA_IFBW      = 10e3    # Hz
CAL_SET       = 'CH1_CALREG'    # set to cal set name string if calibration is needed

SETTLE_TIME_S = 5.0     # s — wait after each voltage step before sweeping


# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------

def list_resources():
    """Print all VISA resources visible to NI-VISA."""
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    if resources:
        print("Found VISA resources:")
        for r in resources:
            print(f"  {r}")
    else:
        print("No VISA resources found.")
    rm.close()


def apply_bias():
    """Apply a DC bias voltage from the SMU FORCE terminals."""
    with Keithley2450(SMU_RESOURCE_STRING) as smu:
        smu.reset()
        smu.source_voltage(
            voltage=BIAS_VOLTAGE,
            compliance_current=COMPLIANCE_CURRENT,
        )
        current = smu.measure_current()
        print(f"Outputting {BIAS_VOLTAGE} V  |  measured current: {current * 1e3:.4f} mA")
        print("Press Enter to turn off.")
        input()
        smu.disable_output()
        print("Output disabled.")


def bias_s11_sweep(optional_name: str = '') -> str:
    """
    Step through BIAS_VOLTAGES on the SMU and record a VNA S11 sweep at
    each voltage. All sweeps are saved to a single .npz file.

    Parameters
    ----------
    optional_name:
        Label prepended to the saved filename.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    bias_voltages = np.asarray(BIAS_VOLTAGES)
    freq_step = (STOP_FREQ - START_FREQ) / (NUM_POINTS - 1)

    print(
        f"Starting bias S11 sweep: {len(bias_voltages)} voltage steps "
        f"({bias_voltages[0]:.2f} to {bias_voltages[-1]:.2f} V), "
        f"{START_FREQ / 1e9:.3f}–{STOP_FREQ / 1e9:.3f} GHz, "
        f"{NUM_POINTS} points, {VNA_IFBW / 1e3:.0f} kHz IFBW"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}bias_s11_sweep_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    all_s11 = []
    freqs = None

    try:
        with VNA(VNA_RESOURCE_STRING) as vna, Keithley2450(SMU_RESOURCE_STRING) as smu:
            vna.configure(
                start_freq=START_FREQ,
                stop_freq=STOP_FREQ,
                freq_step=freq_step,
                power_dbm=VNA_POWER_DBM,
                ifbw=VNA_IFBW,
                cal_set=CAL_SET,
            )
            if CAL_SET is not None:
                vna.apply_calibration()

            smu.reset()
            smu.source_voltage(
                voltage=bias_voltages[0],
                compliance_current=COMPLIANCE_CURRENT,
                voltage_range=float(np.max(np.abs(bias_voltages))),
            )
            smu.measure_current()

            for i, v in enumerate(bias_voltages):
                smu.set_voltage(v)
                time.sleep(SETTLE_TIME_S)
                current = smu.measure_current()
                print(f"Step {i + 1}/{len(bias_voltages)}: {v:.3f} V  |  {current * 1e3:.4f} mA")

                f, s11 = vna.sweep_s11()
                all_s11.append(s11)
                if freqs is None:
                    freqs = f

            smu.disable_output()

    except Exception as exc:
        print(f"ERROR at step {len(all_s11) + 1}/{len(bias_voltages)}: {exc}")
        if not all_s11:
            raise
        print(f"Saving partial data ({len(all_s11)} of {len(bias_voltages)} steps)...")

    completed_voltages = bias_voltages[:len(all_s11)]
    s11_arr = np.array(all_s11)

    np.savez_compressed(
        full_path,
        voltages=completed_voltages,
        freqs=freqs,
        s11_real=s11_arr.real,
        s11_imag=s11_arr.imag,
    )

    print(f"Done. Saved {len(all_s11)}/{len(bias_voltages)} steps to {full_path}")
    return full_path


def main():
    # list_resources()
    # apply_bias()
    bias_s11_sweep()


if __name__ == '__main__':
    main()
