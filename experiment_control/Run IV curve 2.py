"""
Keithley 2450 SMU - IV Curve Measurement
Requires: pyvisa, pyvisa-py (or NI-VISA backend)
Install: pip install pyvisa pyvisa-py
"""

import csv
import os
import pyvisa
import sys
from datetime import datetime
from time import perf_counter, sleep

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

RESOURCE_ADDRESS = 'USB0::0x05E6::0x2450::04615671::INSTR'

DEV_NAME       = "TEST"           # Folder name for this device
VOLTAGE_RANGE  = [-10, 4]         # [V_min, V_max] in volts
STEP_SIZE      = 0.1              # Voltage step size in volts (always positive)
SWEEP_DIR      = "UP"         # UP, DOWN, UPDOWN, or DOWNUP
CURRENT_LIMIT  = 0.1              # Compliance current limit in amps
NPLC           = 1                # Integration time in power line cycles (0.01–10)
SWEEPS         = 2                # Number of consecutive IV sweeps to perform

WARM_UP_VOLTAGE = -1             # Voltage held before the first sweep (V)
WARM_UP_TIME    = 10             # Duration to hold warm-up voltage (s)

DWELL_VOLTAGE  = -10              # Voltage held before each sweep (V)
DWELL_TIME     = 5              # Duration to hold dwell voltage (s)

REST_VOLTAGE   = 4              # Voltage held between sweeps (V)
REST_TIME      = 5.0              # Duration to hold rest voltage between sweeps (s)

RAMP_STEPS     = 20               # Number of steps for all voltage ramps

SAVE_ROOT      = r"C:\Users\Admin\Documents\Alex\IV curve measurements"

# ==============================================================================


def build_sweep(v_min: float, v_max: float, step: float, direction: str) -> list[float]:
    """Build the voltage sweep sequence from the configured parameters."""
    if direction not in ("UP", "DOWN", "UPDOWN", "DOWNUP"):
        print(f"[ERROR] Invalid SWEEP_DIR '{direction}'. Must be UP, DOWN, UPDOWN, or DOWNUP.")
        sys.exit(1)

    n_steps = round((v_max - v_min) / step)
    up   = [round(v_min + i * step, 10) for i in range(n_steps + 1)]
    down = list(reversed(up))

    return {"UP": up, "DOWN": down, "UPDOWN": up + down[1:], "DOWNUP": down + up[1:]}[direction]


# def build_save_path(root: str, dev_name: str) -> str:
#     """Construct the output directory path using current date and time."""
#     now  = datetime.now()
#     date = now.strftime("%m%d")   # MMDD
#     time = now.strftime("%H%M")   # HHMM military
#     path = os.path.join(root, dev_name, date, time)
#     os.makedirs(path, exist_ok=True)
#     return path


def connect(rm: pyvisa.ResourceManager) -> pyvisa.resources.Resource:
    try:
        smu = rm.open_resource(RESOURCE_ADDRESS)
    except pyvisa.errors.VisaIOError as e:
        print(f"[ERROR] Could not open resource: {e}")
        sys.exit(1)

    smu.timeout           = 15000  # ms — bumped from 10000
    smu.read_termination  = "\n"
    smu.write_termination = "\n"
    return smu


def reset(smu: pyvisa.resources.Resource) -> None:
    smu.clear()   # VISA device clear — flushes buffers on both ends
    sleep(0.5)    # brief settle after clear before issuing SCPI
    smu.write("*RST")
    smu.write("*CLS")


def identify(smu: pyvisa.resources.Resource) -> str:
    return smu.query("*IDN?").strip()



def configure(smu: pyvisa.resources.Resource, current_limit: float, nplc: float) -> None:
    """Configure the SMU for voltage sourcing and current measurement."""
    v_range  = max(abs(VOLTAGE_RANGE[0]), abs(VOLTAGE_RANGE[1]))
    hw_range = 200.0 if v_range > 20.0 else 20.0

    smu.write("SOUR:FUNC VOLT")
    smu.write(f"SOUR:VOLT:RANG {hw_range}")
    smu.write(f"SOUR:VOLT:ILIM {current_limit}")
    smu.write('SENS:FUNC "CURR"')
    smu.write("SENS:CURR:RANG:AUTO ON")
    smu.write(f"SENS:CURR:NPLC {nplc}")


def ramp_voltage(smu: pyvisa.resources.Resource, v_from: float, v_to: float, n_steps: int) -> float:
    """
    Ramp output voltage from v_from to v_to in n_steps steps, as fast as the
    instrument responds. Returns the duration of the ramp in seconds.
    If v_from == v_to, sets the voltage immediately and returns 0.
    """
    t_start = perf_counter()

    if v_from == v_to or n_steps <= 0:
        smu.write(f"SOUR:VOLT:LEV {v_to}")
        return perf_counter() - t_start

    # n_steps intervals means n_steps+1 points including both endpoints
    steps = [round(v_from + (v_to - v_from) * i / n_steps, 10) for i in range(n_steps + 1)]
    for v in steps:
        smu.write(f"SOUR:VOLT:LEV {v}")

    return perf_counter() - t_start


def hold_voltage(smu: pyvisa.resources.Resource, voltage: float, duration: float) -> None:
    """Set voltage and hold for the specified duration (seconds)."""
    smu.write(f"SOUR:VOLT:LEV {voltage}")
    sleep(duration)


def run_sweep(smu: pyvisa.resources.Resource, voltages: list[float]) -> tuple[list[float], float]:
    """
    Step through voltages and measure current at each point.
    Returns the list of currents and the sweep duration in seconds.
    """
    currents = []
    t_start  = perf_counter()

    for v in voltages:
        smu.write(f"SOUR:VOLT:LEV {v}")
        current = float(smu.query("MEAS:CURR?").strip())
        currents.append(current)
        print(f"  V = {v:+.4f} V    I = {current:+.6e} A")

    sweep_time = perf_counter() - t_start
    return currents, sweep_time


def save_csv(path: str, voltages: list[float], all_currents: list[list[float]]) -> str:
    """Write IV data to CSV: one voltage column, one current column per sweep."""
    filepath = os.path.join(path, "iv_curve.csv")
    n_sweeps = len(all_currents)

    header = ["Voltage (V)"] + [f"Current_{i+1} (A)" for i in range(n_sweeps)]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for j, v in enumerate(voltages):
            row = [v] + [all_currents[i][j] for i in range(n_sweeps)]
            writer.writerow(row)

    return filepath


def save_metadata(
    path: str,
    sweep_times: list[float],
    total_time: float,
    idn: str,
    warm_up_ramp_time: float,
    pre_sweep_ramp_times: list[float],
    rest_ramp_times: list[float],
) -> str:
    """Write timing and configuration metadata to metadata.txt."""
    filepath = os.path.join(path, "metadata.txt")

    def fmt(seconds: float) -> str:
        return f"{seconds:.3f} s"

    with open(filepath, "w") as f:
        f.write("=== Keithley 2450 IV Curve Metadata ===\n\n")

        f.write("[Instrument]\n")
        f.write(f"  IDN:             {idn}\n")
        f.write(f"  Address:         {RESOURCE_ADDRESS}\n\n")

        f.write("[Configuration]\n")
        f.write(f"  Device name:     {DEV_NAME}\n")
        f.write(f"  Voltage range:   {VOLTAGE_RANGE[0]} V to {VOLTAGE_RANGE[1]} V\n")
        f.write(f"  Step size:       {STEP_SIZE} V\n")
        f.write(f"  Sweep direction: {SWEEP_DIR}\n")
        f.write(f"  Current limit:   {CURRENT_LIMIT} A\n")
        f.write(f"  NPLC:            {NPLC}\n")
        f.write(f"  Sweeps:          {SWEEPS}\n\n")

        f.write("[Warm-Up]\n")
        f.write(f"  Warm-up voltage:      {WARM_UP_VOLTAGE} V\n")
        f.write(f"  Warm-up hold time:    {WARM_UP_TIME} s\n")
        f.write(f"  Ramp to warm-up:      {fmt(warm_up_ramp_time)}\n\n")

        f.write("[Dwell]\n")
        f.write(f"  Dwell voltage:        {DWELL_VOLTAGE} V\n")
        f.write(f"  Dwell hold time:      {DWELL_TIME} s\n\n")

        f.write("[Rest]\n")
        f.write(f"  Rest voltage:         {REST_VOLTAGE} V\n")
        f.write(f"  Rest hold time:       {REST_TIME} s\n\n")

        f.write("[Ramp]\n")
        f.write(f"  Ramp steps:           {RAMP_STEPS}\n\n")

        f.write("[Timing]\n")
        for i in range(SWEEPS):
            f.write(f"  Sweep {i+1}:\n")
            f.write(f"    Ramp to dwell:      {fmt(pre_sweep_ramp_times[i])}\n")
            f.write(f"    Dwell hold:         {DWELL_TIME:.3f} s\n")
            ramp_to_sweep_start = pre_sweep_ramp_times[i]  # already logged above
            # Ramp dwell→sweep start is included in pre_sweep_ramp_times as two sub-ramps;
            # the sweep time is measured independently.
            f.write(f"    Sweep duration:     {fmt(sweep_times[i])}\n")
            if i < SWEEPS - 1:
                f.write(f"    Ramp to rest:       {fmt(rest_ramp_times[i])}\n")
                f.write(f"    Rest hold:          {REST_TIME:.3f} s\n")
        f.write(f"\n  Total sweep time:    {fmt(sum(sweep_times))}\n")
        f.write(f"  Total script time:   {fmt(total_time)}\n")

    return filepath


def main() -> None:
    t_script_start = perf_counter()

    voltages  = build_sweep(VOLTAGE_RANGE[0], VOLTAGE_RANGE[1], STEP_SIZE, SWEEP_DIR)
    # save_path = build_save_path(SAVE_ROOT, DEV_NAME)

    print(f"Device:          {DEV_NAME}")
    print(f"Sweep:           {VOLTAGE_RANGE[0]} V to {VOLTAGE_RANGE[1]} V, "
          f"step {STEP_SIZE} V, direction {SWEEP_DIR}")
    print(f"Points:          {len(voltages)} per sweep")
    print(f"Sweeps:          {SWEEPS}")
    print(f"NPLC:            {NPLC}")
    print(f"Warm-up:         {WARM_UP_VOLTAGE} V for {WARM_UP_TIME} s")
    print(f"Dwell:           {DWELL_VOLTAGE} V for {DWELL_TIME} s (before each sweep)")
    print(f"Rest:            {REST_VOLTAGE} V for {REST_TIME} s (between sweeps)")
    print(f"Ramp steps:      {RAMP_STEPS}")
    # print(f"Save path:       {save_path}")
    print(f"Connecting to:   {RESOURCE_ADDRESS}\n")

    rm  = pyvisa.ResourceManager()
    smu = connect(rm)
    idn = identify(smu)
    print(f"Connected:   {idn}\n")

    reset(smu)
    configure(smu, CURRENT_LIMIT, NPLC)

    all_currents       = []
    sweep_times        = []
    pre_sweep_ramp_times = []  # ramp rest→dwell + ramp dwell→sweep_start, per sweep
    rest_ramp_times    = []    # ramp sweep_end→rest, per inter-sweep gap

    sweep_start_voltage = voltages[0]

    smu.write("OUTP ON")
    # hold_voltage(smu, 10.000, 5)
    try:
        # ------------------------------------------------------------------
        # Warm-up phase: ramp from 0 V → warm-up voltage, hold, then rest
        # ------------------------------------------------------------------
        print("--- Warm-up ---")
        print(f"  Ramping 0 V → {WARM_UP_VOLTAGE} V ({RAMP_STEPS} steps)...")
        warm_up_ramp_time = ramp_voltage(smu, 0.0, WARM_UP_VOLTAGE, RAMP_STEPS)
        print(f"  Holding {WARM_UP_VOLTAGE} V for {WARM_UP_TIME} s...")
        hold_voltage(smu, WARM_UP_VOLTAGE, WARM_UP_TIME)
        print(f"  Warm-up complete ({warm_up_ramp_time:.3f} s ramp + {WARM_UP_TIME:.3f} s hold)\n")

        # After warm-up, ramp to rest voltage to start the inter-sweep cadence
        print(f"  Ramping warm-up → rest ({WARM_UP_VOLTAGE} V → {REST_VOLTAGE} V)...")
        ramp_voltage(smu, WARM_UP_VOLTAGE, REST_VOLTAGE, RAMP_STEPS)

        for i in range(SWEEPS):
            print(f"--- Pre-sweep {i+1}/{SWEEPS} ---")

            # Ramp: rest voltage → dwell voltage
            print(f"  Ramping rest → dwell ({REST_VOLTAGE} V → {DWELL_VOLTAGE} V)...")
            t_ramp_start = perf_counter()
            ramp_voltage(smu, REST_VOLTAGE, DWELL_VOLTAGE, RAMP_STEPS)

            # Hold dwell voltage
            print(f"  Holding dwell voltage {DWELL_VOLTAGE} V for {DWELL_TIME} s...")
            hold_voltage(smu, DWELL_VOLTAGE, DWELL_TIME)

            # Ramp: dwell voltage → sweep start voltage
            print(f"  Ramping dwell → sweep start ({DWELL_VOLTAGE} V → {sweep_start_voltage} V)...")
            ramp_voltage(smu, DWELL_VOLTAGE, sweep_start_voltage, RAMP_STEPS)
            pre_sweep_ramp_time = perf_counter() - t_ramp_start
            pre_sweep_ramp_times.append(pre_sweep_ramp_time)

            # Run sweep
            print(f"--- Sweep {i+1}/{SWEEPS} ---")
            currents, sweep_time = run_sweep(smu, voltages)
            all_currents.append(currents)
            sweep_times.append(sweep_time)
            print(f"  Sweep {i+1} completed in {sweep_time:.3f} s\n")

            # Rest between sweeps (skip after final sweep)
            if i < SWEEPS - 1:
                print(f"--- Rest {i+1} ---")
                print(f"  Ramping sweep end → rest ({voltages[-1]} V → {REST_VOLTAGE} V)...")
                rest_ramp_time = ramp_voltage(smu, voltages[-1], REST_VOLTAGE, RAMP_STEPS)
                rest_ramp_times.append(rest_ramp_time)
                print(f"  Holding rest voltage {REST_VOLTAGE} V for {REST_TIME} s...")
                hold_voltage(smu, REST_VOLTAGE, REST_TIME)
                print()

    finally:
        smu.write("OUTP OFF")

    # smu.close()
    # rm.close()

    # total_time = perf_counter() - t_script_start

    # csv_path  = save_csv(save_path, voltages, all_currents)
    # meta_path = save_metadata(
    #     save_path, sweep_times, total_time, idn,
    #     warm_up_ramp_time, pre_sweep_ramp_times, rest_ramp_times,
    # )

    # print(f"Data saved to:     {csv_path}")
    # print(f"Metadata saved to: {meta_path}")
    # print(f"\nTotal script time: {total_time:.3f} s")


if __name__ == "__main__":
    main()