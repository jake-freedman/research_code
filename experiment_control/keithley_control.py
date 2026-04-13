"""
Keithley 2450 SourceMeter control library.

Connect over VISA (USB, GPIB, or LAN). Example resource strings:
    'USB0::0x05E6::0x2450::04049675::INSTR'
    'GPIB0::24::INSTR'
    'TCPIP0::192.168.1.10::inst0::INSTR'

Example
-------
    from keithley_control import Keithley2450

    with Keithley2450('USB0::0x05E6::0x2450::04049675::INSTR') as smu:
        smu.configure_voltage_source(voltage=1.0, compliance_current=10e-3)
        smu.enable_output()
        i = smu.measure_current()
        print(f"Current: {i * 1e3:.4f} mA")
        smu.disable_output()
"""

from __future__ import annotations

import time

import numpy as np
import pyvisa


class Keithley2450:
    """Interface to a Keithley 2450 SourceMeter over VISA."""

    def __init__(self, resource_name: str, timeout_ms: int = 10000, backend: str = ''):
        """
        Parameters
        ----------
        resource_name:
            VISA resource string.
        timeout_ms:
            VISA timeout in milliseconds. Default 10 s.
        backend:
            pyvisa ResourceManager backend string.
            Use '' (default) for NI-VISA, which handles USB/IVI/USBTMC devices
            natively on Windows. Use '@py' for pyvisa-py (requires libusb).
        """
        self._rm = pyvisa.ResourceManager(backend)
        self._inst = self._rm.open_resource(resource_name)
        self._inst.timeout = timeout_ms
        self._inst.read_termination = '\n'
        self._inst.write_termination = '\n'

        idn = self._inst.query('*IDN?')
        print(f"Connected to {idn.strip()}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'Keithley2450':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Disable output and close the connection."""
        try:
            self.disable_output()
        except Exception:
            pass
        self._inst.close()
        self._rm.close()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the instrument to factory defaults and clear the status registers.

        Requires SCPI mode to be active. If the instrument shows "undefined
        header" errors, go to Menu → System → Settings → Command Set → SCPI
        on the front panel before connecting.
        """
        self._inst.clear()   # VISA Device Clear — flushes buffers on both ends
        time.sleep(0.5)
        self._inst.write('*RST')
        self._inst.write('*CLS')
        self._inst.query('*OPC?')

    # ------------------------------------------------------------------
    # Source configuration
    # ------------------------------------------------------------------

    def configure_voltage_source(
        self,
        voltage: float,
        compliance_current: float,
        voltage_range: float | None = None,
        auto_range: bool = True,
        nplc: float = 1.0,
        four_wire: bool = False,
    ) -> None:
        """
        Configure the instrument to source voltage and measure current.

        Parameters
        ----------
        voltage:
            Source voltage in V.
        compliance_current:
            Current compliance (limit) in A.
        voltage_range:
            Maximum voltage the source range must accommodate in V.
            Defaults to abs(voltage). Pass the maximum absolute voltage
            of a sweep here so the range is not set too narrow.
        auto_range:
            If True, enable auto-range on the current measurement.
        nplc:
            Integration time in power-line cycles. Higher = less noise.
            Typical values: 0.01 (fast), 1 (default), 10 (quiet).
        four_wire:
            If True, use 4-wire (remote) sense for the measurement.
        """
        v_range = abs(float(voltage_range if voltage_range is not None else voltage))
        hw_range = 0.2 if v_range <= 0.2 else 2.0 if v_range <= 2.0 else 20.0 if v_range <= 20.0 else 200.0
        self._inst.write('SOUR:FUNC VOLT')
        self._inst.write(f'SOUR:VOLT:RANG {hw_range}')
        self._inst.write(f'SOUR:VOLT {float(voltage)}')
        self._inst.write(f'SOUR:VOLT:ILIM {float(compliance_current)}')
        self._inst.write('SENS:FUNC "CURR"')
        self._inst.write(f'SENS:CURR:NPLC {float(nplc)}')
        if auto_range:
            self._inst.write('SENS:CURR:RANG:AUTO ON')
        self._inst.write(f'SENS:CURR:RSEN {"ON" if four_wire else "OFF"}')

    def configure_current_source(
        self,
        current: float,
        compliance_voltage: float,
        auto_range: bool = True,
        nplc: float = 1.0,
        four_wire: bool = False,
    ) -> None:
        """
        Configure the instrument to source current and measure voltage.

        Parameters
        ----------
        current:
            Source current in A.
        compliance_voltage:
            Voltage compliance (limit) in V.
        auto_range:
            If True, enable auto-range on the voltage measurement.
        nplc:
            Integration time in power-line cycles.
        four_wire:
            If True, use 4-wire (remote) sense.
        """
        self._inst.write('SOUR:FUNC CURR')
        self._inst.write(f'SOUR:CURR {float(current)}')
        self._inst.write(f'SOUR:CURR:VLIM {float(compliance_voltage)}')
        self._inst.write('SENS:FUNC "VOLT"')
        self._inst.write(f'SENS:VOLT:NPLC {float(nplc)}')
        if auto_range:
            self._inst.write('SENS:VOLT:RANG:AUTO ON')
        self._inst.write(f'SENS:VOLT:RSEN {"ON" if four_wire else "OFF"}')

    def source_voltage(
        self,
        voltage: float,
        compliance_current: float,
        voltage_range: float | None = None,
        nplc: float = 1.0,
        four_wire: bool = False,
    ) -> None:
        """
        Configure and immediately enable a voltage output in one call.

        Convenience wrapper around configure_voltage_source() + enable_output().

        Parameters
        ----------
        voltage:
            Output voltage in V from the FORCE terminals.
        compliance_current:
            Current compliance (limit) in A.
        voltage_range:
            Maximum voltage the source range must accommodate in V.
            Defaults to abs(voltage). Pass the maximum absolute voltage
            of a sweep here so the range is not set too narrow.
        nplc:
            Integration time in power-line cycles. Default 1.
        four_wire:
            Use 4-wire sense. Default False.
        """
        self.configure_voltage_source(
            voltage=voltage,
            compliance_current=compliance_current,
            voltage_range=voltage_range,
            nplc=nplc,
            four_wire=four_wire,
        )
        self.enable_output()

    def set_voltage(self, voltage: float) -> None:
        """Update the source voltage without changing any other setting."""
        self._inst.write(f'SOUR:VOLT {float(voltage)}')

    def set_current(self, current: float) -> None:
        """Update the source current without changing any other setting."""
        self._inst.write(f'SOUR:CURR {float(current)}')

    # ------------------------------------------------------------------
    # Output control
    # ------------------------------------------------------------------

    def enable_output(self) -> None:
        """Turn the source output on."""
        self._inst.write('OUTP ON')

    def disable_output(self) -> None:
        """Turn the source output off."""
        self._inst.write('OUTP OFF')

    # ------------------------------------------------------------------
    # Single measurements
    # ------------------------------------------------------------------

    def measure_current(self) -> float:
        """
        Trigger a single current measurement and return the result in A.

        The instrument must already be configured with configure_voltage_source().
        """
        return float(self._inst.query(':MEAS:CURR?'))

    def measure_voltage(self) -> float:
        """
        Trigger a single voltage measurement and return the result in V.

        The instrument must already be configured with configure_current_source().
        """
        return float(self._inst.query(':MEAS:VOLT?'))

    def measure_resistance(self, four_wire: bool = False) -> float:
        """
        Trigger a single resistance measurement and return the result in Ω.

        Parameters
        ----------
        four_wire:
            If True, use 4-wire (FRES) measurement. Default False (2-wire).
        """
        cmd = ':MEAS:FRES?' if four_wire else ':MEAS:RES?'
        return float(self._inst.query(cmd))

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    def voltage_sweep(
        self,
        start: float,
        stop: float,
        num_points: int,
        compliance_current: float,
        nplc: float = 1.0,
        settle_time_s: float = 0.0,
        four_wire: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step through a voltage range and measure current at each point.

        Output is enabled at the start and disabled at the end.

        Parameters
        ----------
        start, stop:
            Voltage range in V.
        num_points:
            Number of voltage steps.
        compliance_current:
            Current compliance in A.
        nplc:
            Integration time in power-line cycles.
        settle_time_s:
            Delay after setting each voltage before measuring. Default 0.
        four_wire:
            Use 4-wire sense. Default False.

        Returns
        -------
        voltages : np.ndarray
            Applied voltages in V, shape (num_points,).
        currents : np.ndarray
            Measured currents in A, shape (num_points,).
        """
        voltages = np.linspace(start, stop, num_points)
        currents = np.empty(num_points)

        self.configure_voltage_source(
            voltage=start,
            compliance_current=compliance_current,
            nplc=nplc,
            four_wire=four_wire,
        )
        self.enable_output()

        try:
            for i, v in enumerate(voltages):
                self.set_voltage(v)
                if settle_time_s > 0:
                    time.sleep(settle_time_s)
                currents[i] = self.measure_current()
        finally:
            self.disable_output()

        return voltages, currents

    def current_sweep(
        self,
        start: float,
        stop: float,
        num_points: int,
        compliance_voltage: float,
        nplc: float = 1.0,
        settle_time_s: float = 0.0,
        four_wire: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step through a current range and measure voltage at each point.

        Output is enabled at the start and disabled at the end.

        Parameters
        ----------
        start, stop:
            Current range in A.
        num_points:
            Number of current steps.
        compliance_voltage:
            Voltage compliance in V.
        nplc:
            Integration time in power-line cycles.
        settle_time_s:
            Delay after setting each current before measuring. Default 0.
        four_wire:
            Use 4-wire sense. Default False.

        Returns
        -------
        currents : np.ndarray
            Applied currents in A, shape (num_points,).
        voltages : np.ndarray
            Measured voltages in V, shape (num_points,).
        """
        currents = np.linspace(start, stop, num_points)
        voltages = np.empty(num_points)

        self.configure_current_source(
            current=start,
            compliance_voltage=compliance_voltage,
            nplc=nplc,
            four_wire=four_wire,
        )
        self.enable_output()

        try:
            for i, c in enumerate(currents):
                self.set_current(c)
                if settle_time_s > 0:
                    time.sleep(settle_time_s)
                voltages[i] = self.measure_voltage()
        finally:
            self.disable_output()

        return currents, voltages
