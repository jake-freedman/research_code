"""
Berkeley Nucleonics Model 855B-12 two-channel signal generator control library.

Connect over USB (USBTMC). The resource string format is:
    'USB0::<VID>::<PID>::<serial>::INSTR'

Use pyvisa list_resources() to find the exact string, e.g.:
    import pyvisa
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())

Example
-------
    from bnc_control import BNC855B

    with BNC855B('USB0::0x03EB::0x6124::00000000::INSTR') as sig:
        sig.configure_channel(channel=1, freq_hz=1e9, power_dbm=0.0, phase_deg=0.0)
        sig.configure_channel(channel=2, freq_hz=2e9, power_dbm=-10.0, phase_deg=90.0)
        sig.enable_output(1)
        sig.enable_output(2)
"""

from __future__ import annotations

import math
import pyvisa


class BNC855B:
    """Interface to a Berkeley Nucleonics 855B two-channel signal generator over USB."""

    NUM_CHANNELS = 2

    def __init__(self, resource_name: str, timeout_ms: int = 10000):
        """
        Parameters
        ----------
        resource_name:
            VISA resource string for the USB-connected instrument.
        timeout_ms:
            VISA timeout in milliseconds. Default 10 s.
        """
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource_name)
        self._inst.timeout = timeout_ms
        self._inst.read_termination = '\n'
        self._inst.write_termination = '\n'

        idn = self._inst.query('*IDN?')
        print(f"Connected to {idn.strip()}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'BNC855B':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Disable all outputs and close the connection."""
        for ch in range(1, self.NUM_CHANNELS + 1):
            try:
                self.disable_output(ch)
            except Exception:
                pass
        self._inst.close()
        self._rm.close()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _check_channel(self, channel: int) -> None:
        if channel not in range(1, self.NUM_CHANNELS + 1):
            raise ValueError(f"channel must be 1 or 2, got {channel}")

    def reset(self) -> None:
        """Reset the instrument to factory defaults."""
        self._inst.write('*RST')
        self._inst.write('*CLS')
        self._inst.query('*OPC?')

    # ------------------------------------------------------------------
    # Per-channel configuration
    # ------------------------------------------------------------------

    def set_frequency(self, channel: int, freq_hz: float) -> None:
        """
        Set the output frequency for one channel.

        Parameters
        ----------
        channel:
            Channel index, 1 or 2.
        freq_hz:
            Frequency in Hz.
        """
        self._check_channel(channel)
        self._inst.write(f'SOUR{channel}:FREQ {float(freq_hz)} HZ')

    def set_power(self, channel: int, power_dbm: float) -> None:
        """
        Set the output power for one channel.

        Parameters
        ----------
        channel:
            Channel index, 1 or 2.
        power_dbm:
            Output power in dBm.
        """
        self._check_channel(channel)
        self._inst.write(f'SOUR{channel}:POW {float(power_dbm)} DBM')

    def set_phase(self, channel: int, phase_deg: float) -> None:
        """
        Set the output phase for one channel.

        Parameters
        ----------
        channel:
            Channel index, 1 or 2.
        phase_deg:
            Phase offset in degrees.
        """
        self._check_channel(channel)
        self._inst.write(f'SOUR{channel}:PHAS {float(phase_deg)} DEG')

    def configure_channel(
        self,
        channel: int,
        freq_hz: float,
        power_dbm: float,
        phase_deg: float = 0.0,
    ) -> None:
        """
        Set frequency, power, and phase for one channel in a single call.

        Parameters
        ----------
        channel:
            Channel index, 1 or 2.
        freq_hz:
            Frequency in Hz.
        power_dbm:
            Output power in dBm.
        phase_deg:
            Phase offset in degrees. Default 0.
        """
        self.set_frequency(channel, freq_hz)
        self.set_power(channel, power_dbm)
        self.set_phase(channel, phase_deg)

    # ------------------------------------------------------------------
    # Output state
    # ------------------------------------------------------------------

    def enable_output(self, channel: int) -> None:
        """Turn the RF output on for the given channel."""
        self._check_channel(channel)
        self._inst.write(f'OUTP{channel} ON')

    def disable_output(self, channel: int) -> None:
        """Turn the RF output off for the given channel."""
        self._check_channel(channel)
        self._inst.write(f'OUTP{channel} OFF')

    def enable_all_outputs(self) -> None:
        """Turn on both channel outputs."""
        for ch in range(1, self.NUM_CHANNELS + 1):
            self.enable_output(ch)

    def disable_all_outputs(self) -> None:
        """Turn off both channel outputs."""
        for ch in range(1, self.NUM_CHANNELS + 1):
            self.disable_output(ch)

    # ------------------------------------------------------------------
    # Reference oscillator
    # ------------------------------------------------------------------

    def set_reference(self, source: str = 'INT', ref_output: bool = False) -> None:
        """
        Configure the shared reference oscillator.

        Parameters
        ----------
        source:
            'INT' for internal reference, 'EXT' for external 10 MHz reference.
        ref_output:
            If True, enable the 10 MHz reference output connector.
        """
        src = source.upper()
        if src not in ('INT', 'EXT'):
            raise ValueError("source must be 'INT' or 'EXT'")
        self._inst.write(f'ROSC:SOUR {src}')
        self._inst.write(f'ROSC:OUTP {"ON" if ref_output else "OFF"}')

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_frequency(self, channel: int) -> float:
        """Return the currently set frequency in Hz for the given channel."""
        self._check_channel(channel)
        return float(self._inst.query(f'SOUR{channel}:FREQ?'))

    def get_power(self, channel: int) -> float:
        """Return the currently set power in dBm for the given channel."""
        self._check_channel(channel)
        return float(self._inst.query(f'SOUR{channel}:POW?'))

    def get_phase(self, channel: int) -> float:
        """Return the currently set phase in degrees for the given channel."""
        self._check_channel(channel)
        return math.degrees(float(self._inst.query(f'SOUR{channel}:PHAS?')))

    def get_output_state(self, channel: int) -> bool:
        """Return True if the RF output is enabled for the given channel."""
        self._check_channel(channel)
        resp = self._inst.query(f'OUTP{channel}?').strip()
        return resp in ('1', 'ON')
