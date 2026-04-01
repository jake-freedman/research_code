"""
Keysight P9374A VNA control library.

The VNA must be addressed over HiSLIP (TCP/IP), NOT via the raw PXI resource.
The correct resource string format is:
    'TCPIP0::<hostname>::hislip_PXI<chassis>_CHASSIS<n>_SLOT<n>_INDEX0::INSTR'

Example
-------
    from vna_control import VNA

    VNA_RESOURCE = 'TCPIP0::HAL9000::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'

    with VNA(VNA_RESOURCE) as vna:
        print(vna.get_available_cal_sets())

        vna.configure(
            start_freq=10e6,
            stop_freq=10e9,
            freq_step=1e6,
            power_dbm=-10,
            ifbw=1000,
            cal_set='MyCalSet',
        )
        vna.apply_calibration()
        freqs, s11 = vna.sweep_s11()
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvisa


class VNA:
    """Interface to a Keysight P9374A VNA over HiSLIP/TCP-IP."""

    def __init__(self, resource_name: str, timeout_ms: int = 60000):
        """
        Parameters
        ----------
        resource_name:
            HiSLIP VISA resource string, e.g.
            'TCPIP0::HAL9000::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'
        timeout_ms:
            VISA operation timeout in milliseconds. Default is 60s to cover
            long sweeps. sweep_s11() extends this further based on sweep time.
        """
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource_name)
        self._inst.timeout = timeout_ms

        idn = self._inst.query('*IDN?')
        print(f"Connected to {idn.strip()}")

        self._start_freq: float | None = None
        self._stop_freq: float | None = None
        self._num_points: int | None = None
        self._power_dbm: float | None = None
        self._ifbw: float | None = None
        self._cal_set: str | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VNA":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._inst.close()
        self._rm.close()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        start_freq: float,
        stop_freq: float,
        freq_step: float,
        power_dbm: float = -10.0,
        ifbw: float = 1000.0,
        cal_set: str | None = None,
    ) -> None:
        """
        Set sweep parameters and push them to the instrument.

        Parameters
        ----------
        start_freq:
            Start frequency in Hz.
        stop_freq:
            Stop frequency in Hz.
        freq_step:
            Frequency step in Hz. Number of points = round((stop - start) / step) + 1.
        power_dbm:
            Output power in dBm.
        ifbw:
            IF bandwidth in Hz.
        cal_set:
            Cal Set name as stored on the VNA. Pass None to skip calibration.
            Use get_available_cal_sets() to list names.
        """
        if stop_freq <= start_freq:
            raise ValueError("stop_freq must be greater than start_freq")
        if freq_step <= 0:
            raise ValueError("freq_step must be positive")

        self._start_freq = float(start_freq)
        self._stop_freq = float(stop_freq)
        self._num_points = round((stop_freq - start_freq) / freq_step) + 1
        self._power_dbm = float(power_dbm)
        self._ifbw = float(ifbw)
        self._cal_set = cal_set

        self._inst.write('SENS1:SWEep:TYPE LIN')
        self._inst.write(f'SENS1:FREQ:STAR {self._start_freq}')
        self._inst.write(f'SENS1:FREQ:STOP {self._stop_freq}')
        self._inst.write(f'SENS1:SWE:POIN {self._num_points}')
        self._inst.write(f'SOURCE1:POW {self._power_dbm}')
        self._inst.write(f'SENS1:BAND {self._ifbw}')

        # Ensure an S11 measurement exists and is selected on channel 1.
        catalog = self._inst.query('CALC1:PAR:CAT:EXT?').strip().strip('"')
        if catalog and catalog != 'NO CATALOG':
            # Catalog format is "name,param,name,param,..." — select the first
            first_meas = catalog.split(',')[0].strip()
        else:
            self._inst.write("CALC1:PAR:DEF:EXT 'S11_MEAS','S11'")
            self._inst.write("DISP:WIND1:TRAC1:FEED 'S11_MEAS'")
            first_meas = 'S11_MEAS'
        self._inst.write(f"CALC1:PAR:SEL '{first_meas}'")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def apply_calibration(self, cal_set: str | None = None) -> None:
        """
        Activate a Cal Set on channel 1.

        Parameters
        ----------
        cal_set:
            Cal Set name. If None, uses the value set in configure().
            The bool argument is 0 so our configured stimulus frequencies
            are preserved (interpolation is used if the Cal Set frequencies
            differ from the channel frequencies).
        """
        name = cal_set or self._cal_set
        if name is None:
            raise ValueError(
                "No Cal Set specified. Pass cal_set= to configure() or apply_calibration()."
            )
        self._inst.write(f'SENS1:CORR:CSET:ACT "{name}",0')
        self._inst.query("*OPC?")

    def active_cal_set(self) -> str:
        """Return the name of the Cal Set currently active on channel 1.
        Returns 'No Calset Selected' if none is active."""
        return self._inst.query("SENS1:CORR:CSET:ACT? NAME").strip().strip('"')

    def get_available_cal_sets(self) -> list[str]:
        """Return a list of Cal Set names stored on the VNA."""
        resp = self._inst.query("SENS1:CORR:CSET:CAT? NAME").strip().strip('"')
        if not resp:
            return []
        return [s.strip() for s in resp.split(",")]

    # ------------------------------------------------------------------
    # Sweep and data collection
    # ------------------------------------------------------------------

    def sweep_s11(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Trigger a single S11 sweep and return the data.

        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz, shape (num_points,).
        s11_complex : np.ndarray
            Complex S11 (linear), shape (num_points,).
        """
        if self._num_points is None:
            raise RuntimeError("Call configure() before sweep_s11().")

        # Extend timeout to cover the sweep: 5 time-constants per point at the
        # IF filter rate, plus 10 s headroom for instrument overhead.
        sweep_time_ms = 10000 # int((5.0 / self._ifbw) * self._num_points * 1000) + 10000
        prev_timeout = self._inst.timeout
        self._inst.timeout = max(prev_timeout, sweep_time_ms)

        # Trigger a single sweep and wait for completion
        self._inst.query('SENS1:SWE:MODE SING;*OPC?')

        # Reconstruct frequency axis from the parameters we already set
        freqs = np.linspace(self._start_freq, self._stop_freq, self._num_points)

        # Read S11 as interleaved real/imag pairs
        raw_data = self._inst.query('CALC1:DATA? SDATA')
        values = np.array([float(v) for v in raw_data.split(',')])
        s11_complex = values[0::2] + 1j * values[1::2]

        # Return to continuous sweep
        self._inst.query('SENS1:SWE:MODE CONT;*OPC?')
        self._inst.timeout = prev_timeout

        return freqs, s11_complex

    def sweep_s11_s21(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trigger a single sweep and return S11 and S21.

        Both measurements must already exist on the VNA channel, or this
        method will create them. Use configure() first to set frequencies,
        power, and IFBW.

        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz, shape (num_points,).
        s11_complex : np.ndarray
            Complex S11 (linear), shape (num_points,).
        s21_complex : np.ndarray
            Complex S21 (linear), shape (num_points,).
        """
        if self._num_points is None:
            raise RuntimeError("Call configure() before sweep_s11_s21().")

        # Ensure both S11 and S21 measurements exist on channel 1
        catalog = self._inst.query('CALC1:PAR:CAT:EXT?').strip().strip('"')
        existing = {}
        if catalog and catalog != 'NO CATALOG':
            tokens = [t.strip() for t in catalog.split(',')]
            # catalog is "name,param,name,param,..."
            for i in range(0, len(tokens) - 1, 2):
                existing[tokens[i + 1].upper()] = tokens[i]

        if 'S11' not in existing:
            self._inst.write("CALC1:PAR:DEF:EXT 'S11_MEAS','S11'")
            self._inst.write("DISP:WIND1:TRAC1:FEED 'S11_MEAS'")
            existing['S11'] = 'S11_MEAS'
        if 'S21' not in existing:
            self._inst.write("CALC1:PAR:DEF:EXT 'S21_MEAS','S21'")
            self._inst.write("DISP:WIND1:TRAC2:FEED 'S21_MEAS'")
            existing['S21'] = 'S21_MEAS'

        sweep_time_ms = 10000
        prev_timeout = self._inst.timeout
        self._inst.timeout = max(prev_timeout, sweep_time_ms)

        self._inst.query('SENS1:SWE:MODE SING;*OPC?')

        freqs = np.linspace(self._start_freq, self._stop_freq, self._num_points)

        def _read_meas(name: str) -> np.ndarray:
            self._inst.write(f"CALC1:PAR:SEL '{name}'")
            raw = self._inst.query('CALC1:DATA? SDATA')
            v = np.array([float(x) for x in raw.split(',')])
            return v[0::2] + 1j * v[1::2]

        s11_complex = _read_meas(existing['S11'])
        s21_complex = _read_meas(existing['S21'])

        self._inst.query('SENS1:SWE:MODE CONT;*OPC?')
        self._inst.timeout = prev_timeout

        return freqs, s11_complex, s21_complex

    def save_s11_s21(
        self,
        freqs: np.ndarray,
        s11: np.ndarray,
        s21: np.ndarray,
        folder: str,
        filename: str | None = None,
        optional_name: str = '',
    ) -> str:
        """
        Save S11 and S21 data to a CSV file.

        Columns: frequency_hz, s11_real, s11_imag, s11_db, s21_real, s21_imag, s21_db.

        Returns
        -------
        str
            Full path of the saved file.
        """
        os.makedirs(folder, exist_ok=True)

        if filename is None:
            filename = 's11_s21_' + optional_name + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'

        s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
        s21_db = 20.0 * np.log10(np.abs(s21) + 1e-300)

        data = np.column_stack([freqs, s11.real, s11.imag, s11_db, s21.real, s21.imag, s21_db])
        full_path = os.path.join(folder, filename)
        np.savetxt(
            full_path,
            data,
            delimiter=',',
            header='frequency_hz,s11_real,s11_imag,s11_db,s21_real,s21_imag,s21_db',
            comments='',
        )
        return full_path

    def sweep_s11_db(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper returning S11 magnitude in dB.

        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz.
        s11_db : np.ndarray
            S11 magnitude in dB.
        """
        freqs, s11 = self.sweep_s11()
        s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
        return freqs, s11_db

    def save_s11(
        self,
        freqs: np.ndarray,
        s11: np.ndarray,
        folder: str,
        filename: str | None = None,
        optional_name: str = '',
    ) -> str:
        """
        Save S11 data to a CSV file.

        Columns: frequency_hz, s11_real, s11_imag, s11_db.
        If s11 is already real (dB), the real column contains those values,
        imag is zeros, and s11_db is the same.

        Parameters
        ----------
        freqs:
            Frequency array in Hz.
        s11:
            S11 array, either complex linear or real dB.
        folder:
            Directory to save into. Created if it does not exist.
        filename:
            File name (without path). If None, a timestamped name is used:
            's11_YYYY-MM-DD-HH-MM-SS.csv'.

        Returns
        -------
        str
            Full path of the saved file.
        """
        os.makedirs(folder, exist_ok=True)

        if filename is None:
            filename = 's11_' + optional_name + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'

        if np.iscomplexobj(s11):
            s11_real = s11.real
            s11_imag = s11.imag
            s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
        else:
            s11_real = np.asarray(s11)
            s11_imag = np.zeros_like(s11_real)
            s11_db = s11_real

        data = np.column_stack([freqs, s11_real, s11_imag, s11_db])
        full_path = os.path.join(folder, filename)
        np.savetxt(
            full_path,
            data,
            delimiter=',',
            header='frequency_hz,s11_real,s11_imag,s11_db',
            comments='',
        )
        return full_path

    # ------------------------------------------------------------------
    # CW mode
    # ------------------------------------------------------------------

    def set_cw_mode(self, freq: float, power: float) -> None:
        """
        Put the VNA into CW (single-frequency) output mode.

        The VNA outputs a continuous wave at the given frequency and power.
        Call set_cw_freq() to step to a new frequency without changing power.
        Call configure() to return to a swept measurement.

        Parameters
        ----------
        freq:
            CW frequency in Hz.
        power:
            Output power in dBm.
        """
        self._inst.write('SENS1:SWE:TYPE CW')
        self._inst.write(f'SENS1:FREQ {float(freq)}')
        self._inst.write(f'SOURCE1:POW {float(power)}')
        self._inst.write('SENS1:SWE:MODE CONT')

    def set_cw_freq(self, freq: float) -> None:
        """Update the CW frequency without changing any other settings."""
        self._inst.write(f'SENS1:FREQ {float(freq)}')

    def set_cw_power(self, power: float) -> None:
        """
        Update the CW output power.

        Briefly pauses and restarts the continuous sweep so the new power
        level is guaranteed to take effect before returning.

        Parameters
        ----------
        power:
            Output power in dBm.
        """
        self._inst.write('SENS1:SWE:MODE HOLD')
        self._inst.write(f'SOURCE1:POW {float(power)}')
        self._inst.write('SENS1:SWE:MODE CONT')

    def cw_off(self) -> None:
        """Stop CW output by putting the VNA sweep into hold."""
        self._inst.write('SENS1:SWE:MODE HOLD')

    def plot_s11_s21(
        self,
        freqs: np.ndarray,
        s11: np.ndarray,
        s21: np.ndarray,
        axes_width_mm: float = 180.0,
        axes_height_mm: float = 100.0,
        s21_ymin: float = -30.0,
        s21_ymax: float = 5.0,
        s11_ymin: float = -30.0,
        s11_ymax: float = 5.0,
        xmin: float = 0.0,
        xmax: float = 10.0,
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plot S21 on the left y-axis and S11 on the right y-axis."""
        return _plot_s11_s21(
            freqs, s11, s21, axes_width_mm, axes_height_mm,
            s21_ymin, s21_ymax, s11_ymin, s11_ymax, xmin, xmax,
        )

    def plot_s11(
        self,
        freqs: np.ndarray,
        s11: np.ndarray,
        axes_width_mm: float = 180.0,
        axes_height_mm: float = 100.0,
        ymin: float = -30.0,
        ymax: float = 5.0,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot S11 vs frequency.

        Parameters
        ----------
        freqs:
            Frequency array in Hz.
        s11:
            S11 values. Complex arrays are converted to dB magnitude;
            real arrays are plotted as-is (assumed already in dB).
        axes_width_mm:
            Width of the axes area in mm (excludes margins). Default 180.
        axes_height_mm:
            Height of the axes area in mm (excludes margins). Default 100.
        ymin:
            Lower y-axis limit in dB. Default -30.
        ymax:
            Upper y-axis limit in dB. Default 5.

        Returns
        -------
        fig, ax
        """
        return _plot_s11(freqs, s11, axes_width_mm, axes_height_mm, ymin, ymax)


# ---------------------------------------------------------------------------
# Module-level plot helper (shared by VNA and S11Data)
# ---------------------------------------------------------------------------

def _plot_s11(
    freqs: np.ndarray,
    s11: np.ndarray,
    axes_width_mm: float = 180.0,
    axes_height_mm: float = 100.0,
    ymin: float = -30.0,
    ymax: float = 5.0,
    is_grid: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    if np.iscomplexobj(s11):
        y = 20.0 * np.log10(np.abs(s11) + 1e-300)
    else:
        y = np.asarray(s11)

    left_mm, right_mm = 14.0, 5.0
    bottom_mm, top_mm = 12.0, 5.0

    mm = 1.0 / 25.4
    fig_w = (left_mm + axes_width_mm + right_mm) * mm
    fig_h = (bottom_mm + axes_height_mm + top_mm) * mm
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fig.subplots_adjust(
        left=left_mm / (left_mm + axes_width_mm + right_mm),
        right=(left_mm + axes_width_mm) / (left_mm + axes_width_mm + right_mm),
        bottom=bottom_mm / (bottom_mm + axes_height_mm + top_mm),
        top=(bottom_mm + axes_height_mm) / (bottom_mm + axes_height_mm + top_mm),
    )

    ax.plot(freqs / 1e9, y, color='#C2B7E9', linewidth=3.00)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Frequency [GHz]', fontsize=10)
    ax.set_ylabel(r'$S_{11}$ [dB]', fontsize=10)
    ax.tick_params(axis='both', direction='in', width=2, labelsize=8)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2)

    if is_grid:
        ax.grid()

    return fig, ax


# ---------------------------------------------------------------------------
# Module-level S11+S21 plot helper
# ---------------------------------------------------------------------------

def _plot_s11_s21(
    freqs: np.ndarray,
    s11: np.ndarray,
    s21: np.ndarray,
    axes_width_mm: float = 180.0,
    axes_height_mm: float = 100.0,
    s21_ymin: float = -30.0,
    s21_ymax: float = 5.0,
    s11_ymin: float = -30.0,
    s11_ymax: float = 5.0,
    xmin: float = 0.0,
    xmax: float = 10.0,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-300)
    s21_db = 20.0 * np.log10(np.abs(s21) + 1e-300)

    left_mm, right_mm = 20.0, 20.0
    bottom_mm, top_mm = 12.0, 5.0

    mm = 1.0 / 25.4
    fig_w = (left_mm + axes_width_mm + right_mm) * mm
    fig_h = (bottom_mm + axes_height_mm + top_mm) * mm
    fig, ax_s21 = plt.subplots(figsize=(fig_w, fig_h))

    fig.subplots_adjust(
        left=left_mm / (left_mm + axes_width_mm + right_mm),
        right=(left_mm + axes_width_mm) / (left_mm + axes_width_mm + right_mm),
        bottom=bottom_mm / (bottom_mm + axes_height_mm + top_mm),
        top=(bottom_mm + axes_height_mm) / (bottom_mm + axes_height_mm + top_mm),
    )

    # S21 on the left axis
    ax_s21.plot(freqs / 1e9, s21_db, color='#9BB9D9', label=r'$S_{21}$')
    ax_s21.set_ylim([s21_ymin, s21_ymax])
    ax_s21.set_xlim([xmin, xmax])
    ax_s21.set_xlabel('Frequency [GHz]', fontsize=10)
    ax_s21.set_ylabel(r'$S_{21}$ [dB]', fontsize=10, color='#000000')
    ax_s21.tick_params(axis='y', direction='in', width=2, labelsize=8, colors='#000000')
    ax_s21.tick_params(axis='x', direction='in', width=2, labelsize=8)

    # S11 on the right axis
    ax_s11 = ax_s21.twinx()
    ax_s11.plot(freqs / 1e9, s11_db, color='#D9B99B', label=r'$S_{11}$')
    ax_s11.set_ylim([s11_ymin, s11_ymax])
    ax_s11.set_ylabel(r'$S_{11}$ [dB]', fontsize=10, color='#000000')
    ax_s11.tick_params(axis='y', direction='in', width=2, labelsize=8, colors='#000000')

    for side in ['top', 'bottom', 'left', 'right']:
        ax_s21.spines[side].set_linewidth(2)
        ax_s11.spines[side].set_linewidth(2)

    lines = [ax_s21.lines[0], ax_s11.lines[0]]
    labels = [r'$S_{21}$', r'$S_{11}$']
    ax_s21.legend(lines, labels, fontsize=8)

    return fig, ax_s21, ax_s11


# ---------------------------------------------------------------------------
# S11Data — load and plot previously saved S11 CSV files
# ---------------------------------------------------------------------------

class S11Data:
    """Load and plot S11 data previously saved by VNA.save_s11()."""

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath:
            Path to a CSV file written by VNA.save_s11().
            Expected columns: frequency_hz, s11_real, s11_imag, s11_db.
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        self.freqs: np.ndarray = data[:, 0]
        self.s11_complex: np.ndarray = data[:, 1] + 1j * data[:, 2]
        self.s11_db: np.ndarray = data[:, 3]
        self.filepath: str = filepath

    @classmethod
    def from_file(cls, filepath: str) -> "S11Data":
        """Load an S11 CSV file by its full path."""
        return cls(filepath)

    def plot_s11(
        self,
        axes_width_mm: float = 180.0,
        axes_height_mm: float = 100.0,
        ymin: float = -30.0,
        ymax: float = 5.0,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the loaded S11 data. Same style as VNA.plot_s11()."""
        return _plot_s11(
            self.freqs, self.s11_complex, axes_width_mm, axes_height_mm, ymin, ymax
        )


class S11S21Data:
    """Load and plot S11+S21 data previously saved by VNA.save_s11_s21()."""

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath:
            Path to a CSV file written by VNA.save_s11_s21().
            Expected columns: frequency_hz, s11_real, s11_imag, s11_db,
                              s21_real, s21_imag, s21_db.
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        self.freqs: np.ndarray = data[:, 0]
        self.s11_complex: np.ndarray = data[:, 1] + 1j * data[:, 2]
        self.s11_db: np.ndarray = data[:, 3]
        self.s21_complex: np.ndarray = data[:, 4] + 1j * data[:, 5]
        self.s21_db: np.ndarray = data[:, 6]
        self.filepath: str = filepath

    @classmethod
    def from_file(cls, filepath: str) -> "S11S21Data":
        """Load an S11+S21 CSV file by its full path."""
        return cls(filepath)

    def plot_s11_s21(
        self,
        axes_width_mm: float = 180.0,
        axes_height_mm: float = 100.0,
        s21_ymin: float = -30.0,
        s21_ymax: float = 5.0,
        s11_ymin: float = -30.0,
        s11_ymax: float = 5.0,
        xmin: float = 0.0,
        xmax: float = 10.0,
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Plot S21 on the left y-axis and S11 on the right y-axis."""
        return _plot_s11_s21(
            self.freqs, self.s11_complex, self.s21_complex,
            axes_width_mm, axes_height_mm, s21_ymin, s21_ymax, s11_ymin, s11_ymax, xmin, xmax,
        )
