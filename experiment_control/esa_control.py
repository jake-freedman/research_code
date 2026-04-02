"""
Rohde & Schwarz FSV Signal and Spectrum Analyzer control library.

Connect over LAN. The resource string format is:
    'TCPIP0::<ip_address>::INSTR'

Example
-------
    from esa_control import ESA

    with ESA('TCPIP0::192.168.1.50::INSTR') as esa:
        esa.configure(
            start_freq=1e6,
            stop_freq=6e9,
            freq_step=1e6,
            res_bw=1e6,
            video_bw=1e6,
            ref_level=0.0,
        )
        freqs, power_db = esa.sweep()
        esa.save(freqs, power_db, folder=r'C:\data\esa')
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvisa
from scipy.optimize import fsolve
from scipy.special import j1, jn
from graphics import (
    GREEN2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)


class ESA:
    """Interface to a Rohde & Schwarz FSV spectrum analyzer over LAN."""

    def __init__(self, resource_name: str, timeout_ms: int = 30000):
        """
        Parameters
        ----------
        resource_name:
            VISA resource string, e.g. 'TCPIP0::192.168.1.50::INSTR'
        timeout_ms:
            VISA timeout in milliseconds. Increased automatically during sweeps.
        """
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource_name)
        self._inst.timeout = timeout_ms
        self._inst.read_termination = '\n'
        self._inst.write_termination = '\n'

        idn = self._inst.query('*IDN?')
        print(f"Connected to {idn.strip()}")

        self._start_freq: float | None = None
        self._stop_freq: float | None = None
        self._num_points: int | None = None
        self._res_bw: float | None = None
        self._video_bw: float | None = None
        self._ref_level: float | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ESA":
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
        res_bw: float,
        video_bw: float | None = None,
        ref_level: float = 0.0,
        attenuation: float = 0.0,
        detector: str = 'RMS',
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
            Frequency step in Hz. Number of points = round((stop - start) / step) + 1,
            clamped to the instrument max of 32001.
        res_bw:
            Resolution bandwidth in Hz.
        video_bw:
            Video bandwidth in Hz. If None, set to equal res_bw.
        ref_level:
            Reference level (top of display) in dBm. Default 0.
        attenuation:
            RF input attenuation in dB. Default 0 (no attenuation).
        detector:
            Detector type. One of: 'RMS', 'PEAK', 'AVER', 'SAMP', 'NEG'.
            Default 'RMS'.
        """
        if stop_freq <= start_freq:
            raise ValueError("stop_freq must be greater than start_freq")
        if freq_step <= 0:
            raise ValueError("freq_step must be positive")

        self._start_freq = float(start_freq)
        self._stop_freq = float(stop_freq)
        self._num_points = min(round((stop_freq - start_freq) / freq_step) + 1, 32001)
        self._res_bw = float(res_bw)
        self._video_bw = float(video_bw) if video_bw is not None else float(res_bw)
        self._ref_level = float(ref_level)

        self._inst.write(f'FREQ:STAR {self._start_freq}')
        self._inst.write(f'FREQ:STOP {self._stop_freq}')
        self._inst.write(f'SWE:POIN {self._num_points}')
        self._inst.write(f'BAND:RES {self._res_bw}')
        self._inst.write(f'BAND:VID {self._video_bw}')
        self._inst.write(f'DISP:WIND:TRAC:Y:RLEV {self._ref_level}')
        self._inst.write(f'DET {detector}')

        self._inst.write('INP:ATT:AUTO OFF')
        self._inst.write(f'INP:ATT {float(attenuation)}')

    # ------------------------------------------------------------------
    # Sweep and data collection
    # ------------------------------------------------------------------

    def sweep(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Trigger a single sweep and return the trace data.

        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz, shape (num_points,).
        power_db : np.ndarray
            Power spectral density in dBm, shape (num_points,).
        """
        if self._num_points is None:
            raise RuntimeError("Call configure() before sweep().")

        # Estimate sweep time: ~1/(RBW) per point plus headroom
        sweep_time_ms = max(int((1.0 / self._res_bw) * self._num_points * 1000) + 5000, 10000)
        prev_timeout = self._inst.timeout
        self._inst.timeout = max(prev_timeout, sweep_time_ms)

        # Single sweep and wait for completion
        self._inst.write('INIT:CONT OFF')
        self._inst.write('INIT:IMM')
        self._inst.query('*OPC?')

        # Read trace
        raw = self._inst.query('TRAC? TRACE1')
        power_db = np.array([float(v) for v in raw.split(',')])

        self._inst.timeout = prev_timeout

        # Use the actual number of returned points — the instrument may have
        # snapped num_points to its own internal value.
        freqs = np.linspace(self._start_freq, self._stop_freq, len(power_db))
        return freqs, power_db

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        freqs: np.ndarray,
        power_db: np.ndarray,
        folder: str,
        filename: str | None = None,
        optional_name: str = '',
    ) -> str:
        """
        Save sweep data to a CSV file.

        Columns: frequency_hz, power_dbm.

        Returns
        -------
        str
            Full path of the saved file.
        """
        os.makedirs(folder, exist_ok=True)

        if filename is None:
            filename = 'esa_' + optional_name + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'

        data = np.column_stack([freqs, power_db])
        full_path = os.path.join(folder, filename)
        np.savetxt(
            full_path,
            data,
            delimiter=',',
            header='frequency_hz,power_dbm',
            comments='',
        )
        return full_path

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(
        self,
        freqs: np.ndarray,
        power_db: np.ndarray,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the spectrum.

        Parameters
        ----------
        freqs:
            Frequency array in Hz.
        power_db:
            Power array in dBm.
        axes_width_mm, axes_height_mm:
            Size of the axes area in mm. Defaults 180 x 100.
        ymin, ymax:
            Y-axis limits in dBm. If None, matplotlib auto-scales.
        """
        return _plot_spectrum(freqs, power_db, axes_width_mm, axes_height_mm, ymin, ymax)


# ---------------------------------------------------------------------------
# Module-level plot helper
# ---------------------------------------------------------------------------

def _plot_spectrum(
    freqs: np.ndarray,
    power_db: np.ndarray,
    axes_width_mm: float = _default_axes_w,
    axes_height_mm: float = _default_axes_h,
    ymin: float | None = None,
    ymax: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    left_mm, right_mm = _left_mm, _right_mm
    bottom_mm, top_mm = _bottom_mm, _top_mm

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

    ax.plot(freqs / 1e9, power_db, color=GREEN2)

    if ymin is not None or ymax is not None:
        ax.set_ylim([ymin, ymax])

    ax.set_xlabel('Frequency [GHz]', fontsize=axis_label_fontsize)
    ax.set_ylabel('Power [dBm]', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', direction=tick_direction, width=tick_width,
                   labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)

    return fig, ax


# ---------------------------------------------------------------------------
# ESAData — load and plot previously saved ESA CSV files
# ---------------------------------------------------------------------------

class ESAData:
    """Load and plot spectrum data previously saved by ESA.save()."""

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath:
            Path to a CSV file written by ESA.save().
            Expected columns: frequency_hz, power_dbm.
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        self.freqs: np.ndarray = data[:, 0]
        self.power_db: np.ndarray = data[:, 1]
        self.filepath: str = filepath

    @classmethod
    def from_file(cls, filepath: str) -> "ESAData":
        """Load an ESA CSV file by its full path."""
        return cls(filepath)

    def modulation_depth(
        self,
        mod_freq: float,
        window_hz: float = 20e6,
        beta_guess: float = 1.0,
    ) -> float:
        """Extract the MZM sinusoidal modulation depth beta from the ESA spectrum.

        Finds the peak power within a window around the fundamental (mod_freq)
        and third harmonic (3 * mod_freq), converts from dBm to linear power,
        then solves sqrt(P1/P3) = J1(beta) / J3(beta) for beta.

        Parameters
        ----------
        mod_freq : float
            Modulation frequency in Hz.
        window_hz : float
            Half-width of the search window around each harmonic in Hz.
            Default 20 MHz.
        beta_guess : float
            Initial guess for the modulation depth in radians. Default 1.0.

        Returns
        -------
        float
            Modulation depth beta in radians.
        """
        def _peak_power_linear(center_hz):
            mask = np.abs(self.freqs - center_hz) <= window_hz
            if not np.any(mask):
                raise ValueError(
                    f"No frequency points within {window_hz/1e6:.1f} MHz of "
                    f"{center_hz/1e9:.4f} GHz."
                )
            peak_dbm = self.power_db[mask].max()
            return 10.0 ** (peak_dbm / 10.0)  # mW

        p1 = _peak_power_linear(mod_freq)
        p3 = _peak_power_linear(3.0 * mod_freq)

        target = np.sqrt(p1 / p3)  # = J1(beta) / J3(beta)

        def residual(beta):
            return j1(beta) / jn(3, beta) - target

        beta_arr = fsolve(residual, beta_guess, full_output=False)
        return float(beta_arr[0])

    def plot(
        self,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the loaded spectrum. Same style as ESA.plot()."""
        return _plot_spectrum(self.freqs, self.power_db, axes_width_mm, axes_height_mm, ymin, ymax)
