"""
Classes for loading and plotting VNA-CW + ESA sweep data files.

Two file types are supported, both produced by the sweep scripts:
  - CW frequency sweep  (vna_cw_esa_script.py)   → CWFreqSweepData
  - CW power sweep      (vna_power_esa_script.py) → PowerSweepData

File format
-----------
Line 1 (header): swept parameter values, e.g.
    cw_freq_hz:2500000000.00,2501000000.00,...
    cw_power_dbm:-20.00,-19.00,...
Line 2 (header): column names
    esa_freq_hz,power_dbm_step0,power_dbm_step1,...
Remaining rows: numeric data

Example
-------
    from esa_sweep_data import CWFreqSweepData, PowerSweepData

    data = CWFreqSweepData.from_file(r'C:\\data\\...cw_freq_sweep....csv')
    data.plot(ymin=-110, ymax=-60)

    data = PowerSweepData.from_file(r'C:\\data\\...power_sweep....csv')
    data.plot(ymin=-110, ymax=-60)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from graphics import (
    LIGHTBLUE2, DARKBLUE2, VIOLET2, BLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

_CMAP_CWFREQ = LinearSegmentedColormap.from_list('cwfreq', [LIGHTBLUE2, DARKBLUE2])
_CMAP_POWER = LinearSegmentedColormap.from_list('power', [VIOLET2, BLUE2])


def _make_figure(axes_width_mm=_default_axes_w, axes_height_mm=_default_axes_h):
    left_mm, right_mm = _left_mm, _right_mm
    bottom_mm, top_mm = _bottom_mm, _top_mm
    mm = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=(
        (left_mm + axes_width_mm + right_mm) * mm,
        (bottom_mm + axes_height_mm + top_mm) * mm,
    ))
    fig.subplots_adjust(
        left=left_mm / (left_mm + axes_width_mm + right_mm),
        right=(left_mm + axes_width_mm) / (left_mm + axes_width_mm + right_mm),
        bottom=bottom_mm / (bottom_mm + axes_height_mm + top_mm),
        top=(bottom_mm + axes_height_mm) / (bottom_mm + axes_height_mm + top_mm),
    )
    return fig, ax


def _style_axes(ax):
    ax.tick_params(axis='both', direction=tick_direction, width=tick_width,
                   labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)


def _parse_file(filepath: str):
    """Return (swept_values, esa_freqs, spectra) from a sweep CSV."""
    with open(filepath, 'r') as f:
        line1 = f.readline().strip()   # e.g. "cw_freq_hz:2.5e9,2.51e9,..."
        f.readline()                    # column name row — skip

    _, values_str = line1.split(':', 1)
    swept_values = np.array([float(v) for v in values_str.split(',')])

    data = np.loadtxt(filepath, delimiter=',', skiprows=2)
    esa_freqs = data[:, 0]
    spectra = data[:, 1:]              # shape (n_esa_points, n_steps)

    return swept_values, esa_freqs, spectra


class CWFreqSweepData:
    """
    Load and plot a CW frequency sweep file produced by vna_cw_esa_script.py.

    Attributes
    ----------
    cw_freqs : np.ndarray
        Array of VNA CW frequencies in Hz.
    esa_freqs : np.ndarray
        ESA frequency axis in Hz.
    spectra : np.ndarray
        Power in dBm, shape (n_esa_points, n_cw_steps).
    """

    def __init__(self, filepath: str):
        self.cw_freqs, self.esa_freqs, self.spectra = _parse_file(filepath)
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> "CWFreqSweepData":
        return cls(filepath)

    def plot(
        self,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
        colormap=_CMAP_CWFREQ,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot all spectra on one axes, coloured across the CW frequency range.

        Parameters
        ----------
        ymin, ymax : float, optional
            Y-axis limits in dBm.
        colormap : colormap or str
            Matplotlib colormap. Default is LIGHTBLUE2→DARKBLUE2.

        Returns
        -------
        fig, ax
        """
        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        n = self.spectra.shape[1]
        cmap = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        colors = [cmap(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

        for i, color in enumerate(colors):
            ax.plot(self.esa_freqs / 1e9, self.spectra[:, i], color=color, linewidth=1.0)

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel('Frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Power [dBm]', fontsize=axis_label_fontsize)
        _style_axes(ax)

        # Colorbar to show which colour = which CW frequency
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=self.cw_freqs[0] / 1e9, vmax=self.cw_freqs[-1] / 1e9),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('VNA CW frequency [GHz]', fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)

        return fig, ax


class PowerSweepData:
    """
    Load and plot a CW power sweep file produced by vna_power_esa_script.py.

    Attributes
    ----------
    cw_powers : np.ndarray
        Array of VNA output powers in dBm.
    esa_freqs : np.ndarray
        ESA frequency axis in Hz.
    spectra : np.ndarray
        Power in dBm, shape (n_esa_points, n_power_steps).
    """

    def __init__(self, filepath: str):
        self.cw_powers, self.esa_freqs, self.spectra = _parse_file(filepath)
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> "PowerSweepData":
        return cls(filepath)

    def plot(
        self,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
        colormap=_CMAP_POWER,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot all spectra on one axes, coloured across the power range.

        Parameters
        ----------
        ymin, ymax : float, optional
            Y-axis limits in dBm.
        colormap : colormap or str
            Matplotlib colormap. Default is VIOLET2→BLUE2.

        Returns
        -------
        fig, ax
        """
        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        n = self.spectra.shape[1]
        cmap = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap
        colors = [cmap(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

        for i, color in enumerate(colors):
            ax.plot(self.esa_freqs / 1e9, self.spectra[:, i], color=color, linewidth=1.0)

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel('Frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Power [dBm]', fontsize=axis_label_fontsize)
        _style_axes(ax)

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=self.cw_powers[0], vmax=self.cw_powers[-1]),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('VNA output power [dBm]', fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)

        return fig, ax

    def peak_power_vs_vna_power(
        self,
        freq_min: float,
        freq_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the peak ESA power within a frequency window for each VNA power step.

        Parameters
        ----------
        freq_min, freq_max : float
            Frequency window in Hz.

        Returns
        -------
        cw_powers : np.ndarray
            VNA output power steps in dBm.
        peak_powers : np.ndarray
            Peak ESA power in dBm at each VNA power step.
        """
        mask = (self.esa_freqs >= freq_min) & (self.esa_freqs <= freq_max)
        if not np.any(mask):
            raise ValueError(
                f"No ESA frequency points between {freq_min/1e9:.4f} GHz "
                f"and {freq_max/1e9:.4f} GHz."
            )
        peak_powers = self.spectra[mask, :].max(axis=0)
        return self.cw_powers, peak_powers

    def plot_peak_vs_power(
        self,
        freq_min: float,
        freq_max: float,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the peak ESA power within [freq_min, freq_max] as a function of
        VNA output power.

        Parameters
        ----------
        freq_min, freq_max : float
            Frequency window in Hz.
        ymin, ymax : float, optional
            Y-axis limits in dBm.

        Returns
        -------
        fig, ax
        """
        cw_powers, peak_powers = self.peak_power_vs_vna_power(freq_min, freq_max)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(cw_powers, peak_powers, color=DARKBLUE2, linewidth=1.5, marker='o',
                markersize=4)

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel('VNA output power [dBm]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Peak ESA power [dBm]', fontsize=axis_label_fontsize)
        _style_axes(ax)

        return fig, ax
