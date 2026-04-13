"""
Data class for bias-voltage S11 sweeps produced by bias_s11_sweep() in
mechanical_resonance_tuning_script.py.
"""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from graphics import (
    VIOLET2,
    axis_label_fontsize, tick_label_fontsize,
    spine_linewidth, tick_width, tick_direction,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)


class BiasS11SweepData:
    """S11 spectra recorded at a set of DC bias voltages."""

    def __init__(
        self,
        voltages: np.ndarray,
        freqs: np.ndarray,
        s11_complex: np.ndarray,
    ):
        """
        Parameters
        ----------
        voltages : np.ndarray, shape (N,)
            Bias voltages in V.
        freqs : np.ndarray, shape (M,)
            Frequency array in Hz.
        s11_complex : np.ndarray, shape (N, M)
            Complex S11 for each voltage step.
        """
        self.voltages = voltages
        self.freqs = freqs
        self.s11_complex = s11_complex

    @classmethod
    def from_file(cls, filepath: str) -> 'BiasS11SweepData':
        """Load a .npz file written by bias_s11_sweep()."""
        d = np.load(filepath)
        s11 = d['s11_real'] + 1j * d['s11_imag']
        return cls(d['voltages'], d['freqs'], s11)

    def plot(
        self,
        ymin: float = -30.0,
        ymax: float = 5.0,
        xmin: float | None = None,
        xmax: float | None = None,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot all S11 sweeps overlaid, coloured by bias voltage.

        A colorbar on the right maps colour to voltage. Low voltages are
        blue, high voltages are red (coolwarm colormap).

        Parameters
        ----------
        ymin, ymax : float
            S11 y-axis limits in dB.
        xmin, xmax : float or None
            Frequency x-axis limits in GHz. None = auto.
        axes_width_mm, axes_height_mm : float
            Axes area size in mm (excludes margins and colorbar).
        """
        norm = mcolors.Normalize(
            vmin=self.voltages.min(), vmax=self.voltages.max()
        )
        cmap = cm.coolwarm

        mm = 1.0 / 25.4
        fig_w = (axes_width_mm + 55) * mm   # room for left label + colorbar
        fig_h = (axes_height_mm + 30) * mm  # room for x label + top margin
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        s11_db = 20.0 * np.log10(np.abs(self.s11_complex) + 1e-300)
        for i, v in enumerate(self.voltages):
            ax.plot(self.freqs / 1e9, s11_db[i], color=cmap(norm(v)), linewidth=1.0)

        ax.set_ylim([ymin, ymax])
        if xmin is not None or xmax is not None:
            ax.set_xlim(
                left=xmin if xmin is not None else self.freqs[0] / 1e9,
                right=xmax if xmax is not None else self.freqs[-1] / 1e9,
            )
        ax.set_xlabel('Frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel(r'$S_{11}$ [dB]', fontsize=axis_label_fontsize)
        ax.tick_params(
            axis='both', direction=tick_direction,
            width=tick_width, labelsize=tick_label_fontsize,
        )
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(spine_linewidth)
        ax.grid()

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Bias voltage [V]', fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)

        fig.tight_layout()
        return fig, ax

    def resonance_freq_vs_voltage(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the frequency of minimum S11 within a frequency window at each
        bias voltage.

        Parameters
        ----------
        freq_min, freq_max : float or None
            Search window in GHz. None = use the full frequency range.

        Returns
        -------
        voltages : np.ndarray, shape (N,)
            Bias voltages in V.
        res_freqs : np.ndarray, shape (N,)
            Resonance frequency (S11 minimum) in Hz at each voltage.
        """
        mask = np.ones(len(self.freqs), dtype=bool)
        if freq_min is not None:
            mask &= self.freqs >= freq_min * 1e9
        if freq_max is not None:
            mask &= self.freqs <= freq_max * 1e9

        s11_db = 20.0 * np.log10(np.abs(self.s11_complex) + 1e-300)
        res_freqs = np.array([
            self.freqs[mask][np.argmin(s11_db[i, mask])]
            for i in range(len(self.voltages))
        ])
        return self.voltages, res_freqs

    def plot_resonance_freq_vs_voltage(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot resonance frequency vs bias voltage.

        Parameters
        ----------
        freq_min, freq_max : float or None
            Frequency search window in GHz passed to resonance_freq_vs_voltage().
        ymin, ymax : float or None
            Y-axis limits in GHz. None = auto.
        axes_width_mm, axes_height_mm : float
            Axes area size in mm.
        """
        voltages, res_freqs = self.resonance_freq_vs_voltage(freq_min, freq_max)

        mm = 1.0 / 25.4
        fig_w = (_left_mm + axes_width_mm + _right_mm) * mm
        fig_h = (_bottom_mm + axes_height_mm + _top_mm) * mm
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(
            left=_left_mm / (_left_mm + axes_width_mm + _right_mm),
            right=(_left_mm + axes_width_mm) / (_left_mm + axes_width_mm + _right_mm),
            bottom=_bottom_mm / (_bottom_mm + axes_height_mm + _top_mm),
            top=(_bottom_mm + axes_height_mm) / (_bottom_mm + axes_height_mm + _top_mm),
        )

        ax.plot(voltages, res_freqs / 1e9, color=VIOLET2, linewidth=2.0, marker='o',
                markersize=4)
        ax.set_xlabel('Bias voltage [V]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Resonance frequency [GHz]', fontsize=axis_label_fontsize)
        if ymin is not None or ymax is not None:
            ax.set_ylim(
                bottom=ymin if ymin is not None else res_freqs.min() / 1e9,
                top=ymax if ymax is not None else res_freqs.max() / 1e9,
            )
        ax.tick_params(
            axis='both', direction=tick_direction,
            width=tick_width, labelsize=tick_label_fontsize,
        )
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(spine_linewidth)
        ax.grid()

        return fig, ax
