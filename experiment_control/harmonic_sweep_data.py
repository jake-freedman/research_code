"""
Reader and analyser for harmonic sweep .npz files produced by
vna_cw_harmonic_esa_script.py.

The .npz file contains:
    cw_freqs      : (M,)    VNA drive frequencies in Hz
    harmonics     : (N,)    harmonic numbers, e.g. [1, 2, 3]
    offsets_hz    : (K,)    frequency offsets from each harmonic centre in Hz
    spectra       : (M, N, K)  ESA power in dBm
    window_hz     : scalar  half-width of each harmonic window in Hz
    esa_freq_step_hz : scalar
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import jn as bessel_jn

from graphics import (
    BLUE1, RED1, GREEN2, VIOLET1, ORANGE1, TEAL1, PINK1,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# Colour assigned to each harmonic number. Harmonics not listed fall back to
# the extras list in order.
_HARMONIC_COLORS = {1: RED1, 2: GREEN2, 3: BLUE1}
_EXTRA_COLORS = [VIOLET1, ORANGE1, TEAL1, PINK1]


# ---------------------------------------------------------------------------
# Figure helpers (shared style)
# ---------------------------------------------------------------------------

def _make_figure(axes_width_mm: float, axes_height_mm: float):
    mm = 1.0 / 25.4
    fig, ax = plt.subplots(figsize=(
        (_left_mm + axes_width_mm + _right_mm) * mm,
        (_bottom_mm + axes_height_mm + _top_mm) * mm,
    ))
    fig.subplots_adjust(
        left=_left_mm / (_left_mm + axes_width_mm + _right_mm),
        right=(_left_mm + axes_width_mm) / (_left_mm + axes_width_mm + _right_mm),
        bottom=_bottom_mm / (_bottom_mm + axes_height_mm + _top_mm),
        top=(_bottom_mm + axes_height_mm) / (_bottom_mm + axes_height_mm + _top_mm),
    )
    return fig, ax


def _style_axes(ax):
    ax.tick_params(
        axis='both', direction=tick_direction,
        width=tick_width, labelsize=tick_label_fontsize,
    )
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)


# ---------------------------------------------------------------------------
# HarmonicSweepData
# ---------------------------------------------------------------------------

class HarmonicSweepData:
    """
    Load and analyse harmonic-tracking sweep data.

    Attributes
    ----------
    cw_freqs : np.ndarray, shape (M,)
        VNA drive frequencies in Hz.
    harmonics : np.ndarray, shape (N,)
        Harmonic numbers recorded, e.g. [1, 2, 3].
    offsets_hz : np.ndarray, shape (K,)
        Frequency offsets from each harmonic centre in Hz.
    spectra : np.ndarray, shape (M, N, K)
        ESA power in dBm for each (CW step, harmonic, frequency point).
    window_hz : float
        Half-width of each harmonic window in Hz.
    """

    def __init__(self, filepath: str):
        d = np.load(filepath)
        self.cw_freqs: np.ndarray = d['cw_freqs']
        self.harmonics: np.ndarray = d['harmonics'].astype(int)
        self.offsets_hz: np.ndarray = d['offsets_hz']
        self.spectra: np.ndarray = d['spectra']
        self.window_hz: float = float(d['window_hz'])
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> 'HarmonicSweepData':
        """Load a .npz file produced by vna_cw_harmonic_esa_script.py."""
        return cls(filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _harmonic_index(self, harmonic: int) -> int:
        idx = np.where(self.harmonics == harmonic)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Harmonic {harmonic} not in dataset. "
                f"Available: {list(self.harmonics)}"
            )
        return int(idx[0])

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def peak_powers_dbm(self) -> np.ndarray:
        """
        Peak power in each harmonic window for every CW step.

        Returns
        -------
        np.ndarray, shape (M, N)
            Peak power in dBm indexed as [cw_step, harmonic].
        """
        return self.spectra.max(axis=2)

    def modulation_depth(
        self,
        harmonic_a: int = 1,
        harmonic_b: int = 3,
        beta_guess: float = 1.0,
    ) -> np.ndarray:
        """
        Extract modulation depth β at each CW frequency.

        Solves Ja(β) / Jb(β) = sqrt(Pa / Pb) at every CW step, where Pa and
        Pb are the peak powers (converted to linear) in the harmonic_a and
        harmonic_b windows respectively.

        Parameters
        ----------
        harmonic_a, harmonic_b : int
            Harmonic numbers to use for the ratio. Defaults 1 and 3.
        beta_guess : float
            Initial guess passed to fsolve at every CW step. Default 1.0.

        Returns
        -------
        np.ndarray, shape (M,)
            Modulation depth β in radians at each CW frequency.
        """
        idx_a = self._harmonic_index(harmonic_a)
        idx_b = self._harmonic_index(harmonic_b)

        peaks = self.peak_powers_dbm()               # (M, N)
        p_a = 10.0 ** (peaks[:, idx_a] / 10.0)      # dBm → mW
        p_b = 10.0 ** (peaks[:, idx_b] / 10.0)

        betas = np.empty(len(self.cw_freqs))
        for i, (pa, pb) in enumerate(zip(p_a, p_b)):
            # Use Jb(β)/Ja(β) = sqrt(Pb/Pa). This ratio is zero at β=0 and
            # rises monotonically at small β, so fsolve is well-conditioned
            # when harmonic_b is near the noise floor.
            target = np.sqrt(pb / pa)
            # print(self.cw_freqs[i], target)

            def residual(beta, t=target):
                return bessel_jn(harmonic_b, beta) / bessel_jn(harmonic_a, beta) - t

            betas[i] = float(fsolve(residual, beta_guess)[0])

        return betas

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_modulation_depth(
        self,
        harmonic_a: int = 1,
        harmonic_b: int = 3,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot modulation depth β vs CW drive frequency.

        Parameters
        ----------
        harmonic_a, harmonic_b : int
            Harmonic pair used for β extraction. Defaults 1 and 3.
        ymin, ymax : float, optional
            Y-axis limits in radians.

        Returns
        -------
        fig, ax
        """
        betas = self.modulation_depth(harmonic_a, harmonic_b)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(
            self.cw_freqs / 1e9, betas,
            color=BLUE1, linewidth=1.5, marker='o', markersize=4,
        )
        ax.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Modulation depth \u03b2 [rad]', fontsize=axis_label_fontsize)
        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        _style_axes(ax)
        return fig, ax

    def plot_harmonic_spectra(
        self,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the peak ESA power at each harmonic vs CW drive frequency.

        All harmonics in the dataset are shown on the same axes, each in a
        distinct colour (red=1, green=2, blue=3, then violet/orange/teal/pink
        for higher harmonics). Only the peak value within the harmonic window
        is plotted for each drive frequency.

        Parameters
        ----------
        ymin, ymax : float, optional
            Y-axis limits in dBm.

        Returns
        -------
        fig, ax
        """
        peaks = self.peak_powers_dbm()   # (M, N)

        extra_iter = iter(_EXTRA_COLORS)
        fig, ax = _make_figure(axes_width_mm, axes_height_mm)

        for j, n in enumerate(self.harmonics):
            color = _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000'))
            ax.plot(
                self.cw_freqs / 1e9,
                peaks[:, j],
                color=color,
                linewidth=1.5,
                marker='o',
                markersize=3,
                label=f'Harmonic {n}',
            )

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Peak power [dBm]', fontsize=axis_label_fontsize)
        ax.legend(fontsize=tick_label_fontsize, frameon=False)
        _style_axes(ax)

        return fig, ax