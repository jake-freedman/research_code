"""
Reader and analyser for heterodyne sweep .npz files produced by
vna_cw_heterodyne_sweep() in vna_cw_harmonic_esa_script.py.

The .npz file contains:
    cw_freqs         : (M,)      VNA drive frequencies in Hz
    harmonics        : (N,)      harmonic numbers, e.g. [0, 1, 2, 3]
    heterodyne_shift : scalar    LO offset in Hz (e.g. 125 MHz)
    offsets_hz       : (K,)      frequency offsets from each window centre in Hz
    spectra          : (M, N, K) ESA power in dBm
    window_hz        : scalar    half-width of each harmonic window in Hz

For harmonic n, the ESA window is centred at n*f_cw + heterodyne_shift.
Harmonic 0 therefore captures the carrier beat at heterodyne_shift itself.
Modulation depth is extracted by default from J1(β)/J0(β) = sqrt(P1/P0).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import jn as bessel_jn

from graphics import (
    BLUE2, DARKBLUE2, RED2, GREEN2, DARKGREEN2, VIOLET2, ORANGE2,
    TAN2, PINK2, DARKGRAY2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# Colour for each harmonic number. 0 = carrier (dark grey), 1 = red, 2 = green,
# 3 = blue, higher harmonics cycle through extras.
_HARMONIC_COLORS = {0: DARKGRAY2, 1: RED2, 2: GREEN2, 3: BLUE2}
_EXTRA_COLORS = [VIOLET2, ORANGE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2]


# ---------------------------------------------------------------------------
# Figure helpers
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
# HeterodyneSweepData
# ---------------------------------------------------------------------------

class HeterodyneSweepData:
    """
    Load and analyse a heterodyne harmonic sweep.

    Attributes
    ----------
    cw_freqs : np.ndarray, shape (M,)
        VNA drive frequencies in Hz.
    harmonics : np.ndarray, shape (N,)
        Harmonic numbers recorded (0 = carrier beat).
    heterodyne_shift : float
        LO offset in Hz. ESA centre for harmonic n is n*f_cw + heterodyne_shift.
    offsets_hz : np.ndarray, shape (K,)
        Frequency offsets from each window centre in Hz.
    spectra : np.ndarray, shape (M, N, K)
        ESA power in dBm for each (CW step, harmonic, frequency point).
    window_hz : float
        Half-width of each harmonic window in Hz.
    """

    def __init__(self, filepath: str):
        d = np.load(filepath)
        self.cw_freqs: np.ndarray = d['cw_freqs']
        self.harmonics: np.ndarray = d['harmonics'].astype(int)
        self.heterodyne_shift: float = float(d['heterodyne_shift'])
        self.offsets_hz: np.ndarray = d['offsets_hz']
        self.spectra: np.ndarray = d['spectra']
        self.window_hz: float = float(d['window_hz'])
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> 'HeterodyneSweepData':
        """Load a .npz file produced by vna_cw_heterodyne_sweep()."""
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
        harmonic_numerator: int = 1,
        harmonic_denominator: int = 0,
        beta_guess: float = 1.0,
    ) -> np.ndarray:
        """
        Extract modulation depth β at each CW frequency.

        Solves J_num(β) / J_den(β) = sqrt(P_num / P_den) at every CW step,
        where P values are the peak powers in each harmonic window.

        The default uses J1(β)/J0(β) = sqrt(P1/P0). Since J1(0)/J0(0) = 0,
        this is well-conditioned at small modulation depths.

        Parameters
        ----------
        harmonic_numerator : int
            Harmonic number for the numerator of the Bessel ratio. Default 1.
        harmonic_denominator : int
            Harmonic number for the denominator of the Bessel ratio. Default 0
            (carrier beat).
        beta_guess : float
            Initial guess for fsolve. Default 1.0.

        Returns
        -------
        np.ndarray, shape (M,)
            Modulation depth β in radians at each CW frequency.
        """
        idx_num = self._harmonic_index(harmonic_numerator)
        idx_den = self._harmonic_index(harmonic_denominator)

        peaks = self.peak_powers_dbm()                     # (M, N)
        p_num = 10.0 ** (peaks[:, idx_num] / 10.0)        # dBm → mW
        p_den = 10.0 ** (peaks[:, idx_den] / 10.0)

        betas = np.empty(len(self.cw_freqs))
        for i, (pn, pd) in enumerate(zip(p_num, p_den)):
            target = np.sqrt(pn / pd)

            def residual(beta, t=target):
                return (
                    bessel_jn(harmonic_numerator, beta)
                    / bessel_jn(harmonic_denominator, beta)
                    - t
                )

            betas[i] = float(fsolve(residual, beta_guess)[0])

        return betas

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_peak_powers(
        self,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot peak power at each harmonic vs CW drive frequency.

        All harmonics are shown on one axes with distinct colours.

        Returns
        -------
        fig, ax
        """
        peaks = self.peak_powers_dbm()   # (M, N)
        extra_iter = iter(_EXTRA_COLORS)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        for j, n in enumerate(self.harmonics):
            color = _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000'))
            label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n}'
            ax.plot(
                self.cw_freqs / 1e9,
                peaks[:, j],
                color=color,
                linewidth=1.5,
                marker='o',
                markersize=3,
                label=label,
            )

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Peak power [dBm]', fontsize=axis_label_fontsize)
        ax.legend(fontsize=tick_label_fontsize, frameon=False)
        _style_axes(ax)
        return fig, ax

    def plot_modulation_depth(
        self,
        harmonic_numerator: int = 1,
        harmonic_denominator: int = 0,
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot modulation depth β vs CW drive frequency.

        Parameters
        ----------
        harmonic_numerator, harmonic_denominator : int
            Harmonic pair for the Bessel ratio. Defaults 1 and 0.
        ymin, ymax : float, optional
            Y-axis limits in radians.

        Returns
        -------
        fig, ax
        """
        betas = self.modulation_depth(harmonic_numerator, harmonic_denominator)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(
            self.cw_freqs / 1e9, betas,
            color=BLUE2, linewidth=1.5, marker='o', markersize=4,
        )
        ax.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
        ax.set_ylabel('Modulation depth \u03b2 [rad]', fontsize=axis_label_fontsize)
        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        _style_axes(ax)
        return fig, ax
