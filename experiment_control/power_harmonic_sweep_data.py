"""
Reader and analyser for power-sweep heterodyne harmonic .npz files produced by
vna_power_harmonic_esa_script.py or bnc_power_harmonic_esa_script.py.

The .npz file contains:
    cw_freq          : scalar    fixed drive frequency in Hz
    cw_powers        : (M,)      output powers in dBm
    harmonics        : (N,)      harmonic numbers, e.g. [0, 1]
    heterodyne_shift : scalar    LO offset in Hz (e.g. 125 MHz)
    offsets_hz       : (K,)      frequency offsets from each window centre in Hz
    spectra          : (M, N, K) ESA power in dBm
    window_hz        : scalar    half-width of each harmonic window in Hz
    cal_spectra      : (M, K)    carrier-beat power (RF off) — optional,
                                 present when per_step_calibration=True

The x-axis for plots is RMS RF voltage at the modulator input, converted from
the drive power assuming a 50 Ω system:
    V_rms = 10 ^ ((P_dBm - 10*log10(20)) / 20)   [V]
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import jn as bessel_jn

from graphics import (
    BLUE2, DARKGRAY2, RED2, GREEN2, DARKBLUE2,
    VIOLET2, ORANGE2, PINK2, TAN2, DARKGREEN2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

_HARMONIC_COLORS = {0: DARKGRAY2, 1: RED2, 2: GREEN2, 3: BLUE2}
_EXTRA_COLORS = [VIOLET2, ORANGE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2]


def _dbm_to_vrms(power_dbm: np.ndarray) -> np.ndarray:
    """Convert dBm (50 Ω sinusoid) to RMS voltage in volts."""
    return 10.0 ** ((power_dbm - 10.0 * np.log10(20.0)) / 20.0)


def _x_axis_values(data, x_axis: str) -> tuple[np.ndarray, str]:
    """Return (x_values, xlabel) for the requested x_axis mode."""
    if x_axis == 'dbm':
        return data.cw_powers, 'Drive power [dBm]'
    return data.rf_voltage_rms(), 'RF voltage [V$_\\mathrm{rms}$]'


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


class PowerHeterodyneSweepData:
    """
    Load and analyse a heterodyne power sweep.

    The VNA is held at a fixed CW frequency and stepped through output powers.
    At each power level, ESA spectra are recorded around each harmonic window.

    Attributes
    ----------
    cw_freq : float
        Fixed VNA drive frequency in Hz.
    cw_powers : np.ndarray, shape (M,)
        VNA output powers in dBm.
    harmonics : np.ndarray, shape (N,)
        Harmonic numbers recorded (0 = carrier beat).
    heterodyne_shift : float
        LO offset in Hz.
    offsets_hz : np.ndarray, shape (K,)
        Frequency offsets from each window centre in Hz.
    spectra : np.ndarray, shape (M, N, K)
        ESA power in dBm for each (power step, harmonic, frequency point).
    window_hz : float
        Half-width of each harmonic window in Hz.
    """

    def __init__(self, filepath: str):
        d = np.load(filepath)
        self.cw_freq: float = float(d['cw_freq'])
        self.cw_powers: np.ndarray = d['cw_powers']
        self.harmonics: np.ndarray = d['harmonics'].astype(int)
        self.heterodyne_shift: float = float(d['heterodyne_shift'])
        self.offsets_hz: np.ndarray = d['offsets_hz']
        self.spectra: np.ndarray = d['spectra']
        self.window_hz: float = float(d['window_hz'])
        self.cal_spectra: np.ndarray | None = (
            d['cal_spectra'] if 'cal_spectra' in d else None
        )
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> 'PowerHeterodyneSweepData':
        """Load a .npz file produced by vna_power_heterodyne_sweep()."""
        return cls(filepath)

    def _harmonic_index(self, harmonic: int) -> int:
        idx = np.where(self.harmonics == harmonic)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Harmonic {harmonic} not in dataset. "
                f"Available: {list(self.harmonics)}"
            )
        return int(idx[0])

    def rf_voltage_rms(self) -> np.ndarray:
        """RMS RF voltage in volts at the VNA output (50 Ω), shape (M,)."""
        return _dbm_to_vrms(self.cw_powers)

    def peak_powers_dbm(self) -> np.ndarray:
        """Peak power in each harmonic window at every power step, shape (M, N)."""
        return self.spectra.max(axis=2)

    def cal_peak_power_dbm(self) -> np.ndarray:
        """
        Peak carrier power from the per-step calibration (RF off). Shape (M,).
        Raises RuntimeError if the file was recorded without per_step_calibration.
        """
        if self.cal_spectra is None:
            raise RuntimeError(
                "No per-step calibration in this file. "
                "Record with per_step_calibration=True to enable normalisation."
            )
        return self.cal_spectra.max(axis=1)

    def normalized_peak_powers_dbm(self) -> np.ndarray:
        """
        Sideband powers relative to the per-step calibration carrier, in dBc.
        Shape (M, N). Requires per_step_calibration=True when recording.
        """
        return self.peak_powers_dbm() - self.cal_peak_power_dbm()[:, np.newaxis]

    def modulation_depth(
        self,
        harmonic_numerator: int = 1,
        harmonic_denominator: int = 0,
        beta_guess: float | np.ndarray = 1.0,
    ) -> np.ndarray:
        """
        Extract modulation depth β at each drive power level.

        Solves J_num(β) / J_den(β) = sqrt(P_num / P_den) at every power step.
        Default uses J1(β)/J0(β) = sqrt(P1/P0).

        Parameters
        ----------
        harmonic_numerator, harmonic_denominator : int
            Harmonic pair for the Bessel ratio. Defaults 1 and 0.
        beta_guess : float or np.ndarray
            Initial guess for fsolve. A scalar is used for the first step and
            then carried forward (each step seeds the next with the previous
            solution), which is robust when β increases monotonically with
            drive power. Pass an array of length M to specify per-step guesses
            explicitly (no carry-forward). Default 1.0.

        Returns
        -------
        np.ndarray, shape (M,)
            Modulation depth β in radians at each power step.
        """
        idx_num = self._harmonic_index(harmonic_numerator)
        idx_den = self._harmonic_index(harmonic_denominator)

        peaks = self.peak_powers_dbm()
        p_num = 10.0 ** (peaks[:, idx_num] / 10.0)
        p_den = 10.0 ** (peaks[:, idx_den] / 10.0)

        carry_forward = np.ndim(beta_guess) == 0
        guesses = (np.full(len(self.cw_powers), float(beta_guess))
                   if carry_forward else np.asarray(beta_guess, dtype=float))

        betas = np.empty(len(self.cw_powers))
        for i, (pn, pd) in enumerate(zip(p_num, p_den)):
            target = np.sqrt(pn / pd)

            def residual(beta, t=target):
                return (
                    bessel_jn(harmonic_numerator, beta)
                    / bessel_jn(harmonic_denominator, beta)
                    - t
                )

            betas[i] = float(fsolve(residual, guesses[i])[0])
            if carry_forward and i + 1 < len(guesses):
                guesses[i + 1] = betas[i]

        return betas

    def plot_peak_powers(
        self,
        normalize=False,
        x_axis: str = 'voltage',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot peak power at each harmonic vs RF drive voltage or power.

        Parameters
        ----------
        normalize : bool or str
            False     → raw dBm (default).
            True      → dBc relative to the per-step calibration carrier.
            'percent' → linear fraction of carrier power, in %.
            Normalization requires the file to have been recorded with
            per_step_calibration=True; raises RuntimeError otherwise.
        x_axis : str
            'voltage' plots RMS voltage in V; 'dbm' plots drive power in dBm.
        """
        if normalize == 'percent':
            peaks  = 10.0 ** (self.normalized_peak_powers_dbm() / 10.0) * 100.0
            ylabel = 'Sideband power [% of carrier]'
        elif normalize:
            peaks  = self.normalized_peak_powers_dbm()
            ylabel = 'Sideband power [dBc]'
        else:
            peaks  = self.peak_powers_dbm()
            ylabel = 'Peak power [dBm]'
        x, xlabel = _x_axis_values(self, x_axis)
        extra_iter = iter(_EXTRA_COLORS)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        for j, n in enumerate(self.harmonics):
            color = _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000'))
            label = 'Carrier (n=0)' if n == 0 else f'Harmonic {n}'
            ax.plot(
                x, peaks[:, j],
                color=color, linewidth=1.5,
                marker='o', markersize=3, label=label,
            )

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.legend(fontsize=tick_label_fontsize, frameon=False)
        _style_axes(ax)
        return fig, ax

    def plot_calibration(
        self,
        x_axis: str = 'voltage',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot per-step calibration carrier power (RF off) vs drive level."""
        x, xlabel = _x_axis_values(self, x_axis)
        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(
            x, self.cal_peak_power_dbm(),
            color=DARKGRAY2, linewidth=1.5, marker='o', markersize=3,
        )
        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel('Carrier power [dBm]', fontsize=axis_label_fontsize)
        _style_axes(ax)
        return fig, ax

    def plot_modulation_depth(
        self,
        harmonic_numerator: int = 1,
        harmonic_denominator: int = 0,
        beta_guess: float | np.ndarray = 1.0,
        x_axis: str = 'voltage',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
        fit: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot modulation depth β vs RF drive voltage or power.

        Parameters
        ----------
        beta_guess : float or np.ndarray
            Initial guess for fsolve. Scalar → carry-forward mode.
            See modulation_depth() for full description.
        x_axis : str
            'voltage' (default) plots RMS voltage in V; 'dbm' plots drive
            power in dBm.
        fit : bool
            If True, overlay a linear fit and annotate the drive level at
            which β = π (V_π in voltage mode, P_π in dBm mode). Default False.
        """
        betas = self.modulation_depth(harmonic_numerator, harmonic_denominator, beta_guess)
        x, xlabel = _x_axis_values(self, x_axis)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(
            x, betas,
            color=BLUE2, linewidth=1.5, marker='o', markersize=4,
        )

        if fit:
            slope, intercept = np.polyfit(x, betas, 1)
            x_pi = (np.pi - intercept) / slope
            x_fit = np.array([x[0], x[-1]])
            if x_axis == 'dbm':
                fit_label = f'Linear fit  $P_\\pi$ = {x_pi:.2f} dBm'
                print(f"Linear fit: slope = {slope:.4f} rad/dBm,  P_π = {x_pi:.4f} dBm")
            else:
                fit_label = f'Linear fit  $V_\\pi$ = {x_pi:.3f} V$_{{\\mathrm{{rms}}}}$'
                print(f"Linear fit: slope = {slope:.4f} rad/V,  V_π = {x_pi:.4f} V_rms")
            ax.plot(
                x_fit, slope * x_fit + intercept,
                color=RED2, linewidth=1.2, linestyle='--', label=fit_label,
            )
            ax.axhline(np.pi, color='gray', linewidth=0.8, linestyle=':')
            ax.legend(fontsize=tick_label_fontsize, frameon=False)

        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel('Modulation depth β [rad]', fontsize=axis_label_fontsize)
        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        _style_axes(ax)
        return fig, ax
