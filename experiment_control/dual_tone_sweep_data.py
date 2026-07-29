"""
Reader and analyser for dual-tone sweep .npz files produced by
bnc_dual_tone_esa_script.py.

The .npz file contains:
    drive_freqs      : (M,)      ch1 frequency f in Hz (ch2 is at 2*f)
    ch1_powers_dbm   : (M,)      channel 1 power in dBm
    ch2_powers_dbm   : (M,)      channel 2 power in dBm
    ch1_phases_deg   : (M,)      channel 1 phase in degrees
    ch2_phases_deg   : (M,)      channel 2 phase in degrees
    harmonics        : (N,)      harmonic numbers n
    heterodyne_shift : scalar    LO offset in Hz
    window_hz        : scalar    half-width of each ESA window in Hz
    esa_freq_step_hz : scalar
    offsets_hz       : (K,)      frequency offsets within each window
    spectra          : (M, N, K) ESA power in dBm with both RF channels on
    cal_spectra      : (M, K)    ESA power in dBm with RF off (carrier beat)
    ref_freq                 : scalar    drive_freqs[0] used for single-tone preamble
    ref_cal_spectrum         : (K,)     preamble: carrier beat with RF off
    ch1_ref_spectra          : (N, K)   preamble: ch1-only on at f, harmonic windows
    ch2_ref_spectra          : (N, K)   preamble: ch2-only on at 2f, harmonic windows
    ch1_ref_carrier_spectrum : (K,)     preamble: carrier window with ch1 on (J₀ ref for β1)
    ch2_ref_carrier_spectrum : (K,)     preamble: carrier window with ch2 on (J₀ ref for β2)
    n_sweep_repeats          : scalar   number of repeats actually completed
    spectra_all              : (R,M,N,K) per-repeat spectra (only present when R > 1)
    cal_spectra_all          : (R,M,K)  per-repeat cal (only when R>1 and per_step_cal)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import jn as bessel_jn

from graphics import (
    BLUE2, DARKGRAY2, RED2, GREEN2, DARKBLUE2,
    VIOLET2, ORANGE2, PINK2, TAN2, DARKGREEN2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

_HARMONIC_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_EXTRA_COLORS = [BLUE2, PINK2, TAN2, DARKGREEN2, DARKBLUE2, DARKGRAY2]

# dBm → V_rms at 50 Ω:  V = 10^((P_dBm - 10·log10(20)) / 20)
_LOG20 = 10.0 * np.log10(20.0)

# Maps x_axis key → (array getter, axis label)
_X_AXIS_DEFS: dict[str, tuple] = {
    'drive_freq':   (lambda d: d.drive_freqs / 1e9,                                      'Drive frequency $f$ [GHz]'),
    'ch1_power':    (lambda d: d.ch1_powers_dbm,                                         'Ch1 drive power [dBm]'),
    'ch2_power':    (lambda d: d.ch2_powers_dbm,                                         'Ch2 drive power [dBm]'),
    'ch1_voltage':  (lambda d, _l=_LOG20: 10.0 ** ((d.ch1_powers_dbm - _l) / 20.0),     r'Ch1 drive voltage [V$_\mathrm{rms}$]'),
    'ch2_voltage':  (lambda d, _l=_LOG20: 10.0 ** ((d.ch2_powers_dbm - _l) / 20.0),     r'Ch2 drive voltage [V$_\mathrm{rms}$]'),
    'ch1_phase':    (lambda d: d.ch1_phases_deg,                                         'Ch1 phase [deg]'),
    'ch2_phase':    (lambda d: d.ch2_phases_deg,                                         'Ch2 phase [deg]'),
    'stability':    (lambda d: np.arange(len(d.drive_freqs)),                            'Measurement step'),
}


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
    if ax.name == 'polar':
        ax.spines['polar'].set_linewidth(spine_linewidth)
    else:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(spine_linewidth)


# ---------------------------------------------------------------------------
# DualToneSweepData
# ---------------------------------------------------------------------------

class DualToneSweepData:
    """
    Load and analyse dual-tone sweep data.

    Attributes
    ----------
    drive_freqs : np.ndarray, shape (M,)
        Ch1 drive frequencies f in Hz. Ch2 is always at 2*f.
    ch1_powers_dbm, ch2_powers_dbm : np.ndarray, shape (M,)
    ch1_phases_deg, ch2_phases_deg : np.ndarray, shape (M,)
    harmonics : np.ndarray, shape (N,)
        Harmonic numbers n. ESA windows centred at n*f + heterodyne_shift.
    heterodyne_shift : float
        LO offset in Hz.
    offsets_hz : np.ndarray, shape (K,)
        Frequency offsets from each window centre in Hz.
    spectra : np.ndarray, shape (M, N, K)
        ESA power in dBm with both RF channels on.
    cal_spectra : np.ndarray, shape (M, K)
        ESA power in dBm with RF off, centred at heterodyne_shift (carrier).
    """

    def __init__(self, filepath: str):
        d = np.load(filepath)
        self.drive_freqs:     np.ndarray = d['drive_freqs']
        self.ch1_powers_dbm:  np.ndarray = d['ch1_powers_dbm']
        self.ch2_powers_dbm:  np.ndarray = d['ch2_powers_dbm']
        self.ch1_phases_deg:  np.ndarray = d['ch1_phases_deg']
        self.ch2_phases_deg:  np.ndarray = d['ch2_phases_deg']
        self.harmonics:       np.ndarray = d['harmonics'].astype(int)
        self.heterodyne_shift: float     = float(d['heterodyne_shift'])
        self.window_hz:        float     = float(d['window_hz'])
        self.offsets_hz:      np.ndarray = d['offsets_hz']
        self.spectra:         np.ndarray = d['spectra']
        self.cal_spectra:     np.ndarray = d['cal_spectra']
        self.ref_freq: float | None = float(d['ref_freq']) if 'ref_freq' in d else None
        self.ch1_ref_spectra: np.ndarray | None = (
            d['ch1_ref_spectra'] if 'ch1_ref_spectra' in d else None
        )
        self.ch2_ref_spectra: np.ndarray | None = (
            d['ch2_ref_spectra'] if 'ch2_ref_spectra' in d else None
        )
        self.ref_cal_spectrum: np.ndarray | None = (
            d['ref_cal_spectrum'] if 'ref_cal_spectrum' in d else None
        )
        self.ch1_ref_carrier_spectrum: np.ndarray | None = (
            d['ch1_ref_carrier_spectrum'] if 'ch1_ref_carrier_spectrum' in d else None
        )
        self.ch2_ref_carrier_spectrum: np.ndarray | None = (
            d['ch2_ref_carrier_spectrum'] if 'ch2_ref_carrier_spectrum' in d else None
        )
        self.ch1_enabled: bool = bool(d['ch1_enabled']) if 'ch1_enabled' in d else True
        self.ch2_enabled: bool = bool(d['ch2_enabled']) if 'ch2_enabled' in d else True
        self.n_repeats: int = int(d['n_sweep_repeats']) if 'n_sweep_repeats' in d else 1
        self.spectra_all: np.ndarray | None = (
            d['spectra_all'] if 'spectra_all' in d else None
        )
        self.cal_spectra_all: np.ndarray | None = (
            d['cal_spectra_all'] if 'cal_spectra_all' in d else None
        )
        self.spectra_avgs: np.ndarray | None = (
            d['spectra_avgs'] if 'spectra_avgs' in d else None
        )  # shape (M, N, A, K) — individual sweeps before per-point averaging
        self.n_averages_per_point: int = (
            int(d['n_averages_per_point']) if 'n_averages_per_point' in d else 1
        )
        self.filepath = filepath

    @classmethod
    def from_file(cls, filepath: str) -> 'DualToneSweepData':
        """Load a .npz file produced by bnc_dual_tone_esa_script.py."""
        return cls(filepath)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def peak_powers_dbm(self) -> np.ndarray:
        """Peak sideband power per step and harmonic. Shape (M, N)."""
        return self.spectra.max(axis=2)

    def cal_peak_power_dbm(self) -> np.ndarray:
        """Peak carrier power from calibration (RF off) spectra. Shape (M,)."""
        return self.cal_spectra.max(axis=1)

    def normalized_peak_powers_dbm(self) -> np.ndarray:
        """
        Sideband powers relative to the calibration carrier level, in dBc.
        Shape (M, N).

        normalized[i, j] = peak_power[i, j] - cal_peak[i]

        Subtracting the per-step calibration removes optical power drifts
        and LO fluctuations common to all sidebands.
        """
        return self.peak_powers_dbm() - self.cal_peak_power_dbm()[:, np.newaxis]

    def peak_powers_all_dbm(self) -> np.ndarray:
        """
        Per-repeat peak sideband powers in dBm. Shape (R, M, N).
        Requires spectra_all (n_sweep_repeats > 1); raises RuntimeError otherwise.
        """
        if self.spectra_all is None:
            raise RuntimeError(
                "Per-repeat spectra not available (file was recorded with n_sweep_repeats=1)."
            )
        return self.spectra_all.max(axis=3)

    def normalized_peak_powers_all_dbm(self) -> np.ndarray:
        """
        Per-repeat peak powers relative to the per-step calibration, in dBc.
        Shape (R, M, N). Requires spectra_all.

        Uses the mean cal_spectra (M, K) as the reference for each repeat, so
        the per-step carrier level is shared across repeats.
        """
        cal_db = self.cal_peak_power_dbm()  # (M,)
        return self.peak_powers_all_dbm() - cal_db[np.newaxis, :, np.newaxis]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_values(self, x_axis: str) -> tuple[np.ndarray, str]:
        if x_axis not in _X_AXIS_DEFS:
            raise ValueError(
                f"x_axis must be one of {list(_X_AXIS_DEFS)}; got '{x_axis}'"
            )
        getter, label = _X_AXIS_DEFS[x_axis]
        return getter(self), label

    def _harmonic_index(self, harmonic: int) -> int:
        idx = np.where(self.harmonics == harmonic)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Harmonic {harmonic} not in dataset. "
                f"Available: {list(self.harmonics)}"
            )
        return int(idx[0])

    def single_tone_modulation_depths(
        self,
        ch1_harm_num: int = 1,
        ch2_harm_num: int = 2,
        ch2_bessel_num: int = 1,
        beta_guess: float = 1.0,
    ) -> tuple[float, float]:
        """
        Compute modulation depths from the single-tone preamble measurements.

        The J₀ denominator uses the carrier measured WITH each tone on
        (ch1_ref_carrier_spectrum / ch2_ref_carrier_spectrum), giving the
        correct ratio J_n(β)/J₀(β) = sqrt(P_sideband / P_carrier_on).

        β1 (ch1-only at f):
            J_{ch1_harm_num}(β1) / J₀(β1) = sqrt(P_sideband_ch1 / P_carrier_ch1_on)
        β2 (ch2-only at 2f):
            J_{ch2_bessel_num}(β2) / J₀(β2) = sqrt(P_sideband_ch2 / P_carrier_ch2_on)
            ch2_harm_num selects the harmonic window in f-space (default 2 = 1st
            sideband of 2f); ch2_bessel_num is its Bessel order in 2f-space (default 1).

        Returns (beta1, beta2) in radians.
        Raises RuntimeError for files without preamble data.
        """
        ch1_ok = (self.ch1_ref_spectra is not None
                  and self.ch1_ref_carrier_spectrum is not None)
        ch2_ok = (self.ch2_ref_spectra is not None
                  and self.ch2_ref_carrier_spectrum is not None)
        if not ch1_ok and not ch2_ok:
            raise RuntimeError("No single-tone preamble data in this file.")

        def _solve(p_sideband_dbm, p_carrier, bessel_num):
            p_s = 10.0 ** (p_sideband_dbm / 10.0)
            target = np.sqrt(p_s / p_carrier)
            def residual(beta, t=target):
                return bessel_jn(bessel_num, abs(beta)) / bessel_jn(0, abs(beta)) - t
            return abs(float(fsolve(residual, beta_guess)[0]))

        beta1 = None
        if ch1_ok:
            p_cal_ch1 = 10.0 ** (float(self.ch1_ref_carrier_spectrum.max()) / 10.0)
            try:
                i1n = self._harmonic_index(ch1_harm_num)
                beta1 = _solve(float(self.ch1_ref_spectra[i1n].max()), p_cal_ch1, abs(ch1_harm_num))
            except ValueError:
                pass

        beta2 = None
        if ch2_ok:
            p_cal_ch2 = 10.0 ** (float(self.ch2_ref_carrier_spectrum.max()) / 10.0)
            try:
                i2n = self._harmonic_index(ch2_harm_num)
                beta2 = _solve(float(self.ch2_ref_spectra[i2n].max()), p_cal_ch2, ch2_bessel_num)
            except ValueError:
                pass

        return beta1, beta2

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_peak_powers(
        self,
        normalize: bool = False,
        x_axis: str = 'drive_freq',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
        show_error: bool = False,
        show_points: bool = False,
        harmonics: list | None = None,
        show_line_markers: bool = True,
        polar: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot peak sideband power vs the swept parameter.

        Parameters
        ----------
        normalize : bool or str
            False    → raw dBm.
            True     → dBc relative to per-step calibration carrier.
            'percent'→ linear fraction of carrier power, in %.
        x_axis : str
            'drive_freq', 'ch1_power', 'ch2_power', 'ch1_phase', 'ch2_phase'.
        show_error : bool
            If True and n_sweep_repeats > 1, shade ± 1 std across repeats
            around each mean curve. Ignored when only one repeat is available.
        show_points : bool
            If True and n_sweep_repeats > 1, scatter individual per-repeat
            points behind the mean curve. Ignored when only one repeat is
            available.
        polar : bool
            If True, plot in polar form instead of cartesian: angle = the
            swept phase (x_axis must be 'ch1_phase' or 'ch2_phase'), radius
            = the sideband power/CE value. Axes are forced square (diameter
            = max(axes_width_mm, axes_height_mm)). Only the radial
            (constant-value) gridlines are shown; the angular
            (constant-phase) rays are suppressed.
        """
        if polar and x_axis not in ('ch1_phase', 'ch2_phase'):
            raise ValueError(
                f"polar=True requires x_axis to be a phase ('ch1_phase' or 'ch2_phase'); got {x_axis!r}"
            )
        if normalize == 'percent':
            peaks  = 10.0 ** (self.normalized_peak_powers_dbm() / 10.0) * 100.0
            ylabel = 'Sideband power [% of carrier]'
        elif normalize:
            peaks  = self.normalized_peak_powers_dbm()
            ylabel = 'Sideband power [dBc]'
        else:
            peaks  = self.peak_powers_dbm()
            ylabel = 'Peak sideband power [dBm]'
        x, xlabel = self._x_values(x_axis)
        x_plot = np.deg2rad(x) if polar else x

        # Per-repeat arrays for error shading / scatter — prefer spectra_all (whole
        # repeats), fall back to spectra_avgs (per-point averages) for scatter.
        peaks_all = None      # (R, M, N) from sweep repeats
        peaks_avgs = None     # (M, N, A) from per-point averages
        if (show_error or show_points) and self.spectra_all is not None:
            if normalize == 'percent':
                peaks_all = 10.0 ** (self.normalized_peak_powers_all_dbm() / 10.0) * 100.0
            elif normalize:
                peaks_all = self.normalized_peak_powers_all_dbm()
            else:
                peaks_all = self.peak_powers_all_dbm()
            # shape: (R, M, N)
        elif show_points and self.spectra_avgs is not None:
            # spectra_avgs: (M, N, A, K) → peak over K → (M, N, A)
            raw_peaks = self.spectra_avgs.max(axis=3)  # (M, N, A)
            if normalize == 'percent':
                cal_db = self.cal_peak_power_dbm()     # (M,)
                raw_dbc = raw_peaks - cal_db[:, np.newaxis, np.newaxis]
                peaks_avgs = 10.0 ** (raw_dbc / 10.0) * 100.0
            elif normalize:
                cal_db = self.cal_peak_power_dbm()
                peaks_avgs = raw_peaks - cal_db[:, np.newaxis, np.newaxis]
            else:
                peaks_avgs = raw_peaks

        peaks_std = peaks_all.std(axis=0) if (show_error and peaks_all is not None) else None

        _show_set = set(harmonics) if harmonics is not None else None
        extra_iter = iter(_EXTRA_COLORS)
        if polar:
            diameter_mm = max(axes_width_mm, axes_height_mm)
            mm = 1.0 / 25.4
            fig = plt.figure(figsize=(diameter_mm * mm, diameter_mm * mm))
            ax = fig.add_subplot(111, projection='polar')
        else:
            fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        for j, n in enumerate(self.harmonics):
            if _show_set is not None and int(n) not in _show_set:
                continue
            color = _HARMONIC_COLORS.get(int(n), next(extra_iter, '#000000'))
            if show_points and peaks_all is not None:
                x_rep = np.tile(x_plot, (self.n_repeats, 1)).ravel()   # (R*M,)
                y_rep = peaks_all[:, :, j].ravel()                 # (R*M,)
                ax.scatter(x_rep, y_rep, color=color, alpha=0.25,
                           s=8, linewidths=0, zorder=1)
            elif show_points and peaks_avgs is not None:
                # A points per step: repeat each x[i] A times
                x_rep = np.repeat(x_plot, self.n_averages_per_point)   # (M*A,)
                y_rep = peaks_avgs[:, j, :].ravel()               # (M*A,)
                ax.scatter(x_rep, y_rep, color=color, alpha=0.25,
                           s=8, linewidths=0, zorder=1)
            _mk = {'marker': 'o', 'markersize': 3} if show_line_markers else {}
            ax.plot(
                x_plot, peaks[:, j],
                color=color, linewidth=1.5,
                label=f'Harmonic {n}', zorder=2,
                **_mk,
            )
            if peaks_std is not None:
                ax.fill_between(
                    x_plot,
                    peaks[:, j] - peaks_std[:, j],
                    peaks[:, j] + peaks_std[:, j],
                    color=color, alpha=0.2, linewidth=0, zorder=1,
                )

        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        if polar:
            ax.xaxis.grid(False)   # no rays of constant phase
            ax.yaxis.grid(True)    # rings of constant value
        else:
            ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.legend(fontsize=tick_label_fontsize, frameon=False)
        _style_axes(ax)
        return fig, ax

    def plot_single_harmonic(
        self,
        harmonic: int,
        normalize: bool = False,
        x_axis: str = 'drive_freq',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot one harmonic's peak power vs the swept parameter.

        Parameters
        ----------
        harmonic : int
            Harmonic number to plot.
        normalize : bool or str
            False → dBm, True → dBc, 'percent' → % of carrier.
        x_axis : str
            Sweep parameter to use as x-axis.
        """
        idx = self._harmonic_index(harmonic)
        if normalize == 'percent':
            peaks  = 10.0 ** (self.normalized_peak_powers_dbm() / 10.0) * 100.0
            ylabel = f'Harmonic {harmonic} power [% of carrier]'
        elif normalize:
            peaks  = self.normalized_peak_powers_dbm()
            ylabel = f'Harmonic {harmonic} power [dBc]'
        else:
            peaks  = self.peak_powers_dbm()
            ylabel = f'Harmonic {harmonic} power [dBm]'
        x, xlabel = self._x_values(x_axis)
        color = _HARMONIC_COLORS.get(harmonic, BLUE2)

        fig, ax = _make_figure(axes_width_mm, axes_height_mm)
        ax.plot(x, peaks[:, idx], color=color, linewidth=1.5, marker='o', markersize=3)
        if ymin is not None or ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        _style_axes(ax)
        return fig, ax

    def plot_calibration(
        self,
        x_axis: str = 'drive_freq',
        axes_width_mm: float = _default_axes_w,
        axes_height_mm: float = _default_axes_h,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the per-step calibration carrier power (RF off) vs swept parameter."""
        x, xlabel = self._x_values(x_axis)
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
