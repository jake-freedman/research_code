"""ring_resonator.py — single-bus optical ring resonator utilities (CMT)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from graphics import (
    BLUE2, DARKGREEN2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

mm_per_inch = 25.4


@dataclass
class RingResonator:
    """Single-bus ring resonator in the coupled-mode theory (CMT) picture.

    Parameters
    ----------
    f_0 : float
        Resonance frequency [Hz].
    gamma_i : float
        Intrinsic power decay rate [rad/s].
    gamma_e : float
        Waveguide (external) coupling power decay rate [rad/s].
    """
    f_0: float
    gamma_i: float
    gamma_e: float

    @classmethod
    def from_Q(cls, f_0: float, Q_i: float, Q_e: float) -> RingResonator:
        """Construct from resonance frequency and intrinsic / external Q factors."""
        omega_0 = 2 * np.pi * f_0
        return cls(f_0=f_0, gamma_i=omega_0 / Q_i, gamma_e=omega_0 / Q_e)

    @property
    def omega_0(self) -> float:
        """Resonance angular frequency [rad/s]."""
        return 2 * np.pi * self.f_0

    @property
    def gamma(self) -> float:
        """Total power decay rate γ = γᵢ + γₑ [rad/s]."""
        return self.gamma_i + self.gamma_e

    @property
    def Q_i(self) -> float:
        """Intrinsic Q factor."""
        return self.omega_0 / self.gamma_i

    @property
    def Q_e(self) -> float:
        """External (waveguide) Q factor."""
        return self.omega_0 / self.gamma_e

    @property
    def Q_total(self) -> float:
        """Loaded Q factor."""
        return self.omega_0 / self.gamma

    @property
    def linewidth_hz(self) -> float:
        """Total FWHM linewidth γ / (2π) [Hz]."""
        return self.gamma / (2 * np.pi)


def field_transmission_coefficient(
    ring: RingResonator,
    f: float | np.ndarray,
) -> np.ndarray:
    """Complex field transmission coefficient t(f) for the through port.

    Derived from CMT for a single-bus ring resonator:

        t(f) = [iΔω + (γᵢ − γₑ)/2] / [iΔω + (γᵢ + γₑ)/2]

    where Δω = 2π(f − f₀).

    Parameters
    ----------
    ring : RingResonator
    f    : float or array-like — probe frequency [Hz]

    Returns
    -------
    complex ndarray of the same shape as f
    """
    delta_omega = 2 * np.pi * (np.asarray(f, dtype=float) - ring.f_0)
    num = 1j * delta_omega + (ring.gamma_i - ring.gamma_e) / 2
    den = 1j * delta_omega + (ring.gamma_i + ring.gamma_e) / 2
    # return num / den
    return (2j*delta_omega / ring.omega_0 + 1 / ring.Q_i - 1/ring.Q_e) / (2j*delta_omega / ring.omega_0 + 1/ring.Q_i + 1/ring.Q_e)
    return (-ring.gamma_i + ring.gamma_e + 1j*delta_omega) / (ring.gamma_i + ring.gamma_e - 1j*delta_omega)


def power_transmission(
    ring: RingResonator,
    f: float | np.ndarray,
) -> np.ndarray:
    """Power transmission |t(f)|²."""
    return np.abs(field_transmission_coefficient(ring, f)) ** 2


def phase_transmission(
    ring: RingResonator,
    f: float | np.ndarray,
) -> np.ndarray:
    """Phase of the field transmission arg(t(f)) [rad]."""
    return np.angle(field_transmission_coefficient(ring, f))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _phase_tick_label(val: float) -> str:
    """Return a LaTeX label for a value that is a multiple of π/2."""
    n = round(val / (np.pi / 2))   # how many half-pi steps
    if n == 0:
        return "$0$"
    num, den = n, 2                 # n * π/2
    # simplify fraction
    from math import gcd
    g = gcd(abs(num), den)
    num, den = num // g, den // g
    sign = "-" if num < 0 else ""
    num = abs(num)
    if den == 1:
        return f"${sign}{num}\\pi$" if num != 1 else f"${sign}\\pi$"
    return f"${sign}\\frac{{{num}\\pi}}{{{den}}}$"


def plot_ring_transmission(
    ring: RingResonator,
    f_span: float | None = None,
    n_points: int = 2000,
    ax_w_mm: float = 70.0,
    ax_h_mm: float = 55.0,
    left_in: float = 0.60,
    mid_in: float = 0.55,
    right_in: float = 0.20,
    bottom_in: float = 0.45,
    top_in: float = 0.25,
    phase_min: float | None = 0.0,
    power_db: bool = False,
    db_floor: float = -10.0,
) -> matplotlib.figure.Figure:
    """phase_min sets the bottom of the phase y-axis window (always 2π wide).
    power_db plots power transmission in dB, clipped at db_floor."""
    """Plot power and phase transmission side by side vs. detuning_ghz from resonance.

    Parameters
    ----------
    ring      : RingResonator
    f_span    : frequency span [Hz] centred on resonance; defaults to 10× linewidth
    n_points  : number of frequency samples
    ax_w_mm   : individual axes width [mm]
    ax_h_mm   : individual axes height [mm]
    left_in   : left figure margin [in]
    mid_in    : gap between the two panels [in]
    right_in  : right figure margin [in]
    bottom_in : bottom figure margin [in]
    top_in    : top figure margin [in]

    Returns
    -------
    matplotlib.figure.Figure
    """
    if f_span is None:
        f_span = 10 * ring.linewidth_hz

    f           = np.linspace(ring.f_0 - f_span / 2, ring.f_0 + f_span / 2, n_points)
    detuning_ghz = (f - ring.f_0) / 1e9
    pwr         = power_transmission(ring, f)
    phase       = phase_transmission(ring, f)

    ax_w_in = ax_w_mm / mm_per_inch
    ax_h_in = ax_h_mm / mm_per_inch
    fig_w   = left_in + ax_w_in + mid_in + ax_w_in + right_in
    fig_h   = bottom_in + ax_h_in + top_in

    fig    = plt.figure(figsize=(fig_w, fig_h))
    ax_pwr = fig.add_axes([
        left_in / fig_w,
        bottom_in / fig_h,
        ax_w_in / fig_w,
        ax_h_in / fig_h,
    ])
    ax_phi = fig.add_axes([
        (left_in + ax_w_in + mid_in) / fig_w,
        bottom_in / fig_h,
        ax_w_in / fig_w,
        ax_h_in / fig_h,
    ])

    fp = {"family": "Arial"}

    # Power panel
    if power_db:
        pwr_plot = 10 * np.log10(np.maximum(pwr, 10 ** (db_floor / 10)))
        pwr_ylabel = "Power transmission [dB]"
        pwr_ylim   = (db_floor - 1, 1)
    else:
        pwr_plot   = pwr
        pwr_ylabel = "Power transmission $|t|^2$"
        pwr_ylim   = (-0.05, 1.05)

    ax_pwr.plot(detuning_ghz, pwr_plot, color=BLUE2, linewidth=1.5)
    ax_pwr.set_xlabel("Detuning $\\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)
    ax_pwr.set_ylabel(pwr_ylabel, fontsize=axis_label_fontsize, **fp)
    ax_pwr.set_xlim(detuning_ghz[0], detuning_ghz[-1])
    ax_pwr.set_ylim(*pwr_ylim)

    # Phase panel
    if phase_min is not None:
        phase = ((phase - phase_min) % (2 * np.pi)) + phase_min
        phase_max  = phase_min + 2 * np.pi
        tick_vals  = np.arange(phase_min, phase_max + 1e-9, np.pi / 2)
        tick_lbls  = [_phase_tick_label(v) for v in tick_vals]
        pad = 0.05
    else:
        phase_min, phase_max = phase.min(), phase.max()
        tick_vals = tick_lbls = None
        pad = 0.1

    ax_phi.plot(detuning_ghz, phase, color=DARKGREEN2, linewidth=1.5)
    ax_phi.set_xlabel("Detuning $\\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)
    ax_phi.set_ylabel("Phase $\\arg(t)$ [rad]", fontsize=axis_label_fontsize, **fp)
    ax_phi.set_xlim(detuning_ghz[0], detuning_ghz[-1])
    ax_phi.set_ylim(phase_min - pad, phase_max + pad)
    if tick_vals is not None:
        ax_phi.set_yticks(tick_vals)
        ax_phi.set_yticklabels(tick_lbls, fontsize=tick_label_fontsize)

    for ax in (ax_pwr, ax_phi):
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_zorder(100)
        ax.tick_params(
            axis="both", which="both",
            direction=tick_direction, width=tick_width,
            labelsize=tick_label_fontsize,
        )
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Arial")
        ax.grid(linewidth=0.4, zorder=0)

    return fig
