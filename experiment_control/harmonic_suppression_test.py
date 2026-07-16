"""
BNC 855B harmonic suppression test.

Drives the BNC signal generator (channel 1) at a fixed frequency through a
range of output powers and records a narrow ESA spectrum centred on each
requested frequency window:
  - harmonics  (1, 2, 3, …) : n * f_drive
  - sub-harmonics (2, 3, …) : f_drive / m   (optional)

The SG output is connected directly to the ESA — no optical or heterodyne path.
Data is saved as an .npz file and can be re-analysed with plot_harmonic_suppression().
"""

from bnc_control import BNC855B
from esa_control import ESA
from cxa_control import CXA
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
    DARKBLUE2, TAN2, PINK2, DARKGRAY2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_w, axes_height_mm as _default_h,
    left_mm as _left, right_mm as _right,
    bottom_mm as _bottom, top_mm as _top,
)

BNC_RESOURCE_STRING = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
ESA_RESOURCE_STRING = 'TCPIP0::169.254.216.47::INSTR'
CXA_RESOURCE_STRING = 'TCPIP0::169.254.222.67::hislip0::INSTR'

DATA_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\signal_generator_harmonic_suppression"

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2,
           DARKGREEN2, DARKBLUE2, TAN2, PINK2, DARKGRAY2]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def voltage_linspace(p_start_dbm: float, p_stop_dbm: float, n: int) -> np.ndarray:
    """
    Return n powers in dBm that are evenly spaced in RMS voltage (50 Ω sinusoid).

    Parameters
    ----------
    p_start_dbm, p_stop_dbm : float
        Power bounds in dBm.
    n : int
        Number of steps.
    """
    _log20 = 10.0 * np.log10(20.0)
    v_start = 10.0 ** ((p_start_dbm - _log20) / 20.0)
    v_stop  = 10.0 ** ((p_stop_dbm  - _log20) / 20.0)
    v_rms = np.linspace(v_start, v_stop, n)
    return 20.0 * np.log10(v_rms) + _log20


def _dbm_to_vrms(power_dbm: np.ndarray) -> np.ndarray:
    return 10.0 ** ((power_dbm - 10.0 * np.log10(20.0)) / 20.0)


def _make_figure(axes_w=_default_w, axes_h=_default_h):
    mm = 1.0 / 25.4
    fig_w = _left + axes_w + _right
    fig_h = _bottom + axes_h + _top
    fig, ax = plt.subplots(figsize=(fig_w * mm, fig_h * mm))
    fig.subplots_adjust(
        left   = _left   / fig_w,
        right  = 1 - _right  / fig_w,
        bottom = _bottom / fig_h,
        top    = 1 - _top    / fig_h,
    )
    return fig, ax


def _style_ax(ax):
    ax.tick_params(direction=tick_direction, width=tick_width,
                   labelsize=tick_label_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def bnc_harmonic_suppression_sweep(
    drive_freq: float,
    drive_powers,
    harmonics=(1, 2, 3),
    sub_harmonics=(),
    window_hz: float = 2e6,
    esa_freq_step: float = 0.25e6,
    esa_res_bw: float = 10e3,
    esa_ref_level: float = 0.0,
    esa_attenuation: float = 30.0,
    settle_time_s: float = 0.1,
    optional_name: str = '',
    use_cxa: bool = False,
    plot: bool = True,
) -> str:
    """
    Step the BNC 855B through output powers and record ESA spectra at each
    harmonic / sub-harmonic of the drive frequency.

    Parameters
    ----------
    drive_freq : float
        BNC channel 1 drive frequency in Hz.
    drive_powers : array-like
        BNC channel 1 output powers in dBm.
    harmonics : sequence of int
        Integer harmonic orders to record.  n=1 is the fundamental.
        Centre frequency for order n is n * drive_freq.  Default (1, 2, 3).
    sub_harmonics : sequence of int
        Sub-harmonic denominators to record.  m=2 gives drive_freq/2, etc.
        Pass an empty tuple (default) to skip sub-harmonics.
    window_hz : float
        Half-width of each ESA window in Hz.  Default 2 MHz.
    esa_freq_step : float
        Frequency step within each window in Hz.  Default 250 kHz.
    esa_res_bw : float
        ESA resolution bandwidth in Hz.  Default 10 kHz.
    esa_ref_level : float
        ESA reference level in dBm.  Default 0.
    settle_time_s : float
        Wait time after setting each power level.  Default 0.1 s.
    optional_name : str
        Label prepended to the saved filename.
    use_cxa : bool
        If True, use the Keysight CXA instead of the R&S ESA.  Default False.
    plot : bool
        If True, call plot_harmonic_suppression() after saving.  Default True.

    Returns
    -------
    str
        Full path to the saved .npz file.
    """
    drive_powers  = np.asarray(drive_powers)
    harmonics     = list(harmonics)
    sub_harmonics = list(sub_harmonics)

    # Build ordered list of (label, center_freq) for each measurement window.
    windows = [(f'n={n}', float(n * drive_freq)) for n in harmonics]
    windows += [(f'f/{m}', float(drive_freq / m)) for m in sub_harmonics]
    labels      = [w[0] for w in windows]
    center_freqs = np.array([w[1] for w in windows])

    sub_note = f', sub-harmonics {sub_harmonics}' if sub_harmonics else ''
    print(
        f"Starting harmonic suppression sweep: {len(drive_powers)} steps "
        f"({drive_powers[0]:+.1f} to {drive_powers[-1]:+.1f} dBm) "
        f"at {drive_freq / 1e9:.4f} GHz"
    )
    print(f"  Windows: {labels}{sub_note}")

    os.makedirs(DATA_FOLDER, exist_ok=True)
    fname = (
        f'{optional_name}harmonic_suppression_'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.npz'
    )
    full_path = os.path.join(DATA_FOLDER, fname)

    all_spectra = []
    offsets_hz  = None
    K           = None

    try:
        esa_cls, esa_addr = (CXA, CXA_RESOURCE_STRING) if use_cxa else (ESA, ESA_RESOURCE_STRING)
        with BNC855B(BNC_RESOURCE_STRING) as sig, esa_cls(esa_addr) as esa:
            sig.disable_all_outputs()
            sig.configure_channel(1, drive_freq, drive_powers[0])
            sig.enable_output(1)

            for i, power in enumerate(drive_powers):
                sig.set_power(1, power)
                time.sleep(settle_time_s)

                step_spectra = []
                for label, center in windows:
                    esa.configure(
                        start_freq=center - window_hz,
                        stop_freq=center + window_hz,
                        freq_step=esa_freq_step,
                        res_bw=esa_res_bw,
                        ref_level=esa_ref_level,
                        attenuation=esa_attenuation,
                    )
                    _, power_db = esa.sweep()
                    step_spectra.append(power_db)
                    if offsets_hz is None:
                        K = len(power_db)
                        offsets_hz = np.linspace(-window_hz, window_hz, K)

                all_spectra.append(step_spectra)
                print(f"Step {i + 1}/{len(drive_powers)}: {power:+.1f} dBm done.")

            sig.disable_output(1)

    except Exception as exc:
        print(f"ERROR at step {len(all_spectra) + 1}/{len(drive_powers)}: {exc}")
        if not all_spectra:
            raise
        print(f"Saving partial data ({len(all_spectra)} of {len(drive_powers)} steps)...")

    n_done      = len(all_spectra)
    spectra_arr = np.array([[s[:K] for s in row] for row in all_spectra])  # (M, N, K)

    np.savez_compressed(
        full_path,
        drive_freq   = np.array(drive_freq),
        drive_powers = drive_powers[:n_done],
        harmonics    = np.array(harmonics),
        sub_harmonics= np.array(sub_harmonics),
        center_freqs = center_freqs,
        labels       = np.array(labels),
        window_hz    = np.array(window_hz),
        esa_freq_step_hz = np.array(esa_freq_step),
        offsets_hz   = offsets_hz,
        spectra      = spectra_arr,
    )

    print(f"Done. Saved {n_done}/{len(drive_powers)} steps to {full_path}")
    if plot:
        plot_harmonic_suppression(full_path)
        plt.show()
    return full_path


# ─────────────────────────────────────────────────────────────────────────────
# Analysis / plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_harmonic_suppression(
    filepath: str,
    x_axis: str = 'dbm',
    ymin_power: float | None = None,
    ymax_power: float | None = None,
    ymin_suppress: float | None = None,
    ymax_suppress: float | None = None,
):
    """
    Load a harmonic suppression .npz file and produce two figures:

    1. Absolute peak power [dBm] vs drive level for every window.
    2. Suppression relative to the fundamental (n=1) [dBc], if n=1 was recorded.

    Parameters
    ----------
    filepath : str
        Path to the .npz file produced by bnc_harmonic_suppression_sweep().
    x_axis : str
        'dbm' (default) or 'voltage' (V_rms).
    ymin_power, ymax_power : float or None
        Y-axis limits for the absolute power plot.
    ymin_suppress, ymax_suppress : float or None
        Y-axis limits for the suppression plot.
    """
    d = np.load(filepath, allow_pickle=True)
    drive_powers  = d['drive_powers']
    spectra       = d['spectra']               # (M, N, K)
    labels        = [str(l) for l in d['labels']]
    center_freqs  = d['center_freqs']
    drive_freq    = float(d['drive_freq'])

    peak_powers = spectra.max(axis=2)          # (M, N) in dBm

    if x_axis == 'dbm':
        x      = drive_powers
        xlabel = 'Drive power [dBm]'
    else:
        x      = _dbm_to_vrms(drive_powers)
        xlabel = r'Drive voltage [V$_\mathrm{rms}$]'

    print(f"Loaded: {filepath}")
    print(f"  Drive freq   : {drive_freq / 1e9:.4f} GHz")
    print(f"  Drive powers : {drive_powers[0]:+.1f} to {drive_powers[-1]:+.1f} dBm "
          f"({len(drive_powers)} steps)")
    print(f"  Windows      : {labels}")

    # ── Figure 1: absolute peak power ────────────────────────────────────────
    fig1, ax1 = _make_figure()
    for j, label in enumerate(labels):
        color     = _COLORS[j % len(_COLORS)]
        freq_ghz  = center_freqs[j] / 1e9
        ax1.plot(x, peak_powers[:, j], color=color, linewidth=1.5,
                 marker='o', markersize=3,
                 label=f'{label}  ({freq_ghz:.4f} GHz)')
    if ymin_power is not None or ymax_power is not None:
        ax1.set_ylim(ymin_power, ymax_power)
    ax1.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax1.set_ylabel('Peak power [dBm]', fontsize=axis_label_fontsize)
    ax1.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_ax(ax1)

    # ── Figure 2: suppression relative to fundamental ────────────────────────
    fund_idx = next((j for j, lbl in enumerate(labels) if lbl == 'n=1'), None)
    if fund_idx is not None:
        fig2, ax2 = _make_figure()
        for j, label in enumerate(labels):
            if j == fund_idx:
                continue
            color    = _COLORS[j % len(_COLORS)]
            freq_ghz = center_freqs[j] / 1e9
            ax2.plot(x, peak_powers[:, j] - peak_powers[:, fund_idx],
                     color=color, linewidth=1.5, marker='o', markersize=3,
                     label=f'{label}  ({freq_ghz:.4f} GHz)')
        if ymin_suppress is not None or ymax_suppress is not None:
            ax2.set_ylim(ymin_suppress, ymax_suppress)
        ax2.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax2.set_ylabel('Suppression relative to n=1 [dBc]',
                       fontsize=axis_label_fontsize)
        ax2.legend(fontsize=tick_label_fontsize, frameon=False)
        _style_ax(ax2)

    return fig1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    drive_powers = np.linspace(-20, 26, 40)

    bnc_harmonic_suppression_sweep(
        drive_freq    = 1.130e9,
        drive_powers  = drive_powers,
        harmonics     = (1, 2, 3, 4),
        sub_harmonics = (2, 3),
        window_hz     = 2e6,
        esa_freq_step = 2e6 / 1001,
        esa_res_bw    = 10e3,
        esa_ref_level = 0,
        settle_time_s = 0.05,
        optional_name = '',
        use_cxa       = True,
    )


if __name__ == '__main__':
    main()
