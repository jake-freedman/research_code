"""
BNC 855B two-channel output power stability test.

Both channels are enabled simultaneously. Each iteration the CXA sweeps a
narrow window around each channel's frequency in turn, records the peak
power, and prints it. After all iterations the script plots the measured
power of both channels vs iteration.
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from bnc_control import BNC855B
from cxa_control import CXA
from graphics import (
    BLUE2, RED2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ── instruments ───────────────────────────────────────────────────────────────
BNC_RESOURCE = 'USB0::0x03EB::0xAFFF::6B5-0B4F2000B-0989::INSTR'
CXA_RESOURCE = 'TCPIP0::169.254.222.67::hislip0::INSTR'

# ── channel settings ──────────────────────────────────────────────────────────
CH1_FREQ_HZ   = 1.146e9   # Hz
CH1_POWER_DBM = 0              # dBm

CH2_FREQ_HZ   = 2 * CH1_FREQ_HZ   # Hz
CH2_POWER_DBM = -10.0              # dBm

# ── test settings ─────────────────────────────────────────────────────────────
N_ITERATIONS = 50       # number of measurement cycles

SETTLE_S = 0.05         # wait after enabling outputs before first sweep

# ── CXA settings (shared for both channels) ───────────────────────────────────
CXA_SPAN_HZ   = 10e6   # narrow span around each tone
CXA_RBW_HZ    = 1e6    # resolution bandwidth
CXA_REF_LEVEL = 10.0   # reference level (dBm)
CXA_ATTN_DB   = 0.0    # input attenuation (dB)

# ── save ──────────────────────────────────────────────────────────────────────
SAVE = True
SAVE_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5"

# ── graphics ──────────────────────────────────────────────────────────────────
axes_width_mm  = _default_axes_w
axes_height_mm = _default_axes_h
left_mm   = _left_mm
right_mm  = _right_mm
bottom_mm = _bottom_mm
top_mm    = _top_mm
markersize = 4.0
# ─────────────────────────────────────────────────────────────────────────────


def _configure_for(cxa: CXA, center_hz: float) -> None:
    """Re-point the CXA to a narrow window around center_hz."""
    cxa.configure(
        start_freq  = center_hz - CXA_SPAN_HZ / 2,
        stop_freq   = center_hz + CXA_SPAN_HZ / 2,
        freq_step   = CXA_RBW_HZ,
        res_bw      = CXA_RBW_HZ,
        ref_level   = CXA_REF_LEVEL,
        attenuation = CXA_ATTN_DB,
        detector    = 'POS',
    )


def _peak_dbm(cxa: CXA) -> float:
    _, power_db = cxa.sweep()
    return float(power_db.max())


def main():
    ch1_measured = np.full(N_ITERATIONS, np.nan)
    ch2_measured = np.full(N_ITERATIONS, np.nan)
    ch1_times    = np.full(N_ITERATIONS, np.nan)
    ch2_times    = np.full(N_ITERATIONS, np.nan)

    with BNC855B(BNC_RESOURCE) as sig, CXA(CXA_RESOURCE) as cxa:
        sig.configure_channel(1, freq_hz=CH1_FREQ_HZ, power_dbm=CH1_POWER_DBM)
        sig.configure_channel(2, freq_hz=CH2_FREQ_HZ, power_dbm=CH2_POWER_DBM)
        sig.enable_all_outputs()
        time.sleep(SETTLE_S)

        t0 = time.time()
        for i in range(N_ITERATIONS):
            _configure_for(cxa, CH1_FREQ_HZ)
            ch1_measured[i] = _peak_dbm(cxa)
            ch1_times[i] = time.time() - t0

            _configure_for(cxa, CH2_FREQ_HZ)
            ch2_measured[i] = _peak_dbm(cxa)
            ch2_times[i] = time.time() - t0

            print(
                f"  [{i+1:>3d}/{N_ITERATIONS}]  "
                f"ch1={ch1_measured[i]:+6.2f} dBm  "
                f"ch2={ch2_measured[i]:+6.2f} dBm"
            )

        sig.disable_all_outputs()

    # ── statistics ────────────────────────────────────────────────────────────
    for label, commanded, measured in [
        ('Ch1', CH1_POWER_DBM, ch1_measured),
        ('Ch2', CH2_POWER_DBM, ch2_measured),
    ]:
        print(f"\n{label}  (commanded {commanded:+.1f} dBm)")
        print(f"  mean  = {np.nanmean(measured):+.3f} dBm")
        print(f"  std   = {np.nanstd(measured):.4f} dBm")
        print(f"  range = {np.nanmax(measured) - np.nanmin(measured):.4f} dBm")

    # ── save ──────────────────────────────────────────────────────────────────
    if SAVE:
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        fname = os.path.join(SAVE_FOLDER, f'power_stability_{ts}.npz')
        np.savez(
            fname,
            ch1_freq_hz=CH1_FREQ_HZ, ch1_power_dbm=CH1_POWER_DBM,
            ch2_freq_hz=CH2_FREQ_HZ, ch2_power_dbm=CH2_POWER_DBM,
            ch1_measured=ch1_measured, ch2_measured=ch2_measured,
            ch1_times=ch1_times, ch2_times=ch2_times,
        )
        print(f"\nSaved: {fname}")

    # ── figure ────────────────────────────────────────────────────────────────
    iterations = np.arange(1, N_ITERATIONS + 1)

    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    ax.plot(iterations, ch1_measured,
            color=BLUE2, linewidth=0.8, marker='o',
            markersize=markersize, markeredgewidth=0,
            label=f'Ch1  {CH1_FREQ_HZ/1e9:.3f} GHz  '
                  f'(cmd {CH1_POWER_DBM:+.0f} dBm, '
                  f'σ={np.nanstd(ch1_measured):.3f} dBm)')
    ax.plot(iterations, ch2_measured,
            color=RED2, linewidth=0.8, marker='o',
            markersize=markersize, markeredgewidth=0,
            label=f'Ch2  {CH2_FREQ_HZ/1e9:.3f} GHz  '
                  f'(cmd {CH2_POWER_DBM:+.0f} dBm, '
                  f'σ={np.nanstd(ch2_measured):.3f} dBm)')

    ax.axhline(CH1_POWER_DBM, color=BLUE2, linewidth=0.6,
               linestyle='--', alpha=0.45, zorder=1)
    ax.axhline(CH2_POWER_DBM, color=RED2, linewidth=0.6,
               linestyle='--', alpha=0.45, zorder=1)

    ax.set_xlabel('Iteration', fontsize=axis_label_fontsize)
    ax.set_ylabel('Measured power [dBm]', fontsize=axis_label_fontsize)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=tick_label_fontsize, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    plt.show()


if __name__ == '__main__':
    main()
