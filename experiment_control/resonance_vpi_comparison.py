"""
Compare Vπ vs frequency for two heterodyne sweeps taken around resonances at
f and ~2f, and report the RF power required to reach target modulation depths.

Loads two .npz files produced by vna_cw_heterodyne_sweep() in
vna_cw_harmonic_esa_script.py. The VNA drive power used during each sweep
must be provided below (it is not stored in the data file).
"""

import numpy as np
import matplotlib.pyplot as plt
from heterodyne_sweep_data import HeterodyneSweepData
from path_utils import local_path
from graphics import (
    BLUE2, RED2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

# FILE_F  = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\testheterodyne_sweep_2026-06-12-11-42-37.npz"
# FILE_2F = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\testheterodyne_sweep_2026-06-12-11-40-02.npz"
FILE_F  = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\testheterodyne_sweep_2026-06-15-19-22-05.npz"
FILE_2F = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\phase_to_amplitude_modulation\data\testheterodyne_sweep_2026-06-15-19-23-03.npz"

# VNA drive power used during each sweep (dBm). Must be set by the user.
DRIVE_POWER_F_DBM  = 10
DRIVE_POWER_2F_DBM = 10

# Target modulation depths to hit
TARGET_BETA_F  = 1.94   # desired β at f
TARGET_BETA_2F = 0.88   # desired β at 2f

# Harmonic pair used for β extraction (J_num / J_den)
HARMONIC_NUMERATOR   = 1
HARMONIC_DENOMINATOR = 0

# Y-axis limits for Vπ plot (V_rms). None = auto.
VPI_YMIN = None
VPI_YMAX = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_LOG20 = 10.0 * np.log10(20.0)   # ≈ 13.01 dB


def _dbm_to_vrms(power_dbm: float) -> float:
    """Convert dBm (50 Ω sinusoid) to RMS voltage in volts."""
    return 10.0 ** ((power_dbm - _LOG20) / 20.0)


def _vrms_to_dbm(v_rms: np.ndarray) -> np.ndarray:
    """Convert RMS voltage in volts (50 Ω sinusoid) to dBm."""
    return 20.0 * np.log10(v_rms) + _LOG20


def _make_figure(axes_width_mm, axes_height_mm):
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
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(spine_linewidth)


def compute_vpi(data: HeterodyneSweepData, drive_power_dbm: float) -> np.ndarray:
    """
    Compute Vπ (V_rms, 50 Ω) at each CW frequency.

    Vπ(f) = V_drive_rms * π / β(f)
    """
    betas = data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR)
    v_drive = _dbm_to_vrms(drive_power_dbm)
    return v_drive * np.pi / betas


def required_power(
    data: HeterodyneSweepData,
    drive_power_dbm: float,
    target_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (power_dbm, power_mw) arrays — the RF power needed at each CW
    frequency to achieve target_beta.

    Scales quadratically from the measured β: P_target = P0 * (β_target/β)².
    """
    betas = data.modulation_depth(HARMONIC_NUMERATOR, HARMONIC_DENOMINATOR)
    p_target_dbm = drive_power_dbm + 20.0 * np.log10(target_beta / betas)
    p_target_mw  = 10.0 ** (p_target_dbm / 10.0)
    return p_target_dbm, p_target_mw


def report_required_power(
    label: str,
    freqs: np.ndarray,
    p_dbm: np.ndarray,
    p_mw: np.ndarray,
    target_beta: float,
) -> None:
    """Print the minimum required power and the frequency at which it occurs."""
    idx_min = np.argmin(p_dbm)
    print(
        f"  {label}: β = {target_beta}  →  "
        f"min power = {p_dbm[idx_min]:.2f} dBm  "
        f"({p_mw[idx_min]:.4f} mW)  "
        f"at {freqs[idx_min] / 1e9:.4f} GHz"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    data_f  = HeterodyneSweepData.from_file(local_path(FILE_F))
    data_2f = HeterodyneSweepData.from_file(local_path(FILE_2F))

    vpi_f  = compute_vpi(data_f,  DRIVE_POWER_F_DBM)
    vpi_2f = compute_vpi(data_2f, DRIVE_POWER_2F_DBM)

    p_dbm_f,  p_mw_f  = required_power(data_f,  DRIVE_POWER_F_DBM,  TARGET_BETA_F)
    p_dbm_2f, p_mw_2f = required_power(data_2f, DRIVE_POWER_2F_DBM, TARGET_BETA_2F)

    # ---- Summary printout ----
    print("Required RF power to reach target modulation depths:")
    report_required_power("f  ", data_f.cw_freqs,  p_dbm_f,  p_mw_f,  TARGET_BETA_F)
    report_required_power("2f ", data_2f.cw_freqs, p_dbm_2f, p_mw_2f, TARGET_BETA_2F)

    # ---- Plot Vπ vs frequency ----
    fig, ax = _make_figure(_default_axes_w, _default_axes_h)
    ax.plot(2 * data_f.cw_freqs / 1e9, vpi_f,
            color=BLUE2, linewidth=1.5, marker='o', markersize=3,
            label=f'f sweep (axis ×2, {data_f.cw_freqs[0]/1e9:.3f}–{data_f.cw_freqs[-1]/1e9:.3f} GHz)')
    ax.plot(data_2f.cw_freqs / 1e9, vpi_2f,
            color=RED2,  linewidth=1.5, marker='o', markersize=3,
            label=f'2f sweep ({data_2f.cw_freqs[0]/1e9:.3f}–{data_2f.cw_freqs[-1]/1e9:.3f} GHz)')
    if VPI_YMIN is not None or VPI_YMAX is not None:
        ax.set_ylim([VPI_YMIN, VPI_YMAX])
    ax.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
    ax.set_ylabel('$V_\\pi$ [V$_\\mathrm{rms}$]', fontsize=axis_label_fontsize)
    ax.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_axes(ax)

    # ---- Plot required power vs frequency ----
    fig2, ax2 = _make_figure(_default_axes_w, _default_axes_h)
    ax2.plot(2 * data_f.cw_freqs / 1e9, p_dbm_f,
             color=BLUE2, linewidth=1.5, marker='o', markersize=3,
             label=f'f (axis ×2, β = {TARGET_BETA_F})')
    ax2.plot(data_2f.cw_freqs / 1e9, p_dbm_2f,
             color=RED2,  linewidth=1.5, marker='o', markersize=3,
             label=f'2f (β = {TARGET_BETA_2F})')
    ax2.set_xlabel('CW drive frequency [GHz]', fontsize=axis_label_fontsize)
    ax2.set_ylabel('Required RF power [dBm]', fontsize=axis_label_fontsize)
    ax2.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_axes(ax2)

    plt.show()


if __name__ == '__main__':
    main()
