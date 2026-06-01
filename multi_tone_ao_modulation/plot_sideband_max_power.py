import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
from scipy.special import j0, j1

DATA_ROOT = Path(r"D:\die2-3_wg19_narrow_suspended_2mm_long_2024-08-08-10-53-27\die2-3_wg19_narrow_suspended_2mm_long_2024-08-08-10-53-27")

# ── graphics.py style ────────────────────────────────────────────────────────
GREEN2      = '#93C572'
LIGHTBLUE2  = '#b2cbf2'
DARKBLUE2   = '#475c6c'
ORANGE2     = '#FBD8A2'
RED2        = '#e5a3a3'
BLUE2       = '#2522d4'
VIOLET2     = '#C2B7E9'
DARKGREEN2  = '#8DB591'
PINK2       = '#e6b8d0'
BEIGE2      = '#d9b99b'

SB_COLORS = [BLUE2, GREEN2, RED2, VIOLET2, DARKGREEN2, PINK2, BEIGE2]

axes_width_mm  = 150.0
axes_height_mm = 80.0
left_mm, right_mm   = 20.0, 5.0
bottom_mm, top_mm   = 12.0, 5.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  = 8.0
# ─────────────────────────────────────────────────────────────────────────────


def _beta_from_ratio(ratio: float) -> float:
    """Solve J1(β)/J0(β) = ratio for β using Brent's method."""
    try:
        return brentq(lambda b: j1(b) / j0(b) - ratio, 1e-6, 2.4)
    except ValueError:
        return np.nan


def compute_beta(p0_dbm: np.ndarray, p1_dbm: np.ndarray) -> np.ndarray:
    """Return β at each point from zeroth- and first-order sideband powers (dBm)."""
    p0_lin = 10 ** (p0_dbm / 10)
    p1_lin = 10 ** (p1_dbm / 10)
    ratios  = np.sqrt(p1_lin / p0_lin)
    return np.array([_beta_from_ratio(r) for r in ratios])


def parse_csv(path: Path):
    """Return (frequencies_Hz, powers_dBm) arrays from one sweep CSV."""
    lines = []
    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith('#') and line.strip():
                lines.append(line.strip())
    freqs  = np.fromstring(lines[0], sep=',')
    powers = np.fromstring(lines[1], sep=',')
    return freqs, powers


def load_sideband(sb_dir: Path):
    """Return sorted (drive_freq_GHz, max_power_dBm) arrays for one sideband."""
    pattern = re.compile(r'freq(\d+\.\d+)_ind\d+\.csv')
    drive_freqs, max_powers = [], []
    for csv_path in sb_dir.glob('*.csv'):
        m = pattern.match(csv_path.name)
        if m is None:
            continue
        drive_freq_GHz = float(m.group(1))
        _, powers = parse_csv(csv_path)
        drive_freqs.append(drive_freq_GHz)
        max_powers.append(np.max(powers))
    order = np.argsort(drive_freqs)
    return np.array(drive_freqs)[order], np.array(max_powers)[order]


def main():
    data_root = DATA_ROOT

    sb_dirs = sorted(data_root.glob('sb_*'),
                     key=lambda p: int(p.name.split('_')[1]))

    # load all sidebands
    sb_data = {}
    for sb_dir in sb_dirs:
        idx = int(sb_dir.name.split('_')[1])
        sb_data[idx] = load_sideband(sb_dir)

    # two-row figure
    between_mm = 8.0
    fig_h2 = bottom_mm + 2 * axes_height_mm + between_mm + top_mm
    fig_w_in = fig_w / 25.4
    fig_h_in = fig_h2 / 25.4
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(fig_w_in, fig_h_in))

    ax_h_frac = axes_height_mm / fig_h2
    gap_frac  = between_mm     / fig_h2
    l_frac    = left_mm        / fig_w
    r_frac    = 1 - right_mm   / fig_w
    b_frac    = bottom_mm      / fig_h2

    ax_bot.set_position([l_frac, b_frac,                   r_frac - l_frac, ax_h_frac])
    ax_top.set_position([l_frac, b_frac + ax_h_frac + gap_frac, r_frac - l_frac, ax_h_frac])

    # ── top: max power per sideband ──────────────────────────────────────────
    for idx, (drive_freqs, max_powers) in sb_data.items():
        color = SB_COLORS[idx % len(SB_COLORS)]
        ax_top.plot(drive_freqs * 1e3, max_powers,
                    color=color, linewidth=1.5,
                    label=f'SB {idx}')

    ax_top.set_ylabel('Max power (dBm)', fontsize=axis_label_fontsize)
    ax_top.legend(fontsize=tick_label_fontsize, frameon=False)
    ax_top.tick_params(labelbottom=False)

    # ── bottom: modulation depth ─────────────────────────────────────────────
    if 0 in sb_data and 1 in sb_data:
        freqs0, p0 = sb_data[0]
        freqs1, p1 = sb_data[1]
        d0 = {round(f, 7): p for f, p in zip(freqs0, p0)}
        d1 = {round(f, 7): p for f, p in zip(freqs1, p1)}
        common_keys = sorted(set(d0) & set(d1))
        common      = np.array(common_keys)
        p0_aligned  = np.array([d0[k] for k in common_keys])
        p1_aligned  = np.array([d1[k] for k in common_keys])
        beta = compute_beta(p0_aligned, p1_aligned)
        ax_bot.plot(common * 1e3, beta,
                    color=SB_COLORS[0], linewidth=1.5)

    ax_bot.set_xlabel('Drive frequency (MHz)', fontsize=axis_label_fontsize)
    ax_bot.set_ylabel('Modulation depth $\\beta$ (rad)', fontsize=axis_label_fontsize)

    for ax in (ax_top, ax_bot):
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax.tick_params(axis='both', direction=tick_direction,
                       width=tick_width, labelsize=tick_label_fontsize)

    out_path = data_root / 'sideband_max_power.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
