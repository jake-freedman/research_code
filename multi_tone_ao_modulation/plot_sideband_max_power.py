import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path(r"D:\current_periodic_support_phase_modulator_data\die2-3_wgN11_periodic_suspension_2mm_2024-07-02-17-21-16")

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

    fig_w_in = fig_w / 25.4
    fig_h_in = fig_h / 25.4
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    for sb_dir in sb_dirs:
        idx = int(sb_dir.name.split('_')[1])
        color = SB_COLORS[idx % len(SB_COLORS)]
        drive_freqs, max_powers = load_sideband(sb_dir)
        ax.plot(drive_freqs * 1e3, max_powers,
                color=color, linewidth=1.5, marker='none', markersize=3,
                label=f'SB {idx}')

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)
    ax.set_xlabel('Drive frequency (MHz)', fontsize=axis_label_fontsize)
    ax.set_ylabel('Max power (dBm)', fontsize=axis_label_fontsize)
    ax.legend(fontsize=tick_label_fontsize, frameon=False)

    out_path = data_root / 'sideband_max_power.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
