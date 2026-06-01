import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
S2P_PATH = Path(r"C:\Users\acous\code\research_code\multi_tone_ao_modulation\S11_Data_Sweep\die2-3_wg5_RU_ultranarrowperiodic_bi_500um_2024-08-20-10-00.s2p")
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
BLUE2 = '#2522d4'

axes_width_mm  = 150.0
axes_height_mm =  80.0
left_mm, right_mm  = 20.0, 5.0
bottom_mm, top_mm  = 12.0, 5.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────


def parse_s2p(path: Path):
    """Return (freq_hz, s11_dB) skipping comment/option lines."""
    freqs, s11 = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            cols = line.split()
            freqs.append(float(cols[0]))
            s11.append(float(cols[1]))
    return np.array(freqs), np.array(s11)


freq, s11 = parse_s2p(S2P_PATH)

fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

ax.plot(freq / 1e9, s11, color=BLUE2, linewidth=1.0)

ax.set_xlabel('Frequency (GHz)', fontsize=axis_label_fontsize)
ax.set_ylabel('$S_{11}$ (dB)', fontsize=axis_label_fontsize)
ax.set_title(S2P_PATH.stem, fontsize=tick_label_fontsize)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = S2P_PATH.with_suffix('.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
