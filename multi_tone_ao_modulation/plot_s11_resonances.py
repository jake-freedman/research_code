import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.signal import find_peaks

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / 'S11_Data_Sweep'
PROMINENCE_DB  = 0.3    # minimum dip depth (dB) to count as a resonance
MIN_SEP_HZ     = 20e6   # minimum separation between two resonances
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm    = 150.0
left_mm, right_mm   = 25.0, 12.0
bottom_mm, top_mm   = 15.0,  5.0
row_height_mm    = 5.0      # height per device row
spine_linewidth  = 2.0
tick_width       = 2.0
tick_direction   = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize =  7.0
marker_size      = 4.0
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


def find_resonances(freq: np.ndarray, s11_db: np.ndarray) -> np.ndarray:
    """Return frequencies of local minima in S11 with sufficient prominence."""
    df = np.median(np.diff(freq))
    min_dist_pts = max(1, int(MIN_SEP_HZ / df))
    peaks, _ = find_peaks(-s11_db, prominence=PROMINENCE_DB,
                          distance=min_dist_pts)
    return freq[peaks]


def device_label(path: Path) -> str:
    """Strip die prefix and timestamp, return compact label."""
    stem = path.stem
    # remove trailing timestamp  _YYYY-MM-DD-HH-MM
    stem = re.sub(r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$', '', stem)
    # remove leading die identifier
    stem = re.sub(r'^die\d+-\d+_', '', stem)
    return stem


# ── load files, deduplicate by device label (keep latest timestamp) ───────────
files = sorted(DATA_DIR.glob('*.s2p'))
device_files: dict[str, Path] = {}
for f in files:
    label = device_label(f)
    device_files[label] = f   # sorted() ensures latest timestamp wins

devices = sorted(device_files.keys())
n_devices = len(devices)

# ── find resonances ───────────────────────────────────────────────────────────
resonances: dict[str, np.ndarray] = {}
for label in devices:
    freq, s11 = parse_s2p(device_files[label])
    resonances[label] = find_resonances(freq, s11)

# ── build figure ──────────────────────────────────────────────────────────────
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + n_devices * row_height_mm + top_mm

fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
fig.subplots_adjust(
    left   = left_mm  / fig_w,
    right  = 1 - right_mm / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm   / fig_h,
)

cmap = cm.viridis
norm = Normalize(vmin=0, vmax=n_devices - 1)

for dev_idx, label in enumerate(devices):
    color = cmap(norm(dev_idx))
    for f_res in resonances[label]:
        ax.plot(f_res / 1e9, dev_idx,
                marker='o', markersize=marker_size,
                color=color, linestyle='none', markeredgewidth=0)
        if 1.0e9 <= f_res <= 1.2e9:
            ax.plot(2 * f_res / 1e9, dev_idx,
                    marker='o', markersize=marker_size * 2,
                    color='none', markeredgecolor='black',
                    markeredgewidth=1.0, linestyle='none')

# y-axis: one tick per device, label with device name
ax.set_yticks(range(n_devices))
ax.set_yticklabels(devices, fontsize=tick_label_fontsize)
ax.set_ylim(-0.5, n_devices - 0.5)
ax.invert_yaxis()

ax.set_xlabel('Resonance frequency (GHz)', fontsize=axis_label_fontsize)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

# colorbar showing device index
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label('Device index', fontsize=axis_label_fontsize)
cbar.ax.tick_params(labelsize=tick_label_fontsize)

for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = DATA_DIR / 's11_resonances.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
