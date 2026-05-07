"""
2-D phasor diagram of a phase-modulator comb spectrum.

Each combline n is drawn as a stick + ball in the complex plane,
emanating from the origin.  One combline can be highlighted as fully
opaque; all others are rendered at a user-specified opacity.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.special import jv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BETA         = 5       # PM modulation depth [rad]
N_MAX        = 6       # show harmonics -N_MAX … +N_MAX

# Optional polynomial dispersion phase
DISPERSION_BETA2 = 0.0
DISPERSION_BETA3 = 0.0

# Random dispersion phase
RANDOM_DISPERSION = False
RANDOM_SEED       = 42

# dB stick length
USE_DB   = True
DB_FLOOR = -40.0

# Highlighted combline (set to None to make all lines equally opaque)
HIGHLIGHT_N   = 0      # harmonic index to draw fully opaque
OTHER_ALPHA   = 0.20   # opacity for all other comblines

# Save path — set to None to skip saving; extension is forced to .svg
SAVE_PATH = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\comb_plot"

# Geometry — axes size in mm; margins in inches
AX_W_MM   = 60.0
AX_H_MM   = 60.0
LEFT_IN   = 0.15
RIGHT_IN  = 0.15
BOTTOM_IN = 0.15
TOP_IN    = 0.15
STICK_LW  = 2.0
BALL_SIZE = 50

# ---------------------------------------------------------------------------
# Compute spectrum  (identical to plot_pm_comb_3d.py)
# ---------------------------------------------------------------------------
harmonics  = np.arange(-N_MAX, N_MAX + 1)
amplitudes = jv(harmonics, BETA).astype(complex)

phi_disp = (DISPERSION_BETA2 / 2) * harmonics**2 + (DISPERSION_BETA3 / 6) * harmonics**3
amplitudes *= np.exp(1j * phi_disp)

if RANDOM_DISPERSION:
    rng = np.random.default_rng(RANDOM_SEED)
    phi_rand = rng.uniform(0, 2 * np.pi, size=len(harmonics))
    amplitudes *= np.exp(1j * phi_rand)

magnitudes = np.abs(amplitudes)
if USE_DB:
    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(np.where(magnitudes > 0, magnitudes**2, 1e-30))
    lengths = np.clip((power_db - DB_FLOOR) / (-DB_FLOOR), 0, None)
else:
    lengths = magnitudes

with np.errstate(invalid="ignore"):
    unit_phase = np.where(magnitudes > 0, amplitudes / magnitudes, 0 + 0j)
display_amplitudes = lengths * unit_phase

# ---------------------------------------------------------------------------
# Colours — rainbow_r matching the 2-D spectrum plots (19-sideband normalisation)
# ---------------------------------------------------------------------------
_cmap  = matplotlib.colormaps["rainbow_r"]
_norm  = mcolors.Normalize(vmin=-N_MAX, vmax=N_MAX)
colors = [_cmap(_norm(n)) for n in harmonics]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"

mm_per_inch = 25.4
ax_w_in = AX_W_MM / mm_per_inch
ax_h_in = AX_H_MM / mm_per_inch
fig_w   = LEFT_IN + ax_w_in + RIGHT_IN
fig_h   = BOTTOM_IN + ax_h_in + TOP_IN

fig = plt.figure(figsize=(fig_w, fig_h))
ax  = fig.add_axes([LEFT_IN / fig_w, BOTTOM_IN / fig_h, ax_w_in / fig_w, ax_h_in / fig_h])

ax_max = float(np.max(lengths)) if np.max(lengths) > 0 else 1.0

def _blend_white(color, alpha):
    """Pre-multiply color against white so the ball can be drawn opaque."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)

# Draw non-highlighted comblines first (background layer)
for n, A, c in zip(harmonics, display_amplitudes, colors):
    if HIGHLIGHT_N is not None and n == HIGHLIGHT_N:
        continue
    alpha = OTHER_ALPHA if HIGHLIGHT_N is not None else 1.0
    re, im = float(A.real), float(A.imag)
    ax.plot([0, re], [0, im], color=c, linewidth=STICK_LW,
            solid_capstyle="round", alpha=alpha, zorder=2)
    ax.scatter([re], [im], color=[_blend_white(c, alpha)], s=BALL_SIZE,
               zorder=3, clip_on=True, edgecolors="none")

# Draw highlighted combline on top
if HIGHLIGHT_N is not None:
    idx = np.where(harmonics == HIGHLIGHT_N)[0]
    if len(idx):
        i = idx[0]
        A = display_amplitudes[i]
        c = colors[i]
        re, im = float(A.real), float(A.imag)
        ax.plot([0, re], [0, im], color=c, linewidth=STICK_LW,
                solid_capstyle="round", alpha=1.0, zorder=4)
        ax.scatter([re], [im], color=[c], s=BALL_SIZE, alpha=1.0,
                   zorder=5, clip_on=True, edgecolors="black", linewidths=1.0)

# Origin marker
ax.scatter([0], [0], color="black", s=10, zorder=6, clip_on=True, edgecolors="none")

# Unit circle for reference (dashed)
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta) * ax_max, np.sin(theta) * ax_max,
        color="lightgray", linewidth=1.5, linestyle="--", zorder=1)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
lim = ax_max * 1.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")

ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticks([]); ax.set_yticks([])
ax.tick_params(axis="both", which="both", length=0, labelsize=0)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

if SAVE_PATH is not None:
    import os
    save_path = os.path.splitext(SAVE_PATH)[0] + ".svg"
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

plt.show()
