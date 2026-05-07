"""
3-D representation of a phase-modulator comb spectrum.

For each harmonic n the complex field amplitude is  A_n = J_n(beta) * exp(i*phi_n),
where phi_n is an optional per-harmonic phase (e.g. from dispersion).

The plot shows one "stick + ball" per harmonic:
  x  = harmonic index n        (frequency axis)
  y  = Re(A_n)                 (complex plane, real axis)
  z  = Im(A_n)                 (complex plane, imaginary axis)

The length of each stick equals |A_n| and its angle in the y-z plane equals arg(A_n).
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3-D projection)
from scipy.special import jv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BETA         = 5     # PM modulation depth [rad]
N_MAX        = 8       # show harmonics -N_MAX … +N_MAX

# Optional per-harmonic phase added on top of the bare PM spectrum.
# Set to None for a plain PM output (all amplitudes real).
# Example: quadratic dispersion -> DISPERSION_BETA2 != 0
DISPERSION_BETA2 = 0.0    # quadratic phase coefficient [rad / harmonic²]
DISPERSION_BETA3 = 0.0    # cubic  phase coefficient [rad / harmonic³]

# Global phase rotation applied to all comblines equally [rad]
GLOBAL_PHASE = np.pi/7

# Random dispersion phase (independent uniform draw per harmonic, as after
# a random dispersive element).  Set to False or None to disable.
RANDOM_DISPERSION = False
RANDOM_SEED       = 42    # integer for reproducibility, or None for random

# dB stick length: if True the stick length encodes 10*log10(|A_n|²),
# normalised so that 0 dB maps to length 1 and DB_FLOOR maps to length 0.
# The stick direction (phase) is always taken from the complex amplitude.
USE_DB   = True
DB_FLOOR = -40.0   # dB floor mapped to zero length

# Viewing angle
ELEV = 20      # elevation [deg]
AZIM = 200     # azimuth  [deg]
ROLL = 0       # roll     [deg]

# Geometry
FIG_W_IN = 5.5
FIG_H_IN = 4.2
STICK_LW  = 2.0
BALL_SIZE = 30

# Highlighted combline (set to None for uniform appearance)
HIGHLIGHT_N = 0       # harmonic index drawn fully opaque with black ball outline
OTHER_ALPHA = 0.35    # opacity for all other comblines

# Shadows / projections
SHOW_FLOOR_SHADOW    = True   # project ball tip onto z = -ax_max (x-Re plane)
SHOW_BACK_SHADOW     = False   # project ball tip onto x = -N_MAX - 0.5 (Re-Im plane)
SHOW_XAXIS_TICK      = False   # mark each harmonic position on the x-axis
SHADOW_ALPHA         = 0.15
SHADOW_BALL_SIZE     = 18
DROP_LINE_LW         = 0.8

# Reference rings — one circle per harmonic at radius = 1 (0 dB / unit amplitude)
SHOW_REFERENCE_RINGS = True
RING_ALPHA           = 0.35
RING_LW              = 0.6
RING_COLOR           = "gray"
RING_COLORED         = False   # use each combline's own color instead of RING_COLOR

SHOW_RING_POINTER    = True   # dashed line from origin to ring along each combline's phase
RING_POINTER_LW      = 0.8
RING_POINTER_ALPHA   = 0.35

# ---------------------------------------------------------------------------
# Compute spectrum
# ---------------------------------------------------------------------------
harmonics  = np.arange(-N_MAX, N_MAX + 1)
amplitudes = jv(harmonics, BETA).astype(complex)

phi_disp = (DISPERSION_BETA2 / 2) * harmonics**2 + (DISPERSION_BETA3 / 6) * harmonics**3
amplitudes *= np.exp(1j * phi_disp)

if RANDOM_DISPERSION:
    rng = np.random.default_rng(RANDOM_SEED)
    phi_rand = rng.uniform(0, 2 * np.pi, size=len(harmonics))
    amplitudes *= np.exp(1j * phi_rand)

amplitudes *= np.exp(1j * GLOBAL_PHASE)

# Compute display lengths: dB-normalised or linear
magnitudes = np.abs(amplitudes)
if USE_DB:
    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(np.where(magnitudes > 0, magnitudes**2, 1e-30))
    lengths = np.clip((power_db - DB_FLOOR) / (-DB_FLOOR), 0, None)
else:
    lengths = magnitudes

# Build display amplitudes: same phase, length replaced
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
fp = {"family": "Arial"}

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
ax  = fig.add_subplot(111, projection="3d")

ax_max = max(1.0, float(np.max(lengths))) if np.max(lengths) > 0 else 1.0
x_back = -N_MAX - 0.5
z_floor = -ax_max

# Reference line along the frequency axis
ax.plot([-N_MAX - 0.5, N_MAX + 0.5], [0, 0], [0, 0],
        color="black", linewidth=0.8, zorder=1)

# Reference rings at radius = 1 (0 dB / unit amplitude)
if SHOW_REFERENCE_RINGS:
    _theta = np.linspace(0, 2 * np.pi, 120)
    _rx, _ry = np.cos(_theta), np.sin(_theta)
    for n, c in zip(harmonics, colors):
        rc = c if RING_COLORED else RING_COLOR
        ax.plot([n] * len(_theta), _rx, _ry,
                color=rc, linewidth=RING_LW, alpha=RING_ALPHA, zorder=1)

# Ring pointers — dashed line from origin to ring along each combline's phase
if SHOW_RING_POINTER:
    for n, A, c in zip(harmonics, display_amplitudes, colors):
        mag = abs(A)
        if mag == 0:
            continue
        ure, uim = float(A.real) / mag, float(A.imag) / mag
        rc = c if RING_COLORED else RING_COLOR
        ax.plot([n, n], [0, ure], [0, uim],
                color=rc, linewidth=RING_POINTER_LW,
                linestyle="--", alpha=RING_POINTER_ALPHA, zorder=1)

# Shadows and drop-lines (drawn first so sticks render on top)
for n, A, c in zip(harmonics, display_amplitudes, colors):
    re, im = float(A.real), float(A.imag)
    rgba_s = (*mcolors.to_rgb(c), SHADOW_ALPHA)

    if SHOW_FLOOR_SHADOW:
        # drop-line: ball tip → floor
        ax.plot([n, n], [re, re], [im, z_floor],
                color="#999999", linewidth=DROP_LINE_LW, linestyle="--", zorder=2)
        ax.scatter([n], [re], [z_floor],
                   color=[rgba_s], s=SHADOW_BALL_SIZE, zorder=2, depthshade=False)

    if SHOW_BACK_SHADOW:
        # project onto the Re-Im plane at x = x_back
        ax.scatter([x_back], [re], [im],
                   color=[rgba_s], s=SHADOW_BALL_SIZE, zorder=2, depthshade=False)

    if SHOW_XAXIS_TICK:
        ax.scatter([n], [0], [0],
                   color="black", s=12, zorder=3, depthshade=False)

def _blend_white(color, alpha):
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)

# Sticks and balls — non-highlighted first
for n, A, c in zip(harmonics, display_amplitudes, colors):
    if HIGHLIGHT_N is not None and n == HIGHLIGHT_N:
        continue
    alpha = OTHER_ALPHA if HIGHLIGHT_N is not None else 1.0
    re, im = float(A.real), float(A.imag)
    ax.plot([n, n], [0, re], [0, im],
            color=c, linewidth=STICK_LW, solid_capstyle="round",
            alpha=alpha, zorder=4)
    ax.scatter([n], [re], [im],
               color=[_blend_white(c, alpha)], s=BALL_SIZE,
               zorder=5, depthshade=False, edgecolors="none")

# Highlighted combline on top
if HIGHLIGHT_N is not None:
    idx = np.where(harmonics == HIGHLIGHT_N)[0]
    if len(idx):
        i = idx[0]
        A, c = display_amplitudes[i], colors[i]
        re, im = float(A.real), float(A.imag)
        ax.plot([HIGHLIGHT_N, HIGHLIGHT_N], [0, re], [0, im],
                color=c, linewidth=STICK_LW, solid_capstyle="round",
                alpha=1.0, zorder=6)
        ax.scatter([HIGHLIGHT_N], [re], [im],
                   color=[c], s=BALL_SIZE, alpha=1.0,
                   zorder=7, depthshade=False,
                   edgecolors="black", linewidths=1.0)

# ---------------------------------------------------------------------------
# Labels and style
# ---------------------------------------------------------------------------
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

ax.set_xlim(-N_MAX - 0.5, N_MAX + 0.5)
ax.set_ylim(-ax_max, ax_max)
ax.set_zlim(z_floor, ax_max)

ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.tick_params(axis="both", which="both", length=0, labelsize=0)

ax.view_init(elev=ELEV, azim=AZIM, roll=ROLL)
ax.set_box_aspect([2.5, 1, 1])

fig.tight_layout()
plt.show()
